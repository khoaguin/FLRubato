import importlib.util
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from syftbox.lib import Client
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from utils import (
    ProjectStateCols,
    add_public_write_permission,
    create_project_state,
    get_app_private_data,
    read_json,
    search_files,
    update_project_state,
)

DATASET_FILE_PATTERN = r"^mnist_label_[0-9]\.pt$"


# Exception name to indicate the state cannot advance
# as there are some pre-requisites that are not met
class StateNotReady(Exception):
    pass


def init_client_app(client: Client) -> None:
    """
    Creates the `fl_client` app in the `api_data` folder
    with the following structure:
    ```
    api_data
    └── fl_client
            └── request
            └── running
    ```
    """
    fl_client = client.api_data("fl_client")

    for folder in ["request", "running", "done"]:
        fl_client_folder = fl_client / folder
        fl_client_folder.mkdir(parents=True, exist_ok=True)

    # Give public write permission to the request folder
    add_public_write_permission(client, fl_client / "request")

    # We additionally create a private folder for the client to place the datasets
    private_folder_path = get_app_private_data(client, "fl_client")
    private_folder_path.mkdir(parents=True, exist_ok=True)


def init_shared_dirs(client: Client, proj_folder: Path) -> None:
    """Creates the shared directories for the project.
    These directories are shared between the client and the aggregator.
    a. round_weights
    b. agg_weights
    c. state
    """

    round_weights_folder = proj_folder / "round_weights"
    agg_weights_folder = proj_folder / "agg_weights"

    round_weights_folder.mkdir(parents=True, exist_ok=True)
    agg_weights_folder.mkdir(parents=True, exist_ok=True)

    # Give public write permission to the round_weights and agg_weights folder
    add_public_write_permission(client, agg_weights_folder)

    # Create a state folder to track progress of the project
    # and give public read permission to the state folder for the aggregator
    create_project_state(client, proj_folder)


def load_model_class(model_path: Path) -> type:
    """Load the model class from the model architecture file"""
    model_class_name = "FLModel"
    spec = importlib.util.spec_from_file_location(model_path.stem, model_path)
    model_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_arch)
    model_class = getattr(model_arch, model_class_name)

    return model_class


def train_model(proj_folder: Path, round_num: int, dataset_path_files: Path) -> None:
    """
    Trains the model for the given round number
    """

    round_weights_folder = proj_folder / "round_weights"
    agg_weights_folder = proj_folder / "agg_weights"

    fl_config_path = proj_folder / "fl_config.json"
    fl_config = read_json(fl_config_path)

    # Load the Model from the model_arch filename
    model_class = load_model_class(proj_folder / fl_config["model_arch"])
    model: nn.Module = model_class()

    # Load the aggregated weights from the previous round
    agg_weights_file = agg_weights_folder / f"agg_model_round_{round_num - 1}.pt"
    model.load_state_dict(torch.load(agg_weights_file, weights_only=True))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=fl_config["learning_rate"])

    all_datasets = []
    for dataset_path_file in dataset_path_files:
        # load the saved mnist subset
        images, labels = torch.load(str(dataset_path_file), weights_only=True)

        # create a tensordataset
        dataset = TensorDataset(images, labels)

        all_datasets.append(dataset)

    combined_dataset = ConcatDataset(all_datasets)

    # create a dataloader for the dataset
    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

    # Open log file for writing
    logs_folder_path = proj_folder / "logs"
    logs_folder_path.mkdir(parents=True, exist_ok=True)
    output_logs_path = logs_folder_path / f"training_logs_round_{round_num}.txt"
    log_file = open(str(output_logs_path), "w")

    # Log training start
    start_msg = f"[{datetime.now().isoformat()}] Starting training...\n"
    log_file.write(start_msg)
    log_file.flush()
    update_project_state(
        proj_folder,
        ProjectStateCols.MODEL_TRAIN_PROGRESS,
        f"Training Started for Round {round_num}",
    )

    # training loop
    for epoch in range(fl_config["epoch"]):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate loss
            running_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        log_msg = f"[{datetime.now().isoformat()}] Epoch {epoch + 1:04d}: Loss = {avg_loss:.6f}\n"
        log_file.write(log_msg)
        log_file.flush()  # Force write to disk
        update_project_state(
            proj_folder,
            ProjectStateCols.MODEL_TRAIN_PROGRESS,
            f"Training InProgress for Round {round_num} (Curr Epoc: {epoch}/{fl_config['epoch']})",
        )

    # Serialize the model
    output_model_path = round_weights_folder / f"trained_model_round_{round_num}.pt"
    torch.save(model.state_dict(), str(output_model_path))

    # Log completion
    final_msg = f"[{datetime.now().isoformat()}] Training completed. Final loss: {avg_loss:.6f}\n"
    log_file.write(final_msg)
    log_file.flush()
    log_file.close()
    update_project_state(
        proj_folder,
        ProjectStateCols.MODEL_TRAIN_PROGRESS,
        f"Training Completed for Round {round_num}",
    )


def shift_project_to_done_folder(
    client: Client, proj_folder: Path, total_rounds: int
) -> None:
    """
    Moves the project to the `done` folder
    a. Create a directory in the `done` folder with the same name as the project
    b. moves the agg weights and round weights to the done folder
    c. delete the project folder from the running folder
    """
    done_proj_folder = client.api_data(f"fl_client/done/{proj_folder.name}")
    done_proj_folder.mkdir(parents=True, exist_ok=True)

    # Move the agg weights and round weights folder to the done project folder
    shutil.move(proj_folder / "agg_weights", done_proj_folder)
    shutil.move(proj_folder / "round_weights", done_proj_folder)

    # Delete the project folder from the running folder
    print(f"Deleting project folder from the running folder: {proj_folder.resolve()}")
    shutil.rmtree(proj_folder)


def get_train_datasets(client: Client, proj_folder: Path) -> list[Path]:
    # Check if datasets are present in the private folder
    dataset_path = get_app_private_data(client, "fl_client")
    dataset_path_files = search_files(DATASET_FILE_PATTERN, dataset_path)

    if len(dataset_path_files) == 0:
        raise StateNotReady(
            f"⛔ No dataset found in private folder: {dataset_path.resolve()}"
            "Skipping training "
        )

    update_project_state(proj_folder, ProjectStateCols.DATASET_ADDED, True)

    return dataset_path_files


def has_project_completed(client: Client, proj_folder: Path, total_rounds: int) -> bool:
    """Check if the project has completed model training all the rounds."""

    agg_weights_folder = proj_folder / "agg_weights"
    agg_weights_cnt = len(list(agg_weights_folder.glob("*.pt")))

    # Aggregated weights folder include round weights + init seed weight
    # If aggregated weights for all rounds are present, then the project is completed
    if agg_weights_cnt == total_rounds + 1:
        print(f"FL project {proj_folder.name} has completed all the rounds ✅ 🚀")
        shift_project_to_done_folder(client, proj_folder, total_rounds)
        return True

    return False


def perform_model_training(
    client: Client,
    proj_folder: Path,
    dataset_files: list[Path],
) -> None:
    """
    Step 2: Has the aggregate sent the weights for the current round x (in the agg_weights folder)
    b. The client trains the model on the given round  and places the trained model in the round_weights folder
    c. It sends the trained model to the aggregator.
    d. repeat a until all round completes
    """
    round_weights_folder = proj_folder / "round_weights"
    agg_weights_folder = proj_folder / "agg_weights"

    fl_config_path = proj_folder / "fl_config.json"
    fl_config = read_json(fl_config_path)

    total_rounds = fl_config["rounds"]
    current_round = len(list(round_weights_folder.iterdir())) + 1

    # Exit if the project has completed all the rounds.
    if has_project_completed(client, proj_folder, total_rounds):
        return

    # Check if the aggregate has sent the weights for the previous round
    # We always use the previous round weights to train the model
    # from the agg_weights folder to train for the current round
    agg_weights_file = agg_weights_folder / f"agg_model_round_{current_round - 1}.pt"
    if not agg_weights_file.is_file():
        raise StateNotReady(
            f"Aggregator has not sent the weights for the round {current_round}"
        )

    # Train the model for the given FL round
    train_model(proj_folder, current_round, dataset_files)

    # Share the trained model to the aggregator
    trained_model_file = (
        round_weights_folder / f"trained_model_round_{current_round}.pt"
    )
    share_model_to_aggregator(
        client,
        fl_config["aggregator"],
        proj_folder,
        trained_model_file,
    )


def share_model_to_aggregator(
    client: Client,
    aggregator_email: str,
    proj_folder: Path,
    model_file: Path,
) -> None:
    """Shares the trained model to the aggregator."""
    fl_aggregator_app_path = (
        client.datasites / f"{aggregator_email}/api_data/fl_aggregator"
    )
    fl_aggregator_running_folder = fl_aggregator_app_path / "running" / proj_folder.name
    fl_aggregator_client_path = (
        fl_aggregator_running_folder / "fl_clients" / client.email
    )

    # Copy the trained model to the aggregator's client folder
    shutil.copy(model_file, fl_aggregator_client_path)


def _advance_fl_project(client: Client, proj_folder: Path) -> None:
    """
    Iterate over all the project folder, it will try to advance its state.
    1. Ensure the project has init directories (like round_weights, agg_weights)
    2. Has the aggregate sent the weights for the current round x (in the agg_weights folder)
    b. The client trains the model on the given round  and places the trained model in the round_weights folder
    c. It sends the trained model to the aggregator.
    d. repeat a until all round completes
    """

    try:
        # Init the shared directories for the project
        init_shared_dirs(client, proj_folder)

        # Retrieve datasets from the private folder if available
        dataset_files = get_train_datasets(client, proj_folder)

        # Train the model for the given FL round
        perform_model_training(client, proj_folder, dataset_files)

    except StateNotReady as e:
        print(e)
        return


def advance_fl_projects(client: Client) -> None:
    """
    Iterates over the `running` folder and tries to advance the FL projects
    """
    running_folder = client.api_data("fl_client") / "running"
    for proj_folder in running_folder.iterdir():
        if proj_folder.is_dir():
            proj_name = proj_folder.name
            print(
                f"Advancing FL project {proj_name} -> proj_folder: {proj_folder.resolve()}"
            )
            _advance_fl_project(client, proj_folder)


def start_app():
    client = Client.load()

    # Step 1: Init the FL Aggregator App
    init_client_app(client)

    # Step 2: Advance the FL Projects.
    # Iterates over the running folder and tries to advance the FL project
    advance_fl_projects(client)


if __name__ == "__main__":
    start_app()
