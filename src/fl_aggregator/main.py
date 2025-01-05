import shutil
from pathlib import Path

import torch
from syftbox.lib import Client, SyftPermission
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    ParticipantStateCols,
    create_participant_json_file,
    get_all_directories,
    get_network_participants,
    has_empty_dirs,
    load_model_class,
    read_json,
    save_json,
    update_json,
    validate_launch_config,
)


# Exception name to indicate the state cannot advance
# as there are some pre-requisites that are not met
class StateNotReady(Exception):
    pass


def add_public_write_permission(client: Client, path: Path) -> None:
    """
    Adds public write permission to the given path
    """
    permission = SyftPermission.mine_with_public_write(client.email)
    permission.ensure(path)


def get_app_private_data(client: Client, app_name: str) -> Path:
    """
    Returns the private data directory of the app
    """
    return client.workspace.data_dir / "private" / app_name


def get_client_proj_state(project_folder: Path) -> dict:
    """
    Returns the path to the state.json file for the project
    """
    project_state = {}
    project_state_file = project_folder / "state/state.json/"

    if project_state_file.is_file():
        project_state = read_json(project_state_file)

    return project_state


def get_participants_metric_file(client: Client, proj_folder: Path):
    """
    Returns the path to the participant metrics file
    """
    return client.my_datasite / "public" / "fl" / proj_folder.name / "participants.json"


def init_aggregator(client: Client) -> None:
    """
    Creates the `fl_aggregator` app in the `api_data` folder
    with the following structure:
    ```
    api_data
    └── fl_aggregator
            └── launch
            └── running
            └── done
    ```
    """
    fl_aggregator = client.api_data("fl_aggregator")

    for folder in ["launch", "running", "done"]:
        fl_aggregator_folder = fl_aggregator / folder
        fl_aggregator_folder.mkdir(parents=True, exist_ok=True)

    # Create the private data directory for the app
    # This is where the private test data will be stored
    app_pvt_dir = get_app_private_data(client, "fl_aggregator")
    app_pvt_dir.mkdir(parents=True, exist_ok=True)


def create_metrics_dashboard(
    client: Client, fl_config: dict, participants: list, proj_name: str
) -> None:
    """Create the metrics dashboard for the project"""

    # Copy the metrics and dashboard files to the project's public folder
    metrics_folder = client.my_datasite / f"public/fl/{proj_name}/"
    metrics_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy("./dashboard/index.html", metrics_folder)
    shutil.copy("./dashboard/syftbox-sdk.js", metrics_folder)
    shutil.copy("./dashboard/index.js", metrics_folder)

    # Create a new participants.json file in the metrics folder
    participant_metrics_file = metrics_folder / "participants.json"

    # Remove the existing participants.json file if it exists
    participant_metrics_file.unlink(missing_ok=True)

    create_participant_json_file(
        participants, fl_config["rounds"], output_path=participant_metrics_file
    )

    # Copy the accuracy_metrics.json file to the project's metrics folder
    shutil.copy("./dashboard/accuracy_metrics.json", metrics_folder)

    print(
        f"Dashboard created for the project: {proj_name} at {metrics_folder.resolve()}"
    )


def init_project_directory(client: Client, fl_config_json_path: Path) -> None:
    """
    Initializes the FL project from the `fl_config.json` file
    If the project with same name already exists in the `running` folder
    then it skips creating the project

    If the project does not exist, it creates a new project with the
    project name and creates the folders for the clients and the aggregator

    api_data
    └── fl_aggregator
            └── launch
            └── running
                └── <fl_project_name>
                    ├── fl_clients
                    │   ├── ..
                    ├── agg_weights
                    ├── fl_config.json
                    ├── global_model_weights.pt
                    ├── model.py
                    └── state.json
            └── done
    """

    # Read the fl_config.json file
    fl_config = read_json(fl_config_json_path)

    proj_name = str(fl_config["project_name"])
    participants = fl_config["participants"]

    fl_aggregator = client.api_data("fl_aggregator")
    running_folder = fl_aggregator / "running"
    proj_folder = running_folder / proj_name

    # If the project already exists and is not empty
    # then skip creating the project
    if proj_folder.is_dir() and not has_empty_dirs(proj_folder):
        print(f"FL project {proj_name} already exists at: {proj_folder.resolve()}")
        return

    # Create the project folder
    print(f"Creating new FL project {proj_name} at {proj_folder.resolve()}")
    proj_folder.mkdir(parents=True, exist_ok=True)
    fl_clients_folder = proj_folder / "fl_clients"
    agg_weights_folder = proj_folder / "agg_weights"
    fl_clients_folder.mkdir(parents=True, exist_ok=True)
    agg_weights_folder.mkdir(parents=True, exist_ok=True)

    # create the folders for the participants
    for participant in participants:
        participant_folder = fl_clients_folder / participant
        participant_folder.mkdir(parents=True, exist_ok=True)

        # Give participant write access to the project folder
        add_public_write_permission(client, participant_folder)

    # Move the config file to the project's running folder
    shutil.move(fl_config_json_path, proj_folder)

    # move the model architecture to the project's running folder
    model_arch_src = fl_aggregator / "launch" / fl_config["model_arch"]
    shutil.move(model_arch_src, proj_folder)

    # copy the global model weights to the project's agg_weights folder as `agg_model_round_0.pt`
    # and move the global model weights to the project's running folder
    model_weights_src = fl_aggregator / "launch" / fl_config["model_weight"]
    shutil.copy(model_weights_src, agg_weights_folder / "agg_model_round_0.pt")
    shutil.move(model_weights_src, proj_folder)

    create_metrics_dashboard(client, fl_config, participants, proj_name)


def launch_fl_project(client: Client) -> None:
    """
    - Checks if `fl_config.json` file is present in the `launch` folder
    - Check if the project exists in the `running` folder with the same `project_name`.
        If not, create a new Project
        a. creates a directory with the project name in running folder
        b. inside the project it creates the folders of clients with a custom syft permissions
        c. copies over the fl_config.json and model.py and global_model_weights.pt

    Example:

    - Manually Copy the `fl_config.json`, `model.py`, `global_model_weights.pt`
        and `mnist_test_dataset.pt` to the `launch` folder
        api_data
        └── fl_aggregator
                └── launch
                    ├── fl_config.json (dragged and dropped by the user)
                    ├── model.py (dragged and dropped by the FL user)
                    ├── global_model_weights.pt (dragged and dropped by the FL user)
                    ├── mnist_test_dataset.pt
    """

    fl_config_json_path = client.api_data("fl_aggregator/launch/fl_config.json")

    if not fl_config_json_path.is_file():
        print(
            f"No launch config found at path: {fl_config_json_path.resolve()}. Skipping !!!"
        )
        return

    # Validate the fl_config.json file
    try:
        validate_launch_config(fl_config=fl_config_json_path)
    except ValueError as e:
        raise StateNotReady("Invalid launch config: " + str(e))

    # If the config is valid, then create the project
    init_project_directory(client, fl_config_json_path)


def create_fl_client_request(client: Client, proj_folder: Path):
    """
    Create the request folder for the fl clients.
    Creates a request folder for each client in the project's fl_clients folder
    and copies the fl_config.json and model.py to the request folder.
    """

    fl_clients = get_all_directories(proj_folder / "fl_clients")
    network_participants = get_network_participants(client)

    for fl_client in fl_clients:
        if fl_client.name not in network_participants:
            print(f"Client {fl_client.name} is not part of the network")
            continue

        fl_client_app_path = (
            client.datasites / fl_client.name / "api_data" / "fl_client"
        )
        fl_client_request_folder = fl_client_app_path / "request" / proj_folder.name
        if not fl_client_request_folder.is_dir():
            # Create a request folder for the client
            fl_client_request_folder.mkdir(parents=True, exist_ok=True)

            # Copy the fl_config.json, model_arch file to the request folder
            shutil.copy(proj_folder / "fl_config.json", fl_client_request_folder)

            # Copy the model architecture file to the request folder
            fl_config = read_json(proj_folder / "fl_config.json")
            model_arch_filename = fl_config["model_arch"]

            shutil.copy(proj_folder / model_arch_filename, fl_client_request_folder)
            print(
                f"Sending request to {fl_client.name} for the project {proj_folder.name}"
            )


def check_pvt_data_added_by_peer(
    peer_name: str,
    peer_client_path: Path,
    project_name: str,
    participant_metrics_file: Path,
):
    """Check if the private data is added by the client for model training."""

    fl_proj_folder = peer_client_path / "running" / project_name
    proj_state = get_client_proj_state(fl_proj_folder)

    participant_added_data = proj_state.get("dataset_added")

    # Skip if the state file is not present
    if participant_added_data is None:
        print(f"Private data not added by the client {peer_name}")
        return

    update_json(
        participant_metrics_file,
        peer_name,
        ParticipantStateCols.ADDED_PRIVATE_DATA,
        participant_added_data,
    )


def track_model_train_progress_for_peers(client: Client, proj_folder: Path):
    """Track the model training progress for the peer."""
    fl_clients = get_all_directories(proj_folder / "fl_clients")
    for fl_client in fl_clients:
        fl_client_running_folder = client.api_data("fl_client/running", fl_client.name)
        fl_proj_folder = fl_client_running_folder / proj_folder.name
        proj_state = get_client_proj_state(fl_proj_folder)
        model_train_progress = proj_state.get("model_train_progress")

        # Skip if the state file is not present
        if model_train_progress is None:
            return

        participants_metrics_file = get_participants_metric_file(client, proj_folder)
        update_json(
            participants_metrics_file,
            fl_client.name,
            ParticipantStateCols.MODEL_TRAINING_PROGRESS,
            model_train_progress,
        )


def aggregate_model(fl_config, proj_folder, trained_model_paths, current_round) -> Path:
    """Aggregate the trained models from the clients and save the aggregated model"""
    print("Aggregating the trained models")
    print(f"Trained model paths: {trained_model_paths}")
    global_model_class = load_model_class(
        proj_folder / fl_config["model_arch"], fl_config["model_class_name"]
    )
    global_model: nn.Module = global_model_class()
    global_model_state_dict = global_model.state_dict()

    aggregated_model_weights = {}

    n_peers = len(trained_model_paths)
    for model_file in trained_model_paths:
        user_model_state = torch.load(str(model_file))
        for key in global_model_state_dict.keys():
            # If user model has a different architecture than my global model.
            # Skip it
            if user_model_state.keys() != global_model_state_dict.keys():
                raise ValueError(
                    "User model has a different architecture than the global model"
                )

            if aggregated_model_weights.get(key, None) is None:
                aggregated_model_weights[key] = user_model_state[key] * (1 / n_peers)
            else:
                aggregated_model_weights[key] += user_model_state[key] * (1 / n_peers)

    global_model.load_state_dict(aggregated_model_weights)
    global_model_output_path = (
        proj_folder / "agg_weights" / f"agg_model_round_{current_round}.pt"
    )
    torch.save(global_model.state_dict(), str(global_model_output_path))

    return global_model_output_path


def shift_project_to_done_folder(
    client: Client, proj_folder: Path, total_rounds: int
) -> None:
    """
    Moves the project to the `done` folder
    a. Create a directory in the `done` folder with the same name as the project
    b. moves the agg weights and fl_clients to the done folder
    c. delete the project folder from the running folder
    """
    done_folder = client.api_data("fl_aggregator") / "done"
    done_proj_folder = done_folder / proj_folder.name
    done_proj_folder.mkdir(parents=True, exist_ok=True)

    # Move the agg weights and round weights folder to the done project folder
    # Move the fl_clients folder to the done project folder
    shutil.move(proj_folder / "agg_weights", done_proj_folder)
    shutil.move(proj_folder / "fl_clients", done_proj_folder)

    # Delete the project folder from the running folder
    shutil.rmtree(proj_folder)


def evaluate_agg_model(agg_model: nn.Module, dataset_path: Path) -> float:
    """Evaluate the aggregated model using the test dataset. We use accuracy as the evaluation metric."""
    agg_model.eval()

    # load the saved mnist subset
    images, labels = torch.load(str(dataset_path))

    # create a tensordataset
    dataset = TensorDataset(images, labels)

    # create a dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # dataset = torch.load(str(dataset_path))
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = agg_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Accuracy is returned as a percentage
    accuracy = correct / total

    return accuracy


def save_model_accuracy_metrics(
    client: Client, proj_folder: Path, current_round: int, accuracy: float
):
    """
    Saves the model accuracy in the public folder of the datasite under project name
    """
    metrics_folder = client.my_datasite / "public" / "fl" / proj_folder.name

    if not metrics_folder.is_dir():
        raise StateNotReady(
            f"Metrics folder not found for the project {proj_folder.name}"
        )

    metrics_file = metrics_folder / "accuracy_metrics.json"
    # Schema of json files
    # [ {round: 1, accuracy: 0.98}, {round: 2, accuracy: 0.99} ]
    # Append the accuracy and round to the json file
    metrics = read_json(metrics_file)

    metrics.append({"round": current_round, "accuracy": accuracy})
    save_json(metrics, metrics_file)


def check_aggregator_added_pvt_data(client: Client, proj_folder: Path):
    """Check if the aggregator has added the test dataset for model evaluation.

    Test dataset location: `api_data/fl_aggregator/private/<test_dataset>.pt`
    """
    fl_config = read_json(proj_folder / "fl_config.json")
    test_dataset_dir = get_app_private_data(client, "fl_aggregator")
    test_dataset_path = test_dataset_dir / fl_config["test_dataset"]

    if not test_dataset_path.exists():
        raise StateNotReady(
            f"Test dataset for model evaluation not found, please add the test dataset to: {test_dataset_path.resolve()}"
        )


def check_fl_client_app_installed(
    peer_name: str,
    peer_client_path: Path,
    participant_metrics_file: Path,
) -> None:
    """Check if the FL client app is installed for the given participant."""

    client_request_folder = peer_client_path / "request"
    client_request_syftperm = client_request_folder / "_.syftperm"

    installed_fl_client_app = True
    if not client_request_syftperm.is_file():
        print(f"FL client {peer_name} has not installed the app yet")
        installed_fl_client_app = False

    # As they have installed, update the participants.json file with state
    update_json(
        participant_metrics_file,
        peer_name,
        ParticipantStateCols.FL_CLIENT_INSTALLED,
        installed_fl_client_app,
    )


def check_proj_requests_status(
    peer_client_path: Path,
    peer_name: str,
    project_name: str,
    participant_metrics_file: Path,
) -> bool:
    """Check if the project requests are sent to the clients and if the clients have approved the project."""
    request_folder = peer_client_path / "request" / project_name
    running_folder = peer_client_path / "running" / project_name

    if not running_folder.is_dir() and not request_folder.is_dir():
        print(f"Request sent to {peer_name} for the project {project_name}.")

    # Check if project is approved by the client
    # If the running folder is not empty, then the project is a valid project
    if running_folder.is_dir() and not has_empty_dirs(running_folder):
        update_json(
            participant_metrics_file,
            peer_name,
            ParticipantStateCols.PROJECT_APPROVED,
            True,
        )
        return True

    return False


def share_agg_model_to_peers(
    client: Client,
    proj_folder: Path,
    agg_model_output_path: Path,
    participants: list,
):
    """Shares the aggregated model to all the participants."""
    for participant in participants:
        client_app_path = client.datasites / participant / "api_data" / "fl_client"
        client_agg_weights_folder = (
            client_app_path / "running" / proj_folder.name / "agg_weights"
        )
        shutil.copy(agg_model_output_path, client_agg_weights_folder)


def aggregate_and_evaluate(client: Client, proj_folder: Path):
    """
    1. Wait for the trained model from the clients
    3. Aggregate the trained model and place it in the `agg_weights` folder
    4. Send the aggregated model to all the clients
    5. Repeat until all the rounds are complete
    """
    agg_weights_folder = proj_folder / "agg_weights"
    current_round = len(list(agg_weights_folder.iterdir()))

    fl_config = read_json(proj_folder / "fl_config.json")

    total_rounds = fl_config["rounds"]
    if current_round >= total_rounds + 1:
        print(f"FL project {proj_folder.name} is complete ✅")
        shift_project_to_done_folder(client, proj_folder, total_rounds)
        return

    participants = fl_config["participants"]

    track_model_train_progress_for_peers(client, proj_folder)

    if current_round == 1:
        for participant in participants:
            client_app_path = client.datasites / participant / "api_data" / "fl_client"
            client_agg_weights_folder = (
                client_app_path / "running" / proj_folder.name / "agg_weights"
            )
            client_round_1_model = client_agg_weights_folder / "agg_model_round_0.pt"
            if not client_round_1_model.is_file():
                shutil.copy(
                    proj_folder / "agg_weights" / "agg_model_round_0.pt",
                    client_agg_weights_folder,
                )

    pending_clients = []
    trained_model_paths = []
    for participant in participants:
        participant_folder = proj_folder / "fl_clients" / participant
        participant_round_folder = (
            participant_folder / f"trained_model_round_{current_round}.pt"
        )
        trained_model_paths.append(participant_round_folder)
        if not participant_round_folder.is_file():
            pending_clients.append(participant)
        else:
            # Update the participants.json file with the current round
            participants_metrics_file = get_participants_metric_file(
                client, proj_folder
            )
            update_json(
                participants_metrics_file,
                participant,
                ParticipantStateCols.ROUND,
                f"{current_round}/{total_rounds}",
            )

    if pending_clients:
        raise StateNotReady(
            f"Waiting for trained model from the clients {pending_clients} for round {current_round}"
        )

    # Aggregate the trained model
    agg_model_output_path = aggregate_model(
        fl_config, proj_folder, trained_model_paths, current_round
    )

    # Test dataset for model evaluation
    test_dataset_dir = get_app_private_data(client, "fl_aggregator")
    test_dataset_path = test_dataset_dir / fl_config["test_dataset"]

    # Evaluate the aggregate model
    model_class = load_model_class(
        proj_folder / fl_config["model_arch"], fl_config["model_class_name"]
    )
    model: nn.Module = model_class()
    model.load_state_dict(torch.load(str(agg_model_output_path), weights_only=True))
    accuracy = evaluate_agg_model(model, test_dataset_path)
    print(f"Accuracy of the aggregated model for round {current_round}: {accuracy}")

    # Save the model accuracy metrics
    save_model_accuracy_metrics(client, proj_folder, current_round, accuracy)

    # Send the aggregated model to all the clients
    share_agg_model_to_peers(client, proj_folder, agg_model_output_path, participants)


def check_model_aggregation_prerequisites(client: Client, proj_folder: Path) -> None:
    """Check if the prerequisites are met before starting model aggregation

    1. Check if the fl client app is installed for all the peers
    2. Check if the project requests are sent to the peers
    3. Check if all the peers have approved the project
    4. Check if the private data is added by the peers
    5. Check if the test dataset is added by the aggregator
    """

    fl_clients = get_all_directories(proj_folder / "fl_clients")
    participant_metrics_file = get_participants_metric_file(client, proj_folder)
    peers_with_pending_requests = []

    for fl_client in fl_clients:
        fl_client_app_path = client.datasites / f"{fl_client.name}/api_data/fl_client"

        # Check if the fl client app is installed for given participant
        check_fl_client_app_installed(
            peer_name=fl_client.name,
            peer_client_path=fl_client_app_path,
            participant_metrics_file=participant_metrics_file,
        )

        # Check if project request is sent to the client
        # and if the client has approved the project
        project_approved = check_proj_requests_status(
            peer_client_path=fl_client_app_path,
            peer_name=fl_client.name,
            project_name=proj_folder.name,
            participant_metrics_file=participant_metrics_file,
        )

        # If the project is not approved by the client, add it to the list
        if not project_approved:
            peers_with_pending_requests.append(fl_client.name)

        # Check if the private data is added by the participant
        check_pvt_data_added_by_peer(
            peer_client_path=fl_client_app_path,
            project_name=proj_folder.name,
            peer_name=fl_client.name,
            participant_metrics_file=participant_metrics_file,
        )

    if peers_with_pending_requests:
        raise StateNotReady(
            "Project requests are pending for the clients: "
            + str(peers_with_pending_requests)
        )

    # Check if the test dataset is added by the aggregator
    check_aggregator_added_pvt_data(client, proj_folder)


def _advance_fl_project(client: Client, proj_folder: Path) -> None:
    """
    Iterate over all the project folder, it will try to advance its state.
    1. Has the client installed the fl_client app or not (api_data/fl_client), if not throw an error message
    2. have we submitted the project request to the clients  (api_data/fl_client/request)
    3. Have all the clients approved the project or not.
    4. let assume the round ix x,  place agg_model_round_x.pt inside all the clients
    5. wait for the trained model from the clients
    6. aggregate the trained model
    7. repeat d until all the rounds are complete
    """

    # Create the request folder for the fl clients
    create_fl_client_request(client, proj_folder)

    # Check if the prerequisites are met before starting model aggregation
    check_model_aggregation_prerequisites(client, proj_folder)

    aggregate_and_evaluate(client, proj_folder)


def advance_fl_projects(client: Client) -> None:
    """
    Iterates over the `running` folder and tries to advance the FL projects
    """
    running_folder = client.api_data("fl_aggregator/running")

    for proj_folder in running_folder.iterdir():
        if proj_folder.is_dir():
            proj_name = proj_folder.name
            print(f"Advancing FL project {proj_name}")
            _advance_fl_project(client, proj_folder)


def start_app():
    """Main function to run the FL Aggregator App"""
    client = Client.load()

    # Step 1: Init the FL Aggregator App
    init_aggregator(client)

    try:
        # Step 2: Launch the FL Project
        # Iterates over the `launch` folder and creates a new FL project
        # if `fl_config.json` exists in the `launch` folder
        launch_fl_project(client)

        # Step 3: Advance the FL Projects.
        # Iterates over the running folder and tries to advance the FL project
        advance_fl_projects(client)
    except StateNotReady as e:
        print(e)


if __name__ == "__main__":
    start_app()
