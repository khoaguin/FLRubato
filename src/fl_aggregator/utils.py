import importlib.util
import json
from enum import Enum
from pathlib import Path

from syftbox.lib import Client


class ParticipantStateCols(Enum):
    EMAIL = "Email"
    FL_CLIENT_INSTALLED = "Fl Client Installed"
    PROJECT_APPROVED = "Project Approved"
    ADDED_PRIVATE_DATA = "Added Private Data"
    ROUND = "Round (current/total)"
    MODEL_TRAINING_PROGRESS = "Training Progress"


def has_empty_dirs(directory: Path):
    return any(
        subdir.is_dir() and is_dir_empty(subdir) for subdir in directory.iterdir()
    )


def is_dir_empty(directory: Path):
    return not any(directory.iterdir())


def read_json(data_path: Path):
    with open(data_path) as fp:
        data = json.load(fp)
    return data


def save_json(data: dict, data_path: Path):
    with open(data_path, "w") as fp:
        json.dump(data, fp, indent=4)


def create_participant_json_file(
    participants: list, total_rounds: int, output_path: Path
):
    data = []
    for participant in participants:
        data.append(
            {
                ParticipantStateCols.EMAIL.value: participant,
                ParticipantStateCols.FL_CLIENT_INSTALLED.value: False,
                ParticipantStateCols.PROJECT_APPROVED.value: False,
                ParticipantStateCols.ADDED_PRIVATE_DATA.value: False,
                ParticipantStateCols.ROUND.value: f"0/{total_rounds}",
                ParticipantStateCols.MODEL_TRAINING_PROGRESS.value: "N/A",
            }
        )

    save_json(data=data, data_path=output_path)


def update_json(
    data_path: Path,
    participant_email: str,
    column_name: ParticipantStateCols,
    column_val: str,
):
    if column_name not in ParticipantStateCols:
        return
    participant_history = read_json(data_path=data_path)
    for participant in participant_history:
        if participant[ParticipantStateCols.EMAIL.value] == participant_email:
            participant[column_name.value] = column_val

    save_json(participant_history, data_path)


def load_model_class(model_path: Path, model_class_name: str) -> type:
    spec = importlib.util.spec_from_file_location(model_path.stem, model_path)
    model_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_arch)
    model_class = getattr(model_arch, model_class_name)

    return model_class


def get_all_directories(path: Path) -> list:
    """
    Returns the list of directories present in the given path
    """
    return [x for x in path.iterdir() if x.is_dir()]


def get_network_participants(client: Client):
    exclude_dir = ["apps", ".syft"]
    entries = client.datasites.iterdir()

    users = []
    for entry in entries:
        if entry.is_dir() and entry not in exclude_dir:
            users.append(entry.name)

    return users


def validate_launch_config(fl_config: Path) -> bool:
    """
    Validates the `fl_config.json` file
    """

    try:
        fl_config = read_json(fl_config)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {fl_config.resolve()}")

    required_keys = [
        "project_name",
        "aggregator",
        "participants",
        "model_arch",
        "model_weight",
        "model_class_name",
        "rounds",
        "epoch",
        "test_dataset",
        "learning_rate",
    ]

    for key in required_keys:
        if key not in fl_config:
            raise ValueError(f"Required key {key} is missing in fl_config.json")

    return True
