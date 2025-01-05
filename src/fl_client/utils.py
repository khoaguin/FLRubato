import json
import re
from enum import Enum
from pathlib import Path

from syftbox.lib import Client, SyftPermission


class ProjectStateCols(Enum):
    DATASET_ADDED = "dataset_added"
    MODEL_TRAIN_PROGRESS = "model_train_progress"


def read_json(file_path: Path) -> dict:
    """Read a json file and return the content as a dictionary"""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data: dict, file_path: Path):
    """Save a dictionary to a json file"""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


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


def search_files(pattern: str, path: Path) -> list[Path]:
    """Return all the files in the path matching the pattern."""
    return [f for f in path.iterdir() if re.match(pattern, f.name)]


def init_project_state(project_state_file: Path) -> dict:
    """Create a initial state for the project"""

    # If the state file already exists, we don't overwrite it
    if project_state_file.is_file():
        return

    state = {
        ProjectStateCols.DATASET_ADDED.value: False,
        ProjectStateCols.MODEL_TRAIN_PROGRESS.value: None,
    }
    save_json(state, project_state_file)


def create_project_state(client: Client, proj_folder: Path) -> None:
    """Creates the state folder for the project"""

    # Create a state folder to track progress of the project
    state_folder = proj_folder / "state"
    state_folder.mkdir(parents=True, exist_ok=True)

    project_state_file = state_folder / "state.json"

    # Give public read permission to the state folder
    add_public_write_permission(client, state_folder)

    # Create a initial state to track progress of the project
    init_project_state(project_state_file)


def update_project_state(proj_folder: Path, key: ProjectStateCols, val: str) -> None:
    """Updates the state of the project in the state.json file"""

    state_folder = proj_folder / "state"
    project_state_file = state_folder / "state.json"

    project_state = {}

    if project_state_file.is_file():
        project_state = read_json(project_state_file)

    project_state[key.value] = val

    save_json(project_state, project_state_file)
