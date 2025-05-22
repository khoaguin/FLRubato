from pathlib import Path

from loguru import logger
from typing_extensions import Union

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_mnist_data(batch_size: int = 64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = datasets.MNIST(
        PROJECT_ROOT / "data", download=True, train=True, transform=transform
    )

    testset = datasets.MNIST(
        PROJECT_ROOT / "data", download=True, train=False, transform=transform
    )

    return trainset, testset


def save_dataset(dataset: Subset, path: Union[str, Path]) -> None:
    torch.save(dataset, path)
    logger.info(f"Dataset saved to {path}")


def load_dataset(path: Union[str, Path], verbose: bool = False) -> Dataset:
    if verbose:
        logger.info(f"Loading dataset from {path}")
    return torch.load(path, weights_only=False)


def include_digits(
    dataset: Subset, included_digits: list[int], save_path: Union[str, Path]
) -> None:
    including_indices = [
        idx for idx in range(len(dataset)) if dataset[idx][1] in included_digits
    ]
    subset = Subset(dataset, including_indices)
    save_dataset(subset, save_path)
    return subset


def exclude_digits(
    dataset: Subset, excluded_digits: list[int], save_path: Union[str, Path]
) -> None:
    including_indices = [
        idx for idx in range(len(dataset)) if dataset[idx][1] not in excluded_digits
    ]
    subset = Subset(dataset, including_indices)
    save_dataset(subset, save_path)
    return subset


if __name__ == "__main__":
    train_set, test_set = load_mnist_data()

    MNIST_DATA_PATH = PROJECT_ROOT / "data" / "MNIST" / "processed"
    if MNIST_DATA_PATH.exists():
        logger.info(f"MNIST data path exists: {MNIST_DATA_PATH}")
        exit()

    MNIST_DATA_PATH.mkdir(parents=True, exist_ok=True)

    train_no_137 = exclude_digits(
        train_set, [1, 3, 7], MNIST_DATA_PATH / "train_no_137.pt"
    )
    train_no_258 = exclude_digits(
        train_set, [2, 5, 8], MNIST_DATA_PATH / "train_no_258.pt"
    )
    train_no_469 = exclude_digits(
        train_set, [4, 6, 9], MNIST_DATA_PATH / "train_no_469.pt"
    )

    testset_137 = include_digits(test_set, [1, 3, 7], MNIST_DATA_PATH / "test_137.pt")
    testset_258 = include_digits(test_set, [2, 5, 8], MNIST_DATA_PATH / "test_258.pt")
    testset_469 = include_digits(test_set, [4, 6, 9], MNIST_DATA_PATH / "test_469.pt")

    save_dataset(train_set, MNIST_DATA_PATH / "train_all.pt")
    save_dataset(test_set, MNIST_DATA_PATH / "test_all.pt")
