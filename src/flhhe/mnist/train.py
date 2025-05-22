import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
import argparse


from flhhe.mnist.model import (
    SimpleMNISTModel,
    save_simple_mnist_model_to_json,
    load_simple_mnist_model_from_json,
)
from flhhe.mnist.dataset import load_dataset
from flhhe.mnist.eval import evaluate_model
from flhhe.mnist.logger import setup_logger

from flhhe.consts import DEVICE, PROJECT_ROOT


NUM_LOCAL_EPOCHS = 1


def train_and_save_weights(
    model: SimpleMNISTModel,
    train_set_path: Path,
    weights_path: Path = None,
    num_epochs: int = NUM_LOCAL_EPOCHS,
    save_weights: bool = False,
):
    logger.info(f"--- Training model on '{train_set_path.name}' ---")
    logger.info(f"Using device: {DEVICE}")

    train_set = load_dataset(train_set_path)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader)}")

    if save_weights:
        save_simple_mnist_model_to_json(model, weights_path)


def train_eval_models():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-weights", action="store_true", help="Save model weights after training"
    )
    args = parser.parse_args()

    DATA_DIR = PROJECT_ROOT / "data" / "MNIST" / "processed"
    WEIGHTS_DIR = PROJECT_ROOT / "weights" / "MNIST" / "plain"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    setup_logger()

    train_set_parts = [
        DATA_DIR / "train_no_137.pt",
        DATA_DIR / "train_no_258.pt",
        DATA_DIR / "train_no_469.pt",
    ]
    test_set_parts = [
        DATA_DIR / "test_137.pt",
        DATA_DIR / "test_258.pt",
        DATA_DIR / "test_469.pt",
        DATA_DIR / "test_all.pt",
    ]
    weights_paths = [
        WEIGHTS_DIR / "weights_no_137.json",
        WEIGHTS_DIR / "weights_no_258.json",
        WEIGHTS_DIR / "weights_no_469.json",
    ]

    model = SimpleMNISTModel()
    logger.info(f"Model: {model}")
    model.to(DEVICE)
    save_simple_mnist_model_to_json(model, WEIGHTS_DIR / "initial_model.json")

    for train_set_path, weights_path in zip(train_set_parts, weights_paths):
        model = load_simple_mnist_model_from_json(
            WEIGHTS_DIR / "initial_model.json"
        ).to(DEVICE)
        train_and_save_weights(
            model, train_set_path, weights_path, save_weights=args.save_weights
        )
        for test_set_path in test_set_parts:
            evaluate_model(model, test_set_path)


if __name__ == "__main__":
    train_eval_models()
