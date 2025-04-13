import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger


from model import SimpleMNISTModel, save_simple_mnist_model_to_json
from dataset import load_dataset
from eval import evaluate_model
from logger import setup_logger


def train_and_save_weights(
    train_set_path: Path, weights_path: Path = None, num_epochs: int = 3
):
    logger.info("")
    logger.info(f"--- Training model on '{train_set_path.name}' ---")
    train_set = load_dataset(train_set_path)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2
    )

    model = SimpleMNISTModel()
    logger.info(f"Model: {model}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader)}")

    save_simple_mnist_model_to_json(model, weights_path)


def train_models():
    PROJECT_ROOT = Path(__file__).parent.parent.parent
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

    for train_set_path, weights_path in zip(train_set_parts, weights_paths):
        train_and_save_weights(train_set_path, weights_path)
        for test_set_path in test_set_parts:
            evaluate_model(weights_path, test_set_path)


if __name__ == "__main__":
    train_models()
