from pathlib import Path
from loguru import logger
import torch

from dataset import load_dataset
from model import load_simple_mnist_model_from_json
from utils import DEVICE


def evaluate_model(weights_path: Path, test_set_path: Path):
    test_set = load_dataset(test_set_path)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=2
    )
    model = load_simple_mnist_model_from_json(weights_path)
    model.eval()
    model.to(DEVICE)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(
        f"Accuracy of '{weights_path.name}' on test set '{test_set_path.name}': {accuracy:.2f}%"
    )
    return accuracy
