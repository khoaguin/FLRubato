from pathlib import Path
from loguru import logger
import torch

from dataset import load_dataset
from model import load_simple_mnist_model_from_json


def evaluate_model(weights_path: Path, test_set_path: Path, device: str = "cpu"):
    logger.info(
        f"Evaluating model '{weights_path.name}' on test set '{test_set_path.name}'"
    )

    test_set = load_dataset(test_set_path)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=2
    )
    model = load_simple_mnist_model_from_json(weights_path)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(
        f"Accuracy of '{weights_path.name}' on test set '{test_set_path.name}': {accuracy:.2f}%"
    )
    return accuracy
