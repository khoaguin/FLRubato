from pathlib import Path
from loguru import logger
import torch

from flhhe.mnist.dataset import load_dataset
from flhhe.mnist.model import SimpleMNISTModel
from flhhe.consts import DEVICE


def evaluate_model(model: SimpleMNISTModel, test_set_path: Path):
    test_set = load_dataset(test_set_path)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=2
    )
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
    logger.info(f"Accuracy on test set '{test_set_path.name}': {accuracy:.2f}%")
    return accuracy
