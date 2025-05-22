import torch
import torch.nn as nn
from pathlib import Path
import json
from loguru import logger

INPUT_SIZE = 784
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10


class SimpleMNISTModel(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        output_size: int = OUTPUT_SIZE,
    ) -> None:
        super(SimpleMNISTModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def load_simple_mnist_model_from_json(weights_path: Path) -> SimpleMNISTModel:
    # Load weights from JSON
    with open(weights_path, "r") as f:
        weight_data = json.load(f)

    # Create model with same architecture
    model = SimpleMNISTModel()

    # Convert weights back to tensors and load into model
    fc1_weights = torch.tensor(weight_data["fc1"])
    fc2_weights = torch.tensor(weight_data["fc2"])

    model.fc1.weight.data = fc1_weights
    model.fc2.weight.data = fc2_weights

    return model


def save_simple_mnist_model_to_json(
    model: SimpleMNISTModel, weights_path: Path
) -> None:
    # Extract weights and convert to nested Python list
    fc1 = model.fc1.weight.data.cpu().numpy()
    fc2 = model.fc2.weight.data.cpu().numpy()

    logger.info(f"fc1 shape: {fc1.shape}")
    logger.info(f"fc2 shape: {fc2.shape}")

    # Save weights and bias in JSON format
    weight_data = {
        "fc1": fc1.tolist(),
        "fc2": fc2.tolist(),
    }

    with open(weights_path, "w") as f:
        json.dump(weight_data, f)

    logger.info(f"Weights saved to {weights_path}")
