import json
import torch
from loguru import logger

from flhhe.mnist.model import SimpleMNISTModel, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from flhhe.mnist.eval import evaluate_model
from flhhe.consts import PROJECT_ROOT


def main():
    # Define paths
    WEIGHTS_DIR = PROJECT_ROOT / "weights/MNIST/decrypted"
    TEST_SET_PATH = PROJECT_ROOT / "data/MNIST/processed"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # List your weight files here
    fc1_path = WEIGHTS_DIR / "hhe_decrypted_avg_fc1.json"
    fc2_path = WEIGHTS_DIR / "hhe_decrypted_avg_fc2.json"

    logger.info(f"Loading fc1 from: {fc1_path}")
    logger.info(f"Loading fc2 from: {fc2_path}")

    # Construct the model
    avg_model = SimpleMNISTModel()

    # Load and reshape fc1 weights (should be HIDDEN_SIZE x INPUT_SIZE = 32 x 784)
    with open(fc1_path, "r") as f:
        weight_data = json.load(f)
        fc1_weights = torch.tensor(weight_data)[: HIDDEN_SIZE * INPUT_SIZE]
        fc1_weights = fc1_weights.reshape(HIDDEN_SIZE, INPUT_SIZE)
        avg_model.fc1.weight.data = fc1_weights

    # Load and reshape fc2 weights (should be OUTPUT_SIZE x HIDDEN_SIZE = 10 x 32)
    with open(fc2_path, "r") as f:
        weight_data = json.load(f)
        fc2_weights = torch.tensor(weight_data)[: OUTPUT_SIZE * HIDDEN_SIZE]
        fc2_weights = fc2_weights.reshape(OUTPUT_SIZE, HIDDEN_SIZE)
        avg_model.fc2.weight.data = fc2_weights

    # Evaluate on all test sets
    test_sets = ["test_all.pt", "test_137.pt", "test_258.pt", "test_469.pt"]

    for test_set in test_sets:
        evaluate_model(avg_model, TEST_SET_PATH / test_set)


if __name__ == "__main__":
    main()
