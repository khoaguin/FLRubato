from pathlib import Path

from flhhe.mnist.eval import evaluate_model
from flhhe.mnist.model import (
    load_simple_mnist_model_from_json,
    save_simple_mnist_model_to_json,
)
from flhhe.consts import PROJECT_ROOT


WEIGHTS_DIR = PROJECT_ROOT / "weights/MNIST/plain"
TEST_SET_PATH = PROJECT_ROOT / "data/MNIST/processed/"


def fed_avg(weight_paths: list[Path]) -> None:
    models = []
    for weight_path in weight_paths:
        model = load_simple_mnist_model_from_json(weight_path)
        models.append(model)

    avg_model = models[0]
    for model in models[1:]:
        for param, avg_param in zip(model.parameters(), avg_model.parameters()):
            avg_param.data += param.data

    for param in avg_model.parameters():
        param.data /= len(models)

    save_simple_mnist_model_to_json(avg_model, WEIGHTS_DIR / "plaintext_avg.json")

    evaluate_model(avg_model, TEST_SET_PATH / "test_all.pt")
    evaluate_model(avg_model, TEST_SET_PATH / "test_137.pt")
    evaluate_model(avg_model, TEST_SET_PATH / "test_258.pt")
    evaluate_model(avg_model, TEST_SET_PATH / "test_469.pt")


if __name__ == "__main__":
    weight_paths = [
        WEIGHTS_DIR / "weights_no_137.json",
        WEIGHTS_DIR / "weights_no_258.json",
        WEIGHTS_DIR / "weights_no_469.json",
    ]
    fed_avg(weight_paths)
