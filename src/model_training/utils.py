import torch
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")


DEVICE = get_device()
