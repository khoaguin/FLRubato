from pathlib import Path

from flhhe.utils import get_device

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEVICE = get_device()
