import sys
from datetime import datetime
from pathlib import Path
from typing import Union

import loguru
from loguru import logger

DEFAULT_LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_FORMAT = loguru


def setup_logger(
    level: Union[str, int] = "DEBUG",
    log_dir: Union[str, Path] = DEFAULT_LOGS_DIR,
    keep_logs: int = 10,
) -> None:
    logger.remove()
    logger.add(level=level, sink=sys.stderr, diagnose=False, backtrace=False)

    # new file per run - no rotation needed
    # always log debug level
    log_file = Path(log_dir, f"flhhe_{int(datetime.now().timestamp())}.log")
    logger.add(
        log_file,
        level="DEBUG",
        rotation=None,
        compression=None,
        colorize=True,
    )

    # keep last 5 logs
    logs_to_delete = sorted(Path(log_dir).glob("flhhe_*.log"))[:-keep_logs]
    for log in logs_to_delete:
        try:
            log.unlink()
        except Exception:
            pass
