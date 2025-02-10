import logging
import os
from typing import Optional

from accelerate.logging import MultiProcessAdapter


def get_logger(name: str, log_level: Optional[str] = None):
    """Reusing the accelerate logging module.

    Returns a `logging.Logger` for `name` that can handle multiprocessing.

    If a log should be called on all processes, pass `main_process_only=False`
    If a log should be called on all
    processes and in order, also pass `in_order=True`

    Args:
        name (`str`):
            The name for the logger, such as `__file__`
        log_level (`str`, *optional*):
            The log level to use. If not passed, will default to the
            `LOG_LEVEL` environment variable, or `INFO` if not
    """
    log_level = log_level or os.getenv("SWIFT_TAILOR_LOG_LEVEL")

    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        "%(asctime)s - (%(name)s): [%(levelname)s] %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_level:
        logger.setLevel(log_level)
        logger.root.setLevel(log_level)

    # If log_level from environment is not set, hide logs using LoggingAdapter
    # MultiProcessAdapter is also used to handle multiprocessing
    return MultiProcessAdapter(logger, {})
