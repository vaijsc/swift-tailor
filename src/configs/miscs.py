from typing import Optional

from .base import BaseConfig


class AccelerateProjectConfig(BaseConfig):
    target_class: str = "accelerate.utils.ProjectConfiguration"

    project_dir: str
    logging_dir: Optional[str] = None
    automatic_checkpoint_naming: bool = True
