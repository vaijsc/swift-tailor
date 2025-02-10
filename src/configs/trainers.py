from typing import Any, Optional

from src.utils import factory

from .base import BaseConfig
from .dataloaders import DataLoaderConfig
from .miscs import AccelerateProjectConfig
from .models import SwiftTailorConfig
from .optimizers import OptimizerConfig
from .schedulers import SchedulerConfig


class TrainerConfig(BaseConfig):
    target_class: str

    def create_target(self, *args, **kwargs) -> Any:
        """Create instance from config (and update if needed)."""
        model_class = factory.create(self.target_class)

        return model_class(self, *args, **kwargs)


class TailorTrainerConfig(TrainerConfig):
    # Training config
    fast_dev_run: bool = False

    # Training loop config
    st_epoch: int = 0
    en_epoch: int = 50
    max_time: Optional[int] = None

    # Callbacks config
    copy_ckpt_from: str = ""
    resume_from: str = ""
    check_val_every_n_epoch: int = 3
    checkpoint_monitor: str = "val/loss"
    early_stop_patience: int = 5
    disable_auto_lr_scale: bool = False

    # Logging config
    logger: str = "wandb"
    logger_kwargs: dict = {}
    wandb_silent: bool = False

    auto_lr_find: bool = False
    gradient_clip_val: Optional[float] = None
    precision: str = "32-true"
    use_tensorcores: bool = True
    strict_loading: bool = True
    extra_kwargs: dict = {}

    seed: int = 42

    # Accelerator project config
    project_config: AccelerateProjectConfig

    # Model config
    model: SwiftTailorConfig
    optimizer: OptimizerConfig
    dataloader: DataLoaderConfig
    scheduler: SchedulerConfig
