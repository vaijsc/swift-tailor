import os
from pathlib import Path
from typing import Any

import torch
from attr import dataclass


@dataclass
class _GlobalConfig:
    # Root / Working dir / Data if local
    # Root / data and Root / Working dir if on uni server
    root = Path(__file__).resolve().parents[1]
    data_dir = (
        root / "test_data"
        if (
            "RUN_MODE" in os.environ
            and os.environ["RUN_MODE"].lower() == "test"
        )
        else root / "data"
    )

    gcd_data_dir = data_dir / "GarmentCodeData_v2"
    ckpt_dir = root / "ckpts"
    config_dir = root / "configs"
    cache_dir = root / "cache"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        # Data dir
        assert Path.is_dir(
            self.data_dir
        ), f"{self.data_dir} is not a directory"

        # Checkpopint dir
        assert Path.is_dir(
            self.ckpt_dir
        ), f"{self.ckpt_dir} is not a directory"

        # Config dir
        assert Path.is_dir(
            self.config_dir
        ), f"{self.config_dir} is not a directory"

        # Cache dir
        if not Path.is_dir(self.cache_dir):
            Path.mkdir(self.cache_dir, exist_ok=True, parents=True)
        assert Path.is_dir(
            self.cache_dir
        ), f"{self.cache_dir} is not a directory"


GLOBAL_CONFIG = _GlobalConfig()
