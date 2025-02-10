from typing import Optional

import torch
from torch import nn

from src.configs.models import ModelConfig
from src.models.base_dit import DiT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SwiftTailor(nn.Module):
    def __init__(
        self,
        dit_config: ModelConfig,
        init_from_ckpt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self._dit_config = dit_config

        self._model: DiT = self._dit_config.create_target()

    def forward(self, x, t, y):
        return self._model(x, y, t)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        return self._model.forward_with_cfg(x, y=y, t=t, cfg_scale=cfg_scale)

    def sample_noise(self, bs):
        return torch.randn(
            bs,
            self._model.in_channels,
            self._model.input_size,
            self._model.input_size,
        )


if __name__ == "__main__":
    pass
