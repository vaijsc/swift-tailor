from .base import BaseConfig


class ModelConfig(BaseConfig):
    pass


class SwiftTailorConfig(ModelConfig):
    dit_config: ModelConfig
