from typing import Type

import yaml
from pydantic import BaseModel

from configs.envs import GLOBAL_CONFIG
from src.utils import factory

conf = yaml.safe_load(open(GLOBAL_CONFIG.config_dir / 'models/default.yaml'))

opts = conf['opts']
gpus = opts['gpus']

trainer_config_class: Type[BaseModel] = (
    factory.create(opts['trainer_config_class'])
)

trainer_conf_dict = conf['trainer']

trainer_conf = trainer_config_class.model_validate(
    trainer_conf_dict,
    strict=True,
)
trainer = trainer_conf.create_target()

trainer.train()
