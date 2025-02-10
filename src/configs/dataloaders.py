from typing import Any

from src.utils import factory

from .base import BaseConfig


class DatasetConfig(BaseConfig):
    pass


class DataLoaderConfig(BaseConfig):
    target_class: str = "torch.utils.data.DataLoader"
    dataset: DatasetConfig
    batch_size: int
    num_workers: int
    shuffle: bool

    def create_target(self, *args, **kwargs) -> Any:
        model_kwargs = self.model_dump()
        del model_kwargs["target_class"]

        dataset = self.dataset.create_target()
        collate_fn = getattr(dataset, "collate_fn", None)

        model_kwargs.update(kwargs)
        model_kwargs['dataset'] = dataset
        model_kwargs['collate_fn'] = collate_fn

        model_class = factory.create(
            self.target_class
        )

        assert len(args) == 0, f"Unexpected args: {args}"
        return model_class(**model_kwargs)
