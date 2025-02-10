from typing import Any

from pydantic import BaseModel, ConfigDict

from src.utils import factory


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    target_class: str

    def _get_all_fields(self) -> dict:
        return {k: getattr(self, k) for k in self.model_dump().keys()}

    def create_target(self, *args, **kwargs) -> Any:
        """Create instance from config (and update if needed)."""
        model_kwargs = self._get_all_fields()
        del model_kwargs["target_class"]

        model_kwargs.update(kwargs)
        model_class = factory.create(self.target_class)

        return model_class(*args, **model_kwargs)
