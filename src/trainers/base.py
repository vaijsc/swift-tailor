from src.configs.trainers import TrainerConfig


class BaseTrainer:
    def __init__(
        self,
        config: TrainerConfig,
        *args,
        **kwargs,
    ):
        self._config = config

    def _train_step(self, *args, **kwargs):
        raise NotImplementedError

    def _eval_step(self, *args, **kwargs):
        raise NotImplementedError

    def _test_step(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def infer(self, *args, **kwargs):
        raise NotImplementedError
