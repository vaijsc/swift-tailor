from typing import List

import torch
from accelerate import Accelerator
from tqdm import tqdm

from src.configs.trainers import TailorTrainerConfig
from src.models.diffusion import create_diffusion
from src.models.swift_tailor import SwiftTailor
from src.utils.logger import get_logger

from .base import BaseTrainer

logger = get_logger(__name__)


class TailorTrainer(BaseTrainer):
    def __init__(
        self,
        config: TailorTrainerConfig,
        gpus: List[int] = [0],
        **kwargs,
    ):
        super().__init__(config)
        self._config = config
        project_config = self._config.project_config.create_target()
        self._accelerator = Accelerator(project_config=project_config)
        self._gpus = gpus
        self._setup()
        self._diffusion = create_diffusion(
            timestep_respacing="", noise_schedule="squaredcos_cap_v2"
        )

        logger.info("Trainer initialized")
        logger.info(f"Accelerator device: {self._accelerator.device}")

    def _setup(self):
        self._model: SwiftTailor = self._config.model.create_target()
        self._optimizer = self._config.optimizer.create_target(
            params=self._model.parameters()
        )
        self._scheduler = self._config.scheduler.create_target(
            optimizer=self._optimizer
        )
        self._dataloader = self._config.dataloader.create_target()

        (
            self._model,
            self._optimizer,
            self._dataloader,
            self._scheduler,
        ) = self._accelerator.prepare(
            self._model,
            self._optimizer,
            self._dataloader,
            self._scheduler,
        )

    # Util functions
    def _sample_timestep(self, num_samples: int) -> torch.Tensor:
        return torch.randint(
            0,
            self._diffusion.num_timesteps,
            (num_samples,),
        )

    # STEP functions
    def _train_step(self, gims, labels, stage="train"):
        device = self._accelerator.device
        bs = gims.shape[0]

        timesteps = self._sample_timestep(bs).to(device)
        loss_dict = self._diffusion.training_losses(
            model=self._model,
            x_start=gims,
            t=timesteps,
            model_kwargs={
                "y": labels,
            },
        )

        loss = loss_dict["loss"].mean()
        return loss

    def _eval_step(self, gims, labels):
        return self._train_step(gims, labels, stage="eval")

    def _test_step(self, gims, labels):
        return self._train_step(gims, labels, stage="test")

    # Main functions
    def train(self):
        device = self._accelerator.device

        # TODO: implement resume
        # TODO: implemt max time each training

        st_epoch = self._config.st_epoch
        en_epoch = self._config.en_epoch

        for epoch in range(st_epoch, en_epoch):
            accum_loss = 0
            logger.info(f"Starting epoch: {epoch}")
            with tqdm(self._dataloader, unit='batch') as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch: {epoch}")
                    self._optimizer.zero_grad()
                    gims, labels = (
                        batch["gim"].to(device), batch["label"].to(device)
                    )
                    loss = self._train_step(gims, labels)
                    accum_loss += loss.item()

                    self._accelerator.backward(loss)
                    self._optimizer.step()
                    tepoch.set_postfix(loss=loss.item())

                self._scheduler.step()
                self._accelerator.save_state()
                logger.info(f"Epoch: {epoch} - "
                            f"Loss: {accum_loss / len(self._dataloader)}")

    def eval(self):
        # TODO: implement
        raise NotImplementedError

    def test(self):
        # TODO: implement
        raise NotImplementedError

    @torch.inference_mode()
    def infer(self, labels, infer_steps: str = "250"):
        """Sampling a single sample"""
        bs = labels.shape[0]
        device = self._accelerator.device

        labels = labels.to(device)
        gim = self._model.sample_noise(bs).to(device)
        gim = torch.cat([gim, gim], dim=0)
        labels_null = labels * 0 + 63
        labels = torch.cat([labels, labels_null], dim=0)

        model_kwargs = {"y": labels}
        infer_diffusion = create_diffusion(
            timestep_respacing=str(infer_steps),
            noise_schedule="squaredcos_cap_v2",
        )

        samples = infer_diffusion.p_sample_loop(
            self._model.forward_with_cfg,
            gim.shape,
            gim,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )

        samples, _ = samples.chunk(2, dim=0)  # Remove the null samples
        logger.info(
            f"Samples shape: {samples.shape}" f"Dtype: {samples.dtype}"
        )

        return samples
