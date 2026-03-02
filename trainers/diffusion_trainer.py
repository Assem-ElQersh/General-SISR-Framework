"""DiffusionTrainer — DDPM noise-prediction training loop for SR3."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.logger import Logger
from .base_trainer import BaseTrainer


class DiffusionTrainer(BaseTrainer):
    """Trains SR3 by predicting the noise added to HR images.

    Loss = MSE(predicted_noise, actual_noise)
    """

    def __init__(self, cfg, model, train_loader: DataLoader, val_loader: DataLoader, logger: Logger) -> None:
        super().__init__(cfg, model, train_loader, val_loader, logger)
        self.sample_every: int = cfg.trainer.get("sample_every", 50)

    def train_step(self, batch: Dict[str, torch.Tensor], accum_step: int) -> Dict[str, float]:
        lr = batch["lr"]
        hr = batch["hr"]
        is_last_accum = (accum_step == self.grad_accum - 1)

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            noise, noise_pred = self.model.forward_train(lr, hr)
            loss = F.mse_loss(noise_pred, noise) / self.grad_accum

        self.scaler.scale(loss).backward()

        if is_last_accum:
            if self.clip_grad is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return {"noise_mse": loss.item() * self.grad_accum}

    @torch.no_grad()
    def validate(self, log_images: bool = False) -> Dict[str, float]:
        """For diffusion models, only run a small number of samples due to inference cost."""
        self.model.eval()
        self.evaluator.reset()
        max_samples = 4  # full denoising chain is expensive

        for i, batch in enumerate(self.val_loader):
            if i >= max_samples:
                break
            lr = batch["lr"].to(self.device)
            hr = batch["hr"].to(self.device)
            sr = self.model(lr)
            self.evaluator.update(sr, hr)

        return self.evaluator.compute()
