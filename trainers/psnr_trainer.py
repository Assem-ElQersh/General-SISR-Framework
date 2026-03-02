"""PSNRTrainer — for pixel-loss-optimized models (SRCNN, VDSR, EDSR, RCAN, SwinIR, HAT, IMDN, ESPCN)."""

from typing import Dict

import torch
from torch.utils.data import DataLoader

from losses import build_losses
from utils.logger import Logger
from .base_trainer import BaseTrainer


class PSNRTrainer(BaseTrainer):
    def __init__(self, cfg, model, train_loader: DataLoader, val_loader: DataLoader, logger: Logger) -> None:
        super().__init__(cfg, model, train_loader, val_loader, logger)
        self.losses = build_losses(cfg)
        # Move loss modules (e.g. perceptual) to device
        for loss_fn in self.losses.values():
            loss_fn.to(self.device)

    def train_step(self, batch: Dict[str, torch.Tensor], accum_step: int) -> Dict[str, float]:
        lr = batch["lr"]
        hr = batch["hr"]
        is_last_accum = (accum_step == self.grad_accum - 1)

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            sr = self.model(lr)
            total_loss = torch.tensor(0.0, device=self.device)
            loss_dict: Dict[str, float] = {}

            for name, loss_fn in self.losses.items():
                l = loss_fn(sr, hr)
                total_loss = total_loss + l
                loss_dict[name] = l.item()

            total_loss = total_loss / self.grad_accum

        self.scaler.scale(total_loss).backward()

        if is_last_accum:
            if self.clip_grad is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        loss_dict["total"] = total_loss.item() * self.grad_accum
        return loss_dict
