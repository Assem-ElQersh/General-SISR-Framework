"""GANTrainer — alternating generator/discriminator training for SRGAN, ESRGAN, Real-ESRGAN.

Training procedure:
    Phase 1 (pretrain_epochs): Train generator with pixel loss only.
    Phase 2 (remaining epochs): Alternate G/D updates with full composite loss.
"""

import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from losses import build_losses
from losses.adversarial_loss import AdversarialLoss
from utils.checkpoint import save_checkpoint
from utils.logger import Logger
from .base_trainer import BaseTrainer


class GANTrainer(BaseTrainer):
    def __init__(self, cfg, model, train_loader: DataLoader, val_loader: DataLoader, logger: Logger) -> None:
        super().__init__(cfg, model, train_loader, val_loader, logger)

        # Generator = model itself; discriminator is stored on model
        self.generator = self.model
        self.discriminator = model.discriminator.to(self.device)

        self.pretrain_epochs: int = cfg.trainer.get("pretrain_epochs", 50)

        # Separate optimizers for G and D
        d_opt_cfg = cfg.optimizer.get("discriminator", cfg.optimizer)
        self.d_optimizer = self._build_optimizer(d_opt_cfg, self.discriminator.parameters())

        self.losses = build_losses(cfg)
        for loss_fn in self.losses.values():
            loss_fn.to(self.device)

        # Extract adversarial loss module
        self.adv_loss: AdversarialLoss = self.losses.get(
            "adversarial",
            AdversarialLoss(loss_type="ragan", weight=0.005),
        )

    def _is_pretrain(self) -> bool:
        return self.current_epoch < self.pretrain_epochs

    def train_step(self, batch: Dict[str, torch.Tensor], accum_step: int) -> Dict[str, float]:
        lr = batch["lr"]
        hr = batch["hr"]
        is_last_accum = (accum_step == self.grad_accum - 1)
        loss_dict: Dict[str, float] = {}

        # ── Generator step ────────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            sr = self.generator(lr)
            g_loss = torch.tensor(0.0, device=self.device)

            if "pixel" in self.losses:
                pl = self.losses["pixel"](sr, hr)
                g_loss = g_loss + pl
                loss_dict["g_pixel"] = pl.item()

            if not self._is_pretrain():
                if "perceptual" in self.losses:
                    perc = self.losses["perceptual"](sr, hr)
                    g_loss = g_loss + perc
                    loss_dict["g_percep"] = perc.item()

                if "frequency" in self.losses:
                    freq = self.losses["frequency"](sr, hr)
                    g_loss = g_loss + freq
                    loss_dict["g_freq"] = freq.item()

                # Adversarial
                fake_pred = self.discriminator(sr)
                real_pred = self.discriminator(hr.detach()) if self.adv_loss.loss_type == "ragan" else None
                adv = self.adv_loss(real_pred, fake_pred)
                g_loss = g_loss + adv
                loss_dict["g_adv"] = adv.item()

            g_loss = g_loss / self.grad_accum

        self.scaler.scale(g_loss).backward()
        if is_last_accum:
            if self.clip_grad is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        loss_dict["g_total"] = g_loss.item() * self.grad_accum

        # ── Discriminator step ────────────────────────────────────────
        if not self._is_pretrain():
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                real_pred = self.discriminator(hr)
                fake_pred = self.discriminator(sr.detach())
                d_loss = self.adv_loss.discriminator_loss(real_pred, fake_pred)
                d_loss = d_loss / self.grad_accum

            self.scaler.scale(d_loss).backward()
            if is_last_accum:
                self.scaler.step(self.d_optimizer)
                self.scaler.update()
                self.d_optimizer.zero_grad()

            loss_dict["d_loss"] = d_loss.item() * self.grad_accum

        return loss_dict

    def _save(self, epoch: int, is_best: bool) -> None:
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_psnr": self.best_psnr,
            "model_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "d_optimizer_state_dict": self.d_optimizer.state_dict(),
        }
        path = os.path.join(self.output_dir, f"epoch_{epoch+1:04d}.pth")
        best_path = os.path.join(self.output_dir, "best.pth")
        save_checkpoint(ckpt, path, is_best=is_best, best_path=best_path)
