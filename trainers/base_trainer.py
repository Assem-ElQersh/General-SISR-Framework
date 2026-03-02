"""BaseTrainer — shared training infrastructure.

Provides:
    - Optimizer / scheduler construction from config
    - Mixed precision (torch.cuda.amp)
    - Gradient accumulation
    - Checkpoint save / resume
    - Validation loop
    - TensorBoard / WandB logging
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from metrics.evaluator import Evaluator
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import Logger
from utils.visualization import make_grid_tensor


class BaseTrainer(ABC):
    def __init__(
        self,
        cfg,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: Logger,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.current_epoch = 0
        self.global_step = 0
        self.best_psnr = 0.0

        self.mixed_precision: bool = cfg.trainer.get("mixed_precision", True)
        self.grad_accum: int = cfg.trainer.get("grad_accumulation_steps", 1)
        self.clip_grad: Optional[float] = cfg.trainer.get("clip_grad_norm", None)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.optimizer = self._build_optimizer(cfg.optimizer, self.model.parameters())
        self.scheduler = self._build_scheduler(cfg.scheduler, self.optimizer)

        self.evaluator = Evaluator(
            scale=cfg.data.scale,
            y_channel_only=True,
            with_lpips=False,
            with_niqe=False,
        )

        self.output_dir = os.path.join(
            cfg.experiment.get("output_dir", "outputs"),
            cfg.experiment.name,
        )
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self, opt_cfg, params) -> torch.optim.Optimizer:
        opt_type = opt_cfg.get("type", "Adam")
        lr = opt_cfg.get("lr", 1e-4)
        wd = opt_cfg.get("weight_decay", 0.0)

        if opt_type == "Adam":
            betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
            return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)
        elif opt_type == "AdamW":
            betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
            return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
        elif opt_type == "SGD":
            momentum = opt_cfg.get("momentum", 0.9)
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

    def _build_scheduler(self, sched_cfg, optimizer) -> Optional[object]:
        sched_type = sched_cfg.get("type", "CosineAnnealingLR")
        if sched_type == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_cfg.get("T_max", 300),
                eta_min=sched_cfg.get("eta_min", 1e-7),
            )
        elif sched_type == "MultiStepLR":
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(sched_cfg.get("milestones", [200])),
                gamma=sched_cfg.get("gamma", 0.5),
            )
        elif sched_type == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.get("step_size", 50),
                gamma=sched_cfg.get("gamma", 0.5),
            )
        return None

    # ------------------------------------------------------------------
    # Abstract train step
    # ------------------------------------------------------------------

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor], accum_step: int) -> Dict[str, float]:
        """Perform one gradient-accumulation sub-step.

        Returns a dict of scalar losses for logging.
        """

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        max_epochs = self.cfg.trainer.max_epochs
        val_every = self.cfg.trainer.get("val_every", 5)
        save_every = self.cfg.trainer.get("save_every", 10)
        log_img_every = self.cfg.logging.get("log_image_every", 10)

        self.logger.info(f"Training on {self.device} for {max_epochs} epochs")

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            self.model.train()
            epoch_losses: Dict[str, float] = {}

            for i, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                accum_step = i % self.grad_accum

                losses = self.train_step(batch, accum_step)
                self.global_step += 1

                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            # Average epoch losses
            n = len(self.train_loader)
            epoch_losses = {k: v / n for k, v in epoch_losses.items()}
            self.logger.log_scalars("train", epoch_losses, epoch)
            self.logger.info(
                f"Epoch [{epoch+1}/{max_epochs}] " +
                "  ".join(f"{k}: {v:.4f}" for k, v in epoch_losses.items())
            )

            if self.scheduler is not None:
                self.scheduler.step()

            # Validation
            if (epoch + 1) % val_every == 0:
                val_metrics = self.validate(log_images=(epoch + 1) % log_img_every == 0)
                self.logger.log_scalars("val", val_metrics, epoch)
                self.logger.info(
                    f"  Val  PSNR: {val_metrics.get('psnr', 0):.2f} dB  "
                    f"SSIM: {val_metrics.get('ssim', 0):.4f}"
                )
                psnr = val_metrics.get("psnr", 0.0)
                is_best = psnr > self.best_psnr
                if is_best:
                    self.best_psnr = psnr

                if (epoch + 1) % save_every == 0 or is_best:
                    self._save(epoch, is_best)

        self.logger.info(f"Training complete. Best PSNR: {self.best_psnr:.2f} dB")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, log_images: bool = False) -> Dict[str, float]:
        self.model.eval()
        self.evaluator.reset()

        for i, batch in enumerate(self.val_loader):
            lr = batch["lr"].to(self.device)
            hr = batch["hr"].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                sr = self.model(lr)
            self.evaluator.update(sr, hr)

            if log_images and i == 0:
                grid = make_grid_tensor(lr, sr, hr)
                self.logger.log_images("val/comparison", grid.unsqueeze(0), self.global_step)

        return self.evaluator.compute()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save(self, epoch: int, is_best: bool) -> None:
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_psnr": self.best_psnr,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        path = os.path.join(self.output_dir, f"epoch_{epoch+1:04d}.pth")
        best_path = os.path.join(self.output_dir, "best.pth")
        save_checkpoint(ckpt, path, is_best=is_best, best_path=best_path)

    def resume(self, checkpoint_path: str) -> None:
        ckpt = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        self.current_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_psnr = ckpt.get("best_psnr", 0.0)
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.logger.info(f"Resumed from {checkpoint_path} at epoch {self.current_epoch}")
