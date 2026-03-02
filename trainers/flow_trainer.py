"""FlowTrainer — normalizing flow training via negative log-likelihood.

Loss = NLL = -log p(HR | LR)
           = -log p_z(z) - sum_k log|det J_k|
           = 0.5 * ||z||² / σ² + const - total_log_det

where z = Flow(HR | LR) and σ² = 1 (standard normal prior).
"""

from typing import Dict

import torch
from torch.utils.data import DataLoader

from utils.logger import Logger
from .base_trainer import BaseTrainer


class FlowTrainer(BaseTrainer):
    """Trains SRFlow with NLL loss on the normalizing flow objective."""

    def train_step(self, batch: Dict[str, torch.Tensor], accum_step: int) -> Dict[str, float]:
        lr = batch["lr"]
        hr = batch["hr"]
        is_last_accum = (accum_step == self.grad_accum - 1)

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            z, log_det = self.model.forward_train(lr, hr)

            # NLL = 0.5 * ||z||² - log_det / (C * H * W)
            B, C, H, W = z.shape
            nll = 0.5 * (z ** 2).mean() - log_det.mean() / (C * H * W)
            loss = nll / self.grad_accum

        self.scaler.scale(loss).backward()

        if is_last_accum:
            if self.clip_grad is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return {"nll": nll.item()}
