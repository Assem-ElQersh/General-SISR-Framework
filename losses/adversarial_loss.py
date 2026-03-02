"""Adversarial losses for GAN-based SR.

Supported types:
    "vanilla"  — BCE with logits (SRGAN original)
    "lsgan"    — Least-Squares GAN (more stable gradients)
    "ragan"    — Relativistic average GAN (ESRGAN)
    "wgan"     — Wasserstein distance (no sigmoid)
    "wgan-gp"  — WGAN with gradient penalty
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    """Unified adversarial loss module.

    Args:
        loss_type: One of "vanilla", "lsgan", "ragan", "wgan", "wgan-gp".
        weight: Loss weight for the generator adversarial term.
        target_real_label: Label value for real images.
        target_fake_label: Label value for fake images.
    """

    def __init__(
        self,
        loss_type: str = "ragan",
        weight: float = 1.0,
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type.lower()
        self.weight = weight
        self.real_label = target_real_label
        self.fake_label = target_fake_label

    def _label_like(self, pred: torch.Tensor, real: bool) -> torch.Tensor:
        val = self.real_label if real else self.fake_label
        return torch.full_like(pred, val)

    # ------------------------------------------------------------------
    # Discriminator loss
    # ------------------------------------------------------------------

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "vanilla":
            d_real = F.binary_cross_entropy_with_logits(
                real_pred, self._label_like(real_pred, True)
            )
            d_fake = F.binary_cross_entropy_with_logits(
                fake_pred, self._label_like(fake_pred, False)
            )
            return (d_real + d_fake) * 0.5

        elif self.loss_type == "lsgan":
            d_real = F.mse_loss(real_pred, self._label_like(real_pred, True))
            d_fake = F.mse_loss(fake_pred, self._label_like(fake_pred, False))
            return (d_real + d_fake) * 0.5

        elif self.loss_type == "ragan":
            # Relativistic average discriminator
            real_avg = real_pred - fake_pred.mean(0, keepdim=True)
            fake_avg = fake_pred - real_pred.mean(0, keepdim=True)
            d_real = F.binary_cross_entropy_with_logits(real_avg, self._label_like(real_avg, True))
            d_fake = F.binary_cross_entropy_with_logits(fake_avg, self._label_like(fake_avg, False))
            return (d_real + d_fake) * 0.5

        elif self.loss_type in ("wgan", "wgan-gp"):
            return fake_pred.mean() - real_pred.mean()

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    # ------------------------------------------------------------------
    # Generator adversarial loss
    # ------------------------------------------------------------------

    def generator_loss(
        self,
        real_pred: Optional[torch.Tensor],
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "vanilla":
            return self.weight * F.binary_cross_entropy_with_logits(
                fake_pred, self._label_like(fake_pred, True)
            )
        elif self.loss_type == "lsgan":
            return self.weight * F.mse_loss(fake_pred, self._label_like(fake_pred, True))
        elif self.loss_type == "ragan":
            assert real_pred is not None, "RaGAN requires real_pred for generator loss"
            fake_avg = fake_pred - real_pred.mean(0, keepdim=True)
            real_avg = real_pred - fake_pred.mean(0, keepdim=True)
            g_fake = F.binary_cross_entropy_with_logits(fake_avg, self._label_like(fake_avg, True))
            g_real = F.binary_cross_entropy_with_logits(real_avg, self._label_like(real_avg, False))
            return self.weight * (g_fake + g_real) * 0.5
        elif self.loss_type in ("wgan", "wgan-gp"):
            return self.weight * (-fake_pred.mean())
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    # ------------------------------------------------------------------
    # Gradient penalty (WGAN-GP)
    # ------------------------------------------------------------------

    def gradient_penalty(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        B = real.shape[0]
        alpha = torch.rand(B, 1, 1, 1, device=real.device)
        interp = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
        d_interp = discriminator(interp)
        gradients = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(B, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return lambda_gp * gp

    def forward(self, real_pred: Optional[torch.Tensor], fake_pred: torch.Tensor) -> torch.Tensor:
        return self.generator_loss(real_pred, fake_pred)
