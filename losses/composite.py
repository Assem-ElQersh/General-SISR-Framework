"""Build a composite loss from OmegaConf config.

Example YAML section::

    losses:
      pixel:
        type: L1
        weight: 1.0
      perceptual:
        type: vgg
        layer: relu3_4
        weight: 1.0
      adversarial:
        type: ragan
        weight: 0.005
      frequency:
        weight: 0.1
"""

from typing import Dict, Optional

import torch.nn as nn

from .pixel_loss import PixelLoss, CharbonnierLoss
from .perceptual_loss import PerceptualLoss
from .adversarial_loss import AdversarialLoss
from .frequency_loss import FrequencyLoss


def build_losses(cfg) -> Dict[str, nn.Module]:
    """Build a dict of loss modules from OmegaConf losses config."""
    from omegaconf import OmegaConf

    losses_cfg = OmegaConf.to_container(cfg.losses, resolve=True)
    modules: Dict[str, nn.Module] = {}

    if "pixel" in losses_cfg:
        c = losses_cfg["pixel"]
        t = c.get("type", "L1").upper()
        w = c.get("weight", 1.0)
        if t == "CHARBONNIER":
            modules["pixel"] = CharbonnierLoss(weight=w)
        else:
            modules["pixel"] = PixelLoss(loss_type=t, weight=w)

    if "perceptual" in losses_cfg:
        c = losses_cfg["perceptual"]
        modules["perceptual"] = PerceptualLoss(
            layer=c.get("layer", "relu3_4"),
            weight=c.get("weight", 1.0),
        )

    if "adversarial" in losses_cfg:
        c = losses_cfg["adversarial"]
        modules["adversarial"] = AdversarialLoss(
            loss_type=c.get("type", "ragan"),
            weight=c.get("weight", 1.0),
        )

    if "frequency" in losses_cfg:
        c = losses_cfg["frequency"]
        modules["frequency"] = FrequencyLoss(
            weight=c.get("weight", 0.1),
            loss_type=c.get("type", "L1"),
        )

    return modules
