"""VGG-based perceptual / feature loss.

Computes the L2 distance between deep features extracted by a pre-trained
VGG19 network.  Different layers capture different levels of abstraction:

    relu1_2  — low-level edges and textures
    relu2_2  — SRGAN-style (early feature match)
    relu3_4  — ESRGAN-style (richer, before activation → sharper gradients)
    relu5_4  — very high-level semantic features
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


_VGG_LAYER_MAP: Dict[str, int] = {
    "relu1_1": 2,
    "relu1_2": 4,
    "relu2_1": 7,
    "relu2_2": 9,
    "relu3_1": 12,
    "relu3_2": 14,
    "relu3_3": 16,
    "relu3_4": 18,
    "relu4_1": 21,
    "relu4_2": 23,
    "relu4_3": 25,
    "relu4_4": 27,
    "relu5_1": 30,
    "relu5_2": 32,
    "relu5_3": 34,
    "relu5_4": 36,
}


class PerceptualLoss(nn.Module):
    """VGG19 perceptual feature loss.

    Args:
        layer: VGG19 layer name at which to extract features.
        weight: Loss weight in composite objective.
        use_input_norm: Normalize input with ImageNet mean/std.
        before_relu: If True, use features before the final ReLU activation
                     (ESRGAN style — richer perceptual gradients).
    """

    def __init__(
        self,
        layer: str = "relu3_4",
        weight: float = 1.0,
        use_input_norm: bool = True,
        before_relu: bool = True,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.use_input_norm = use_input_norm

        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1)
        idx = _VGG_LAYER_MAP[layer]
        if before_relu and idx > 0:
            idx -= 1  # step back one layer to get pre-activation features
        self.features = nn.Sequential(*list(vgg.features)[:idx + 1])

        for param in self.features.parameters():
            param.requires_grad = False

        if use_input_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_norm:
            return (x - self.mean) / self.std
        return x

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_feat = self.features(self._normalize(sr))
        with torch.no_grad():
            hr_feat = self.features(self._normalize(hr))
        return self.weight * F.mse_loss(sr_feat, hr_feat)
