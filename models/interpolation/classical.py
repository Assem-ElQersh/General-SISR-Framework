"""Classical interpolation-based SR baselines.

These are non-learning methods and serve as baselines.  They require no
training and produce deterministic outputs.
"""

import torch
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model


class _InterpolationSR(BaseSRModel):
    """Shared base for interpolation-based SR models."""

    _MODE: str = "bicubic"

    def __init__(self, scale: int = 4, in_channels: int = 3, out_channels: int = 3, **kwargs) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        # No learnable parameters
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if self._MODE in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
        return F.interpolate(lr, scale_factor=self.scale, mode=self._MODE, **kwargs).clamp(0, 1)

    def param_count(self) -> int:
        return 0


@register_model("nearest")
class NearestNeighborSR(_InterpolationSR):
    """Nearest-neighbor upsampling."""
    _MODE = "nearest"


@register_model("bilinear")
class BilinearSR(_InterpolationSR):
    """Bilinear interpolation upsampling."""
    _MODE = "bilinear"


@register_model("bicubic")
class BicubicSR(_InterpolationSR):
    """Bicubic interpolation upsampling (standard baseline)."""
    _MODE = "bicubic"
