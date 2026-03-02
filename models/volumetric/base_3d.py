"""Base class for 3D volumetric super-resolution models.

Unlike BaseSRModel (2D, (B, C, H, W)), 3D models operate on
(B, C, D, H, W) tensors where D is the depth/z-axis dimension.

`scale` may be:
    - int:   isotropic scale (all axes)
    - tuple: (sd, sh, sw) for anisotropic scale (e.g., MRI z-axis only)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn


ScaleFactor = Union[int, Tuple[int, int, int]]


class Base3DSRModel(nn.Module, ABC):
    """Abstract base class for 3D SR models.

    Subclasses must implement:
        forward(lr: Tensor[B,C,D,H,W]) -> Tensor[B,C,D*sd,H*sh,W*sw]
    """

    def __init__(
        self,
        scale: ScaleFactor = 2,
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(scale, int):
            self.scale = (scale, scale, scale)
        else:
            self.scale = tuple(scale)
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            lr: (B, C, D, H, W) low-resolution volume.

        Returns:
            (B, C, D*sd, H*sh, W*sw) super-resolved volume.
        """

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scale={self.scale}, "
            f"in={self.in_channels}, "
            f"out={self.out_channels}, "
            f"params={self.param_count():,})"
        )
