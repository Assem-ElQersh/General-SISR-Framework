"""Abstract base class for all SISR models."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseSRModel(nn.Module, ABC):
    """Base class for all SISR models.

    Subclasses must implement:
        - forward(lr) → sr
        - param_count() (optional, has default)

    Subclasses may also override:
        - preprocess(lr) → lr (e.g. convert RGB → Y channel)
        - postprocess(sr) → sr
    """

    def __init__(self, scale: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lr: LR image tensor (B, C, H, W), values in [0, 1].

        Returns:
            sr: SR image tensor (B, C, H*scale, W*scale), values in [0, 1].
        """

    def preprocess(self, lr: torch.Tensor) -> torch.Tensor:
        return lr

    def postprocess(self, sr: torch.Tensor) -> torch.Tensor:
        return sr.clamp(0.0, 1.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scale={self.scale}, "
            f"in_ch={self.in_channels}, "
            f"out_ch={self.out_channels}, "
            f"params={self.param_count():,})"
        )
