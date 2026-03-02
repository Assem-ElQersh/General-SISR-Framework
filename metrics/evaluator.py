"""Evaluator — accumulates metrics over a dataset split and returns averages."""

from typing import Dict, List, Optional

import torch

from .psnr import compute_psnr
from .ssim import compute_ssim
from .lpips_metric import compute_lpips
from .niqe import compute_niqe


class Evaluator:
    """Accumulate PSNR / SSIM / LPIPS / NIQE over a batch of images.

    Usage::

        evaluator = Evaluator(compute_lpips=True)
        for batch in loader:
            sr = model(batch["lr"])
            evaluator.update(sr, batch["hr"])
        results = evaluator.compute()
        evaluator.reset()
    """

    def __init__(
        self,
        scale: int = 4,
        y_channel_only: bool = True,
        with_lpips: bool = False,
        with_niqe: bool = False,
    ) -> None:
        self.scale = scale
        self.y_channel_only = y_channel_only
        self.with_lpips = with_lpips
        self.with_niqe = with_niqe
        self.reset()

    def reset(self) -> None:
        self._psnr: List[float] = []
        self._ssim: List[float] = []
        self._lpips: List[float] = []
        self._niqe: List[float] = []

    def update(self, sr: torch.Tensor, hr: torch.Tensor) -> Dict[str, float]:
        """Update running metrics for one batch (returns per-sample averages)."""
        border = self.scale
        B = sr.shape[0]
        results = {}

        for i in range(B):
            psnr = compute_psnr(sr[i], hr[i], border=border, y_channel_only=self.y_channel_only)
            ssim = compute_ssim(sr[i], hr[i], border=border, y_channel_only=self.y_channel_only)
            self._psnr.append(psnr)
            self._ssim.append(ssim)

        results["psnr"] = float(torch.tensor(self._psnr[-B:]).mean())
        results["ssim"] = float(torch.tensor(self._ssim[-B:]).mean())

        if self.with_lpips:
            for i in range(B):
                self._lpips.append(compute_lpips(sr[i], hr[i]))
            results["lpips"] = float(torch.tensor(self._lpips[-B:]).mean())

        if self.with_niqe:
            for i in range(B):
                self._niqe.append(compute_niqe(sr[i]))
            results["niqe"] = float(torch.tensor(self._niqe[-B:]).mean())

        return results

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self._psnr:
            out["psnr"] = float(sum(self._psnr) / len(self._psnr))
        if self._ssim:
            out["ssim"] = float(sum(self._ssim) / len(self._ssim))
        if self._lpips:
            out["lpips"] = float(sum(self._lpips) / len(self._lpips))
        if self._niqe:
            out["niqe"] = float(sum(self._niqe) / len(self._niqe))
        return out
