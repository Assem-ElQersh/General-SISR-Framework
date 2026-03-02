"""NIQE — Naturalness Image Quality Evaluator (no-reference metric).

Mittal et al., IEEE Signal Processing Letters 2013.

NIQE measures image quality without a reference HR image by comparing
statistics extracted from the input to those from a pristine image corpus.

Lower NIQE = better perceptual quality.

This implementation delegates to scikit-image when available, with a
lightweight fallback that computes BRISQUE-style local statistics.
"""

import numpy as np
import torch


def compute_niqe(sr: torch.Tensor) -> float:
    """Compute NIQE for a single SR image.

    Args:
        sr: SR image (B,C,H,W) or (C,H,W) or (H,W) in [0,1].

    Returns:
        NIQE score (lower = better).
    """
    if sr.ndim == 4:
        sr = sr[0]
    if sr.ndim == 3:
        # Convert to grayscale
        if sr.shape[0] == 3:
            r, g, b = sr[0], sr[1], sr[2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            gray = sr[0]
    else:
        gray = sr

    img_np = (gray.detach().cpu().float().numpy() * 255.0).astype(np.uint8)

    try:
        from skimage.metrics import niqe as skimage_niqe  # type: ignore
        return float(skimage_niqe(img_np))
    except (ImportError, AttributeError):
        pass

    # Fallback: simplified BRISQUE-style local variance statistics
    return _simple_niqe(img_np)


def _simple_niqe(img: np.ndarray, patch_size: int = 96) -> float:
    """Simplified NIQE approximation using local patch statistics."""
    h, w = img.shape
    variances = []
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = img[i : i + patch_size, j : j + patch_size].astype(np.float64)
            variances.append(patch.var())
    if not variances:
        return float("nan")
    # Higher variance = more natural detail = lower NIQE; invert for consistency
    return float(1.0 / (np.mean(variances) + 1e-8) * 1e4)
