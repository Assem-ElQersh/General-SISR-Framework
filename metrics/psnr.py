"""PSNR — Peak Signal-to-Noise Ratio.

PSNR = 10 * log10(MAX² / MSE)

Standard convention:
    - Images are uint8 in [0, 255]  (MAX = 255)
    - Evaluate on Y channel only (for YCbCr) or on full RGB
    - Crop borders by `scale` pixels to match benchmark protocols
"""

import torch
import torch.nn.functional as F


def compute_psnr(
    sr: torch.Tensor,
    hr: torch.Tensor,
    max_val: float = 1.0,
    y_channel_only: bool = True,
    border: int = 0,
) -> float:
    """Compute PSNR between SR and HR tensors.

    Args:
        sr: Super-resolved image (B, C, H, W) or (C, H, W) in [0, max_val].
        hr: Ground-truth HR image, same shape.
        max_val: Maximum pixel value (1.0 for [0,1] tensors).
        y_channel_only: Evaluate on luminance channel only (standard protocol).
        border: Number of pixels to crop from each border before computing.

    Returns:
        Average PSNR in dB.
    """
    if sr.ndim == 3:
        sr = sr.unsqueeze(0)
        hr = hr.unsqueeze(0)

    if border > 0:
        sr = sr[..., border:-border, border:-border]
        hr = hr[..., border:-border, border:-border]

    if y_channel_only and sr.shape[1] == 3:
        sr = _rgb_to_y(sr)
        hr = _rgb_to_y(hr)

    mse = F.mse_loss(sr.float(), hr.float(), reduction="mean")
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10((max_val ** 2) / mse).item()


def _rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB (B,3,H,W) in [0,1] to Y channel."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y
