"""SSIM — Structural Similarity Index Measure.

Wang et al., IEEE TIP 2004.

Computed using a Gaussian window over spatial patches.
Standard protocol: Y channel only, borders cropped by scale.
"""

import torch
import torch.nn.functional as F


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g[:, None] * g[None, :]
    return kernel / kernel.sum()


def compute_ssim(
    sr: torch.Tensor,
    hr: torch.Tensor,
    y_channel_only: bool = True,
    border: int = 0,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = (0.01) ** 2,
    C2: float = (0.03) ** 2,
) -> float:
    """Compute SSIM between SR and HR.

    Args:
        sr: SR tensor (B,C,H,W) or (C,H,W) in [0,1].
        hr: HR tensor.
        y_channel_only: Use Y channel only (standard).
        border: Crop pixels from each border.

    Returns:
        Mean SSIM across the batch.
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

    sr, hr = sr.float(), hr.float()
    C = sr.shape[1]
    kernel = _gaussian_kernel(window_size, sigma).to(sr.device)
    kernel = kernel.expand(C, 1, window_size, window_size)

    mu_sr = F.conv2d(sr, kernel, padding=window_size // 2, groups=C)
    mu_hr = F.conv2d(hr, kernel, padding=window_size // 2, groups=C)

    mu_sr_sq = mu_sr ** 2
    mu_hr_sq = mu_hr ** 2
    mu_sr_hr = mu_sr * mu_hr

    sigma_sr_sq = F.conv2d(sr * sr, kernel, padding=window_size // 2, groups=C) - mu_sr_sq
    sigma_hr_sq = F.conv2d(hr * hr, kernel, padding=window_size // 2, groups=C) - mu_hr_sq
    sigma_sr_hr = F.conv2d(sr * hr, kernel, padding=window_size // 2, groups=C) - mu_sr_hr

    ssim_map = (
        (2 * mu_sr_hr + C1) * (2 * sigma_sr_hr + C2)
        / ((mu_sr_sq + mu_hr_sq + C1) * (sigma_sr_sq + sigma_hr_sq + C2))
    )
    return ssim_map.mean().item()


def _rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
