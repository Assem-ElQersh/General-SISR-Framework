"""Real-world degradation pipeline for Real-ESRGAN.

Reference: Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
with Pure Synthetic Data", ICCV 2021.

The pipeline models realistic image degradation as a sequence of:
    1. Gaussian / anisotropic blur
    2. Resize (up or down)
    3. Additive Gaussian noise
    4. JPEG compression

A second-order variant applies the full chain twice, matching more complex
real-world degradation patterns.
"""

import io
import math
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Kernel utilities
# ---------------------------------------------------------------------------

def _isotropic_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    half = kernel_size // 2
    ax = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _anisotropic_gaussian_kernel(
    kernel_size: int, sigma_x: float, sigma_y: float, angle: float
) -> np.ndarray:
    half = kernel_size // 2
    ax = np.arange(-half, half + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    xr = cos_a * xx + sin_a * yy
    yr = -sin_a * xx + cos_a * yy
    kernel = np.exp(-(xr ** 2 / (2 * sigma_x ** 2) + yr ** 2 / (2 * sigma_y ** 2)))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def random_blur(
    img: np.ndarray,
    kernel_size_range: Tuple[int, int] = (7, 21),
    sigma_range: Tuple[float, float] = (0.2, 3.0),
    anisotropic_prob: float = 0.5,
) -> np.ndarray:
    ksize = random.randrange(kernel_size_range[0], kernel_size_range[1] + 1, 2)
    if random.random() < anisotropic_prob:
        sx = random.uniform(*sigma_range)
        sy = random.uniform(*sigma_range)
        angle = random.uniform(0, math.pi)
        kernel = _anisotropic_gaussian_kernel(ksize, sx, sy, angle)
    else:
        sigma = random.uniform(*sigma_range)
        kernel = _isotropic_gaussian_kernel(ksize, sigma)

    return cv2.filter2D(img, -1, kernel)


# ---------------------------------------------------------------------------
# Resize degradation
# ---------------------------------------------------------------------------

_INTERP_MODES = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}


def random_resize(
    img: np.ndarray,
    up_range: Tuple[float, float] = (1.0, 2.0),
    down_range: Tuple[float, float] = (0.15, 1.0),
    modes: Optional[List[str]] = None,
) -> np.ndarray:
    if modes is None:
        modes = ["area", "bilinear", "bicubic"]
    mode = random.choice(modes)
    interp = _INTERP_MODES.get(mode, cv2.INTER_CUBIC)

    h, w = img.shape[:2]
    if random.random() < 0.5:
        scale = random.uniform(*up_range)
    else:
        scale = random.uniform(*down_range)

    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def add_gaussian_noise(img: np.ndarray, sigma_range: Tuple[float, float] = (1, 30)) -> np.ndarray:
    sigma = random.uniform(*sigma_range) / 255.0
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img.astype(np.float32) + noise, 0.0, 1.0).astype(img.dtype)


# ---------------------------------------------------------------------------
# JPEG compression
# ---------------------------------------------------------------------------

def random_jpeg(img: np.ndarray, quality_range: Tuple[int, int] = (30, 95)) -> np.ndarray:
    quality = random.randint(*quality_range)
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR), encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return (dec.astype(np.float32) / 255.0)


# ---------------------------------------------------------------------------
# Full degradation pipeline
# ---------------------------------------------------------------------------

class DegradationPipeline:
    """Applies a configurable chain of degradation operations to an HR image
    to produce a synthetic LR image.

    Usage::

        pipeline = DegradationPipeline.from_config(cfg.data.degradation, scale=4)
        lr_tensor = pipeline(hr_tensor)
    """

    def __init__(
        self,
        scale: int = 4,
        blur_kernel_range: Tuple[int, int] = (7, 21),
        blur_sigma_range: Tuple[float, float] = (0.2, 3.0),
        blur_prob: float = 1.0,
        resize_up: Tuple[float, float] = (1.0, 2.0),
        resize_down: Tuple[float, float] = (0.15, 1.0),
        noise_sigma_range: Tuple[float, float] = (1, 30),
        noise_prob: float = 0.5,
        jpeg_quality_range: Tuple[int, int] = (30, 95),
        jpeg_prob: float = 0.5,
        second_order: bool = True,
    ) -> None:
        self.scale = scale
        self.blur_kernel_range = blur_kernel_range
        self.blur_sigma_range = blur_sigma_range
        self.blur_prob = blur_prob
        self.resize_up = resize_up
        self.resize_down = resize_down
        self.noise_sigma_range = noise_sigma_range
        self.noise_prob = noise_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.jpeg_prob = jpeg_prob
        self.second_order = second_order

    @classmethod
    def from_config(cls, cfg, scale: int = 4) -> "DegradationPipeline":
        return cls(
            scale=scale,
            blur_kernel_range=tuple(cfg.get("kernel_size", [7, 21])),
            blur_sigma_range=tuple(cfg.get("sigma", [0.2, 3.0])),
            blur_prob=cfg.get("prob", 1.0),
            noise_sigma_range=tuple(cfg.get("noise", {}).get("sigma", [1, 30])),
            noise_prob=cfg.get("noise", {}).get("prob", 0.5),
            jpeg_quality_range=tuple(cfg.get("jpeg", {}).get("quality", [30, 95])),
            jpeg_prob=cfg.get("jpeg", {}).get("prob", 0.5),
            second_order=cfg.get("second_order", True),
        )

    def _apply_once(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.blur_prob:
            img = random_blur(img, self.blur_kernel_range, self.blur_sigma_range)
        img = random_resize(img, self.resize_up, self.resize_down)
        if random.random() < self.noise_prob:
            img = add_gaussian_noise(img, self.noise_sigma_range)
        if random.random() < self.jpeg_prob:
            img = random_jpeg(img, self.jpeg_quality_range)
        return img

    def __call__(self, hr_tensor: torch.Tensor) -> torch.Tensor:
        """Degrade an HR tensor (CHW, float32, [0,1]) → LR tensor."""
        img = hr_tensor.permute(1, 2, 0).numpy()  # CHW → HWC
        img = self._apply_once(img)
        if self.second_order:
            img = self._apply_once(img)
        # Final bicubic downscale to target scale
        h, w = img.shape[:2]
        lr_h, lr_w = h // self.scale, w // self.scale
        lr_h, lr_w = max(1, lr_h), max(1, lr_w)
        img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1)
