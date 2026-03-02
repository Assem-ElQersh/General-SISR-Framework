"""Blind blur kernel generation utilities for physics-informed SR training.

Provides:
    - Isotropic Gaussian kernels
    - Anisotropic Gaussian kernels (rotated ellipses)
    - Motion blur kernels (line kernels)
    - Random kernel sampler (mixes above types)
    - Batch kernel generator for IKC / DASR training

Usage::

    from data.kernel_estimation import random_kernel, batch_random_kernels

    # Single kernel (numpy)
    k = random_kernel(kernel_size=21)

    # Batch of kernels as torch tensors (B, kH, kW)
    kernels = batch_random_kernels(batch_size=8, kernel_size=21)
"""

import math
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Isotropic Gaussian kernel
# ---------------------------------------------------------------------------

def isotropic_gaussian_kernel(
    kernel_size: int = 21,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """Isotropic (symmetric) Gaussian blur kernel.

    Args:
        kernel_size: Odd integer side length.
        sigma: Gaussian sigma. If None, sampled uniformly from [0.5, 4.0].

    Returns:
        (kernel_size, kernel_size) normalized numpy array.
    """
    if sigma is None:
        sigma = random.uniform(0.5, 4.0)
    half = kernel_size // 2
    ax = np.arange(-half, half + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Anisotropic Gaussian kernel
# ---------------------------------------------------------------------------

def anisotropic_gaussian_kernel(
    kernel_size: int = 21,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
    theta: Optional[float] = None,
) -> np.ndarray:
    """Anisotropic (rotated ellipse) Gaussian blur kernel.

    Args:
        kernel_size: Odd integer side length.
        sigma_x: Sigma along x-axis. If None, sampled from [0.5, 4.0].
        sigma_y: Sigma along y-axis. If None, sampled from [0.5, sigma_x].
        theta: Rotation angle in radians. If None, sampled from [0, π].

    Returns:
        (kernel_size, kernel_size) normalized numpy array.
    """
    if sigma_x is None:
        sigma_x = random.uniform(0.5, 4.0)
    if sigma_y is None:
        sigma_y = random.uniform(0.5, sigma_x)
    if theta is None:
        theta = random.uniform(0, math.pi)

    half = kernel_size // 2
    ax = np.arange(-half, half + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_rot = cos_t * xx + sin_t * yy
    y_rot = -sin_t * xx + cos_t * yy

    kernel = np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2)))
    return (kernel / kernel.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Motion blur kernel
# ---------------------------------------------------------------------------

def motion_blur_kernel(
    kernel_size: int = 21,
    angle: Optional[float] = None,
    length: Optional[int] = None,
) -> np.ndarray:
    """Linear motion blur kernel.

    Args:
        kernel_size: Odd integer side length.
        angle: Angle in degrees. If None, sampled from [0, 180).
        length: Number of pixels in motion path. If None, sampled from [1, kernel_size//2].

    Returns:
        (kernel_size, kernel_size) normalized numpy array.
    """
    if angle is None:
        angle = random.uniform(0, 180)
    if length is None:
        length = random.randint(1, max(1, kernel_size // 2))

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    angle_rad = math.radians(angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    for i in range(-length, length + 1):
        x = int(round(center + i * cos_a))
        y = int(round(center + i * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    total = kernel.sum()
    if total == 0:
        kernel[center, center] = 1.0
    else:
        kernel /= total
    return kernel


# ---------------------------------------------------------------------------
# Random kernel sampler
# ---------------------------------------------------------------------------

def random_kernel(
    kernel_size: int = 21,
    iso_prob: float = 0.5,
    motion_prob: float = 0.1,
) -> np.ndarray:
    """Sample a random kernel from the mixture distribution.

    Args:
        kernel_size: Kernel side length (must be odd).
        iso_prob: Probability of isotropic Gaussian.
        motion_prob: Probability of motion blur.
        Remaining probability → anisotropic Gaussian.

    Returns:
        (kernel_size, kernel_size) numpy array.
    """
    r = random.random()
    if r < iso_prob:
        return isotropic_gaussian_kernel(kernel_size)
    elif r < iso_prob + motion_prob:
        return motion_blur_kernel(kernel_size)
    else:
        return anisotropic_gaussian_kernel(kernel_size)


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------

def batch_random_kernels(
    batch_size: int,
    kernel_size: int = 21,
    iso_prob: float = 0.5,
    motion_prob: float = 0.1,
    device: Union[torch.device, str] = "cpu",
) -> torch.Tensor:
    """Generate a batch of random kernels as a PyTorch tensor.

    Returns:
        (B, kernel_size, kernel_size) float32 tensor on `device`.
    """
    kernels = np.stack([
        random_kernel(kernel_size, iso_prob, motion_prob)
        for _ in range(batch_size)
    ])
    return torch.from_numpy(kernels).to(device)


# ---------------------------------------------------------------------------
# Kernel visualization helper
# ---------------------------------------------------------------------------

def kernel_to_image(kernel: np.ndarray) -> np.ndarray:
    """Scale kernel values to [0, 255] uint8 for visualization."""
    k = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
    return (k * 255).astype(np.uint8)
