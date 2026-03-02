"""Data augmentation and preprocessing for SISR."""

import random
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class SRTransform:
    """Paired (LR, HR) augmentation pipeline.

    Steps (training):
        1. Random crop to patch_size (HR) / patch_size//scale (LR)
        2. Random horizontal flip
        3. Random 90° rotation
        4. Convert to tensor in [0, 1]

    Steps (validation/test):
        1. Convert to tensor in [0, 1]
    """

    def __init__(
        self,
        patch_size: int,
        scale: int,
        augmentation: Dict = None,
    ) -> None:
        self.patch_size = patch_size  # HR patch size
        self.lr_patch = patch_size // scale
        self.scale = scale
        aug = augmentation or {}
        self.do_flip = aug.get("random_flip", True)
        self.do_rotate = aug.get("random_rotate", True)

    def __call__(
        self, lr: Image.Image, hr: Image.Image, is_train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_train:
            lr, hr = self._random_crop(lr, hr)
            if self.do_flip and random.random() > 0.5:
                lr = TF.hflip(lr)
                hr = TF.hflip(hr)
            if self.do_rotate and random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                lr = TF.rotate(lr, angle)
                hr = TF.rotate(hr, angle)

        lr_t = _pil_to_tensor(lr)
        hr_t = _pil_to_tensor(hr)
        return lr_t, hr_t

    # ------------------------------------------------------------------
    def _random_crop(
        self, lr: Image.Image, hr: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        hr_w, hr_h = hr.size
        p = self.patch_size
        lp = self.lr_patch

        if hr_w < p or hr_h < p:
            hr = TF.resize(hr, (max(hr_h, p), max(hr_w, p)), Image.BICUBIC)
            lr_size = (hr.size[0] // self.scale, hr.size[1] // self.scale)
            lr = hr.resize(lr_size, Image.BICUBIC)
            hr_w, hr_h = hr.size

        top_hr = random.randint(0, hr_h - p)
        left_hr = random.randint(0, hr_w - p)
        top_lr = top_hr // self.scale
        left_lr = left_hr // self.scale

        hr = TF.crop(hr, top_hr, left_hr, p, p)
        lr = TF.crop(lr, top_lr, left_lr, lp, lp)
        return lr, hr


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    """Merge YCbCr channels back to RGB (all tensors CHW in [0,1])."""
    delta = 0.5
    cb = cb - delta
    cr = cr - delta
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=0).clamp(0, 1)


def rgb_to_y_channel(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor (CHW, [0,1]) to Y channel only."""
    r, g, b = img[0], img[1], img[2]
    y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y.unsqueeze(0)
