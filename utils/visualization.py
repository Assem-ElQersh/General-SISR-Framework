from typing import List, Optional, Sequence

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW float tensor in [0,1] to HWC uint8 numpy array."""
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return arr


def save_comparison_grid(
    images: Sequence[torch.Tensor],
    labels: Optional[Sequence[str]],
    path: str,
    nrow: int = 4,
) -> None:
    """Save a grid of images (each a CHW tensor in [0,1]) side-by-side."""
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, img, lbl in zip(axes, images, labels or [""] * n):
        arr = tensor_to_image(img)
        ax.imshow(arr)
        ax.set_title(lbl, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def make_grid_tensor(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    max_images: int = 4,
) -> torch.Tensor:
    """Stack LR (upsampled), SR, and HR into a single display grid."""
    import torch.nn.functional as F

    lr_up = F.interpolate(
        lr[:max_images], size=hr.shape[-2:], mode="bicubic", align_corners=False
    )
    grid = torch.cat([lr_up, sr[:max_images], hr[:max_images]], dim=0)
    return vutils.make_grid(grid.clamp(0, 1), nrow=max_images)
