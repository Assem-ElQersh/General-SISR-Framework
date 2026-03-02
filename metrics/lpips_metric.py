"""LPIPS — Learned Perceptual Image Patch Similarity.

Zhang et al., CVPR 2018.

Lower LPIPS = more perceptually similar.

Requires the `lpips` package (pip install lpips).
"""

from typing import Optional

import torch

_lpips_model = None


def _get_model(net: str = "vgg", device: Optional[torch.device] = None):
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips

            _lpips_model = lpips.LPIPS(net=net)
            if device is not None:
                _lpips_model = _lpips_model.to(device)
        except ImportError:
            raise ImportError("lpips package not installed. Run: pip install lpips")
    return _lpips_model


def compute_lpips(
    sr: torch.Tensor,
    hr: torch.Tensor,
    net: str = "vgg",
) -> float:
    """Compute LPIPS between SR and HR.

    Args:
        sr: SR image (B,C,H,W) or (C,H,W) in [0,1].
        hr: HR image.
        net: Backbone network ("vgg" or "alex").

    Returns:
        Mean LPIPS score (lower = better).
    """
    if sr.ndim == 3:
        sr = sr.unsqueeze(0)
        hr = hr.unsqueeze(0)

    model = _get_model(net, sr.device)
    # LPIPS expects images in [-1, 1]
    sr_norm = sr * 2.0 - 1.0
    hr_norm = hr * 2.0 - 1.0
    with torch.no_grad():
        score = model(sr_norm, hr_norm)
    return score.mean().item()
