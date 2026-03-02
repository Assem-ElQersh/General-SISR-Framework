"""Tiled (overlap-tile) inference for memory-limited GPUs.

Splits a large LR image into overlapping tiles, runs super-resolution on each
tile, then blends them back together using a weighted average.

This allows any SISR model to run on an RTX 3050 Ti (4GB) regardless of
input resolution.

Usage::

    sr = tiled_infer(model, lr_tensor, tile_size=256, overlap=32)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def tiled_infer(
    model: nn.Module,
    lr: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 16,
    scale: Optional[int] = None,
    device: Optional[torch.device] = None,
    amp: bool = True,
) -> torch.Tensor:
    """Run model on overlapping tiles and stitch results.

    Args:
        model: SISR model (must have model.scale attribute).
        lr: LR input tensor (1, C, H, W) or (C, H, W) in [0, 1].
        tile_size: Tile size in LR pixels.
        overlap: Overlap between adjacent tiles in LR pixels (must be even).
        scale: Upscaling factor. Inferred from model.scale if not provided.
        device: Target device. Uses model's device if not provided.
        amp: Use automatic mixed precision.

    Returns:
        SR tensor (1, C, H*scale, W*scale) in [0, 1].
    """
    if lr.ndim == 3:
        lr = lr.unsqueeze(0)

    if scale is None:
        scale = getattr(model, "scale", 4)
    if device is None:
        device = next(model.parameters()).device

    lr = lr.to(device)
    B, C, H, W = lr.shape
    assert B == 1, "Tiled inference only supports batch size 1"

    if H <= tile_size and W <= tile_size:
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            return model(lr)

    step = tile_size - overlap
    assert step > 0, "tile_size must be greater than overlap"

    out_H, out_W = H * scale, W * scale
    output = torch.zeros(1, C, out_H, out_W, device=device)
    weight = torch.zeros(1, 1, out_H, out_W, device=device)

    # Build overlap blend mask (linear ramp at edges)
    tile_weight = _make_blend_mask(tile_size, overlap, device)

    h_starts = list(range(0, H - tile_size, step)) + [H - tile_size]
    w_starts = list(range(0, W - tile_size, step)) + [W - tile_size]

    for hs in h_starts:
        for ws in w_starts:
            tile = lr[:, :, hs : hs + tile_size, ws : ws + tile_size]
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                with torch.no_grad():
                    sr_tile = model(tile)  # (1, C, ts*s, ts*s)

            hs_out = hs * scale
            ws_out = ws * scale
            ts_out = tile_size * scale

            w_tile = F.interpolate(
                tile_weight.unsqueeze(0).unsqueeze(0),
                size=(ts_out, ts_out),
                mode="bilinear",
                align_corners=False,
            )
            output[:, :, hs_out : hs_out + ts_out, ws_out : ws_out + ts_out] += sr_tile * w_tile
            weight[:, :, hs_out : hs_out + ts_out, ws_out : ws_out + ts_out] += w_tile

    return (output / weight.clamp(min=1e-8)).clamp(0.0, 1.0)


def _make_blend_mask(tile_size: int, overlap: int, device: torch.device) -> torch.Tensor:
    """Create a 2D blending weight mask with linear ramps at tile borders."""
    mask = torch.ones(tile_size, tile_size, device=device)
    ramp = torch.linspace(0.0, 1.0, overlap, device=device)

    mask[:overlap, :] *= ramp[:, None]
    mask[-overlap:, :] *= ramp.flip(0)[:, None]
    mask[:, :overlap] *= ramp[None, :]
    mask[:, -overlap:] *= ramp.flip(0)[None, :]

    return mask
