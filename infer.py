"""Single image super-resolution inference entry point.

Usage:
    python infer.py \\
        --config configs/edsr_x4.yaml \\
        --checkpoint outputs/edsr_x4/best.pth \\
        --img path/to/lr_image.png \\
        --out path/to/sr_output.png

    # Use tiled inference for large images on limited VRAM:
    python infer.py \\
        --config configs/hat_x4.yaml \\
        --checkpoint outputs/hat_x4/best.pth \\
        --img large_lr.png \\
        --out large_sr.png \\
        --tiled --tile_size 128 --overlap 16
"""

import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from rich.console import Console

import models  # noqa: F401
from inference.tiled_inference import tiled_infer
from models.registry import build_model
from utils.checkpoint import load_checkpoint

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="SISR Single Image Inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img", required=True, help="Path to input LR image")
    parser.add_argument("--out", default=None, help="Path to save SR output")
    parser.add_argument("--tiled", action="store_true", help="Use tiled inference")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    return parser.parse_args()


def load_lr(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_sr(sr: torch.Tensor, path: str) -> None:
    arr = (sr.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(arr).save(path)
    console.print(f"Saved SR image to [green]{path}[/green]")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    scale = cfg.data.scale
    console.print(f"Model: {model.__class__.__name__} ×{scale}  |  Device: {device}")

    lr = load_lr(args.img).to(device)
    console.print(f"LR size: {lr.shape[-2]}×{lr.shape[-1]}")

    amp = not args.no_amp and device.type == "cuda"

    with torch.no_grad():
        if args.tiled:
            sr = tiled_infer(
                model, lr,
                tile_size=args.tile_size,
                overlap=args.overlap,
                scale=scale,
                amp=amp,
            )
        else:
            with torch.cuda.amp.autocast(enabled=amp):
                sr = model(lr)

    console.print(f"SR  size: {sr.shape[-2]}×{sr.shape[-1]}")

    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.img)
        out_path = f"{base}_sr_x{scale}{ext}"
    save_sr(sr, out_path)


if __name__ == "__main__":
    main()
