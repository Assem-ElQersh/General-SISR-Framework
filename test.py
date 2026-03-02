"""Evaluation entry point — run a trained model on benchmark test sets.

Usage:
    python test.py \\
        --config configs/edsr_x4.yaml \\
        --checkpoint outputs/edsr_x4/best.pth \\
        --datasets Set5 Set14 BSD100 Urban100

    python test.py \\
        --config configs/swinir_x4.yaml \\
        --checkpoint outputs/swinir_x4/best.pth \\
        --tiled --tile_size 256 --overlap 16
"""

import argparse
import os

import torch
from omegaconf import OmegaConf
from rich.console import Console

import models  # noqa: F401
from data.datasets import BenchmarkDataset
from inference.tiled_inference import tiled_infer
from metrics import Evaluator
from models.registry import build_model
from utils.checkpoint import load_checkpoint
from utils.visualization import save_comparison_grid, tensor_to_image

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="SISR Evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", nargs="+", default=["Set5", "Set14", "BSD100", "Urban100"])
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--tiled", action="store_true", help="Use tiled inference")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--output_dir", default="test_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    scale = cfg.data.scale

    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    console.print(f"[bold green]Model:[/bold green] {model.__class__.__name__} ×{scale}")
    console.print(f"[bold green]Checkpoint:[/bold green] {args.checkpoint}")

    from torch.utils.data import DataLoader

    for dset_name in args.datasets:
        console.print(f"\n[bold cyan]Dataset: {dset_name}[/bold cyan]")
        try:
            dataset = BenchmarkDataset(args.data_root, dset_name, scale)
        except FileNotFoundError as e:
            console.print(f"  [yellow]Skipping: {e}[/yellow]")
            continue

        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        evaluator = Evaluator(scale=scale, y_channel_only=True, with_lpips=True)

        out_dir = os.path.join(args.output_dir, cfg.experiment.name, dset_name)
        if args.save_images:
            os.makedirs(out_dir, exist_ok=True)

        with torch.no_grad():
            for batch in loader:
                lr = batch["lr"].to(device)
                hr = batch["hr"].to(device)

                if args.tiled:
                    sr = tiled_infer(model, lr, tile_size=args.tile_size, overlap=args.overlap, scale=scale)
                else:
                    with torch.cuda.amp.autocast(enabled=True):
                        sr = model(lr)

                metrics = evaluator.update(sr, hr)

                if args.save_images:
                    img_name = os.path.basename(batch["path"][0])
                    save_comparison_grid(
                        images=[lr[0], sr[0], hr[0]],
                        labels=["LR (bicubic)", "SR", "HR"],
                        path=os.path.join(out_dir, img_name),
                    )

        results = evaluator.compute()
        console.print(
            f"  PSNR: [bold]{results.get('psnr', 0):.2f}[/bold] dB  "
            f"SSIM: [bold]{results.get('ssim', 0):.4f}[/bold]  "
            f"LPIPS: [bold]{results.get('lpips', 0):.4f}[/bold]"
        )


if __name__ == "__main__":
    main()
