"""Benchmark runner — evaluate one or multiple checkpoints on standard test sets.

Usage::

    python -m inference.benchmark \\
        --config configs/edsr_x4.yaml \\
        --checkpoint outputs/edsr_x4/best.pth \\
        --datasets Set5 Set14 BSD100 Urban100

    python -m inference.benchmark \\
        --checkpoint_dir outputs/ \\
        --datasets Set5

Results are printed as a rich table and saved to a CSV file.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

from data.datasets import BenchmarkDataset
from inference.tiled_inference import tiled_infer
from metrics import Evaluator
from models import build_model
from utils.checkpoint import load_checkpoint

console = Console()


def evaluate_model(
    model: torch.nn.Module,
    dataset_name: str,
    data_root: str,
    scale: int,
    tile_size: int = 256,
    overlap: int = 16,
    use_tiled: bool = True,
) -> Dict[str, float]:
    from torch.utils.data import DataLoader

    dataset = BenchmarkDataset(data_root=data_root, dataset_name=dataset_name, scale=scale)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    device = next(model.parameters()).device
    evaluator = Evaluator(scale=scale, y_channel_only=True, with_lpips=True)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            if use_tiled:
                sr = tiled_infer(model, lr, tile_size=tile_size, overlap=overlap, scale=scale)
            else:
                sr = model(lr)
            evaluator.update(sr, hr)

    return evaluator.compute()


def run_benchmark(
    config_path: str,
    checkpoint_path: str,
    datasets: List[str],
    data_root: str = "datasets",
    tile_size: int = 256,
    overlap: int = 16,
    output_csv: str = "benchmark_results.csv",
) -> Dict[str, Dict[str, float]]:
    cfg = OmegaConf.load(config_path)
    model = build_model(cfg)
    load_checkpoint(checkpoint_path, model, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    results: Dict[str, Dict[str, float]] = {}
    for dset in datasets:
        console.print(f"Evaluating on [bold]{dset}[/bold] ...")
        metrics = evaluate_model(
            model=model,
            dataset_name=dset,
            data_root=data_root,
            scale=cfg.data.scale,
            tile_size=tile_size,
            overlap=overlap,
        )
        results[dset] = metrics

    # Display table
    table = Table(title=f"Benchmark Results — {Path(checkpoint_path).parent.name}")
    table.add_column("Dataset", style="bold cyan")
    table.add_column("PSNR (dB)", justify="right")
    table.add_column("SSIM", justify="right")
    table.add_column("LPIPS", justify="right")
    for dset, m in results.items():
        table.add_row(
            dset,
            f"{m.get('psnr', 0):.2f}",
            f"{m.get('ssim', 0):.4f}",
            f"{m.get('lpips', 0):.4f}",
        )
    console.print(table)

    # Save CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "psnr", "ssim", "lpips"])
        writer.writeheader()
        for dset, m in results.items():
            writer.writerow({"dataset": dset, **m})
    console.print(f"Results saved to [green]{output_csv}[/green]")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SISR Benchmark Runner")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pth")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Set5", "Set14", "BSD100", "Urban100"],
        help="Dataset names to evaluate on",
    )
    parser.add_argument("--data_root", default="datasets", help="Root directory of benchmark datasets")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--output_csv", default="benchmark_results.csv")
    args = parser.parse_args()

    run_benchmark(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        datasets=args.datasets,
        data_root=args.data_root,
        tile_size=args.tile_size,
        overlap=args.overlap,
        output_csv=args.output_csv,
    )
