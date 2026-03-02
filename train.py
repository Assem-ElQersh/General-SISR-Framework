"""Main training entry point.

Usage:
    python train.py --config configs/edsr_x4.yaml
    python train.py --config configs/esrgan_x4.yaml --resume outputs/esrgan_x4/epoch_0050.pth
    python train.py --config configs/swinir_x4.yaml --seed 1234
"""

import argparse
import random

import numpy as np
import torch
from omegaconf import OmegaConf

import models  # noqa: F401 — triggers all @register_model decorators
from data.datasets import build_dataloader
from models.registry import build_model
from trainers import build_trainer
from utils.logger import Logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="SISR Training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and merge config with base
    cfg = OmegaConf.load(args.config)
    seed = args.seed or cfg.experiment.get("seed", 42)
    set_seed(seed)

    # Build components
    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")
    model = build_model(cfg)

    log_dir = f"{cfg.experiment.get('output_dir', 'outputs')}/{cfg.experiment.name}/logs"
    logger = Logger(
        name=cfg.experiment.name,
        log_dir=log_dir,
        use_tensorboard=cfg.logging.get("use_tensorboard", True),
        use_wandb=cfg.logging.get("use_wandb", False),
        wandb_project=cfg.logging.get("wandb_project", "sisr_framework"),
        wandb_config=OmegaConf.to_container(cfg, resolve=True),
    )

    logger.info(f"Model: {model}")
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Val dataset size:   {len(val_loader.dataset)}")

    trainer = build_trainer(cfg, model, train_loader, val_loader, logger)

    if args.resume:
        trainer.resume(args.resume)

    trainer.fit()
    logger.close()


if __name__ == "__main__":
    main()
