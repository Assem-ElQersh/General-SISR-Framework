from .base_trainer import BaseTrainer
from .psnr_trainer import PSNRTrainer
from .gan_trainer import GANTrainer
from .diffusion_trainer import DiffusionTrainer
from .flow_trainer import FlowTrainer


def build_trainer(cfg, model, train_loader, val_loader, logger):
    trainer_type = cfg.trainer.type.lower()
    if trainer_type == "psnr":
        return PSNRTrainer(cfg, model, train_loader, val_loader, logger)
    elif trainer_type == "gan":
        return GANTrainer(cfg, model, train_loader, val_loader, logger)
    elif trainer_type == "diffusion":
        return DiffusionTrainer(cfg, model, train_loader, val_loader, logger)
    elif trainer_type == "flow":
        return FlowTrainer(cfg, model, train_loader, val_loader, logger)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


__all__ = [
    "BaseTrainer", "PSNRTrainer", "GANTrainer",
    "DiffusionTrainer", "FlowTrainer", "build_trainer",
]
