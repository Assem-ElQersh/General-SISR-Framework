import os
import logging
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler
from torch.utils.tensorboard import SummaryWriter

_console = Console()


class Logger:
    """Unified logger supporting console (rich), TensorBoard, and optionally WandB."""

    def __init__(
        self,
        name: str,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=_console, rich_tracebacks=True)],
        )
        self._log = logging.getLogger(name)

        self.tb_writer: Optional[SummaryWriter] = None
        if use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

        self.wandb_run = None
        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project or name,
                    config=wandb_config or {},
                    dir=log_dir,
                )
            except ImportError:
                self._log.warning("wandb not installed; skipping WandB logging.")

    # ------------------------------------------------------------------
    def info(self, msg: str) -> None:
        self._log.info(msg)

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def error(self, msg: str) -> None:
        self._log.error(msg)

    # ------------------------------------------------------------------
    def log_scalars(self, tag: str, scalars: Dict[str, float], step: int) -> None:
        for k, v in scalars.items():
            full_tag = f"{tag}/{k}"
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(full_tag, v, step)
            if self.wandb_run is not None:
                self.wandb_run.log({full_tag: v}, step=step)

    def log_images(self, tag: str, images: Any, step: int) -> None:
        """Log a grid of images (CHW or NCHW tensor, values in [0,1])."""
        if self.tb_writer is not None:
            self.tb_writer.add_images(tag, images, step)
        if self.wandb_run is not None:
            import wandb

            self.wandb_run.log({tag: [wandb.Image(img) for img in images]}, step=step)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
