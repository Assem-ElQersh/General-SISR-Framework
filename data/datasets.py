"""Dataset classes for SISR training and evaluation.

Supported datasets:
    Training:  DIV2K (800 high-res images)
    Testing:   Set5, Set14, BSD100, Urban100
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import SRTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATASET_URLS: Dict[str, str] = {
    "DIV2K_train_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "Set5": "https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip",
    "Set14": "https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsplnd5ls.zip",
    "BSD100": "https://uofi.box.com/shared/static/qgctsplb8txwoszvjnjywr3tnx3wt70n.zip",
    "Urban100": "https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip",
}

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_images(root: str) -> List[str]:
    paths = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in IMG_EXTENSIONS:
            paths.append(str(p))
    return paths


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Base paired dataset
# ---------------------------------------------------------------------------

class PairedSRDataset(Dataset):
    """Generic paired (LR, HR) dataset.

    Expects the following structure:
        root/
            HR/   (or hr/)
            LR/   (or lr/X{scale}/)

    LR images can be pre-generated or created on-the-fly via bicubic downscaling.
    """

    def __init__(
        self,
        hr_dir: str,
        lr_dir: Optional[str] = None,
        scale: int = 4,
        patch_size: int = 96,
        transform: Optional[SRTransform] = None,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.hr_paths = _collect_images(hr_dir)
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_size = patch_size  # HR patch size
        self.transform = transform
        self.is_train = is_train

        if not self.hr_paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")

        if lr_dir is not None:
            self.lr_paths = _collect_images(lr_dir)
            if len(self.lr_paths) != len(self.hr_paths):
                raise ValueError(
                    f"HR/LR count mismatch: {len(self.hr_paths)} vs {len(self.lr_paths)}"
                )
        else:
            self.lr_paths = None

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hr = _load_image(self.hr_paths[idx])

        if self.lr_paths is not None:
            lr = _load_image(self.lr_paths[idx])
        else:
            lr_size = (hr.size[0] // self.scale, hr.size[1] // self.scale)
            lr = hr.resize(lr_size, Image.BICUBIC)

        if self.transform is not None:
            lr, hr = self.transform(lr, hr, is_train=self.is_train)

        return {"lr": lr, "hr": hr, "path": self.hr_paths[idx]}


# ---------------------------------------------------------------------------
# DIV2K dataset
# ---------------------------------------------------------------------------

class DIV2KDataset(PairedSRDataset):
    """DIV2K dataset wrapper.

    Expects data at:
        <data_root>/DIV2K/DIV2K_train_HR/
        <data_root>/DIV2K/DIV2K_valid_HR/
    """

    def __init__(
        self,
        data_root: str,
        scale: int = 4,
        patch_size: int = 96,
        split: str = "train",
        transform: Optional[SRTransform] = None,
    ) -> None:
        split_name = "DIV2K_train_HR" if split == "train" else "DIV2K_valid_HR"
        hr_dir = os.path.join(data_root, "DIV2K", split_name)
        lr_dir_candidate = os.path.join(
            data_root, "DIV2K", split_name.replace("HR", f"LR_bicubic"), f"X{scale}"
        )
        lr_dir = lr_dir_candidate if os.path.isdir(lr_dir_candidate) else None

        super().__init__(
            hr_dir=hr_dir,
            lr_dir=lr_dir,
            scale=scale,
            patch_size=patch_size,
            transform=transform,
            is_train=(split == "train"),
        )


# ---------------------------------------------------------------------------
# Benchmark test sets (Set5, Set14, BSD100, Urban100)
# ---------------------------------------------------------------------------

class BenchmarkDataset(Dataset):
    """Standard benchmark test sets.  Returns full images (no cropping)."""

    KNOWN_SETS = {"Set5", "Set14", "BSD100", "Urban100"}

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        scale: int = 4,
    ) -> None:
        super().__init__()
        if dataset_name not in self.KNOWN_SETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {self.KNOWN_SETS}")

        hr_dir = os.path.join(data_root, dataset_name, "HR")
        lr_dir = os.path.join(data_root, dataset_name, f"LR_bicubic", f"X{scale}")

        self.hr_paths = _collect_images(hr_dir)
        if not self.hr_paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")

        self.lr_paths = _collect_images(lr_dir) if os.path.isdir(lr_dir) else None
        self.scale = scale
        self.to_tensor = _pil_to_tensor

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hr = _load_image(self.hr_paths[idx])
        if self.lr_paths is not None:
            lr = _load_image(self.lr_paths[idx])
        else:
            lr_size = (hr.size[0] // self.scale, hr.size[1] // self.scale)
            lr = hr.resize(lr_size, Image.BICUBIC)

        return {
            "lr": self.to_tensor(lr),
            "hr": self.to_tensor(hr),
            "path": self.hr_paths[idx],
        }


# ---------------------------------------------------------------------------
# Unpaired dataset (for Real-ESRGAN — HR only; LR generated on-the-fly)
# ---------------------------------------------------------------------------

class UnpairedSRDataset(Dataset):
    """Dataset that loads only HR images.

    LR is produced at collation time via the degradation pipeline.
    """

    def __init__(
        self,
        hr_dir: str,
        patch_size: int = 256,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.hr_paths = _collect_images(hr_dir)
        self.patch_size = patch_size
        self.transform = transform

        if not self.hr_paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hr = _load_image(self.hr_paths[idx])
        hr_tensor = _pil_to_tensor(hr)

        # Random crop
        h, w = hr_tensor.shape[-2:]
        p = self.patch_size
        if h >= p and w >= p:
            top = random.randint(0, h - p)
            left = random.randint(0, w - p)
            hr_tensor = hr_tensor[:, top : top + p, left : left + w]

        if self.transform is not None:
            hr_tensor = self.transform(hr_tensor)

        return {"hr": hr_tensor, "path": self.hr_paths[idx]}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # HWC → CHW


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(cfg, split: str = "train") -> DataLoader:
    """Build a DataLoader from OmegaConf config."""
    from omegaconf import OmegaConf

    data_cfg = cfg.data
    transform = SRTransform(
        patch_size=data_cfg.patch_size,
        scale=data_cfg.scale,
        augmentation=OmegaConf.to_container(data_cfg.augmentation, resolve=True)
        if hasattr(data_cfg, "augmentation")
        else {},
    )

    if split == "train":
        dataset_name = data_cfg.get("train_dataset", "DIV2K")
        if dataset_name == "DIV2K":
            dataset = DIV2KDataset(
                data_root=data_cfg.data_root,
                scale=data_cfg.scale,
                patch_size=data_cfg.patch_size,
                split="train",
                transform=transform,
            )
        else:
            raise ValueError(f"Unknown train dataset: {dataset_name}")
        shuffle = True
        drop_last = True
    else:
        dataset_name = data_cfg.get("val_dataset", "Set5")
        dataset = BenchmarkDataset(
            data_root=data_cfg.data_root,
            dataset_name=dataset_name,
            scale=data_cfg.scale,
        )
        shuffle = False
        drop_last = False

    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size if split == "train" else 1,
        shuffle=shuffle,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=drop_last,
    )
