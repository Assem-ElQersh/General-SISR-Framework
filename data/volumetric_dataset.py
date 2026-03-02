"""VolumetricSRDataset — 3D patch dataset for volumetric super-resolution.

Supports:
    - NIfTI format (.nii / .nii.gz) via nibabel
    - DICOM directories via pydicom (optional)

Output patches:
    {"lr": (C, D, H, W), "hr": (C, D, H, W)}

Training strategy:
    Full-volume training is impractical at ≤4 GB VRAM.
    All samples are 3D patches extracted at random positions.
    Patch size is controlled by `patch_size_3d` in the config.

Degradation:
    HR patches are downsampled to LR patches using trilinear interpolation
    (or along z-axis only for anisotropic MRI SR).

Usage::

    dataset = VolumetricSRDataset(
        data_root="data/MRI",
        scale=2,
        patch_size=32,
        split="train",
    )
    lr, hr = dataset[0]["lr"], dataset[0]["hr"]
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Volume loaders
# ---------------------------------------------------------------------------

def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI volume to a float32 numpy array (D, H, W) or (D, H, W, C).

    Requires ``nibabel``.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required to load NIfTI files. "
            "Install with: pip install nibabel>=5.0.0"
        ) from exc

    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data


def load_dicom_series(folder: str) -> np.ndarray:
    """Load a DICOM series from a folder to a float32 numpy array (D, H, W).

    Requires ``pydicom``.
    """
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "pydicom is required to load DICOM files. "
            "Install with: pip install pydicom>=2.4.0"
        ) from exc

    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".dcm")],
        key=lambda p: pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber,
    )
    slices = [pydicom.dcmread(f).pixel_array.astype(np.float32) for f in files]
    return np.stack(slices, axis=0)


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1]."""
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


# ---------------------------------------------------------------------------
# VolumetricSRDataset
# ---------------------------------------------------------------------------

class VolumetricSRDataset(Dataset):
    """3D patch dataset for volumetric super-resolution.

    Args:
        data_root: Root directory of NIfTI files (or DICOM series folders).
        scale: SR scale factor (int or tuple (sd, sh, sw)).
        patch_size: Side length of 3D training patches (default 32).
        split: "train" or "val".
        val_ratio: Fraction of volumes used for validation (default 0.1).
        patches_per_volume: Patches sampled per volume per epoch (default 50).
        anisotropic_axis: "z" for z-axis only SR (MRI). None for isotropic.
        file_ext: File extension filter (default ".nii.gz").
        transform: Optional callable applied to (lr, hr) tuple.
        seed: Random seed for train/val split.
    """

    def __init__(
        self,
        data_root: str,
        scale: Union[int, Tuple[int, int, int]] = 2,
        patch_size: int = 32,
        split: str = "train",
        val_ratio: float = 0.1,
        patches_per_volume: int = 50,
        anisotropic_axis: Optional[str] = None,
        file_ext: str = ".nii.gz",
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.split = split
        self.patches_per_volume = patches_per_volume
        self.anisotropic_axis = anisotropic_axis
        self.transform = transform

        if isinstance(scale, int):
            self.scale = (scale, scale, scale)
        else:
            self.scale = tuple(scale)

        # Discover volume files
        all_files = sorted(self.data_root.rglob(f"*{file_ext}"))
        if len(all_files) == 0:
            # Also try DICOM directories
            all_files = sorted(
                [p for p in self.data_root.iterdir() if p.is_dir()],
            )
        if len(all_files) == 0:
            raise FileNotFoundError(f"No {file_ext} files found in {data_root}")

        # Train/val split
        rng = random.Random(seed)
        files_shuffled = list(all_files)
        rng.shuffle(files_shuffled)
        n_val = max(1, int(len(files_shuffled) * val_ratio))
        if split == "val":
            self.files = files_shuffled[:n_val]
        else:
            self.files = files_shuffled[n_val:]

        # Pre-load volumes (small datasets only; for large datasets, use lazy load)
        self._volumes: List[np.ndarray] = []
        for f in self.files:
            try:
                if str(f).endswith(".dcm") or f.is_dir():
                    vol = load_dicom_series(str(f))
                else:
                    vol = load_nifti(str(f))
                vol = normalize_volume(vol)
                # Ensure shape (D, H, W) — squeeze extra dims
                if vol.ndim == 4 and vol.shape[-1] <= 4:
                    vol = vol[..., 0]   # first channel
                self._volumes.append(vol)
            except Exception as e:
                print(f"[VolumetricSRDataset] Warning: could not load {f}: {e}")

        if len(self._volumes) == 0:
            raise RuntimeError("No volumes could be loaded.")

        self._length = len(self._volumes) * patches_per_volume

    def __len__(self) -> int:
        return self._length

    def _random_patch(self, vol: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract a random 3D HR patch and produce the corresponding LR patch."""
        D, H, W = vol.shape
        ps = self.patch_size

        # Ensure enough room for the patch
        if D < ps or H < ps or W < ps:
            # Pad if volume is too small
            pad_d = max(0, ps - D)
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            vol = np.pad(vol, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="reflect")
            D, H, W = vol.shape

        d0 = random.randint(0, D - ps)
        h0 = random.randint(0, H - ps)
        w0 = random.randint(0, W - ps)

        hr_patch = vol[d0:d0+ps, h0:h0+ps, w0:w0+ps]
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0)  # (1, D, H, W)

        # Downsample to LR
        sd, sh, sw = self.scale
        lr_tensor = F.interpolate(
            hr_tensor.unsqueeze(0).float(),
            size=(ps // sd, ps // sh, ps // sw),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        return lr_tensor, hr_tensor.float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx = idx % len(self._volumes)
        vol = self._volumes[vol_idx]
        lr, hr = self._random_patch(vol)

        if self.transform is not None:
            lr, hr = self.transform(lr, hr)

        return {"lr": lr, "hr": hr}
