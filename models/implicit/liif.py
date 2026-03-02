"""LIIF — Learning Continuous Image Representation with Local Implicit Image Function.

Chen et al., CVPR 2021.

Core idea:
    An image is represented as a continuous function f: R² → R^C.
    Given a query 2D coordinate x_q, the function returns the RGB value at that point.

    f(x_q) = MLP(concat[z*, x_q - x_z*, cell_size])

    where z* is the nearest encoded feature vector to x_q.

Key advantage:
    Supports arbitrary scale factors — including non-integer and asymmetric scales —
    with a single trained model.

Architecture:
    Encoder: EDSR backbone (without the upsampler) → feature map F (B, C_feat, H_lr, W_lr)
    Decoder: MLP queried at arbitrary HR coordinates

Training:
    Randomly sample HR pixel coordinates, extract corresponding LR features,
    and supervise with pixel-level L1 loss.

Inference:
    Generate a regular grid of HR coordinates, query the MLP for each one.

Usage::

    model = LIIF(scale=4)
    # Standard inference (generates scale×4 output)
    sr = model(lr)

    # Custom scale inference
    coords = LIIF.make_coords(lr, scale=6)
    sr = model.query(lr, coords)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model
from models.cnn.edsr import ResBlock


# ---------------------------------------------------------------------------
# EDSR encoder (no upsampler)
# ---------------------------------------------------------------------------

class EDSREncoder(nn.Module):
    """EDSR feature encoder — extracts dense LR features."""

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_resblocks: int = 16,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(num_features, res_scale) for _ in range(num_resblocks)],
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )
        self.out_channels = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        return x + self.body(x)


# ---------------------------------------------------------------------------
# MLP decoder
# ---------------------------------------------------------------------------

class ImplicitDecoder(nn.Module):
    """MLP that maps (feature, relative_coord, cell_size) → RGB."""

    def __init__(self, feat_dim: int, out_channels: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = feat_dim + 2 + 2  # feature + 2D coord + 2D cell
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, feat: torch.Tensor, coord: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat:  (B, N, C_feat)  — features sampled from encoder
            coord: (B, N, 2)       — query coordinates in [-1, 1]
            cell:  (B, N, 2)       — cell size (1/H_hr, 1/W_hr)

        Returns:
            (B, N, out_channels)
        """
        return self.net(torch.cat([feat, coord, cell], dim=-1))


# ---------------------------------------------------------------------------
# LIIF model
# ---------------------------------------------------------------------------

@register_model("liif")
class LIIF(BaseSRModel):
    """Local Implicit Image Function for arbitrary-scale SR.

    Args:
        scale: Default upsampling factor used in forward().
        in_channels: Input LR image channels.
        out_channels: Output HR image channels.
        num_features: EDSR encoder feature width.
        num_resblocks: EDSR encoder depth.
        hidden_dim: MLP hidden dimension.
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_resblocks: int = 16,
        hidden_dim: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        self.encoder = EDSREncoder(in_channels, num_features, num_resblocks)
        self.decoder = ImplicitDecoder(num_features, out_channels, hidden_dim)

    # ------------------------------------------------------------------
    # Coordinate utilities
    # ------------------------------------------------------------------

    @staticmethod
    def make_coords(
        lr: torch.Tensor,
        scale: Optional[int] = None,
        hr_h: Optional[int] = None,
        hr_w: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a regular HR coordinate grid and cell sizes.

        Returns:
            coords: (B, H_hr * W_hr, 2) in [-1, 1]
            cell:   (B, H_hr * W_hr, 2)
        """
        B, _, H_lr, W_lr = lr.shape
        if hr_h is None:
            hr_h = H_lr * (scale or 4)
        if hr_w is None:
            hr_w = W_lr * (scale or 4)

        ys = torch.linspace(-1 + 1 / hr_h, 1 - 1 / hr_h, hr_h, device=lr.device)
        xs = torch.linspace(-1 + 1 / hr_w, 1 - 1 / hr_w, hr_w, device=lr.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        coords = coords.unsqueeze(0).expand(B, -1, -1)

        cell = torch.tensor([2.0 / hr_w, 2.0 / hr_h], device=lr.device)
        cell = cell.view(1, 1, 2).expand(B, coords.shape[1], -1)

        return coords, cell

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, lr: torch.Tensor, coords: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """Query arbitrary coordinates.

        Args:
            lr:     (B, C, H_lr, W_lr) LR image
            coords: (B, N, 2) query coordinates in [-1, 1]
            cell:   (B, N, 2) cell sizes

        Returns:
            (B, C_out, H_hr, W_hr) HR image reconstructed from queried pixels
        """
        feat_map = self.encoder(lr)  # (B, C_feat, H_lr, W_lr)

        # Sample features at query coordinates (nearest-neighbor feature lookup)
        # grid_sample expects coords in (B, H, W, 2) format
        B, N, _ = coords.shape
        # We need sqrt(N) to reconstruct; for non-square use caller to reshape
        H_hr = W_hr = int(math.isqrt(N))

        coords_grid = coords.view(B, H_hr, W_hr, 2)
        feat_sampled = F.grid_sample(feat_map, coords_grid, mode="nearest", align_corners=False)
        feat_sampled = feat_sampled.permute(0, 2, 3, 1).view(B, N, -1)

        rgb = self.decoder(feat_sampled, coords, cell)  # (B, N, C_out)
        return rgb.view(B, H_hr, W_hr, self.out_channels).permute(0, 3, 1, 2).clamp(0, 1)

    # ------------------------------------------------------------------
    # Standard forward (fixed scale)
    # ------------------------------------------------------------------

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        coords, cell = self.make_coords(lr, scale=self.scale)
        return self.query(lr, coords, cell)
