"""Model package — importing submodules triggers @register_model decorators."""

from .base_model import BaseSRModel
from .registry import build_model, list_models, register_model

# Auto-import all model families to populate the registry
from .interpolation import classical  # noqa: F401
from .cnn import srcnn, vdsr, edsr, rcan, espcn  # noqa: F401
from .cnn import fsrcnn, drrn, rdn, dbpn  # noqa: F401
from .gan import srgan, esrgan, real_esrgan  # noqa: F401
from .transformer import swinir, hat, restormer  # noqa: F401
from .diffusion import sr3, ddnm  # noqa: F401
from .lightweight import imdn  # noqa: F401
from .flow import srflow  # noqa: F401
from .implicit import liif  # noqa: F401
from .mamba import mambair  # noqa: F401
from .physics import dasr, ikc, dpsr, unrolled_sr  # noqa: F401
from .volumetric import srcnn3d, edsr3d, med_sr  # noqa: F401

__all__ = ["BaseSRModel", "build_model", "list_models", "register_model"]
