"""Global model registry.

Usage::

    from models.registry import register_model, build_model

    @register_model("my_model")
    class MyModel(BaseSRModel):
        ...

    model = build_model(cfg)  # cfg.model.name == "my_model"
"""

from typing import Callable, Dict, Type

import torch.nn as nn

_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable:
    """Class decorator that registers a model under a given name."""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_model(cfg) -> nn.Module:
    """Instantiate a model from OmegaConf config (cfg.model)."""
    from omegaconf import OmegaConf

    model_cfg = cfg.model
    name = model_cfg.name
    if name not in _REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    kwargs = OmegaConf.to_container(model_cfg, resolve=True)
    kwargs.pop("name")
    return _REGISTRY[name](**kwargs)


def list_models():
    return sorted(_REGISTRY.keys())
