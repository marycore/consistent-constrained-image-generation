from __future__ import annotations

from .base import OpenImageModel
from .flux import FluxDevModel, FluxSchnellModel
from .qwen_image import QwenImageModel
from .sd35_large import SD35LargeModel

MODEL_REGISTRY: dict[str, type[OpenImageModel]] = {
    "sd3.5-large": SD35LargeModel,
    "flux.1-dev": FluxDevModel,
    "flux.1-schnell": FluxSchnellModel,
    "qwen-image": QwenImageModel,
}


def build_model(name: str, checkpoint: str | None = None) -> OpenImageModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](checkpoint=checkpoint)
