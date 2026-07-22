from __future__ import annotations

from .bagel import BagelModel
from .base import OpenImageModel
from .flux import FluxDevModel, FluxSchnellModel
from .hidream import HiDreamI1Model
from .janus_pro import JanusProModel
from .pixart_sigma import PixArtSigmaModel
from .qwen_image import QwenImageModel
from .sana import SanaModel
from .sd35_large import SD35LargeModel
from .showo import ShowoModel

MODEL_REGISTRY: dict[str, type[OpenImageModel]] = {
    "pixart-sigma": PixArtSigmaModel,
    "sd3.5-large": SD35LargeModel,
    "flux.1-dev": FluxDevModel,
    "flux.1-schnell": FluxSchnellModel,
    "sana": SanaModel,
    "hidream-i1": HiDreamI1Model,
    "qwen-image": QwenImageModel,
    "janus-pro": JanusProModel,
    "show-o": ShowoModel,
    "bagel": BagelModel,
}


def build_model(name: str, checkpoint: str | None = None) -> OpenImageModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](checkpoint=checkpoint)
