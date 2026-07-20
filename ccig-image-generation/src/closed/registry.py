from __future__ import annotations

from .base import ClosedImageModel
from .gemini import GeminiFlashImage
from .gpt_image import GPTImage1

MODEL_REGISTRY: dict[str, type[ClosedImageModel]] = {
    "gemini-2.0-flash": GeminiFlashImage,
    "gpt-image-1": GPTImage1,
}


def build_model(name: str) -> ClosedImageModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()
