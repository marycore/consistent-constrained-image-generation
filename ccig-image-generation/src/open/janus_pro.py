from __future__ import annotations

from PIL import Image

from .base import OpenImageModel


class JanusProModel(OpenImageModel):
    """Janus-Pro is a unified multimodal transformer (DeepSeek), not a diffusers
    pipeline -- it needs the `janus` package (github.com/deepseek-ai/Janus) and its
    own image-token sampling loop rather than a `DiffusersImageModel.generate()`.
    Registered as a placeholder; implement `generate()` when this model is needed.
    """

    name = "janus-pro"
    hf_repo = "deepseek-ai/Janus-Pro-7B"

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        raise NotImplementedError(
            "JanusProModel is a registry placeholder. Implement generate() using the "
            "`janus` package's MultiModalityCausalLM + image-token sampling loop -- "
            "see https://github.com/deepseek-ai/Janus."
        )

    def generate(self, prompt: str) -> Image.Image:
        raise NotImplementedError
