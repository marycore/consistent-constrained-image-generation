from __future__ import annotations

from PIL import Image

from .base import OpenImageModel


class BagelModel(OpenImageModel):
    """BAGEL (ByteDance-Seed) is a unified multimodal understanding+generation
    transformer (mixture-of-transformer-experts over a Qwen2.5-backbone), not a
    diffusers pipeline -- it needs the `bagel` repo's own model/generation code.
    Registered as a placeholder; implement `generate()` when this model is needed.
    """

    name = "bagel"
    hf_repo = "ByteDance-Seed/BAGEL-7B-MoT"

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        raise NotImplementedError(
            "BagelModel is a registry placeholder. Implement generate() using the "
            "BAGEL repo's inference code -- see https://github.com/ByteDance-Seed/Bagel."
        )

    def generate(self, prompt: str) -> Image.Image:
        raise NotImplementedError
