from __future__ import annotations

from PIL import Image

from .base import OpenImageModel


class ShowoModel(OpenImageModel):
    """Show-o is a unified autoregressive-diffusion transformer (showlab), shipped
    as its own repo/codebase rather than a `diffusers` pipeline -- it needs its own
    generation loop rather than `DiffusersImageModel.generate()`.
    Registered as a placeholder; implement `generate()` when this model is needed.
    """

    name = "show-o"
    hf_repo = "showlab/show-o"

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        raise NotImplementedError(
            "ShowoModel is a registry placeholder. Implement generate() using the "
            "Show-o repo's inference pipeline -- see https://github.com/showlab/Show-o."
        )

    def generate(self, prompt: str) -> Image.Image:
        raise NotImplementedError
