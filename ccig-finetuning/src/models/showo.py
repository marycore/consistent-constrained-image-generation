from __future__ import annotations

from pathlib import Path

from src.common.types import TrainConfig

from .base import LoraTrainer


class ShowoTrainer(LoraTrainer):
    """Show-o is a unified autoregressive-diffusion transformer (showlab), shipped
    as its own repo/codebase rather than a `diffusers` pipeline -- LoRA finetuning
    needs its own training code rather than `DiffusersLoraTrainer`. Registered as a
    placeholder.
    """

    name = "show-o"
    hf_repo = "showlab/show-o"

    def train(self, config: TrainConfig) -> Path:
        raise NotImplementedError(
            "ShowoTrainer is a registry placeholder. Implement train() using the "
            "Show-o repo's training code -- see https://github.com/showlab/Show-o."
        )
