from __future__ import annotations

from pathlib import Path

from src.common.types import TrainConfig

from .base import LoraTrainer


class BagelTrainer(LoraTrainer):
    """BAGEL (ByteDance-Seed) is a unified multimodal understanding+generation
    transformer, not a diffusers pipeline -- LoRA finetuning needs the `bagel` repo's
    own model/training code rather than `DiffusersLoraTrainer`. Registered as a
    placeholder.
    """

    name = "bagel"
    hf_repo = "ByteDance-Seed/BAGEL-7B-MoT"

    def train(self, config: TrainConfig) -> Path:
        raise NotImplementedError(
            "BagelTrainer is a registry placeholder. Implement train() using the "
            "BAGEL repo's training code -- see https://github.com/ByteDance-Seed/Bagel."
        )
