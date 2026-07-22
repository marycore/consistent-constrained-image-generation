from __future__ import annotations

from pathlib import Path

from src.common.types import TrainConfig

from .base import LoraTrainer


class JanusProTrainer(LoraTrainer):
    """Janus-Pro is a unified multimodal transformer (DeepSeek), not a diffusers
    pipeline -- LoRA finetuning needs the `janus` package's own model/training code
    rather than `DiffusersLoraTrainer`. Registered as a placeholder.
    """

    name = "janus-pro"
    hf_repo = "deepseek-ai/Janus-Pro-7B"

    def train(self, config: TrainConfig) -> Path:
        raise NotImplementedError(
            "JanusProTrainer is a registry placeholder. Implement train() using the "
            "`janus` package -- see https://github.com/deepseek-ai/Janus."
        )
