from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .types import GenerationResult


class ImageGenerationModel(ABC):
    model_id: str
    provider: str
    seed_supported: bool = False

    @abstractmethod
    def generate(
        self,
        *,
        prompt: str,
        output_path: str,
        prompt_id: str,
        complexity_level: str,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        timeout_seconds: int = 180,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> GenerationResult:
        raise NotImplementedError

    @staticmethod
    def ensure_parent(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
