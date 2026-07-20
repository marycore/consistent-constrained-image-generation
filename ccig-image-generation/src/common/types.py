from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PromptRecord:
    id: str
    text: str
    status: str
    complexity_class: str


@dataclass
class GenerationRecord:
    id: str
    model: str
    prompt: str
    image_path: str | None
    success: bool
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
