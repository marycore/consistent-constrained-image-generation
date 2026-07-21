from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PromptRecord:
    id: str
    text: str
    number_of_objects: int
    domain: str
    status: str
    complexity_class: str


@dataclass
class GenerationRecord:
    id: str
    model: str
    prompt: str
    prompt_field: str
    scene_generation_setup: str
    image_path: str | None
    success: bool
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
