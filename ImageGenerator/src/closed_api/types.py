from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PromptItem:
    prompt_id: str
    prompt: str
    complexity_level: str
    constraint_family: str = ""
    constraints_general: str = ""
    constraints_specific: str = ""
    template_family: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def full_prompt(self) -> str:
        parts = [self.prompt, self.constraints_general, self.constraints_specific]
        return "\n".join([p for p in parts if p]).strip()


@dataclass
class GenerationResult:
    success: bool
    model_id: str
    provider: str
    prompt_id: str
    complexity_level: str
    constraint_family: str
    prompt: str
    constraints_general: str
    constraints_specific: str
    template_family: str | None
    seed_requested: int | None
    seed_supported: bool
    image_path: str | None
    generation_parameters: dict[str, Any]
    latency_seconds: float
    estimated_cost_usd: float | None
    raw_response_metadata: dict[str, Any] | None
    error_message: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
