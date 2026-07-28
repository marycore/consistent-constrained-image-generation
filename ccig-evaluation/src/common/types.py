from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PromptRecord:
    """One record from ccig_eval_dataset_{SAT,UNSAT}.jsonl, as produced by ccig-dataset-gen."""

    id: str
    domain: str
    complexity_class: str
    constraint_family: str
    prompts: dict[str, str]  # {"short": ..., "medium": ..., "long": ...}
    instantiated_rule: str
    asp_template_file: str
    status: str  # "SAT" | "UNSAT", the ground-truth label for the *original* symbolic scene
    number_of_objects: int


@dataclass
class MatchedItem:
    """A generated image joined to the prompt record that produced it."""

    id: str
    prompt_field: str  # "short" | "medium" | "long"
    image_path: Path
    record: PromptRecord

    @property
    def prompt_text(self) -> str:
        return self.record.prompts[self.prompt_field]


@dataclass
class ClipScoreResult:
    id: str
    prompt_field: str
    prompt: str
    image_path: str
    clipscore: float | None
    success: bool
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeResult:
    id: str
    prompt_field: str
    prompt: str
    image_path: str
    score: float | None  # normalized to 0..1
    raw_score: float | int | None  # backend's native scale, for audit
    rationale: str | None
    success: bool
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerceptionResult:
    id: str
    prompt_field: str
    image_path: str
    instantiated_rule: str
    dataset_status: str
    predicted_status: str | None
    agrees_with_dataset: bool | None
    scene_graph: dict[str, Any] | None
    clingo_program: str | None
    success: bool
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
