from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Literal, TypedDict, Dict, Any


Category = Literal[
    "latent_diffusion_open",
    "latent_diffusion_accelerated_distilled",
    "rectified_flow_open",
    "autoregressive_open",
    "closed_multimodal_transformer_api",
    "closed_diffusion_api",
]

Mode = Literal["general", "general_specific"]


class PromptRecord(TypedDict):
    id: str
    prompt: str
    constraints_general: str
    constraints_specific: str


@dataclass
class GenerationMetadata:
    model_id: str
    category: Category
    mode: Mode
    full_prompt: str
    seed: int
    steps: int | None
    guidance_scale: float | None
    resolution: tuple[int, int] | None
    dtype: str
    device: str
    scheduler: str | None
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.resolution is not None:
            data["resolution"] = list(self.resolution)
        return data


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning runs."""

    max_steps: int
    lr: float
    batch_size: int
    grad_accum: int
    seed: int
    val_ratio: float = 0.05
    resolution: int | None = None  # optional override (e.g. 512 for SDXL when OOM)
    caption_key: str = "text"  # dataset caption field to train on (e.g. text or pred)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Runner:
    """Simple interface implemented by all model runners."""

    model_id: str
    category: Category

    def run(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
    ) -> None:  # pragma: no cover - small glue method
        raise NotImplementedError

    def finetune(
        self,
        *,
        dataset_path: str,
        images_root: str,
        out_dir: str,
        config: FinetuneConfig,
        init_ckpt_dir: str | None = None,
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def run_finetuned(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
        ckpt_dir: str,
    ) -> None:  # pragma: no cover
        raise NotImplementedError


def now_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

