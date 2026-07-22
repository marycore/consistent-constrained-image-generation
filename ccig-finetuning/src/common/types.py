from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainExample:
    id: int
    image_path: Path
    prompt: str


@dataclass
class TrainConfig:
    dataset_path: str
    images_dir: str
    output_dir: str
    run_name: str = "run1"
    prompt_field: str = "medium"
    resolution: int = 1024
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 16
    batch_size: int = 1
    max_steps: int = 1000
    seed: int = 42

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name
