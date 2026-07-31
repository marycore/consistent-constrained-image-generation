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
    resolution: int = 1024
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 16
    batch_size: int = 1
    max_steps: int = 1000
    seed: int = 42
    # Save a checkpoint every N steps (and always at max_steps), deleting the previous
    # one -- see DiffusersLoraTrainer._save_checkpoint. Keeps disk usage to one
    # checkpoint at a time instead of one per save.
    checkpoint_every: int = 500

    def checkpoint_dir_for_step(self, step: int) -> Path:
        return Path(self.output_dir) / f"{self.run_name}-step{step:06d}"
