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
    # None (the default) means "one full epoch over dataset_path" -- DiffusersLoraTrainer.train()
    # resolves this to len(dataloader) once the dataset is loaded, since that depends on
    # dataset_path's actual size. Set an explicit number of steps to override (e.g. to train
    # for more than one epoch, or stop partway through one).
    max_steps: int | None = None
    seed: int = 42
    # Save a checkpoint every N steps (and always at max_steps), deleting the previous
    # one -- see DiffusersLoraTrainer._save_checkpoint. Keeps disk usage to one
    # checkpoint at a time instead of one per save.
    checkpoint_every: int = 500
    # Path to a previous run's checkpoint directory (as returned by train(), e.g.
    # outputs/flux.1-dev/batch1-step002000) to continue LoRA training from, instead of
    # initializing fresh LoRA weights. lora_rank/lora_alpha are ignored when set -- the
    # saved adapter's own config is used. See DiffusersLoraTrainer.train().
    init_ckpt: str | None = None

    def checkpoint_dir_for_step(self, step: int) -> Path:
        return Path(self.output_dir) / f"{self.run_name}-step{step:06d}"
