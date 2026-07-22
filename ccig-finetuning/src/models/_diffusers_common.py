from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.common.dataset import load_examples
from src.common.types import TrainConfig

from .base import LoraTrainer


class _ImagePromptDataset(Dataset):
    def __init__(self, config: TrainConfig, resolution: int) -> None:
        self.examples = list(
            load_examples(config.dataset_path, config.images_dir, config.prompt_field)
        )
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        image = Image.open(ex.image_path).convert("RGB").resize((self.resolution, self.resolution))
        pixel_values = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        return {"pixel_values": pixel_values, "prompt": ex.prompt}


class DiffusersLoraTrainer(LoraTrainer):
    """Shared LoRA finetuning scaffold for models built on a `diffusers` pipeline.

    Handles the model-agnostic parts once -- pipeline loading, LoRA injection via
    `peft`, the dataset/dataloader, the optimizer loop, and checkpoint saving (via
    PEFT's `save_pretrained`, matching ccig-image-generation's `--checkpoint` loader).

    `_training_step` (forward pass + loss) is deliberately NOT given a generic
    default: diffusers architectures differ in scheduler type (flow-matching vs.
    DDPM) and transformer forward signature (single vs. packed latents, extra
    positional ids, pooled projections, ...), so a one-size-fits-all step would
    silently produce wrong gradients for at least half of these models. Each
    subclass implements its own, informed by that model's actual pipeline.
    """

    pipeline_cls: ClassVar[Any]
    torch_dtype: ClassVar[Any] = torch.bfloat16
    lora_target_modules: ClassVar[list[str]] = ["to_k", "to_q", "to_v", "to_out.0"]

    def _load_pipeline(self) -> Any:
        pipe = self.pipeline_cls.from_pretrained(self.hf_repo, torch_dtype=self.torch_dtype)
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipe

    def train(self, config: TrainConfig) -> Path:
        torch.manual_seed(config.seed)
        pipe = self._load_pipeline()

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=self.lora_target_modules,
        )
        transformer = get_peft_model(pipe.transformer, lora_config)
        pipe.transformer = transformer
        transformer.train()
        pipe.vae.eval()
        pipe.vae.requires_grad_(False)

        dataset = _ImagePromptDataset(config, config.resolution)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            (p for p in transformer.parameters() if p.requires_grad), lr=config.learning_rate
        )

        def _cycle(dl):
            while True:
                yield from dl

        step = 0
        for batch in _cycle(loader):
            if step >= config.max_steps:
                break
            loss = self._training_step(pipe, transformer, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 50 == 0 or step == config.max_steps:
                print(f"[{self.name}] step {step}/{config.max_steps} loss={loss.item():.4f}")

        out_dir = config.checkpoint_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        transformer.save_pretrained(str(out_dir))
        return out_dir

    def _training_step(self, pipe: Any, transformer: Any, batch: dict) -> torch.Tensor:
        raise NotImplementedError
