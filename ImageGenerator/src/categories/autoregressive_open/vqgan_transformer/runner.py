"""VQGAN + Transformer runner: minimal transformer prior fine-tuning (Taming Transformers–style)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ....common import io, prompts, seeds
from ....common.dataset import get_train_val_datasets
from ....common.registry import register
from ....common.types import (
    Runner,
    Category,
    PromptRecord,
    Mode,
    FinetuneConfig,
)


# Minimal transformer prior: causal LM over discrete image tokens conditioned on text.
# Uses a dummy tokenization (16x16 patch indices) when no VQGAN is available.
VOCAB_SIZE = 256
MAX_SEQ_LEN = 256
TEXT_DIM = 512
PRIOR_DIM = 512
PRIOR_LAYERS = 4
PRIOR_HEADS = 8


def _dummy_image_to_tokens(image: Image.Image, size: int = 16) -> np.ndarray:
    """Placeholder: resize image, flatten to grid, quantize to [0, VOCAB_SIZE-1]."""
    img = image.resize((size, size)).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr * (VOCAB_SIZE - 1)).astype(np.int64)
    return arr.flatten()


class _MinimalTransformerPrior(nn.Module):
    """Small causal transformer: text condition + previous tokens -> next token logits."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = MAX_SEQ_LEN,
        text_dim: int = TEXT_DIM,
        d_model: int = PRIOR_DIM,
        n_layers: int = PRIOR_LAYERS,
        n_heads: int = PRIOR_HEADS,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.text_proj = nn.Linear(text_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, S), text_embeds: (B, text_dim)
        B, S = token_ids.shape
        x = self.token_embed(token_ids) + self.pos_embed[:, :S]
        cond = self.text_proj(text_embeds).unsqueeze(1)
        x = x + cond
        causal_mask = torch.triu(
            torch.ones(S, S, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        out = self.transformer(x, mask=causal_mask)
        logits = self.head(out)
        return logits


def _collate_vqgan(batch: list[tuple[Image.Image, str]], max_len: int = MAX_SEQ_LEN):
    """Batch (PIL, text) into token_ids and dummy text embeddings (placeholder)."""
    import numpy as np
    images, texts = zip(*batch)
    token_list = []
    for im in images:
        ids = _dummy_image_to_tokens(im)
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = np.pad(ids, (0, max_len - len(ids)), constant_values=0)
        token_list.append(ids)
    token_ids = torch.from_numpy(np.stack(token_list)).long()
    text_embeds = torch.randn(len(texts), TEXT_DIM)
    return token_ids, text_embeds, list(texts)


@dataclass
class VQGANTransformerRunner(Runner):
    model_id: str = "vqgan_transformer"
    category: Category = "autoregressive_open"

    def run(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
    ) -> None:
        raise NotImplementedError(
            "vqgan_transformer inference is not implemented yet. "
            "Planned: load VQGAN + transformer prior, sample tokens autoregressively, decode with VQGAN."
        )

    def finetune(
        self,
        *,
        dataset_path: str,
        images_root: str,
        out_dir: str,
        config: FinetuneConfig,
    ) -> None:
        seeds.set_seed(config.seed)

        resolution = config.resolution or 256
        try:
            train_ds, val_ds = get_train_val_datasets(
                dataset_path, images_root,
                val_ratio=config.val_ratio,
                seed=config.seed,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Dataset or images root invalid: {e}. Check --data and --images_root."
            ) from e

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_collate_vqgan,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        prior = _MinimalTransformerPrior().to(device)
        optimizer = torch.optim.AdamW(prior.parameters(), lr=config.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_steps
        )

        train_config = {
            "model_id": self.model_id,
            "dataset_path": dataset_path,
            "images_root": images_root,
            **config.to_dict(),
            "resolution": resolution,
            "vocab_size": VOCAB_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
        }
        io.save_train_config(out_dir, train_config)

        global_step = 0
        accum_loss = 0.0
        optimizer.zero_grad()

        def _cycle(loader):
            while True:
                for batch in loader:
                    yield batch

        try:
            pbar = tqdm(total=config.max_steps, desc="Transformer prior finetune")
            for token_ids, text_embeds, _ in _cycle(train_loader):
                if global_step >= config.max_steps:
                    break
                token_ids = token_ids.to(device)
                text_embeds = text_embeds.to(device)
                logits = prior(token_ids[:, :-1], text_embeds)
                target = token_ids[:, 1:]
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    target.reshape(-1),
                    ignore_index=0,
                )
                (loss / config.grad_accum).backward()
                accum_loss += loss.item()

                if (global_step + 1) % config.grad_accum == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    io.append_train_log(out_dir, {
                        "step": global_step + 1,
                        "loss": accum_loss / config.grad_accum,
                        "lr": lr_scheduler.get_last_lr()[0],
                    })
                    accum_loss = 0.0

                global_step += 1
                pbar.update(1)
            pbar.close()
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                "GPU OOM during transformer prior fine-tuning. "
                "Try --batch_size 1, --grad_accum 8."
            ) from e

        ckpt_path = Path(out_dir) / "transformer.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": prior.state_dict(), "config": train_config}, ckpt_path)
        print(f"Saved transformer prior to {ckpt_path}")

    def run_finetuned(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
        ckpt_dir: str,
    ) -> None:
        raise NotImplementedError(
            "run_finetuned for vqgan_transformer is not implemented yet. "
            "Full generation requires a VQGAN decoder and autoregressive sampling."
        )


runner = VQGANTransformerRunner()
register(runner)
