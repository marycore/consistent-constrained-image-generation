"""Stable Diffusion 3.5 runner: base inference + LoRA fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ....common import io, prompts as prompt_utils, seeds
from ....common.dataset import get_train_val_datasets
from ....common.registry import register
from ....common.types import (
    Runner,
    Category,
    PromptRecord,
    Mode,
    GenerationMetadata,
    FinetuneConfig,
    now_timestamp,
)


def _collate_sd35(batch: list[tuple[Image.Image, str]], resolution: int = 1024):
    """Batch (PIL, text) into pixel_values tensor and list of texts."""
    images, texts = zip(*batch)
    pixel_values = []
    for im in images:
        im = im.resize((resolution, resolution), Image.BILINEAR)
        arr = np.array(im).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        pixel_values.append(torch.from_numpy(arr).permute(2, 0, 1))
    pixel_values = torch.stack(pixel_values)
    return pixel_values, list(texts)


@dataclass
class SD35Runner(Runner):
    model_id: str = "sd3_5"
    category: Category = "latent_diffusion_open"

    _pipe: Optional[StableDiffusion3Pipeline] = None

    def _get_pipeline(self) -> StableDiffusion3Pipeline:
        if self._pipe is not None:
            return self._pipe

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pipe.enable_attention_slicing()

        self._pipe = pipe
        return pipe

    def _get_pipeline_with_lora(self, ckpt_dir: str) -> StableDiffusion3Pipeline:
        """Load base pipeline and apply LoRA from <ckpt_dir>/adapters/ (PEFT format)."""
        from peft import PeftModel

        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        adapters_path = Path(ckpt_dir) / "adapters"
        if not adapters_path.exists():
            raise FileNotFoundError(
                f"Adapters not found at {adapters_path}. Run finetune first."
            )
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(adapters_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pipe.enable_attention_slicing()
        return pipe

    def _generate_image(
        self,
        full_prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        pipe: Optional[StableDiffusion3Pipeline] = None,
    ) -> tuple[Image.Image, GenerationMetadata]:
        pipe = pipe or self._get_pipeline()
        device = pipe._execution_device  # type: ignore[attr-defined]
        dtype = str(getattr(pipe, "transformer", pipe).dtype)

        generator = torch.Generator(device=device).manual_seed(seed)

        result = pipe(
            prompt=full_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=1024,
            width=1024,
            generator=generator,
        )
        image = result.images[0]

        metadata = GenerationMetadata(
            model_id=self.model_id,
            category=self.category,
            mode="general",
            full_prompt=full_prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            resolution=(1024, 1024),
            dtype=dtype,
            device=str(device),
            scheduler=pipe.scheduler.__class__.__name__,
            timestamp=now_timestamp(),
        )
        return image, metadata

    def run(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
    ) -> None:
        steps = 28
        guidance_scale = 5.0

        for record in prompts:
            full_prompt = prompt_utils.build_full_prompt(record, mode)
            image, metadata = self._generate_image(
                full_prompt=full_prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
            )
            metadata.mode = mode
            io.save_image_and_metadata(
                image=image,
                metadata=metadata,
                output_root=output_root,
                category=self.category,
                model_id=self.model_id,
                mode=mode,
            )

    def finetune(
        self,
        *,
        dataset_path: str,
        images_root: str,
        out_dir: str,
        config: FinetuneConfig,
        init_ckpt_dir: str | None = None,
    ) -> None:
        _sd35_finetune_impl(
            self,
            dataset_path=dataset_path,
            images_root=images_root,
            out_dir=out_dir,
            config=config,
            init_ckpt_dir=init_ckpt_dir,
        )

    def run_finetuned(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
        ckpt_dir: str,
    ) -> None:
        _sd35_run_finetuned_impl(
            self,
            prompts=prompts,
            mode=mode,
            seed=seed,
            output_root=output_root,
            ckpt_dir=ckpt_dir,
        )


def _encode_sd35_prompt(pipe: StableDiffusion3Pipeline, texts: list[str], device: str):
    """
    Normalize SD3 prompt encodings across diffusers versions.

    Returns:
        (prompt_embeds, pooled_prompt_embeds)
    """
    encoded = pipe.encode_prompt(
        texts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    # Common outputs:
    # - (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
    # - (prompt_embeds, pooled_prompt_embeds)
    if isinstance(encoded, tuple):
        if len(encoded) >= 3:
            prompt_embeds = encoded[0]
            pooled_prompt_embeds = encoded[2]
            return prompt_embeds, pooled_prompt_embeds
        if len(encoded) == 2:
            return encoded[0], encoded[1]
        if len(encoded) == 1:
            return encoded[0], None
    return encoded, None


def _sd35_finetune_impl(
    runner: SD35Runner,
    *,
    dataset_path: str,
    images_root: str,
    out_dir: str,
    config: FinetuneConfig,
    init_ckpt_dir: str | None = None,
) -> None:
    from peft import LoraConfig, PeftModel, get_peft_model

    seeds.set_seed(config.seed)

    resolution = config.resolution or 1024
    try:
        train_ds, _val_ds = get_train_val_datasets(
            dataset_path,
            images_root,
            val_ratio=config.val_ratio,
            seed=config.seed,
            caption_key=config.caption_key,
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
        collate_fn=lambda b: _collate_sd35(b, resolution=resolution),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()

    if init_ckpt_dir:
        adapters_path = Path(init_ckpt_dir) / "adapters"
        if not adapters_path.exists():
            raise FileNotFoundError(
                f"init_ckpt adapters not found at {adapters_path}. "
                "Point --init_ckpt to a previous checkpoint directory containing adapters/."
            )
        transformer = PeftModel.from_pretrained(
            pipe.transformer, str(adapters_path), is_trainable=True
        )
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer = transformer
    transformer.train()
    pipe.vae.eval()
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.eval()
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2.eval()
    if getattr(pipe, "text_encoder_3", None) is not None:
        pipe.text_encoder_3.eval()
    if device == "cuda":
        transformer.enable_gradient_checkpointing()

    noise_scheduler = pipe.scheduler
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.lr)
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.max_steps,
    )

    train_config = {
        "model_id": runner.model_id,
        "dataset_path": dataset_path,
        "images_root": images_root,
        "init_ckpt_dir": init_ckpt_dir,
        **config.to_dict(),
        "resolution": resolution,
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
        pbar = tqdm(total=config.max_steps, desc="LoRA finetune sd3_5")
        for pixel_values, texts in _cycle(train_loader):
            if global_step >= config.max_steps:
                break
            pixel_values = pixel_values.to(device, dtype=dtype)
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
                scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
                latents = (latents - shift_factor) * scaling_factor

                noise = torch.randn_like(latents, device=device, dtype=dtype)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompt_embeds, pooled_prompt_embeds = _encode_sd35_prompt(
                    pipe, texts, device
                )
                if pooled_prompt_embeds is None:
                    raise RuntimeError(
                        "SD3 prompt encoding did not return pooled_prompt_embeds. "
                        "Update diffusers/transformers to recent versions."
                    )

            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=True,
            ).sample
            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
            (loss / config.grad_accum).backward()
            accum_loss += loss.item()

            if (global_step + 1) % config.grad_accum == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                io.append_train_log(
                    out_dir,
                    {
                        "step": global_step + 1,
                        "loss": accum_loss / config.grad_accum,
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                )
                accum_loss = 0.0

            global_step += 1
            pbar.update(1)
        pbar.close()
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            "GPU OOM during SD3.5 fine-tuning. Try: --resolution 768, "
            "--batch_size 1, --grad_accum 8."
        ) from e

    adapters_path = io.adapters_dir(out_dir)
    pipe.transformer.save_pretrained(str(adapters_path))
    print(f"Saved LoRA adapters to {adapters_path}")


def _sd35_run_finetuned_impl(
    runner: SD35Runner,
    *,
    prompts: list[PromptRecord],
    mode: Mode,
    seed: int,
    output_root: str,
    ckpt_dir: str,
) -> None:
    pipe = runner._get_pipeline_with_lora(ckpt_dir)
    steps = 28
    guidance_scale = 5.0
    for record in prompts:
        full_prompt = prompt_utils.build_full_prompt(record, mode)
        image, metadata = runner._generate_image(
            full_prompt=full_prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            pipe=pipe,
        )
        metadata.mode = mode
        io.save_image_and_metadata(
            image=image,
            metadata=metadata,
            output_root=output_root,
            category=runner.category,
            model_id=runner.model_id,
            mode=mode,
        )


runner = SD35Runner()
register(runner)

