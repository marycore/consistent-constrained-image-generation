"""SDXL base runner: base inference + LoRA fine-tuning (UNet required; text-encoder LoRA optional)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
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


def _collate_sdxl(batch: list[tuple[Image.Image, str]], resolution: int = 1024):
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
class SDXLBaseRunner(Runner):
    model_id: str = "sdxl_base"
    category: Category = "latent_diffusion_open"

    _pipe: Optional[StableDiffusionXLPipeline] = None

    def _get_pipeline(self) -> StableDiffusionXLPipeline:
        if self._pipe is not None:
            return self._pipe

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pipe.enable_model_cpu_offload()

        self._pipe = pipe
        return pipe

    def _get_pipeline_with_lora(self, ckpt_dir: str) -> StableDiffusionXLPipeline:
        """Load base pipeline and apply LoRA from <ckpt_dir>/adapters/ (PEFT format)."""
        from peft import PeftModel

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16",
        )
        adapters_path = Path(ckpt_dir) / "adapters"
        if not adapters_path.exists():
            raise FileNotFoundError(
                f"Adapters not found at {adapters_path}. Run finetune first."
            )
        pipe.unet = PeftModel.from_pretrained(pipe.unet, str(adapters_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pipe.enable_model_cpu_offload()
        return pipe

    def _generate_image(
        self,
        full_prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        pipe: Optional[StableDiffusionXLPipeline] = None,
    ) -> tuple[Image.Image, GenerationMetadata]:
        pipe = pipe or self._get_pipeline()
        device = pipe._execution_device  # type: ignore[attr-defined]
        dtype = str(pipe.unet.dtype)

        generator = torch.Generator(device=device).manual_seed(seed)

        result = pipe(
            prompt=full_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
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
            resolution=(height, width),
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
        steps = 30
        guidance_scale = 7.0
        height, width = 1024, 1024

        for record in prompts:
            full_prompt = prompt_utils.build_full_prompt(record, mode)
            image, metadata = self._generate_image(
                full_prompt=full_prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
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
    ) -> None:
        from peft import LoraConfig, get_peft_model

        seeds.set_seed(config.seed)

        resolution = config.resolution or 1024
        try:
            train_ds, val_ds = get_train_val_datasets(
                dataset_path, images_root,
                val_ratio=config.val_ratio,
                seed=config.seed,
<<<<<<< HEAD
                caption_key=config.caption_key,
=======
>>>>>>> 944ef832ccc5c8e13f4cb8c0be1cb6304a2ad873
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
            collate_fn=lambda b: _collate_sdxl(b, resolution=resolution),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            variant="fp16",
        )
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pipe.enable_model_cpu_offload()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet = get_peft_model(pipe.unet, lora_config)
        pipe.unet = unet
        unet.train()
        pipe.vae.eval()
        pipe.text_encoder.eval()
        pipe.text_encoder_2.eval()
        if device == "cuda":
            unet.enable_gradient_checkpointing()

        noise_scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler",
        )
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.lr)
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=config.max_steps,
        )

        train_config = {
            "model_id": self.model_id,
            "dataset_path": dataset_path,
            "images_root": images_root,
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
            pbar = tqdm(total=config.max_steps, desc="LoRA finetune sdxl_base")
            for pixel_values, texts in _cycle(train_loader):
                if global_step >= config.max_steps:
                    break
                pixel_values = pixel_values.to(device, dtype=dtype)
                with torch.no_grad():
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor
                    noise = torch.randn_like(latents, device=device, dtype=dtype)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,),
                        device=device,
                    )
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    (
                        prompt_embeds,
                        _,
                        pooled_embeds,
                        _,
                    ) = pipe.encode_prompt(
                        texts,
                        device=device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    add_time_ids = torch.tensor(
                        [[resolution, resolution, 0, 0, resolution, resolution]],
                        device=device,
                        dtype=prompt_embeds.dtype,
                    ).repeat(bsz, 1)
                    added_cond_kwargs = {
                        "text_embeds": pooled_embeds,
                        "time_ids": add_time_ids,
                    }
                pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
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
                "GPU OOM during SDXL fine-tuning. Try: --resolution 512, "
                "--batch_size 1, --grad_accum 8."
            ) from e

        adapters_path = io.adapters_dir(out_dir)
        pipe.unet.save_pretrained(str(adapters_path))
        print(f"Saved LoRA adapters to {adapters_path}")

    def run_finetuned(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
        ckpt_dir: str,
    ) -> None:
        pipe = self._get_pipeline_with_lora(ckpt_dir)
        steps = 30
        guidance_scale = 7.0
        height, width = 1024, 1024
        for record in prompts:
            full_prompt = prompt_utils.build_full_prompt(record, mode)
            image, metadata = self._generate_image(
                full_prompt=full_prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                pipe=pipe,
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


runner = SDXLBaseRunner()
register(runner)
