"""FLUX.1-dev runner: base inference + QLoRA/LoRA fine-tuning (parameter-efficient)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import FluxPipeline
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


# FIXED: FLUX uses a packed-patch latent format; these helpers convert between
# the standard (B, C, H, W) VAE latent space and FLUX's (B, H//2*W//2, C*4) packed form.
def _flux_pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H//2 * W//2, C*4) – 2×2 spatial patch packing for FLUX."""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 2) * (W // 2), C * 4)
    return latents


def _flux_unpack_latents(packed: torch.Tensor, H: int, W: int, C: int) -> torch.Tensor:
    """(B, H//2*W//2, C*4) -> (B, C, H, W)."""
    B = packed.shape[0]
    packed = packed.reshape(B, H // 2, W // 2, C, 2, 2)
    packed = packed.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
    return packed


def _flux_prepare_img_ids(bsz: int, latent_h: int, latent_w: int, device) -> torch.Tensor:
    """Build spatial position IDs for FLUX RoPE: (B, H//2*W//2, 3)."""
    h_p, w_p = latent_h // 2, latent_w // 2
    img_ids = torch.zeros(h_p, w_p, 3, device=device)
    img_ids[..., 1] = torch.arange(h_p, device=device).float()[:, None]
    img_ids[..., 2] = torch.arange(w_p, device=device).float()[None, :]
    return img_ids.reshape(h_p * w_p, 3).unsqueeze(0).expand(bsz, -1, -1)


def _collate_flux(batch: list[tuple[Image.Image, str]], resolution: int = 1024):
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
class Flux1DevRunner(Runner):
    model_id: str = "flux_1_dev"
    category: Category = "rectified_flow_open"

    _pipe: Optional[FluxPipeline] = None

    def _get_pipeline(self) -> FluxPipeline:
        if self._pipe is not None:
            return self._pipe

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pipe.enable_model_cpu_offload()

        self._pipe = pipe
        return pipe

    def _get_pipeline_with_lora(self, ckpt_dir: str) -> FluxPipeline:
        """Load base pipeline and apply LoRA from <ckpt_dir>/adapters/ (PEFT format)."""
        from peft import PeftModel

        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
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
        pipe: Optional[FluxPipeline] = None,
    ) -> tuple[Image.Image, GenerationMetadata]:
        pipe = pipe or self._get_pipeline()
        device = pipe._execution_device  # type: ignore[attr-defined]
        dtype = str(getattr(pipe.transformer, "dtype", "unknown"))

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
            scheduler=getattr(pipe, "scheduler", None).__class__.__name__ if getattr(pipe, "scheduler", None) else None,
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
        guidance_scale = 4.5
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
        init_ckpt_dir: str | None = None,
    ) -> None:
        from peft import LoraConfig, PeftModel, get_peft_model

        seeds.set_seed(config.seed)

        resolution = config.resolution or 1024
        max_sequence_length = config.max_sequence_length
        try:
            train_ds, val_ds = get_train_val_datasets(
                dataset_path, images_root,
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
            collate_fn=lambda b: _collate_flux(b, resolution=resolution),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Try QLoRA (4-bit) first; fall back to full LoRA if not supported
        transformer = None
        use_qlora = False
        try:
            from transformers import BitsAndBytesConfig
            from diffusers import FluxTransformer2DModel

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="transformer",
                quantization_config=bnb_config,
                torch_dtype=dtype,
            )
            use_qlora = True
        except Exception:
            pass

        if transformer is None:
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=dtype,
            )
            transformer = pipe.transformer
        else:
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                transformer=transformer,
                torch_dtype=dtype,
            )

        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pipe.enable_model_cpu_offload()

        if init_ckpt_dir:
            adapters_path = Path(init_ckpt_dir) / "adapters"
            if not adapters_path.exists():
                raise FileNotFoundError(
                    f"init_ckpt adapters not found at {adapters_path}. "
                    "Point --init_ckpt to a previous checkpoint directory containing adapters/."
                )
            transformer = PeftModel.from_pretrained(
                pipe.transformer,
                str(adapters_path),
                is_trainable=True,
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
        # FIXED: FLUX has both CLIP (text_encoder) and T5 (text_encoder_2); both must be frozen
        if getattr(pipe, "text_encoder_2", None) is not None:
            pipe.text_encoder_2.eval()
        if device == "cuda":
            transformer.enable_gradient_checkpointing()

        # FIXED: FLUX uses FlowMatchEulerDiscreteScheduler, not DDPM; use pipe.scheduler directly
        noise_scheduler = pipe.scheduler
        optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.lr)
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
            "init_ckpt_dir": init_ckpt_dir,
            **config.to_dict(),
            "resolution": resolution,
            "use_qlora": use_qlora,
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
            pbar = tqdm(total=config.max_steps, desc="QLoRA/LoRA finetune flux_1_dev")
            for pixel_values, texts in _cycle(train_loader):
                if global_step >= config.max_steps:
                    break
                pixel_values = pixel_values.to(device, dtype=dtype)
                with torch.no_grad():
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor
                    _lat_C, _lat_H, _lat_W = latents.shape[1], latents.shape[2], latents.shape[3]

                    # FIXED: FLUX requires packing latents from (B,C,H,W) to (B,H//2*W//2,C*4) patches
                    packed_latents = _flux_pack_latents(latents)
                    noise = torch.randn_like(latents, device=device, dtype=dtype)
                    packed_noise = _flux_pack_latents(noise)

                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,),
                        device=device,
                    )
                    # FIXED: flow-matching noisy = sigma*noise + (1-sigma)*latents (on packed tensors)
                    sigma = (timesteps.float() / noise_scheduler.config.num_train_timesteps).view(
                        -1, 1, 1
                    ).to(device=device, dtype=dtype)
                    noisy_packed = sigma * packed_noise + (1.0 - sigma) * packed_latents

                    # FIXED: encode_prompt returns (prompt_embeds, pooled_prompt_embeds, text_ids)
                    _enc = pipe.encode_prompt(
                        texts,
                        device=device,
                        num_images_per_prompt=1,
                        max_sequence_length=max_sequence_length or 512,
                    )
                    if isinstance(_enc, (tuple, list)) and len(_enc) >= 3:
                        prompt_embeds, pooled_prompt_embeds, text_ids = _enc[0], _enc[1], _enc[2]
                    elif isinstance(_enc, (tuple, list)) and len(_enc) == 2:
                        prompt_embeds, pooled_prompt_embeds = _enc[0], _enc[1]
                        text_ids = torch.zeros(bsz, prompt_embeds.shape[1], 3, device=device)
                    else:
                        raise RuntimeError("Unexpected return shape from FluxPipeline.encode_prompt()")

                    # FIXED: generate image positional IDs required by FLUX RoPE
                    img_ids = _flux_prepare_img_ids(bsz, _lat_H, _lat_W, device)

                # FIXED: transformer requires packed latents + explicit positional IDs + pooled projections
                # NOTE: target_modules ["to_k","to_q","to_v","to_out.0"] may miss FLUX double-stream
                #       text projections (add_q_proj etc.); flag for review if convergence is slow
                pred_packed = transformer(
                    hidden_states=noisy_packed,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=timesteps,
                    img_ids=img_ids,
                    txt_ids=text_ids,
                    return_dict=True,
                ).sample
                # FIXED: flow-matching target is the velocity field (noise - latents) in packed form
                flow_target = packed_noise - packed_latents
                loss = torch.nn.functional.mse_loss(pred_packed.float(), flow_target.float())
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
                    # FIXED: periodic in-place checkpoint so a crashed/terminated pod loses at most N steps
                    if config.save_every_n_steps > 0 and (global_step + 1) % config.save_every_n_steps == 0:
                        _ckpt = io.adapters_dir(out_dir)
                        pipe.transformer.save_pretrained(str(_ckpt))
                        print(f"[step {global_step + 1}] Saved intermediate checkpoint to {_ckpt}")

                global_step += 1
                pbar.update(1)
            pbar.close()
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                "GPU OOM during FLUX fine-tuning. Try: --resolution 512, "
                "--batch_size 1, --grad_accum 8. QLoRA (4-bit) reduces VRAM if supported."
            ) from e

        adapters_path = io.adapters_dir(out_dir)
        pipe.transformer.save_pretrained(str(adapters_path))
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
        steps = 28
        guidance_scale = 4.5
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


runner = Flux1DevRunner()
register(runner)
