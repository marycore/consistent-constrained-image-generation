"""DeepFloyd IF stage-I runner (inference-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from PIL import Image

from ....common import io, prompts as prompt_utils
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


@dataclass
class DeepFloydIFRunner(Runner):
    model_id: str = "deepfloyd_if"
    category: Category = "latent_diffusion_open"

    _pipe: Optional[DiffusionPipeline] = None

    def _get_pipeline(self) -> DiffusionPipeline:
        if self._pipe is not None:
            return self._pipe

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0",
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()

        self._pipe = pipe
        return pipe

    def _generate_image(
        self,
        full_prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
    ) -> tuple[Image.Image, GenerationMetadata]:
        pipe = self._get_pipeline()
        device = pipe._execution_device  # type: ignore[attr-defined]
        unet = getattr(pipe, "unet", None)
        dtype = str(getattr(unet, "dtype", torch.float32))

        generator = torch.Generator(device=device).manual_seed(seed)

        result = pipe(
            prompt=full_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
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
            resolution=image.size[::-1],
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
        # IF stage-I outputs low-res images; designed for strong text alignment.
        steps = 50
        guidance_scale = 7.0

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
        raise NotImplementedError(
            "Fine-tuning is not implemented for deepfloyd_if in this benchmark runner yet. "
            "Use pixart_sigma or flux_1_dev for training."
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
        raise NotImplementedError("run_finetuned is not supported for deepfloyd_if (inference-only).")


runner = DeepFloydIFRunner()
register(runner)

