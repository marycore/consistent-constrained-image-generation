from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import ZImagePipeline
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
class ZImageTurboRunner(Runner):
    model_id: str = "z_image_turbo"
    category: Category = "latent_diffusion_accelerated_distilled"

    _pipe: Optional[ZImagePipeline] = None

    def _get_pipeline(self) -> ZImagePipeline:
        if self._pipe is not None:
            return self._pipe

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Z-Image Turbo recommends bfloat16 on GPU.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        pipe = pipe.to(device)

        # Use SDPA by default; try enabling a more efficient backend if available.
        try:
            pipe.transformer.set_attention_backend("flash")  # type: ignore[call-arg]
        except Exception:
            # Fallback: standard SDPA / CPU offload if needed.
            if device == "cuda":
                pipe.enable_model_cpu_offload()

        self._pipe = pipe
        return pipe

    def _generate_image(
        self,
        full_prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        height: int,
        width: int,
    ) -> tuple[Image.Image, GenerationMetadata]:
        pipe = self._get_pipeline()
        device = pipe._execution_device  # type: ignore[attr-defined]
        dtype = str(getattr(pipe, "transformer", pipe).dtype)

        generator = torch.Generator(device=device).manual_seed(seed)

        result = pipe(
            prompt=full_prompt,
            height=height,
            width=width,
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
            resolution=(height, width),
            dtype=dtype,
            device=str(device),
            scheduler=None,
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
        # Defaults aligned with the HF quick-start: 9 steps, guidance_scale=0, 1024x1024.
        steps = 9
        guidance_scale = 0.0
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
        raise NotImplementedError(
            "Fine-tuning for z_image_turbo is not wired up yet. "
            "Consider training LoRA/adapter weights on top of Tongyi-MAI/Z-Image-Turbo."
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
        raise NotImplementedError("run_finetuned for z_image_turbo is not implemented yet.")


runner = ZImageTurboRunner()
register(runner)

