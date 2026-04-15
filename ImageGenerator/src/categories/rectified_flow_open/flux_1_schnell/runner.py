from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import FluxPipeline
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
class Flux1SchnellRunner(Runner):
    model_id: str = "flux_1_schnell"
    category: Category = "rectified_flow_open"

    _pipe: Optional[FluxPipeline] = None

    def _get_pipeline(self) -> FluxPipeline:
        if self._pipe is not None:
            return self._pipe

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
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
        steps = 4
        guidance_scale = 1.0
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
        raise NotImplementedError(
            "Fine-tuning is not supported for flux_1_schnell (inference-only). "
            "Use flux_1_dev for fine-tuning."
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
        raise NotImplementedError("run_finetuned is not supported for flux_1_schnell (inference-only).")


runner = Flux1SchnellRunner()
register(runner)

