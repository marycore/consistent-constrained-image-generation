from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from ....common import io as common_io, prompts as prompt_utils
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
class Dalle3Runner(Runner):
    model_id: str = "dalle3"
    category: Category = "closed_multimodal_transformer_api"

    def run(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Required for dalle3 inference.")

        # Lazy import so users without the dependency/key can still use open models.
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)

        for record in prompts:
            full_prompt = prompt_utils.build_full_prompt(record, mode)

            # DALL·E 3 does not support user-controlled deterministic seeds via the public API.
            # We still record the provided seed in metadata for benchmarking consistency.
            resp = client.images.generate(
                model="dall-e-3",
                prompt=full_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json",
            )

            b64 = resp.data[0].b64_json
            img_bytes = base64.b64decode(b64)
            image = Image.open(BytesIO(img_bytes)).convert("RGB")

            metadata = GenerationMetadata(
                model_id=self.model_id,
                category=self.category,
                mode=mode,
                full_prompt=full_prompt,
                seed=seed,
                steps=None,
                guidance_scale=None,
                resolution=(1024, 1024),
                dtype="api",
                device="api",
                scheduler=None,
                timestamp=now_timestamp(),
            )

            common_io.save_image_and_metadata(
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
        raise NotImplementedError("Fine-tuning is not supported for dalle3 (closed API).")

    def run_finetuned(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
        ckpt_dir: str,
    ) -> None:
        raise NotImplementedError("run_finetuned is not supported for dalle3 (closed API).")


runner = Dalle3Runner()
register(runner)

