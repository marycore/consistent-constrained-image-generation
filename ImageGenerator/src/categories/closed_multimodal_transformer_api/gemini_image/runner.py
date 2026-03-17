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
class GeminiImageRunner(Runner):
    model_id: str = "gemini_image"
    category: Category = "closed_multimodal_transformer_api"

    def run(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Required for gemini_image inference.")

        # Lazy import so users without the dependency/key can still use open models.
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)

        # Model name is intentionally configurable via env var to avoid hard-coding vendor naming changes.
        model_name = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-exp")
        model = genai.GenerativeModel(model_name)

        for record in prompts:
            full_prompt = prompt_utils.build_full_prompt(record, mode)

            # Deterministic seeding is generally not guaranteed for hosted APIs.
            # We record the requested seed in metadata for benchmark reproducibility.
            resp = model.generate_content(
                full_prompt,
                generation_config={"temperature": 0.0},
            )

            # Attempt to extract an image from the response.
            # Different SDK versions expose slightly different shapes; handle common cases.
            image: Image.Image | None = None
            if getattr(resp, "parts", None):
                for part in resp.parts:
                    b = getattr(part, "inline_data", None)
                    if b and getattr(b, "data", None):
                        img_bytes = base64.b64decode(b.data)
                        image = Image.open(BytesIO(img_bytes)).convert("RGB")
                        break

            if image is None:
                raise RuntimeError(
                    "Gemini response did not contain an image payload. "
                    "Check GEMINI_IMAGE_MODEL and SDK version."
                )

            metadata = GenerationMetadata(
                model_id=self.model_id,
                category=self.category,
                mode=mode,
                full_prompt=full_prompt,
                seed=seed,
                steps=None,
                guidance_scale=None,
                resolution=None,
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
    ) -> None:
        raise NotImplementedError("Fine-tuning is not supported for gemini_image (closed API).")

    def run_finetuned(
        self,
        *,
        prompts: list[PromptRecord],
        mode: Mode,
        seed: int,
        output_root: str,
        ckpt_dir: str,
    ) -> None:
        raise NotImplementedError("run_finetuned is not supported for gemini_image (closed API).")


runner = GeminiImageRunner()
register(runner)

