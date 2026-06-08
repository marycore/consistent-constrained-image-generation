from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from typing import Any

from PIL import Image

from .base import ImageGenerationModel
from .types import GenerationResult


COST_DEFAULTS: dict[str, float | None] = {
    "openai_dalle_3": 0.04,
    "openai_gpt_image_1_5": 0.034,
    "openai_gpt_image_2": 0.053,
    "google_nano_banana": 0.04,
    "google_imagen_4": 0.04,
    "google_imagen_4_ultra": 0.06,
    "ideogram_3": None,
    "recraft": None,
    "firefly": None,
    "nova_canvas": None,
    "seedream": None,
    "grok_imagine": None,
}


def _retry(fn, retries: int, backoff: float = 2.0):
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception:
            if attempt > retries:
                raise
            time.sleep(backoff * attempt)


class _OpenAIImageModel(ImageGenerationModel):
    api_model: str
    model_id: str
    provider: str = "openai"
    seed_supported: bool = False

    def generate(
        self,
        *,
        prompt: str,
        output_path: str,
        prompt_id: str,
        complexity_level: str,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        timeout_seconds: int = 180,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> GenerationResult:
        start = time.monotonic()
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            from openai import OpenAI  # type: ignore
            import urllib.request

            client = OpenAI(api_key=api_key)
            self.ensure_parent(output_path)
            response = _retry(
                lambda: client.images.generate(
                    model=self.api_model,
                    prompt=prompt,
                    size=f"{width}x{height}",
                ),
                retries=max_retries,
            )
            item = response.data[0]
            if item.b64_json:
                img = Image.open(BytesIO(base64.b64decode(item.b64_json))).convert("RGB")
            elif item.url:
                with urllib.request.urlopen(item.url) as resp:
                    img = Image.open(BytesIO(resp.read())).convert("RGB")
            else:
                raise RuntimeError("OpenAI image response contains neither b64_json nor url.")
            img.save(output_path)
            return GenerationResult(
                success=True,
                model_id=self.model_id,
                provider=self.provider,
                prompt_id=prompt_id,
                complexity_level=complexity_level,
                constraint_family=str(kwargs.get("constraint_family", "")),
                prompt=prompt,
                constraints_general=str(kwargs.get("constraints_general", "")),
                constraints_specific=str(kwargs.get("constraints_specific", "")),
                template_family=kwargs.get("template_family"),
                seed_requested=seed,
                seed_supported=self.seed_supported,
                image_path=output_path,
                generation_parameters={"width": width, "height": height, "num_images": num_images},
                latency_seconds=time.monotonic() - start,
                estimated_cost_usd=COST_DEFAULTS.get(self.model_id),
                raw_response_metadata={"provider_response": "openai.images.generate"},
                error_message=None,
            )
        except Exception as e:
            return GenerationResult(
                success=False,
                model_id=self.model_id,
                provider=self.provider,
                prompt_id=prompt_id,
                complexity_level=complexity_level,
                constraint_family=str(kwargs.get("constraint_family", "")),
                prompt=prompt,
                constraints_general=str(kwargs.get("constraints_general", "")),
                constraints_specific=str(kwargs.get("constraints_specific", "")),
                template_family=kwargs.get("template_family"),
                seed_requested=seed,
                seed_supported=self.seed_supported,
                image_path=None,
                generation_parameters={"width": width, "height": height, "num_images": num_images},
                latency_seconds=time.monotonic() - start,
                estimated_cost_usd=COST_DEFAULTS.get(self.model_id),
                raw_response_metadata=None,
                error_message=repr(e),
            )


class OpenAIGPTImage15(_OpenAIImageModel):
    model_id = "openai_gpt_image_1_5"
    api_model = "gpt-image-1"


class OpenAIGPTImage2(_OpenAIImageModel):
    model_id = "openai_gpt_image_2"
    api_model = "gpt-image-1"


class OpenAIDalle3(_OpenAIImageModel):
    model_id = "openai_dalle_3"
    # dall-e-3 was removed from the v2 images endpoint; gpt-image-1 is the replacement
    api_model = "gpt-image-1"


class GoogleNanoBanana(ImageGenerationModel):
    model_id = "google_nano_banana"
    provider = "google"
    seed_supported = False

    def generate(self, **kwargs: Any) -> GenerationResult:
        return _generate_google_gemini(
            model_id=self.model_id,
            api_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-preview-image-generation"),
            cost=COST_DEFAULTS[self.model_id],
            **kwargs,
        )


class GoogleImagen4(ImageGenerationModel):
    model_id = "google_imagen_4"
    provider = "google"
    seed_supported = False

    def generate(self, **kwargs: Any) -> GenerationResult:
        # TODO: Replace with official Imagen 4 API wiring for your environment.
        return _stub_result(
            model_id=self.model_id,
            provider=self.provider,
            cost=COST_DEFAULTS[self.model_id],
            message="Google Imagen 4 wrapper is a stub. Configure provider-specific API client.",
            **kwargs,
        )


def _generate_google_gemini(
    *,
    model_id: str,
    api_model: str,
    cost: float | None,
    prompt: str,
    output_path: str,
    prompt_id: str,
    complexity_level: str,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
    num_images: int = 1,
    timeout_seconds: int = 180,
    max_retries: int = 3,
    **kwargs: Any,
) -> GenerationResult:
    start = time.monotonic()
    try:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) or GOOGLE_APPLICATION_CREDENTIALS.")
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore

        client = genai.Client(api_key=key)

        def _call():
            return client.models.generate_content(
                model=api_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

        resp = _retry(_call, retries=max_retries)
        img = _extract_gemini_image(resp)
        if img is None:
            raise RuntimeError("Gemini response did not contain image data.")
        ImageGenerationModel.ensure_parent(output_path)
        img.save(output_path)
        return GenerationResult(
            success=True,
            model_id=model_id,
            provider="google",
            prompt_id=prompt_id,
            complexity_level=complexity_level,
            constraint_family=str(kwargs.get("constraint_family", "")),
            prompt=prompt,
            constraints_general=str(kwargs.get("constraints_general", "")),
            constraints_specific=str(kwargs.get("constraints_specific", "")),
            template_family=kwargs.get("template_family"),
            seed_requested=seed,
            seed_supported=False,
            image_path=output_path,
            generation_parameters={"width": width, "height": height, "num_images": num_images},
            latency_seconds=time.monotonic() - start,
            estimated_cost_usd=cost,
            raw_response_metadata={"provider_response": "google.genai"},
            error_message=None,
        )
    except Exception as e:
        return GenerationResult(
            success=False,
            model_id=model_id,
            provider="google",
            prompt_id=prompt_id,
            complexity_level=complexity_level,
            constraint_family=str(kwargs.get("constraint_family", "")),
            prompt=prompt,
            constraints_general=str(kwargs.get("constraints_general", "")),
            constraints_specific=str(kwargs.get("constraints_specific", "")),
            template_family=kwargs.get("template_family"),
            seed_requested=seed,
            seed_supported=False,
            image_path=None,
            generation_parameters={"width": width, "height": height, "num_images": num_images},
            latency_seconds=time.monotonic() - start,
            estimated_cost_usd=cost,
            raw_response_metadata=None,
            error_message=repr(e),
        )


def _extract_gemini_image(resp: object) -> Image.Image | None:
    parts = getattr(resp, "parts", None)
    if parts:
        for part in parts:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline else None
            if data:
                return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    cands = getattr(resp, "candidates", None)
    if cands:
        for cand in cands:
            content = getattr(cand, "content", None)
            cparts = getattr(content, "parts", None) if content else None
            if not cparts:
                continue
            for part in cparts:
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline else None
                if data:
                    return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    return None


class _StubModel(ImageGenerationModel):
    model_id: str
    provider: str

    def generate(self, **kwargs: Any) -> GenerationResult:
        return _stub_result(
            model_id=self.model_id,
            provider=self.provider,
            cost=COST_DEFAULTS.get(self.model_id),
            message=f"{self.model_id} is not wired yet. TODO: add provider call wrapper.",
            **kwargs,
        )


def _stub_result(
    *,
    model_id: str,
    provider: str,
    message: str,
    cost: float | None,
    prompt: str,
    output_path: str,
    prompt_id: str,
    complexity_level: str,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
    num_images: int = 1,
    **kwargs: Any,
) -> GenerationResult:
    return GenerationResult(
        success=False,
        model_id=model_id,
        provider=provider,
        prompt_id=prompt_id,
        complexity_level=complexity_level,
        constraint_family=str(kwargs.get("constraint_family", "")),
        prompt=prompt,
        constraints_general=str(kwargs.get("constraints_general", "")),
        constraints_specific=str(kwargs.get("constraints_specific", "")),
        template_family=kwargs.get("template_family"),
        seed_requested=seed,
        seed_supported=False,
        image_path=None,
        generation_parameters={"width": width, "height": height, "num_images": num_images},
        latency_seconds=0.0,
        estimated_cost_usd=cost,
        raw_response_metadata=None,
        error_message=message,
    )


class Ideogram3(_StubModel):
    model_id = "ideogram_3"
    provider = "ideogram"


class Recraft(_StubModel):
    model_id = "recraft"
    provider = "recraft"


class Firefly(_StubModel):
    model_id = "firefly"
    provider = "adobe"


class NovaCanvas(_StubModel):
    model_id = "nova_canvas"
    provider = "aws"


class Seedream(_StubModel):
    model_id = "seedream"
    provider = "byteplus"


class GrokImagine(_StubModel):
    model_id = "grok_imagine"
    provider = "xai"
