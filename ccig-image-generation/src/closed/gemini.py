from __future__ import annotations

import os
from io import BytesIO

from PIL import Image

from .base import ClosedImageModel


class GeminiFlashImage(ClosedImageModel):
    name = "gemini-2.0-flash"
    api_model = "gemini-2.0-flash-preview-image-generation"

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> Image.Image:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.api_model,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return Image.open(BytesIO(part.inline_data.data)).convert("RGB")
        raise RuntimeError("Gemini response did not contain image data.")
