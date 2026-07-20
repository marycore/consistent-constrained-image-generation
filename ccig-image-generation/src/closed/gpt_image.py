from __future__ import annotations

import base64
import os
from io import BytesIO

from PIL import Image

from .base import ClosedImageModel


class GPTImage1(ClosedImageModel):
    name = "gpt-image-1"

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> Image.Image:
        response = self._client.images.generate(
            model=self.name,
            prompt=prompt,
            size="1024x1024",
        )
        b64 = response.data[0].b64_json
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
