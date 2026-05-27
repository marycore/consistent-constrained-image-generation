from __future__ import annotations

from .base import ImageGenerationModel
from .models import (
    Firefly,
    GoogleImagen4,
    GoogleNanoBanana,
    GrokImagine,
    Ideogram3,
    NovaCanvas,
    OpenAIDalle3,
    OpenAIGPTImage15,
    OpenAIGPTImage2,
    Recraft,
    Seedream,
)


def build_closed_registry() -> dict[str, ImageGenerationModel]:
    return {
        "openai_gpt_image_2": OpenAIGPTImage2(),
        "openai_gpt_image_1_5": OpenAIGPTImage15(),
        "openai_dalle_3": OpenAIDalle3(),
        "google_nano_banana": GoogleNanoBanana(),
        "google_imagen_4": GoogleImagen4(),
        "ideogram_3": Ideogram3(),
        "recraft": Recraft(),
        "firefly": Firefly(),
        "nova_canvas": NovaCanvas(),
        "seedream": Seedream(),
        "grok_imagine": GrokImagine(),
    }
