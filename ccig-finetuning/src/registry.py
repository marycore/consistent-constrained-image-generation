from __future__ import annotations

from src.models.bagel import BagelTrainer
from src.models.base import LoraTrainer
from src.models.flux import FluxDevTrainer, FluxSchnellTrainer
from src.models.hidream import HiDreamI1Trainer
from src.models.janus_pro import JanusProTrainer
from src.models.pixart_sigma import PixArtSigmaTrainer
from src.models.qwen_image import QwenImageTrainer
from src.models.sana import SanaTrainer
from src.models.sd35_large import SD35LargeTrainer
from src.models.showo import ShowoTrainer

TRAINER_REGISTRY: dict[str, type[LoraTrainer]] = {
    "pixart-sigma": PixArtSigmaTrainer,
    "sd3.5-large": SD35LargeTrainer,
    "flux.1-dev": FluxDevTrainer,
    "flux.1-schnell": FluxSchnellTrainer,
    "sana": SanaTrainer,
    "hidream-i1": HiDreamI1Trainer,
    "qwen-image": QwenImageTrainer,
    "janus-pro": JanusProTrainer,
    "show-o": ShowoTrainer,
    "bagel": BagelTrainer,
}


def build_trainer(name: str) -> LoraTrainer:
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(TRAINER_REGISTRY)}")
    return TRAINER_REGISTRY[name]()
