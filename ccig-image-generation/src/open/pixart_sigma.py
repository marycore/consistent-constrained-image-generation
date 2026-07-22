from __future__ import annotations

from diffusers import PixArtSigmaPipeline

from ._diffusers_common import DiffusersImageModel


class PixArtSigmaModel(DiffusersImageModel):
    name = "pixart-sigma"
    hf_repo = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    pipeline_cls = PixArtSigmaPipeline
    _call_kwargs = {"num_inference_steps": 20, "guidance_scale": 4.5}
