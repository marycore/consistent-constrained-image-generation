from __future__ import annotations

from diffusers import StableDiffusion3Pipeline

from ._diffusers_common import DiffusersImageModel


class SD35LargeModel(DiffusersImageModel):
    name = "sd3.5-large"
    hf_repo = "stabilityai/stable-diffusion-3.5-large"
    pipeline_cls = StableDiffusion3Pipeline
    _call_kwargs = {"num_inference_steps": 28, "guidance_scale": 4.5}
