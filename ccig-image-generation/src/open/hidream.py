from __future__ import annotations

from diffusers import HiDreamImagePipeline

from ._diffusers_common import DiffusersImageModel


class HiDreamI1Model(DiffusersImageModel):
    # Note: HiDream-I1's text encoder is a gated Llama-3.1-8B-Instruct checkpoint.
    # `from_pretrained` below assumes the environment is authenticated for both
    # repos (`huggingface-cli login`) with access to the Llama checkpoint granted.
    name = "hidream-i1"
    hf_repo = "HiDream-ai/HiDream-I1-Full"
    pipeline_cls = HiDreamImagePipeline
    _call_kwargs = {"num_inference_steps": 50, "guidance_scale": 5.0}
