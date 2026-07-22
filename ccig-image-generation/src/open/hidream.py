from __future__ import annotations

from diffusers import HiDreamImagePipeline

from ._diffusers_common import DiffusersImageModel


class HiDreamI1Model(DiffusersImageModel):
    # Its Llama-3.1 text encoder (text_encoder_4) ships bundled inside this repo, not
    # pulled from the separately-gated meta-llama/Llama-3.1-8B-Instruct repo -- this
    # repo itself is not gated, verified via the HF Hub API (`gated: false`).
    name = "hidream-i1"
    hf_repo = "HiDream-ai/HiDream-I1-Full"
    pipeline_cls = HiDreamImagePipeline
    _call_kwargs = {"num_inference_steps": 50, "guidance_scale": 5.0}
