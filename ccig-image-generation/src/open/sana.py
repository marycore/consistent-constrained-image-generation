from __future__ import annotations

import torch
from diffusers import SanaPipeline

from ._diffusers_common import DiffusersImageModel


class SanaModel(DiffusersImageModel):
    name = "sana"
    hf_repo = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
    pipeline_cls = SanaPipeline
    torch_dtype = torch.float16
    _call_kwargs = {"num_inference_steps": 20, "guidance_scale": 4.5}
