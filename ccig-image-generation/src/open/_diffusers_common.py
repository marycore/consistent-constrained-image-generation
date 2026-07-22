from __future__ import annotations

from typing import Any, ClassVar

import torch
from PIL import Image

from .base import OpenImageModel


class DiffusersImageModel(OpenImageModel):
    """Shared implementation for models built on a `diffusers` pipeline.

    Subclasses set `pipeline_cls` + `hf_repo`, and override `_call_kwargs` for
    model-specific sampling parameters (steps, guidance scale, ...). This is the
    single place that knows how to load base weights, optionally apply a LoRA
    checkpoint, and run the pipeline -- every diffusers-based model below reuses it
    instead of re-implementing `generate()`.
    """

    pipeline_cls: ClassVar[Any]
    torch_dtype: ClassVar[Any] = torch.bfloat16
    _call_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        self._pipe = self.pipeline_cls.from_pretrained(self.hf_repo, torch_dtype=self.torch_dtype)
        self._pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint is not None:
            # Convention: ccig-finetuning saves the LoRA-wrapped transformer via PEFT's
            # `save_pretrained`; load it back the same way (PeftModel.from_pretrained),
            # not diffusers' `pipe.load_lora_weights` (a different state-dict format).
            from peft import PeftModel

            self._pipe.transformer = PeftModel.from_pretrained(self._pipe.transformer, checkpoint)

    def generate(self, prompt: str) -> Image.Image:
        return self._pipe(prompt=prompt, **self._call_kwargs).images[0]
