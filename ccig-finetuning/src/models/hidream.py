from __future__ import annotations

from typing import Any

import torch
from diffusers import HiDreamImagePipeline

from ._diffusers_common import DiffusersLoraTrainer


class HiDreamI1Trainer(DiffusersLoraTrainer):
    # UNVERIFIED: no prior working reference for HiDream-I1 in this repo (unlike
    # pixart-sigma/sd3.5-large/flux, ported from proven runners). HiDream uses a
    # flow-matching scheduler and FOUR text encoders (incl. a gated Llama-3.1-8B);
    # `pipe.encode_prompt` below assumes a similar (embeds, pooled_embeds, ...) return
    # shape to SD3.5/Flux, which may not hold. Requires `huggingface-cli login` with
    # access granted to both HiDream and the Llama checkpoint. Verify before trusting.
    name = "hidream-i1"
    hf_repo = "HiDream-ai/HiDream-I1-Full"
    pipeline_cls = HiDreamImagePipeline

    def _training_step(self, pipe: Any, transformer: Any, batch: dict) -> torch.Tensor:
        device = next(transformer.parameters()).device
        dtype = self.torch_dtype
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        texts = batch["prompt"]

        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=device
            )
            sigma = (
                (timesteps.float() / pipe.scheduler.config.num_train_timesteps)
                .view(-1, 1, 1, 1)
                .to(device=device, dtype=dtype)
            )
            noisy_latents = sigma * noise + (1.0 - sigma) * latents

            encoded = pipe.encode_prompt(
                texts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            prompt_embeds = encoded[0]
            pooled_prompt_embeds = encoded[2] if len(encoded) >= 3 else encoded[1]

        model_pred = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=True,
        ).sample
        flow_target = noise - latents
        return torch.nn.functional.mse_loss(model_pred.float(), flow_target.float())
