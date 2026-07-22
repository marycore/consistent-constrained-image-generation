from __future__ import annotations

from typing import Any

import torch
from diffusers import SanaPipeline

from ._diffusers_common import DiffusersLoraTrainer


class SanaTrainer(DiffusersLoraTrainer):
    # UNVERIFIED: unlike pixart-sigma/sd3.5-large/flux (ported from proven runners in
    # ImageGenerator/src/categories/), there is no prior working reference for Sana in
    # this repo. Sana uses a flow-matching scheduler like SD3.5, so this step follows
    # the same pattern -- but has not been run end-to-end. Verify against the
    # installed diffusers version's SanaTransformer2DModel forward signature before
    # trusting the result.
    name = "sana"
    hf_repo = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
    pipeline_cls = SanaPipeline
    torch_dtype = torch.float16

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

            prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
                texts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )[:2]

        model_pred = transformer(
            noisy_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timesteps,
            return_dict=True,
        ).sample
        flow_target = noise - latents
        return torch.nn.functional.mse_loss(model_pred.float(), flow_target.float())
