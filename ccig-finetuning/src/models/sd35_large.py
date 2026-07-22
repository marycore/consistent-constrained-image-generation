from __future__ import annotations

from typing import Any

import torch
from diffusers import StableDiffusion3Pipeline

from ._diffusers_common import DiffusersLoraTrainer


class SD35LargeTrainer(DiffusersLoraTrainer):
    # SD3.5's scheduler is FlowMatchEulerDiscreteScheduler, which has no add_noise() --
    # confirmed by introspecting the installed diffusers version. Uses the flow-matching
    # forward process directly instead: x_t = sigma*noise + (1-sigma)*x_0.
    name = "sd3.5-large"
    hf_repo = "stabilityai/stable-diffusion-3.5-large"
    pipeline_cls = StableDiffusion3Pipeline

    def _training_step(self, pipe: Any, transformer: Any, batch: dict) -> torch.Tensor:
        device = next(transformer.parameters()).device
        dtype = self.torch_dtype
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        texts = batch["prompt"]

        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
            scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
            latents = (latents - shift_factor) * scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=device
            )
            # SD3.5's FlowMatchEulerDiscreteScheduler has no add_noise(); use the
            # flow-matching forward process directly: x_t = sigma*noise + (1-sigma)*x_0.
            sigma = (
                (timesteps.float() / pipe.scheduler.config.num_train_timesteps)
                .view(-1, 1, 1, 1)
                .to(device=device, dtype=dtype)
            )
            noisy_latents = sigma * noise + (1.0 - sigma) * latents

            encoded = pipe.encode_prompt(
                texts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            # encode_prompt returns either (embeds, neg_embeds, pooled, neg_pooled) or (embeds, pooled).
            prompt_embeds = encoded[0]
            pooled_prompt_embeds = encoded[2] if len(encoded) >= 3 else encoded[1]

        model_pred = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=True,
        ).sample
        # Flow-matching target is the velocity field (noise - x_0), not noise alone.
        flow_target = noise - latents
        return torch.nn.functional.mse_loss(model_pred.float(), flow_target.float())
