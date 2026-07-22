from __future__ import annotations

from typing import Any

import torch
from diffusers import SanaPipeline

from ._diffusers_common import DiffusersLoraTrainer


class SanaTrainer(DiffusersLoraTrainer):
    # Sana's default scheduler is DPMSolverMultistepScheduler (has add_noise(), unlike
    # SD3.5/FLUX's flow-matching scheduler), and SanaPipeline.__call__ scales the
    # timestep by `transformer.config.timestep_scale` before calling the transformer
    # (default 1.0, so a no-op unless the checkpoint overrides it) -- mirrored below.
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
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            scaled_timesteps = timesteps * getattr(transformer.config, "timestep_scale", 1.0)

            prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
                texts, do_classifier_free_guidance=False, device=device, num_images_per_prompt=1
            )

        model_pred = transformer(
            noisy_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=scaled_timesteps,
            return_dict=True,
        ).sample
        return torch.nn.functional.mse_loss(model_pred.float(), noise.float())
