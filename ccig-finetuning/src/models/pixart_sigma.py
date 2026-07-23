from __future__ import annotations

from typing import Any

import torch
from diffusers import DDPMScheduler, PixArtSigmaPipeline

from ._diffusers_common import DiffusersLoraTrainer


class PixArtSigmaTrainer(DiffusersLoraTrainer):
    # Forward-pass args verified against PixArtTransformer2DModel.forward's actual
    # signature (hidden_states, encoder_hidden_states, timestep, added_cond_kwargs,
    # encoder_attention_mask, return_dict) in the installed diffusers version.
    name = "pixart-sigma"
    hf_repo = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    pipeline_cls = PixArtSigmaPipeline
    torch_dtype = torch.float16

    def _training_step(self, pipe: Any, transformer: Any, batch: dict) -> torch.Tensor:
        if not hasattr(self, "_noise_scheduler"):
            self._noise_scheduler = DDPMScheduler.from_pretrained(self.hf_repo, subfolder="scheduler")

        device = next(transformer.parameters()).device
        dtype = self.torch_dtype
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        texts = batch["prompt"]

        with torch.no_grad():

            # Encode the images to latents and sample noise
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, self._noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            )
            # Add noise to the latents according to the noise schedule and the sampled timesteps.
            noisy_latents = self._noise_scheduler.add_noise(latents, noise, timesteps)

            # Encode the prompts to embeddings and build the added_cond_kwargs for PixArt's adaln_single.
            prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
                texts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                max_sequence_length=120,
            )
            # PixArt's adaln_single expects resolution/aspect-ratio micro-conditions during training too.
            resolution = pixel_values.shape[-1]
            resolution_cond = torch.tensor(
                [resolution, resolution], device=device, dtype=prompt_embeds.dtype
            ).repeat(bsz, 1)
            aspect_ratio_cond = torch.tensor([1.0], device=device, dtype=prompt_embeds.dtype).repeat(
                bsz, 1
            )
            added_cond_kwargs = {"resolution": resolution_cond, "aspect_ratio": aspect_ratio_cond}

        # Forward pass through the transformer and compute the MSE loss against the noise.
        pred = transformer(
            noisy_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=True,
        ).sample
        if pred.shape[1] == latents.shape[1] * 2:
            # learned-variance models predict [noise_pred, variance_pred] concatenated.
            pred = pred.chunk(2, dim=1)[0]
            
        # The loss is computed against the original noise, not the noisy latents, as per PixArt's training objective.
        return torch.nn.functional.mse_loss(pred.float(), noise.float())
