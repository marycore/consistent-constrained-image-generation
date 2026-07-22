from __future__ import annotations

from typing import Any

import torch
from diffusers import FluxPipeline

from ._diffusers_common import DiffusersLoraTrainer


# FLUX uses a packed-patch latent format; these convert between the standard
# (B, C, H, W) VAE latent space and FLUX's (B, H//2*W//2, C*4) packed form.
# Ported from ImageGenerator/src/categories/rectified_flow_open/flux_1_dev/runner.py.
def _flux_pack_latents(latents: torch.Tensor) -> torch.Tensor:
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 2) * (W // 2), C * 4)
    return latents


def _flux_prepare_img_ids(bsz: int, latent_h: int, latent_w: int, device) -> torch.Tensor:
    """Spatial position IDs for FLUX RoPE: (B, H//2*W//2, 3)."""
    h_p, w_p = latent_h // 2, latent_w // 2
    img_ids = torch.zeros(h_p, w_p, 3, device=device)
    img_ids[..., 1] = torch.arange(h_p, device=device).float()[:, None]
    img_ids[..., 2] = torch.arange(w_p, device=device).float()[None, :]
    return img_ids.reshape(h_p * w_p, 3).unsqueeze(0).expand(bsz, -1, -1)


class FluxBaseTrainer(DiffusersLoraTrainer):
    pipeline_cls = FluxPipeline

    def _training_step(self, pipe: Any, transformer: Any, batch: dict) -> torch.Tensor:
        device = next(transformer.parameters()).device
        dtype = self.torch_dtype
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        texts = batch["prompt"]

        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            _, _, lat_h, lat_w = latents.shape

            packed_latents = _flux_pack_latents(latents)
            noise = torch.randn_like(latents)
            packed_noise = _flux_pack_latents(noise)

            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=device
            )
            sigma = (
                (timesteps.float() / pipe.scheduler.config.num_train_timesteps)
                .view(-1, 1, 1)
                .to(device=device, dtype=dtype)
            )
            noisy_packed = sigma * packed_noise + (1.0 - sigma) * packed_latents

            encoded = pipe.encode_prompt(
                texts, device=device, num_images_per_prompt=1, max_sequence_length=512
            )
            if len(encoded) >= 3:
                prompt_embeds, pooled_prompt_embeds, text_ids = encoded[0], encoded[1], encoded[2]
            else:
                prompt_embeds, pooled_prompt_embeds = encoded[0], encoded[1]
                text_ids = torch.zeros(bsz, prompt_embeds.shape[1], 3, device=device)

            img_ids = _flux_prepare_img_ids(bsz, lat_h, lat_w, device)

        pred_packed = transformer(
            hidden_states=noisy_packed,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps,
            img_ids=img_ids,
            txt_ids=text_ids,
            return_dict=True,
        ).sample
        # Note: lora_target_modules ["to_k","to_q","to_v","to_out.0"] may miss FLUX's
        # double-stream text projections (add_q_proj etc.) -- revisit if convergence is slow.
        flow_target = packed_noise - packed_latents
        return torch.nn.functional.mse_loss(pred_packed.float(), flow_target.float())


class FluxDevTrainer(FluxBaseTrainer):
    name = "flux.1-dev"
    hf_repo = "black-forest-labs/FLUX.1-dev"


class FluxSchnellTrainer(FluxBaseTrainer):
    name = "flux.1-schnell"
    hf_repo = "black-forest-labs/FLUX.1-schnell"
