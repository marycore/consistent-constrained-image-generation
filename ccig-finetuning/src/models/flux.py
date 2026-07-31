from __future__ import annotations

from typing import Any

import torch
from diffusers import FluxPipeline

from ._diffusers_common import DiffusersLoraTrainer


class FluxBaseTrainer(DiffusersLoraTrainer):
    pipeline_cls = FluxPipeline
    frozen_module_names = ["vae", "text_encoder", "text_encoder_2"]
    # FLUX.1-dev is guidance-distilled (transformer.config.guidance_embeds == True) and
    # requires a `guidance` tensor at every forward pass, including training; schnell is
    # not guidance-distilled and passes guidance=None. See FluxPipeline.__call__'s
    # "handle guidance" block for the reference behavior this mirrors.
    guidance_scale: float = 3.5

    def _training_step(self, pipe: Any, transformer: Any, batch: dict) -> torch.Tensor:
        device = next(transformer.parameters()).device
        dtype = self.torch_dtype
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        texts = batch["prompt"]

        # VAE + text encoders live on CPU by default (see frozen_module_names) and are
        # moved to GPU only for this block, then back -- keeps the resident, trainable
        # transformer's memory budget from having to share the GPU with the ~24GB of
        # frozen FLUX weights at the same time.
        def _encode():
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                bsz, num_channels, lat_h, lat_w = latents.shape

                # Use FluxPipeline's own packing helpers (`_pack_latents` converts
                # (B,C,H,W) -> (B, H//2*W//2, C*4) patches; `_prepare_latent_image_ids`
                # builds the matching RoPE position ids) instead of a hand-rolled
                # reimplementation, so this stays correct if diffusers changes the format.
                packed_latents = pipe._pack_latents(latents, bsz, num_channels, lat_h, lat_w)
                noise = torch.randn_like(latents)
                packed_noise = pipe._pack_latents(noise, bsz, num_channels, lat_h, lat_w)
                img_ids = pipe._prepare_latent_image_ids(bsz, lat_h // 2, lat_w // 2, device, dtype)

                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=device
                )
                sigma = (
                    (timesteps.float() / pipe.scheduler.config.num_train_timesteps)
                    .view(-1, 1, 1)
                    .to(device=device, dtype=dtype)
                )
                noisy_packed = sigma * packed_noise + (1.0 - sigma) * packed_latents

                # encode_prompt(prompt, prompt_2=None, ...) -> (prompt_embeds, pooled_prompt_embeds,
                # text_ids); prompt_2 defaults to prompt when omitted. text_ids is already
                # unbatched (seq_len, 3), matching what the transformer expects.
                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=texts, device=device, num_images_per_prompt=1, max_sequence_length=512
                )

                guidance = None
                if transformer.config.guidance_embeds:
                    guidance = torch.full((bsz,), self.guidance_scale, device=device, dtype=torch.float32)

            return (
                noisy_packed, packed_noise, packed_latents, img_ids, timesteps,
                prompt_embeds, pooled_prompt_embeds, text_ids, guidance,
            )

        (
            noisy_packed, packed_noise, packed_latents, img_ids, timesteps,
            prompt_embeds, pooled_prompt_embeds, text_ids, guidance,
        ) = self._run_frozen_on_cuda(pipe, _encode)

        pred_packed = transformer(
            hidden_states=noisy_packed,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps,
            img_ids=img_ids,
            txt_ids=text_ids,
            guidance=guidance,
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
