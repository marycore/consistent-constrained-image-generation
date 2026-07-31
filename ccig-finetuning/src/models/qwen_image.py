from __future__ import annotations

from typing import Any

import torch
from diffusers import QwenImagePipeline

from ._diffusers_common import DiffusersLoraTrainer


class QwenImageTrainer(DiffusersLoraTrainer):
    # Qwen-Image is guidance-distilled; its pipeline raises if guidance_scale is None
    # and transformer.config.guidance_embeds is True. See QwenImagePipeline.__call__'s
    # "handle guidance" block for the reference behavior this mirrors. The exact tuned
    # value for LoRA training isn't verified end-to-end here -- adjust if needed.
    guidance_scale: float = 1.0

    name = "qwen-image"
    hf_repo = "Qwen/Qwen-Image"
    pipeline_cls = QwenImagePipeline
    frozen_module_names = ["vae", "text_encoder"]

    def _training_step(self, pipe: Any, transformer: Any, batch: dict) -> torch.Tensor:
        device = next(transformer.parameters()).device
        dtype = self.torch_dtype
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        texts = batch["prompt"]

        # VAE + text encoder live on CPU by default (see frozen_module_names) and are
        # moved to GPU only for this block, then back -- keeps the resident, trainable
        # transformer's memory budget from having to share the GPU with the ~20B+ params
        # of frozen Qwen-Image weights at the same time.
        def _encode():
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                bsz, num_channels, lat_h, lat_w = latents.shape

                # Use QwenImagePipeline's own `_pack_latents` (same 2x2 patch packing as
                # FLUX) instead of reimplementing it.
                packed_latents = pipe._pack_latents(latents, bsz, num_channels, lat_h, lat_w)
                noise = torch.randn_like(latents)
                packed_noise = pipe._pack_latents(noise, bsz, num_channels, lat_h, lat_w)
                # img_shapes / txt_seq_lens as built in QwenImagePipeline.__call__: one
                # (frames, h//2, w//2) tuple per batch item, and per-sample real token
                # counts (post-padding-mask) for the text sequence.
                img_shapes = [[(1, lat_h // 2, lat_w // 2)]] * bsz

                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=device
                )
                sigma = (
                    (timesteps.float() / pipe.scheduler.config.num_train_timesteps)
                    .view(-1, 1, 1)
                    .to(device=device, dtype=dtype)
                )
                noisy_packed = sigma * packed_noise + (1.0 - sigma) * packed_latents

                prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                    prompt=texts, device=device, num_images_per_prompt=1, max_sequence_length=1024
                )
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                guidance = None
                if transformer.config.guidance_embeds:
                    guidance = torch.full((bsz,), self.guidance_scale, device=device, dtype=torch.float32)

            return (
                noisy_packed, packed_noise, packed_latents, img_shapes, timesteps,
                prompt_embeds, prompt_embeds_mask, txt_seq_lens, guidance,
            )

        (
            noisy_packed, packed_noise, packed_latents, img_shapes, timesteps,
            prompt_embeds, prompt_embeds_mask, txt_seq_lens, guidance,
        ) = self._run_frozen_on_cuda(pipe, _encode)

        model_pred = transformer(
            hidden_states=noisy_packed,
            timestep=timesteps,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=True,
        ).sample
        flow_target = packed_noise - packed_latents
        return torch.nn.functional.mse_loss(model_pred.float(), flow_target.float())
