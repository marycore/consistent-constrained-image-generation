from __future__ import annotations

from typing import Any

import torch
from diffusers import HiDreamImagePipeline

from ._diffusers_common import DiffusersLoraTrainer


class HiDreamI1Trainer(DiffusersLoraTrainer):
    # HiDream-I1's transformer forward takes `timesteps`/`encoder_hidden_states_t5`/
    # `encoder_hidden_states_llama3`/`pooled_embeds` (not the single `encoder_hidden_states`
    # + `pooled_projections` used by SD3.5/FLUX) and does its own internal patchify --
    # HiDreamImagePipeline.__call__ passes unpacked (B,C,H,W) latents directly, no
    # img_ids/packing needed. Its default scheduler is UniPCMultistepScheduler
    # (prediction_type="epsilon", has add_noise() -- standard DDPM forward process, not
    # flow-matching like SD3.5/FLUX). The pipeline also negates the raw transformer
    # output (`noise_pred = -noise_pred`) before treating it as the predicted noise, so
    # the training target here is -noise, not noise.
    # The Llama-3.1 text encoder (text_encoder_4) ships inside this repo itself (not
    # pulled from the separately-gated meta-llama/Llama-3.1-8B-Instruct repo), and this
    # repo is not gated -- no extra HF access/login needed beyond the usual.
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
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # prompt_2/3/4 (CLIP-G, T5, Llama3) default to `prompt` (CLIP-L) when omitted.
            (
                prompt_embeds_t5,
                _,
                prompt_embeds_llama3,
                _,
                pooled_prompt_embeds,
                _,
            ) = pipe.encode_prompt(
                prompt=texts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )

        model_pred = transformer(
            hidden_states=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states_t5=prompt_embeds_t5,
            encoder_hidden_states_llama3=prompt_embeds_llama3,
            pooled_embeds=pooled_prompt_embeds,
            return_dict=True,
        ).sample
        return torch.nn.functional.mse_loss(model_pred.float(), (-noise).float())
