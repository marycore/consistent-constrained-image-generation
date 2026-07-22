# CCIG Finetuning

LoRA finetunes open-source text-to-image models on `data/finetune-dataset/` (CLEVR images
paired with constraint descriptions). Produces checkpoints consumed by
`ccig-image-generation`'s `--checkpoint` flag -- finetuning and generation are separate,
decoupled modules connected only by that checkpoint-directory convention.

## Structure

```
src/
├── common/          # shared types + dataset loading, model-agnostic
├── models/          # one LoRA trainer per model (mirrors ccig-image-generation/src/open/)
├── registry.py       # model name -> trainer class
└── run.py            # CLI entry point
configs/               # one YAML per model: dataset paths, LoRA/training hyperparameters
outputs/               # checkpoints land here: outputs/<model>/<run_name>/
```

## Models

| model            | status      | gated? | notes                                     |
|-------------------|-------------|--------|----------------------------------------------|
| `pixart-sigma`    | signature-checked | no | `PixArtSigmaPipeline`, DDPM training step |
| `sd3.5-large`     | signature-checked | yes | `StableDiffusion3Pipeline`, flow-matching training step |
| `flux.1-dev`      | signature-checked | yes | `FluxPipeline`, packed-latent flow-matching step, guidance-distilled |
| `flux.1-schnell`  | signature-checked | yes | `FluxPipeline`, same training step as `flux.1-dev`, not guidance-distilled |
| `sana`            | signature-checked | no | `SanaPipeline`, DPM-solver (non-flow-matching) training step |
| `hidream-i1`      | signature-checked | no | `HiDreamImagePipeline`, dual T5/Llama3 text encoders (Llama bundled in this repo, not gated), DPM-solver (non-flow-matching) training step, no latent packing |
| `qwen-image`      | signature-checked | no | `QwenImagePipeline`, packed-latent flow-matching step, guidance-distilled |
| `janus-pro`       | stub        | no | not diffusers-based; see `src/models/janus_pro.py`   |
| `show-o`          | stub        | no | not diffusers-based; see `src/models/showo.py`       |
| `bagel`           | stub        | no | not diffusers-based; see `src/models/bagel.py`       |

"gated" is per the Hugging Face Hub API (checked directly, not assumed) -- gated repos need you
to accept the model's license on its Hub page while logged in, then `huggingface-cli login`
locally before `from_pretrained` will succeed.

"Signature-checked" means each model's `_training_step` was written and cross-checked against
the *actual* transformer `forward()` signature, scheduler class, and pipeline `encode_prompt`
return values by introspecting the installed `diffusers` package (`python -c "import
inspect, diffusers; ..."`) -- not assumed by analogy between models. That catches real
mismatches: e.g. `SanaPipeline`'s scheduler is DPMSolverMultistep (has `add_noise()`), not
flow-matching like SD3.5/FLUX; `HiDreamImageTransformer2DModel.forward` takes
`encoder_hidden_states_t5`/`encoder_hidden_states_llama3`/`pooled_embeds` (not the
`encoder_hidden_states`/`pooled_projections` used elsewhere) and needs no latent packing;
FLUX.1-dev's guidance-distilled transformer requires a `guidance` tensor at every forward call
or it raises inside `time_text_embed`. None of these have been run end-to-end on real GPU
hardware yet, though -- that's the remaining gap between "should be correct" and "verified by
a completed training run." If a run surfaces a mismatch, fix it in that model's file and
update this table.

All models share the model-agnostic parts in `src/models/_diffusers_common.py` (pipeline
loading, LoRA injection via `peft`, dataset/dataloader, optimizer loop, checkpoint saving).
There's deliberately no generic training step: diffusers architectures differ in scheduler
type (flow-matching vs. DDPM) and transformer call signature (packed vs. unpacked latents,
extra positional ids, pooled projections, ...), so each model implements its own
`_training_step`.

## Setup

```bash
cd ccig-finetuning
pip install -r requirements.txt
huggingface-cli login   # required for gated repos: sd3.5-large, flux.1-dev, flux.1-schnell
```

## Run

```bash
python -m src.run --model sd3.5-large --config configs/sd3.5-large.yaml
python -m src.run --model flux.1-schnell --config configs/flux.1-schnell.yaml --run-name run2 --max-steps 500
```

Writes a LoRA checkpoint to `outputs/<model>/<run_name>/`. Feed that path straight into
image generation:

```bash
cd ../ccig-image-generation
python -m src.run --model sd3.5-large --checkpoint ../ccig-finetuning/outputs/sd3.5-large/run1
```

## Adding a model

- **Diffusers-based**: add a class in `src/models/<model>.py` subclassing
  `DiffusersLoraTrainer` (set `pipeline_cls`, `hf_repo`), register it in `src/registry.py`,
  add a `configs/<model>.yaml`.
- **Non-diffusers**: subclass `LoraTrainer` directly and implement `train()` with the model's
  own training code (see `src/models/janus_pro.py` / `showo.py` for the placeholder shape).
