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

| model            | status      | notes                                              |
|-------------------|-------------|------------------------------------------------------|
| `pixart-sigma`    | verified    | `PixArtSigmaPipeline`, DDPM training step ported from a proven runner |
| `sd3.5-large`     | verified    | `StableDiffusion3Pipeline`, flow-matching training step ported from a proven runner |
| `flux.1-dev`      | verified    | `FluxPipeline`, packed-latent flow-matching step ported from a proven runner |
| `flux.1-schnell`  | verified    | `FluxPipeline`, same training step as `flux.1-dev`   |
| `sana`            | unverified  | `SanaPipeline`, best-effort flow-matching step, not run end-to-end |
| `hidream-i1`      | unverified  | `HiDreamImagePipeline`, best-effort flow-matching step; needs gated Llama text encoder access |
| `janus-pro`       | stub        | not diffusers-based; see `src/models/janus_pro.py`   |
| `show-o`          | stub        | not diffusers-based; see `src/models/showo.py`       |

"Verified" models port their training step (noise process, transformer forward signature,
prompt encoding) from a working prior implementation in
`ImageGenerator/src/categories/{latent_diffusion_open,rectified_flow_open}/`. "Unverified"
models follow the same pattern by analogy but haven't been run end-to-end -- check the
installed `diffusers` version's transformer forward signature before trusting the result.

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
huggingface-cli login   # required for gated repos (e.g. hidream-i1's Llama text encoder)
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
