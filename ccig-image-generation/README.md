# CCIG Image Generation

Generates images from CCIG eval prompts (`data/ccig_eval_dataset_{SAT,UNSAT}.jsonl`) using
text-to-image models. Generation only — scoring/constraint checking against the prompts is a
separate, future module.

## Structure

```
src/
├── common/          # shared types + dataset I/O, provider-agnostic
├── closed/          # closed-source, API-based models (OpenAI, Google, ...)
├── open/            # open-weight, locally-run models (diffusers, ...)
└── run.py           # CLI entry point
```

## Open-source models

Registered in `src/open/registry.py`, implemented on top of `diffusers`:

| model            | status  | gated? | notes                                   |
|-------------------|---------|--------|-------------------------------------------|
| `pixart-sigma`    | working | no     | `PixArtSigmaPipeline`                     |
| `sd3.5-large`     | working | yes    | `StableDiffusion3Pipeline` -- accept the license on the model page, then `huggingface-cli login` |
| `flux.1-dev`      | working | yes    | `FluxPipeline`, guidance-distilled -- accept the license on the model page, then `huggingface-cli login` |
| `flux.1-schnell`  | working | yes    | `FluxPipeline`, few-step, not guidance-distilled -- same login requirement as `flux.1-dev` |
| `sana`            | working | no     | `SanaPipeline`                            |
| `hidream-i1`      | working | no     | `HiDreamImagePipeline`; bundles its own Llama-3.1 text encoder (not the separately-gated meta-llama repo) |
| `qwen-image`      | working | no     | `QwenImagePipeline`, guidance-distilled; needs a recent `diffusers` version |
| `janus-pro`       | stub    | no     | not diffusers-based; see `src/open/janus_pro.py` |
| `show-o`          | stub    | no     | not diffusers-based; see `src/open/showo.py`     |
| `bagel`           | stub    | no     | not diffusers-based; see `src/open/bagel.py`     |

"gated" is per the Hugging Face Hub API (checked directly, not assumed) -- gated repos need you
to accept the model's license on its Hub page while logged in, then `huggingface-cli login`
locally before `from_pretrained` will succeed.

All working models share one implementation (`src/open/_diffusers_common.py`) that loads
base weights from Hugging Face Hub and, when `--checkpoint` is passed, loads a LoRA
finetuned checkpoint on top via `peft`'s `PeftModel.from_pretrained` -- the same format
`ccig-finetuning` writes checkpoints in (`transformer.save_pretrained(...)`).

## Setup

```bash
cd ccig-image-generation

# closed, API-based models only
pip install -r requirements.txt

# adds the open-source / diffusers stack (torch, diffusers, transformers, peft, ...)
pip install -r requirements-open.txt
huggingface-cli login   # required for gated repos: sd3.5-large, flux.1-dev, flux.1-schnell

export OPENAI_API_KEY=...    # for gpt-image-1
export GEMINI_API_KEY=...    # for gemini-2.0-flash
```

## Run

```bash
python -m src.run --model gpt-image-1 --dataset ../data/ccig_eval_dataset_SAT.jsonl --limit 5
python -m src.run --model gemini-2.0-flash --dataset ../data/ccig_eval_dataset_SAT.jsonl
python -m src.run --model gpt-image-1 --limit 5

# open-source model, base weights
python -m src.run --model flux.1-schnell --limit 5

# open-source model, LoRA finetuned checkpoint (from ccig-finetuning)
python -m src.run --model sd3.5-large --checkpoint ../ccig-finetuning/outputs/sd3.5-large/run1
```

Options:
- `--prompt-field {short,medium,long}` — which NL description to use as the prompt (default `medium`)
- `--out` — output root (default `../data/generated_images`, i.e. the repo-root `data/` folder)
- `--limit` — cap the number of prompts processed
- `--dataset` — path to the existing eval-dataset (default `../data/ccig_eval_dataset.jsonl`)
- `--checkpoint` — path to a LoRA finetuned checkpoint directory (open-source models only)

## Output

Images and a manifest are written to `data/generated_images/<model>/`:
- `data/generated_images/<model>/<prompt_id>.png`
- `data/generated_images/<model>/manifest.jsonl` — one record per prompt: `id`, `model`, `prompt`,
  `image_path`, `success`, `error`

## Adding a model

- **Closed (API-based)**: add a class in `src/closed/<provider>.py` implementing
  `ClosedImageModel.generate(prompt) -> Image`, register it in `src/closed/registry.py`.
- **Open (diffusers-based)**: add a class in `src/open/<model>.py` subclassing
  `DiffusersImageModel` (set `pipeline_cls`, `hf_repo`, `_call_kwargs`), register it in
  `src/open/registry.py`.
- **Open (non-diffusers)**: subclass `OpenImageModel` directly and implement `generate()` with
  the model's own inference code (see `src/open/janus_pro.py` / `showo.py` for the placeholder
  shape to follow).
