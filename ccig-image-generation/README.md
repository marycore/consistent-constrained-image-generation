# CCIG Image Generation

Generates images from CCIG eval prompts (`data/ccig_eval_dataset/*.json`, one JSONL file per
domain/complexity/status combination, e.g. `clevr_3_scenes_SAT.json`) using text-to-image models.
Generation only — scoring/constraint checking against the prompts is a separate, future module.

## Structure

```
src/
├── common/          # shared types + dataset I/O, provider-agnostic
├── closed/          # closed-source, API-based models (OpenAI, Google, ...)
├── open/            # open-weight, locally-run models (diffusers, ...)
└── run.py           # CLI entry point
```

## Closed-source models

Registered in `src/closed/registry.py`:

| model                | provider | notes                                                              |
|-----------------------|----------|---------------------------------------------------------------------|
| `gpt-image-1`         | OpenAI   | legacy -- kept for comparison, superseded by `gpt-image-2`          |
| `gpt-image-2`         | OpenAI   | current flagship (released Apr 2026); `quality` set on the class (`low`/`medium`/`high`) -- see pricing note below |
| `gemini-2.0-flash`    | Google   | legacy -- kept for comparison; despite the registry name, actually calls `gemini-2.5-flash-image` ("Nano Banana") |
| `gemini-3-pro-image`  | Google   | current flagship, aka "Nano Banana Pro"; `image_size` set on the class (`1K`/`2K`/`4K`) |

Both `gpt-image-2` and `gemini-3-pro-image` price per image by quality/resolution tier rather than
a flat rate -- check `quality`/`image_size` on the model class in `src/closed/gpt_image.py` /
`src/closed/gemini.py` before a large run, since it directly drives cost (e.g. `gpt-image-2` at
1024x1024: ~$0.006/image low, ~$0.05-0.08/image medium, ~$0.21/image high -- verify against your
own OpenAI usage dashboard, list prices/estimates drift).

## Open-source models

Registered in `src/open/registry.py`, implemented on top of `diffusers`:

| model            | status  | gated? | notes                                   |
|-------------------|---------|--------|-------------------------------------------|
| `sd3.5-large`     | working | yes    | `StableDiffusion3Pipeline` -- accept the license on the model page, then `huggingface-cli login` |
| `flux.1-dev`      | working | yes    | `FluxPipeline`, guidance-distilled -- accept the license on the model page, then `huggingface-cli login` |
| `flux.1-schnell`  | working | yes    | `FluxPipeline`, few-step, not guidance-distilled -- same login requirement as `flux.1-dev` |
| `qwen-image`      | working | no     | `QwenImagePipeline`, guidance-distilled; needs a recent `diffusers` version |

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

export OPENAI_API_KEY=...    # for gpt-image-1, gpt-image-2
export GEMINI_API_KEY=...    # for gemini-2.0-flash, gemini-3-pro-image (or GOOGLE_API_KEY)
```

`.env.example` in this directory lists both keys; the code does not auto-load a `.env` file
(no `python-dotenv` dependency), so `set -a && source .env && set +a` (or export manually)
before running.

## Run

```bash
python -m src.run --model gpt-image-2 --dataset ../data/ccig_eval_dataset/clevr_3_scenes_SAT.json --limit 5
python -m src.run --model gemini-3-pro-image --dataset ../data/ccig_eval_dataset/clevr_3_scenes_SAT.json --limit 5

# open-source model, base weights
python -m src.run --model flux.1-schnell --limit 5

# open-source model, LoRA finetuned checkpoint (from ccig-finetuning)
python -m src.run --model sd3.5-large --checkpoint ../outputs/checkpoints-finetuning/sd3.5-large/run1
```

Options:
- `--prompt-field {short,medium,long}` — which NL description to use as the prompt (default `medium`)
- `--out` — output root (default `../data/generated_images`, i.e. the repo-root `data/` folder)
- `--limit` — cap the number of prompts processed
- `--dataset` — path to one of the eval-dataset files under `data/ccig_eval_dataset/`
  (the CLI default is stale -- always pass this explicitly)
- `--checkpoint` — path to a LoRA finetuned checkpoint directory (open-source models only)

## Output

Images and a manifest are written to `data/generated_images/<model>[-<variant>]/<dataset>/`,
where `<variant>` is the checkpoint name (open models) or quality/resolution tier (closed models
that expose one, e.g. `gpt-image-2-low`) and `<dataset>` is the stem of `--dataset`
(e.g. `clevr_3_scenes_SAT`):
- `data/generated_images/<model>[-<variant>]/<dataset>/<prompt_id>-<prompt_field>.png`
- `data/generated_images/<model>[-<variant>]/<dataset>/manifest.jsonl` — one record per prompt:
  `id`, `model`, `prompt`, `prompt_field`, `scene_generation_setup`, `image_path`, `success`,
  `error`, `variant`

## Adding a model

- **Closed (API-based)**: add a class in `src/closed/<provider>.py` implementing
  `ClosedImageModel.generate(prompt) -> Image`, register it in `src/closed/registry.py`.
- **Open (diffusers-based)**: add a class in `src/open/<model>.py` subclassing
  `DiffusersImageModel` (set `pipeline_cls`, `hf_repo`, `_call_kwargs`), register it in
  `src/open/registry.py`.
- **Open (non-diffusers)**: subclass `OpenImageModel` directly and implement `generate()` with
  the model's own inference code.
