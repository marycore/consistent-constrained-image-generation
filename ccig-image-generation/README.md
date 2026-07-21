# CCIG Image Generation

Generates images from CCIG eval prompts (`data/ccig_eval_dataset_{SAT,UNSAT}.jsonl`) using
text-to-image models. Generation only — scoring/constraint checking against the prompts is a
separate, future module.

## Structure

```
src/
├── common/          # shared types + dataset I/O, provider-agnostic
├── closed/          # closed-source, API-based models (OpenAI, Google, ...)
└── run.py           # CLI entry point
```

Open-source (locally run) models will live in a sibling `src/open/` package once needed.

## Setup

```bash
cd ccig-image-generation
pip install -r requirements.txt

export OPENAI_API_KEY=...    # for gpt-image-1
export GEMINI_API_KEY=...    # for gemini-2.0-flash
```

## Run

```bash
python -m src.run --model gpt-image-1 --dataset ../data/ccig_eval_dataset_SAT.jsonl --limit 5
python -m src.run --model gemini-2.0-flash --dataset ../data/ccig_eval_dataset_SAT.jsonl
python -m src.run --model gpt-image-1 --limit 5
```

Options:
- `--prompt-field {short,medium,long}` — which NL description to use as the prompt (default `medium`)
- `--out` — output root (default `../data/generated_images`, i.e. the repo-root `data/` folder)
- `--limit` — cap the number of prompts processed
- `--dataset` — path to the existing eval-dataset (default `../data/ccig_eval_dataset.jsonl`)

## Output

Images and a manifest are written to `data/generated_images/<model>/`:
- `data/generated_images/<model>/<prompt_id>.png`
- `data/generated_images/<model>/manifest.jsonl` — one record per prompt: `id`, `model`, `prompt`,
  `image_path`, `success`, `error`

## Adding a model

1. Add a class in `src/closed/<provider>.py` implementing `ClosedImageModel.generate(prompt) -> Image`
2. Register it in `src/closed/registry.py`
