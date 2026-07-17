# Finetune Dataset Generation Pipeline

Compiles `finetune-dataset.json`: for every existing CLEVR-CCIG image, reconstructs its scene and
grounds one or more true C1–C9 constraint statements against it, using the exact same NL phrasing
as `eval_dataset_gen` (`../common/verbalize.py`) so training captions and eval prompts share
vocabulary. Unlike `eval_dataset_gen`, this pipeline generates no new scenes or images — it only
captions images that already exist.

## How it works

**Important:** no spatial relation is inferred here. Every fact this pipeline uses (object
attributes, who is left/right/front/behind of whom) is *already written down* in
`original-clevr-train-scenes.json`, computed once by the original image-rendering pipeline from
real 3D coordinates. This code only reads those existing facts and rephrases them — it never
looks at pixels. That's what makes it deterministic: same input text → same output caption, always.

Running example — one record already in `original-clevr-train-scenes.json`:
```
image: "CLEVR_train_000000.png"
pred:  "object(o_0). color(o_0, blue). shape(o_0, cube). ... object(o_2). color(o_2, cyan). shape(o_2, cube). ..."
text:  "... The objects to the right of blue cube are: cyan cube. ..."
```

1. **Reconstruct** (`scene_reconstruct.py`) — parse `pred` into a table: `o_0 → {color: blue,
   shape: cube, ...}`, `o_2 → {color: cyan, shape: cube, ...}`. Parse `text`'s "objects to the
   right/front of X are: ..." sentences into relation facts: *"o_2 is right of o_0"* (and, since
   right/left are opposites, *"o_0 is left of o_2"* for free). Output:
   `{"objects": {...}, "relations": [{"from": "o_2", "to": "o_0", "direction": "right"}, ...]}`.
2. **Query** (`scene_queries.py`) — small lookup helpers over that table/relation list, e.g.
   "which objects are cubes", "what is right of o_0".
3. **Ground** (`grounders.py`) — for each requested constraint class (C1–C9), use the Step-2
   helpers to find a combination of facts that makes the class's pattern true *for this specific
   scene* (e.g. C5: "a cube standing to the right of another cube" → true here via o_2/o_0), then
   render it into English with `verbalize.verbalize(...)` — the same phrasing function
   `eval_dataset_gen` uses for the evaluation prompts.
4. **Compile** (`compile_dataset.py`) — pick N grounded sentences per image (config-controlled),
   prefix with a one-line object count, write one `{id, image, text, ...}` record per image.

To modify behavior: change what counts as true in Step 3 (`grounders.py`), add new constraint
classes there, or change how facts are read from the source JSON in Step 1
(`scene_reconstruct.py`) if the upstream data format changes.

## Files

| File | Role |
|------|------|
| `scene_reconstruct.py` | Parses `pred`/`text` into a structured scene |
| `scene_queries.py` | Scene-query primitives shared by all grounders |
| `grounders.py` | Per-class (C1–C9) grounders + `ground_constraints()` entry point |
| `compile_dataset.py` | CLI: scenes + images → `finetune-dataset.json` |

## Run

```bash
cd clevr-ccig-dataset-gen

# Default: compiles every image in data/finetune-dataset/images against its scene record
python -m src.finetune_dataset_gen.compile_dataset

# Control how many grounded constraints go into each caption
python -m src.finetune_dataset_gen.compile_dataset --constraints_per_image 1       # exactly 1
python -m src.finetune_dataset_gen.compile_dataset --constraints_per_image random  # 1..max_constraints, randomized per image
python -m src.finetune_dataset_gen.compile_dataset --constraints_per_image all     # every satisfiable constraint found (long captions)

# Restrict to specific classes, change verbalization granularity, smoke-test on a few images
python -m src.finetune_dataset_gen.compile_dataset --classes C1 C8 --granularity short --limit 50
```

Defaults (overridable via `--scenes`/`--images`/`--out`) resolve to the repo-root `data/`
folder: `data/finetune-dataset/original-clevr-train-scenes.json`, `data/finetune-dataset/images/`,
`data/finetune-dataset/finetune-dataset.json`.

## Output format

```json
{
  "id": 0,
  "image": "CLEVR_train_000000.png",
  "text": "There are 6 objects in the scene. There must be at least one object in r0 standing to the right of a distinct small object. For every such pair, the latter must also be cube.",
  "n_objects": 6,
  "constraints": [
    {"class": "C5", "variant": "pair_propRelA_propC", "short": "...", "medium": "...", "long": "..."}
  ]
}
```
