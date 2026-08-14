# ccig-human-evaluation

A browser-based tool for hand-annotating scene graphs on generated CLEVR/COCO
images. It lets a human's labels be compared against `ccig-evaluation`'s
automated perception pipeline, so you can measure how accurate that pipeline
actually is, image by image.

For each image you draw a bounding box per object and pick its
shape/color/material/size from a fixed vocabulary; region and left/right/
front/behind relations are computed automatically from the boxes. If
perception has already run on the batch, its (imperfect) detections
pre-populate the boxes so you're correcting, not starting from scratch.

## Setup

```bash
cd ccig-human-evaluation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# only needed to run the "Run perception on all images" button:
pip install torch "transformers>=4.46" accelerate scipy
```

`clingo` must be on `PATH` for "Run perception" (it isn't needed for hand
annotation alone).

## Running it

```bash
python -m src.app --port 5001
```

Open `http://localhost:5001` and fill in the setup form:

| Field | Example |
|---|---|
| Images dir | `data/generated_images/gpt-image-2-low/clevr_1_scenes_SAT` |
| Prompts file | `data/ccig_eval_dataset/clevr_1_scenes_SAT.json` |

Both must be from the same generation batch. Output location is derived
automatically, not entered by hand (see below).

## Input / output

**Input:** an images dir of `{id}-{short|medium|long}.png` files, and the
matching `ccig_eval_dataset_*.json` prompts/ground-truth file (one JSON
record per line) that batch was generated from.

**Output:** two JSON files, written to a location derived from the images
dir (`<model>` = the folder right after `generated_images`) and the prompts
file (`<dataset>` = its filename stem):

```
data/evaluation/perception/<model>/<dataset>_auto-perception.json
data/evaluation/perception/<model>/<dataset>_human-perception.json
```

Both share the same entry shape (one object per image, `results`/
`annotations` list) and the same field names as the ground-truth file where
the concept is shared: `id`, `prompt_field`, `image_path`, `prompt`,
`instantiated_rule`, `status`, `number_of_objects`, `scene_graph`. Neither
file references the other's path. `auto-perception` additionally carries
`predicted_status`, `agrees_with_dataset`, `clingo_program`, `success`,
`error` (solver output, no human equivalent). `human-perception` additionally
carries two independent human-only judgment calls, each `true`/`false`/`null`,
set per image in the annotation view (Yes/No toggle):
- `reasonable_scene` — is the generated image a sensible scene at all. Its
  starting value (before that image has ever been saved) is whatever you
  picked on the setup form.
- `valid_scene` — distinct from the above; starts `true` by default (fixed,
  not configurable on the setup form).

If a human entry is missing but an automated one exists for that image, it's
auto-copied over as a starting point the next time perception runs or the
batch is loaded — so every autosaved edit always starts from perception's
guess unless you clear it (or click "Mark scene as empty").

## Not yet implemented

- Running the human scene graph through the ASP/clingo constraint check for
  a SAT/UNSAT verdict (needed to compare against judge/CLIP scores) — scene
  graph capture only, for now.
- Batch/aggregate agreement metrics (perception vs. human, judge vs. human,
  CLIP score vs. human) — this tool only produces the raw annotation files;
  scoring them is a separate step.
