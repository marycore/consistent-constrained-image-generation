# ccig-human-evaluation

A browser-based tool for hand-annotating scene graphs on generated CLEVR/COCO
images. It lets a human's labels be compared against `ccig-evaluation`'s
automated perception pipeline, so you can measure how accurate that pipeline
actually is, image by image.

For each image you draw a bounding box per object and pick its
shape/color/material from a fixed vocabulary; region and left/right/
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

## Running it

```bash
python -m src.app --port 5001
```

Open `http://localhost:5001` and fill in the setup form:

| Field | Example |
|---|---|
| Images dir | `data/generated_images/gpt-image-2-low/clevr_1_scenes_SAT` |
| Prompts file | `data/ccig_eval_dataset/clevr_1_scenes_SAT.jsonl` |

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
file references the other's path, and neither carries anything ASP/clingo-
related — this pipeline only detects objects and their properties, it never
runs the constraint-satisfaction check ccig-evaluation's own perception
pipeline does. `auto-perception` additionally carries `success`/`error` (did
detection succeed on this image). `human-perception` additionally carries
`reasonable_scene` (`true`/`false`/`null`) — a human-only judgment of whether
the generated image is a sensible scene at all, set per image in the
annotation view (Yes/No toggle); its starting value (before that image has
ever been saved) is whatever you picked on the setup form.

If a human entry is missing but an automated one exists for that image, it's
auto-copied over as a starting point the next time perception runs or the
batch is loaded — so every autosaved edit always starts from perception's
guess unless you clear it (or click "Mark scene as empty").

## Browse page

`http://localhost:5001/browse` is a second, independent page for opening any
`*_human-perception.json` file directly, without going through the setup
form: a dropdown lists every one found under `data/evaluation/perception/*/`,
picking one populates a second dropdown of its images, and picking an image
opens the same box/property editor as the main page. No prompt, no
constraint, no automated-perception comparison, no file paths shown anywhere
— just the file, the image, and the annotation. It edits the same
`human-perception.json` files described above (nothing separate to keep in
sync), and doesn't require or touch a configured batch on the main page.

## Password protection

There's no authentication by default -- fine for local use, not safe to expose
as-is (`/api/setup` reads arbitrary local file paths; the save endpoints let
anyone with the URL overwrite your annotations). Set a password to turn it on:

```bash
CCIG_HUMAN_EVAL_PASSWORD=some-long-random-string python -m src.app --port 5001
```

Every route (main page, `/browse`, all `/api/...` endpoints) then requires
HTTP Basic Auth with that password (any username) -- your browser will prompt
for it once and remember it for the session. Leaving the variable unset keeps
the server exactly as before, no prompt, nothing changed.

## Deploying to Cloud Run (remote annotators)

`deploy.sh` builds a self-contained image (annotation only — no torch/
transformers, no ASP/clingo) and deploys it to Cloud Run, with your `data/`
folder uploaded once to a Cloud Storage bucket and mounted into the container
so reads/writes work the same as local files:

```bash
gcloud auth login   # once, if not already
CCIG_HUMAN_EVAL_PASSWORD='choose-a-real-password' ./deploy.sh
```

See the comments at the top of `deploy.sh` for the project/region/bucket
defaults and how to override them. It prints a public `https://*.run.app` URL
when done — share that and the password with your annotators. In the deployed
app's setup form, paths live under `/srv/data/...` (see `deploy.sh`'s final
output for exact examples) rather than wherever `data/` sits on your machine.

## Not yet implemented

- Batch/aggregate agreement metrics (perception vs. human, judge vs. human,
  CLIP score vs. human) — this tool only produces the raw annotation files;
  scoring them is a separate step.
