# clipscore

CLIPScore: cosine similarity between a generated image's CLIP embedding and its
prompt's CLIP text embedding, using a CLIP checkpoint pretrained on the target domain
(`--clip-checkpoint`, a local path or HF repo id -- one checkpoint for CLEVR, a
different one for COCO; this method does not ship a checkpoint of its own).

Score range is `[-1, 1]` (raw cosine similarity, not rescaled) -- the CLIP literature's
"CLIPScore" popular rescaling (`2.5 * max(cos, 0)`) is intentionally not applied here so
the raw signal is preserved for downstream analysis; apply any rescaling downstream if
needed.

Output: `clipscore/results.json` — `{method, domain, checkpoint, results: [{id,
prompt_field, prompt, image_path, clipscore, success, error}]}`.
