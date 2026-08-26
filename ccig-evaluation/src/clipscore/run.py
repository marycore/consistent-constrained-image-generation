from __future__ import annotations

from pathlib import Path

from ..common.io import write_json
from ..common.types import ClipScoreResult, MatchedItem


def run_clipscore(items: list[MatchedItem], domain: str, checkpoint: str, out_path: str | Path) -> None:
    """Compute CLIPScore (CLIP image-text embedding cosine similarity) for each
    generated image against the prompt that generated it.

    checkpoint: path (or HF repo id) to a CLIP model pretrained on `domain` --
    one checkpoint for clevr, a different one for coco, passed explicitly by the
    caller since neither is a general-purpose CLIP checkpoint.
    """
    # Imported lazily so selecting `--method vlm-judge` / `--method perception` doesn't
    # force-load torch/transformers when clipscore isn't requested.
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(checkpoint).to(device).eval()
    processor = CLIPProcessor.from_pretrained(checkpoint)
     
    results: list[ClipScoreResult] = []
    for item in items:
        try:
            image = Image.open(item.image_path).convert("RGB")
            inputs = processor(text=["A photo depicts an image where: " + item.prompt_text], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            image_embed = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            text_embed = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            #768d embeddings
            cosine = (image_embed @ text_embed.T).item()
            clipscore = 2.5 * max(cosine, 0.0)
            results.append(
                ClipScoreResult(
                    id=item.id,
                    prompt_field=item.prompt_field,
                    prompt=item.prompt_text,
                    image_path=str(item.image_path),
                    clipscore=clipscore,
                    cosine=cosine,
                    success=True,
                    error=None,
                )
            )
            print(f"[ok]   {item.id}: {clipscore:.4f}")
        except Exception as e:
            results.append(
                ClipScoreResult(
                    id=item.id,
                    prompt_field=item.prompt_field,
                    prompt=item.prompt_text,
                    image_path=str(item.image_path),
                    clipscore=None,
                    cosine=None,
                    success=False,
                    error=repr(e),
                )
            )
            print(f"[fail] {item.id}: {e}")
        

    write_json(
        out_path,
        {
            "method": "clipscore",
            "domain": domain,
            "checkpoint": checkpoint,
            "results": [r.to_json() for r in results],
        },
    )
