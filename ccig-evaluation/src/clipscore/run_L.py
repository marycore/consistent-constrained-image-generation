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
    import torch, sys
    from PIL import Image
    #from transformers import AutoProcessor, AutoModel
    sys.path.append("/users/sbsh670/Long-CLIP")
    from model import longclip
    checkpoint = "/users/sbsh670/Long-CLIP/checkpoints/longclip-L.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = longclip.load(checkpoint,device=device,)

    model.eval()
    
    '''
    checkpoint = "Ambarella/LongCLIP"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(checkpoint,trust_remote_code=True)

    model = AutoModel.from_pretrained(checkpoint,trust_remote_code=True).to(device)

    model.eval()
    '''
    
    
    results: list[ClipScoreResult] = []
    len_77 = 0
    for item in items:
        try:
            
            image = Image.open(item.image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            text = "A photo depicts an image where: " + item.prompt_text
            try:
                text_input = longclip.tokenize([text]).to(device)

            except RuntimeError as e:
                if "too long for context length 248" in str(e):
                    len_77 = len_77+1
                    print('gretaer:',item.id, item.prompt_field, len_77)
                    # Re-tokenize with truncation
                    text_input = longclip.tokenize(
                        [text], truncate=True).to(device)
                else:
                    raise
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_input)

        
            image_features = image_features / image_features.norm(dim=-1,keepdim=True,)

            text_features = text_features / text_features.norm(dim=-1,keepdim=True,)
            cosine = (image_features @ text_features.T).item()

            # Same scoring convention as your original code
            clipscore = 2.5 * max(cosine, 0.0)
            
            
            results.append(
                ClipScoreResult(
                    clipmodel=checkpoint,
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
                    clipmodel = checkpoint,
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
