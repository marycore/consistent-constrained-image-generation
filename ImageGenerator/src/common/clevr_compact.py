"""
Convert long CLEVR-style captions to the compact format used for CLIP (≤77 tokens).
Use this when your model was fine-tuned on dataset_clip77.json so inference
prompts use the same style.
"""

from __future__ import annotations

import re
from typing import Optional

# Lazy-load CLIP tokenizer for exact token count (avoids import/cache issues)
_tokenizer: Optional["CLIPTokenizer"] = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import CLIPTokenizer
            _tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        except Exception:
            pass
    return _tokenizer


def count_tokens(text: str) -> int:
    """Return CLIP token count for text (or conservative estimate if tokenizer unavailable)."""
    tok = _get_tokenizer()
    if tok is not None:
        return tok(text, return_tensors="pt", truncation=False).input_ids.shape[1]
    # Conservative: CLIP often gives ~1.5-1.7 tokens/word for this style; overestimate to stay under 77
    return int(len(text.split()) * 1.7)


# Abbreviations (same as in compress_clevr_to_clip77.py)
SIZE = {"large": "L", "small": "S"}
MAT = {"metal": "met", "rubber": "rb"}
SHAPE = {"cylinder": "cyl", "sphere": "sph", "cube": "cube"}
COLORS = {
    "blue": "bl", "green": "gr", "cyan": "cy", "brown": "br",
    "gray": "gy", "yellow": "yl", "purple": "pu", "red": "rd",
}


def _abbrev(obj: dict) -> str:
    c = COLORS.get(obj["color"], obj["color"])
    s = SIZE.get(obj["size"], obj["size"])
    m = MAT.get(obj["material"], obj["material"])
    sh = SHAPE.get(obj["shape"], obj["shape"])
    return f"{c} {s} {m} {sh}"


def _obj_str(i: int, obj: dict, region: str) -> str:
    return f"{i}: {_abbrev(obj)} r{region}"


def _resolve_list(s: str, objects: list, obj_desc) -> list:
    indices = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        for j in range(len(objects)):
            if obj_desc(j) == part:
                indices.append(j)
                break
    return indices


def parse_clevr(text: str) -> dict:
    """Parse CLEVR long caption into structured data (objects, right_of, front_of)."""
    m = re.search(r"There are (\d+) objects? in the scene", text)
    n_objs = int(m.group(1)) if m else 0

    pattern = re.compile(
        r"There is a (blue|green|cyan|brown|gray|yellow|purple|red) "
        r"(large|small) (metal|rubber) (cube|sphere|cylinder) in region r(\d+)"
    )
    objects = []
    for match in pattern.finditer(text):
        objects.append({
            "color": match.group(1),
            "size": match.group(2),
            "material": match.group(3),
            "shape": match.group(4),
            "region": match.group(5),
        })

    def obj_desc(i: int) -> str:
        o = objects[i]
        return f"{o['color']} {o['size']} {o['material']} {o['shape']}"

    right_of = {}
    front_of = {}
    for i in range(len(objects)):
        needle = obj_desc(i)
        prefix_r = "The objects to the right of " + needle + " are:"
        if prefix_r in text:
            start = text.index(prefix_r) + len(prefix_r)
            end = text.find(".", start)
            if end == -1:
                end = len(text)
            right_str = text[start:end].strip()
            right_of[i] = _resolve_list(right_str, objects, obj_desc)
        prefix_f = "The objects to the front of " + needle + " are:"
        if prefix_f in text:
            start = text.index(prefix_f) + len(prefix_f)
            end = text.find(".", start)
            if end == -1:
                end = len(text)
            front_str = text[start:end].strip()
            front_of[i] = _resolve_list(front_str, objects, obj_desc)

    return {
        "n_objs": len(objects),
        "objects": objects,
        "right_of": right_of,
        "front_of": front_of,
    }


def _obj_str_minimal(i: int, obj: dict, region: str) -> str:
    """Most compact object line: no colon, region as digit (saves tokens)."""
    a = _abbrev(obj)
    return f"{i} {a} {region}"


def _build_level1(data: dict) -> str:
    """Standard compact: 'CLEVR 4 regions. N objects.' + '0: bl L rb cube r1' + 'R0:2,5 F0:1,3'."""
    n = data["n_objs"]
    objects = data["objects"]
    right_of, front_of = data["right_of"], data["front_of"]
    parts = [f"CLEVR 4 regions. {n} objects."]
    for i, o in enumerate(objects):
        parts.append(_obj_str(i, o, o["region"]))
    so_far = " ".join(parts)
    rel_parts = []
    for i in range(n):
        r_list = right_of.get(i, [])
        f_list = front_of.get(i, [])
        if r_list:
            rel_parts.append(f"R{i}:" + ",".join(map(str, r_list)))
        if f_list:
            rel_parts.append(f"F{i}:" + ",".join(map(str, f_list)))
    if rel_parts:
        so_far += " " + " ".join(rel_parts)
    return so_far


def _build_level2(data: dict) -> str:
    """Shorter: no colons in relations, region as digit, shorter preamble (fewer tokens)."""
    n = data["n_objs"]
    objects = data["objects"]
    right_of, front_of = data["right_of"], data["front_of"]
    parts = [f"CLEVR 4r {n} objs."]
    for i, o in enumerate(objects):
        parts.append(_obj_str_minimal(i, o, o["region"]))
    so_far = " ".join(parts)
    rel_parts = []
    for i in range(n):
        r_list = right_of.get(i, [])
        f_list = front_of.get(i, [])
        if r_list:
            rel_parts.append("R" + str(i) + " " + " ".join(map(str, r_list)))
        if f_list:
            rel_parts.append("F" + str(i) + " " + " ".join(map(str, f_list)))
    if rel_parts:
        so_far += " " + " ".join(rel_parts)
    return so_far


def _build_level3(data: dict) -> str:
    """Shorter relations: R0,2,5 F0,1,3 (comma-separated, no space after R/F)."""
    n = data["n_objs"]
    objects = data["objects"]
    right_of, front_of = data["right_of"], data["front_of"]
    parts = [f"CLEVR 4r {n} objs."]
    for i, o in enumerate(objects):
        parts.append(_obj_str_minimal(i, o, o["region"]))
    so_far = " ".join(parts)
    rel_parts = []
    for i in range(n):
        r_list = right_of.get(i, [])
        f_list = front_of.get(i, [])
        if r_list:
            rel_parts.append("R" + str(i) + "," + ",".join(map(str, r_list)))
        if f_list:
            rel_parts.append("F" + str(i) + "," + ",".join(map(str, f_list)))
    if rel_parts:
        so_far += " " + " ".join(rel_parts)
    return so_far


def _obj_str_compact(i: int, obj: dict, region: str) -> str:
    """Even shorter: index fused with color to save a token (e.g. '0bl L rb cube 1')."""
    c = COLORS.get(obj["color"], obj["color"])
    s = SIZE.get(obj["size"], obj["size"])
    m = MAT.get(obj["material"], obj["material"])
    sh = SHAPE.get(obj["shape"], obj["shape"])
    return f"{i}{c} {s} {m} {sh} {region}"


def _build_level4(data: dict) -> str:
    """Shortest object lines (index+color fused) + level3 relations."""
    n = data["n_objs"]
    objects = data["objects"]
    right_of, front_of = data["right_of"], data["front_of"]
    parts = [f"CLEVR 4r {n} objs."]
    for i, o in enumerate(objects):
        parts.append(_obj_str_compact(i, o, o["region"]))
    so_far = " ".join(parts)
    rel_parts = []
    for i in range(n):
        r_list = right_of.get(i, [])
        f_list = front_of.get(i, [])
        if r_list:
            rel_parts.append("R" + str(i) + "," + ",".join(map(str, r_list)))
        if f_list:
            rel_parts.append("F" + str(i) + "," + ",".join(map(str, f_list)))
    if rel_parts:
        so_far += " " + " ".join(rel_parts)
    return so_far


def build_compact(data: dict, max_tokens: int = 77) -> str:
    """Build compact caption from parsed CLEVR data, under max_tokens. Uses progressively shorter encodings; only drops relation text if even the shortest full form exceeds the limit."""
    n = data["n_objs"]
    objects = data["objects"]
    right_of, front_of = data["right_of"], data["front_of"]

    for build_fn in [_build_level1, _build_level2, _build_level3, _build_level4]:
        candidate = build_fn(data)
        if count_tokens(candidate) <= max_tokens:
            return candidate

    # Still over 77: keep full object list, add relations until at limit (so we only drop tail relations if unavoidable)
    parts = [f"CLEVR 4r {n} objs."]
    for i, o in enumerate(objects):
        parts.append(_obj_str_compact(i, o, o["region"]))
    so_far = " ".join(parts)
    rel_parts = []
    for i in range(n):
        r_list = right_of.get(i, [])
        f_list = front_of.get(i, [])
        if r_list:
            rel_parts.append("R" + str(i) + "," + ",".join(map(str, r_list)))
        if f_list:
            rel_parts.append("F" + str(i) + "," + ",".join(map(str, f_list)))
    if not rel_parts:
        return so_far
    best = so_far
    for k in range(1, len(rel_parts) + 1):
        candidate = so_far + " " + " ".join(rel_parts[:k])
        if count_tokens(candidate) <= max_tokens:
            best = candidate
        else:
            break
    return best


def clevr_long_to_compact(
    long_text: str,
    max_tokens: int = 77,
    *,
    on_parse_error: str = "pass_through",
) -> str:
    """
    Convert long CLEVR-style caption to compact form (≤ max_tokens).

    Use this for inference when the model was fine-tuned on dataset_clip77.json
    so prompts match the training style.

    Args:
        long_text: Full CLEVR caption (e.g. "Generate a CLEVR style image... There are N objects...").
        max_tokens: Target max token count (default 77 for CLIP).
        on_parse_error: If parsing fails, "pass_through" returns long_text (truncated to ~500 chars),
            "raise" raises ValueError, "empty" returns "".

    Returns:
        Compact string in the same style as dataset_clip77.json.
    """
    try:
        parsed = parse_clevr(long_text)
        if not parsed["objects"]:
            raise ValueError("No objects parsed")
        return build_compact(parsed, max_tokens=max_tokens)
    except Exception as e:
        if on_parse_error == "raise":
            raise ValueError(f"CLEVR parse failed: {e}") from e
        if on_parse_error == "empty":
            return ""
        # pass_through: return original, truncated to reduce tokens
        return long_text[:500].strip() if len(long_text) > 500 else long_text
