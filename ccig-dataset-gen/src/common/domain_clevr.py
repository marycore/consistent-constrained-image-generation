"""
CLEVR domain: property/value vocabulary shared by eval_dataset_gen and finetune_dataset_gen,
reflecting what's actually in the CLEVR-CCIG scenes: 8 colors, 3 shapes, 2 sizes, 2 materials,
4 quadrant regions (labeled "r0".."r3", matching the region atoms already baked into the
rendered scene data).

eval_dataset_gen deliberately excludes "material" from its ASP constraint search space (see
eval_dataset_gen/domain.py) -- that's a one-sided override on top of this domain, not a second
independent copy of it.

A future domain (e.g. COCO) would live as a sibling module, e.g. src/common/domain_coco.py,
exposing the same PROPERTIES/DIRECTIONS shape; run.py's --domain flag picks which one is active.
"""

from __future__ import annotations

PROPERTIES: dict[str, list[str]] = {
    "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "shape": ["cube", "cylinder", "sphere"],
    "size": ["small", "large"],
    "material": ["rubber", "metal"],
    "region": ["r0", "r1", "r2", "r3"],
}

DIRECTIONS: list[str] = ["left", "right", "front", "behind"]
