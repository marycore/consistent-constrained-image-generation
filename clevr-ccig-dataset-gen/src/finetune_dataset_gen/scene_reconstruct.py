"""
Reconstruct a structured scene {"objects": {...}, "relations": [...]} from one record of
data/finetune-dataset/original-clevr-train-scenes.json.

`pred` gives clean per-object attributes as ASP-style atoms:
    object(o_0). color(o_0, blue). size(o_0, large). material(o_0, rubber). shape(o_0, cube). region(o_0, r1).

`text` narratively lists, for every object, all objects "to the right of" and "to the front of"
it. That fully determines the pairwise relation graph (left/behind are the inverse of
right/front) without needing 3D coordinates.

Output shape mirrors ../eval_dataset_gen/solve.py::format_scene():
    {
      "objects": {"o_0": {"color": "blue", "size": "large", "material": "rubber",
                           "shape": "cube", "region": "r1"}, ...},
      "relations": [{"from": "o_0", "to": "o_2", "direction": "right"}, ...],
    }
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

_OBJ_ATOM = re.compile(r"object\((o_\d+)\)")
_PROP_ATOM = re.compile(r"(color|size|material|shape|region)\((o_\d+),\s*([\w]+)\)")

_INVERSE_DIRECTION = {"right": "left", "left": "right", "front": "behind", "behind": "front"}

# "The objects to the right of X are:a,b,c." / "The objects to the front of X are: a, b, c."
_RELATION_SENTENCE = re.compile(
    r"The objects to the (right|front) of ([a-z][\w\s]*?) are:\s*([^.]*)\."
)


def parse_objects(pred: str) -> Dict[str, Dict[str, str]]:
    """Parse `pred` into {obj_id: {color, size, material, shape, region}}."""
    objects: Dict[str, Dict[str, str]] = {}
    for obj_id in _OBJ_ATOM.findall(pred):
        objects.setdefault(obj_id, {})
    for prop, obj_id, val in _PROP_ATOM.findall(pred):
        objects.setdefault(obj_id, {})[prop] = val
    return objects


def _object_phrase(attrs: Dict[str, str]) -> str:
    """'blue large rubber cube' from an object's attribute dict, matching the phrasing in `text`."""
    return f"{attrs['color']} {attrs['size']} {attrs['material']} {attrs['shape']}"


def _phrase_to_id_map(objects: Dict[str, Dict[str, str]]) -> Dict[str, Optional[str]]:
    """
    Map each object's attribute-phrase to its id. If two objects share an identical phrase
    (color+size+material+shape), the phrase is ambiguous — map it to None so relation sentences
    referencing it are skipped rather than silently attributed to the wrong object.
    """
    phrase_to_ids: Dict[str, List[str]] = {}
    for obj_id, attrs in objects.items():
        phrase_to_ids.setdefault(_object_phrase(attrs), []).append(obj_id)
    return {
        phrase: (ids[0] if len(ids) == 1 else None)
        for phrase, ids in phrase_to_ids.items()
    }


def parse_relations(text: str, objects: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Parse right/front relation sentences from `text` into a full relation list including the
    left/behind inverses. Sentences referencing an ambiguous (duplicate-attribute) object phrase
    are skipped and not silently guessed.
    """
    phrase_to_id = _phrase_to_id_map(objects)
    relations: List[Dict[str, str]] = []
    seen = set()

    def _add(from_id: str, to_id: str, direction: str) -> None:
        key = (from_id, to_id, direction)
        if key in seen:
            return
        seen.add(key)
        relations.append({"from": from_id, "to": to_id, "direction": direction})

    for direction, subject_phrase, targets_raw in _RELATION_SENTENCE.findall(text):
        subject_id = phrase_to_id.get(subject_phrase.strip())
        if subject_id is None:
            continue
        target_phrases = [p.strip() for p in targets_raw.split(",") if p.strip()]
        for target_phrase in target_phrases:
            target_id = phrase_to_id.get(target_phrase)
            if target_id is None:
                continue
            # "objects to the right of X are Y" means Y is to the right of X, i.e. Y -right-> X
            # (edge {from: A, to: B, direction: d} means "A is d of B" -- verified empirically
            # against region-grid ground truth: 30113/30113 unambiguous cases match this
            # direction, 0/30113 matched the reversed (subject, target) storage used before).
            _add(target_id, subject_id, direction)
            _add(subject_id, target_id, _INVERSE_DIRECTION[direction])

    return relations


def reconstruct_scene(record: dict) -> Dict[str, object]:
    """Build {"objects": ..., "relations": ...} from one original-clevr-train-scenes.json record."""
    objects = parse_objects(record["pred"])
    relations = parse_relations(record["text"], objects)
    return {"objects": objects, "relations": relations}
