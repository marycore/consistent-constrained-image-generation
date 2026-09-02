from __future__ import annotations

from pathlib import Path

from ..common.dataset_gen import load_domain, solve
from ..common.io import write_json
from ..common.types import MatchedItem, PerceptionResult
from .attributes.registry import build_attribute_classifier
from .crop import crop_and_neutralize, make_object_views
from .detectors.registry import build_detector
from .regions import bbox_center, region_of
from .scene_graph import build_scene_facts, to_graph_dict
from .types import DetectedObject


def _build_property_classifiers(domain_module, domain: str, attribute_classifier: str, device: str | None) -> dict:
    """One classifier instance per property, built once per run (not per object --
    reloading a CLIP checkpoint per crop would be prohibitively slow). Properties
    come from domain_module.PROPERTIES, never hardcoded, so this generalizes to
    whatever properties a domain defines (e.g. CLEVR's "size", which some
    constraints reference despite the sibling pipeline's ASP generator excluding it
    from its own choice rules)."""
    if domain == "clevr":
        properties = ["shape", "color", "material"]
    else:  # coco: category comes from the detector label itself, only color is classified
        properties = ["color"]
    return {
        prop: build_attribute_classifier(attribute_classifier, prop, domain_module.PROPERTIES[prop], device)
        for prop in properties
    }

from PIL import Image


def _classify_object_properties(
    domain: str,
    views: dict[str, Image.Image],
    classifiers: dict,
) -> dict[str, str]:
    """
    Classification order for CLEVR:
      1. shape
      2. color conditioned on predicted shape
      3. material
      4. size

    Different attributes use different image views.
    """

    properties: dict[str, str] = {}

    if domain == "clevr":

        # Shape: emphasize geometry with tighter views.
        predicted_shape, _ = classifiers["shape"].classify_ensemble(
            [
                views["tight"],
                views["medium"],
            ]
        )
        properties["shape"] = predicted_shape

        # Color: use multiple views, conditioned on predicted shape.
        # Example competing prompts:
        #   "a red cube"
        #   "a blue cube"
        #   "a green cube"
        #   ...
        predicted_color, _ = classifiers["color"].classify_ensemble([views["tight"],views["medium"],views["wide"],],
            context=predicted_shape,
        )
        properties["color"] = predicted_color

        # Material: preserve RGB information because lighting,
        # highlights and reflections can be useful.
        predicted_material, _ = classifiers["material"].classify_ensemble(
            [
                views["rgb"],
                #views["medium"],
            ]
        )
        properties["material"] = predicted_material

        # Size: use medium/wide views.
        #predicted_size, _ = classifiers["size"].classify_ensemble(
        #    [
        #        views["medium"],
        #        views["wide"],
        #    ]
        #)
        #properties["size"] = predicted_size

    else:
        # Non-CLEVR: only color is classified here.
        predicted_color, _ = classifiers["color"].classify_ensemble(
            [
                views["tight"],
                views["medium"],
                views["wide"],
            ]
        )
        properties["color"] = predicted_color

    return properties

def _perceive_scene(
    image,
    domain: str,
    domain_module,
    detector,
    classifiers: dict,
) -> list[DetectedObject]:

    
    class_prompts = domain_module.PROPERTIES["shape"]

    boxes = detector.detect(image, class_prompts)

    objects: list[DetectedObject] = []

    for obj_id, bbox in enumerate(boxes):

        # Generate multiple representations of the detected object.
        views = make_object_views(image, bbox)

        properties = _classify_object_properties(
            domain,
            views,
            classifiers,
        )

        if domain == "coco":
            # Detector's matched class prompt is the category.
            properties["shape"] = bbox.label

        cx, cy = bbox_center(bbox)
        region = region_of(
            cx,
            cy,
            image.width,
            image.height,
        )

        objects.append(
            DetectedObject(
                obj_id=obj_id,
                bbox=bbox,
                properties=properties,
                region=region,
            )
        )

    return objects



def run_perception(
    items: list[MatchedItem],
    domain: str,
    detector_name: str,
    attribute_classifier: str,
    device: str | None,
    out_path: str | Path,
) -> None:
    from PIL import Image

    domain_module = load_domain(domain)
    detector = build_detector(detector_name, device=device)
    classifiers = _build_property_classifiers(domain_module, domain, attribute_classifier, device)

    results: list[PerceptionResult] = []
    for item in items:
        try:
            image = Image.open(item.image_path).convert("RGB")
            print('Processing:', item.id, flush=True)
            objects = _perceive_scene(image, domain, domain_module, detector, classifiers)
            facts = build_scene_facts(objects)
            # instantiated_rule uses free ASP variables (X, Y, ...) bound over object(X),
            # not literal object ids -- it applies unchanged no matter how perception
            # numbered the detected objects here.
            program = f"{facts}\n" + "\n".join(item.record.instantiated_rule)

            predicted_status, _ = solve(program, n_models=1, time_limit=10)
            results.append(
                PerceptionResult(
                    id=item.id,
                    prompt_field=item.prompt_field,
                    image_path=str(item.image_path),
                    instantiated_rule=item.record.instantiated_rule,
                    dataset_status=item.record.status,
                    predicted_status=predicted_status,
                    agrees_with_dataset=predicted_status == item.record.status,
                    scene_graph=to_graph_dict(objects),
                    clingo_program=program,
                    success=True,
                    error=None,
                )
            )
            print(f"[ok]   {item.id}: predicted={predicted_status} dataset={item.record.status}")
        except Exception as e:
            results.append(
                PerceptionResult(
                    id=item.id,
                    prompt_field=item.prompt_field,
                    image_path=str(item.image_path),
                    instantiated_rule=item.record.instantiated_rule,
                    dataset_status=item.record.status,
                    predicted_status=None,
                    agrees_with_dataset=None,
                    scene_graph=None,
                    clingo_program=None,
                    success=False,
                    error=repr(e),
                )
            )
            print(f"[fail] {item.id}: {e}")

    write_json(
        out_path,
        {
            "method": "perception",
            "domain": domain,
            "detector": detector_name,
            "attribute_classifier": attribute_classifier,
            "results": [r.to_json() for r in results],
        },
    )
