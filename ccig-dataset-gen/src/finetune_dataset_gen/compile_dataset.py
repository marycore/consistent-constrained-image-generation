import os, sys, json
from collections import Counter
from pathlib import Path
file_path = Path(__file__).resolve()
path_current = file_path.parents[0]
path_root = file_path.parents[1]
sys.path.append(".")
sys.path.append(str(path_root))
sys.path.append(str(path_current))
import numpy as np
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

#prepare data for image generative model pretraining


def annotate_with_regions(src_images_train, destination_train_images):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_train_images):
        os.makedirs(destination_train_images)
    # Iterate over all images in the source directory
    for img_name in os.listdir(src_images_train):
        # Full path to the image
        img_path = os.path.join(src_images_train, img_name)

        # Check if it's an image file (you can extend this check if needed)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open the image
            image = Image.open(img_path)
            width, height = image.size
            # Draw grey lines dividing the image into four regions
            draw = ImageDraw.Draw(image)
            vertical_line_x = width // 2
            horizontal_line_y = height // 2

            # Grey color for the lines
            grey_color = (180, 180, 180)

            # Draw the lines
            draw.line([(vertical_line_x, 0), (vertical_line_x, height)], fill=grey_color, width=3)
            draw.line([(0, horizontal_line_y), (width, horizontal_line_y)], fill=grey_color, width=3)

            # Add text for region labels
            font = ImageFont.load_default()  # Using the default font

            # Coordinates for placing text at each corner (regions r0, r1, r2, r3)
            text_positions = {
                'r0': (5, 3),
                'r1': (width - 10, 3),
                'r2': (5, height - 10),
                'r3': (width-10, height-10)
            }

            # Add text to each region
            for region, position in text_positions.items():
                draw.text(position, region, fill=(255, 255, 255), font=font)

            # Save the annotated image
            annotated_image_path = os.path.join(destination_train_images, img_name)
            image.save(annotated_image_path)

            print(f"Annotated image saved to: {annotated_image_path}")
            



def find_region(x, y):
   
    image_width = 480
    image_height = 320
    
    REGIONS = {
    "reg_0": {"x": [0, 240], "y": [0, 160]}, 
    "reg_1": {"x": [240, 480], "y": [0, 160]},
    "reg_2": {"x": [0, 240], "y": [160, 320]},
    "reg_3": {"x": [240, 480], "y": [160, 320]}
    }
    
    for region_id, region in REGIONS.items():
        if region["x"][0] <= x < region["x"][1] and region["y"][0] <= y < region["y"][1]:
            return region_id
    

def reconstruct_scene(scene):
    #return a dict:{
    #  "objects": {"o_0": {"color": "blue", "size": "large", "material": "rubber",
    #                       "shape": "cube", "region": "r1"}, ...},
    #  "relations": [{"from": "o_0", "to": "o_2", "direction": "right"}, ...],
    #}
    reconstructed_scene = {}
    objects_scene = scene['objects']
    num= len(objects_scene)
    relationships_scene = scene['relationships']
    objects = {}
    relationships = []
    for idx, o in enumerate(objects):
        pos = o['pixel_coords']
        x=  pos[0]
        y = pos[1]
        region = find_region(x, y)
        objects['o_'+idx] = {"color":o["color"], "size":o["size"], "material": o["material"] , "shape": o["shape"], "region": region }
        
    for idx, rr in enumerate(relationships['right']): ## rr is the list of objects to the right of idx
        for r, o in enumerate(rr):
            relation ={}
            relation['from'] = 'o_'+o
            relation['to'] = 'o_'+idx
            relation['direction'] = 'right'
            relationships.append(relation)

            relation_inv ={}
            relation_inv['from'] = 'o_'+idx
            relation_inv['to'] = 'o_'+o
            relation_inv['direction'] = 'left'
            relationships.append(relation_inv)
    
    for idx, rr in enumerate(relationships['front']): ## rr is the list of objects to the front of idx
        for r, o in enumerate(rr):
            relation ={}
            relation['from'] = 'o_'+o
            relation['to'] = 'o_'+idx
            relation['direction'] = 'front'
            relationships.append(relation)

            relation_inv ={}
            relation_inv['from'] = 'o_'+idx
            relation_inv['to'] = 'o_'+o
            relation_inv['direction'] = 'behind'
            relationships.append(relation_inv)
    return {"objects": objects, "relations": relationships}

def compile_dataset(
    scenes_path: Path,
    out_path: Path,
    *,
    constraints_per_image,
    granularity: str = "medium",
    classes: Optional[List[str]] = None,
    seed: int = 0,
    limit: Optional[int] = None,
    max_constraints: int = 3,
) -> List[dict]:
    
    rng = random.Random(seed)
    out_records = []
    n_no_constraints = 0

    # Load the CLEVR scenes JSON file
    with scenes_path.open("r", encoding="utf-8") as f:
        data = json.load(file)
    scene_records = data['scenes']
    i=0
    for rec in scene_records:
        scene = reconstruct_scene(rec)
        n_objects = len(scene["objects"])
        grounded = ground_constraints(scene, classes=classes, rng=rng)
        selected = _select_constraints(grounded, constraints_per_image, rng, max_constraints)
        if not selected:
            n_no_constraints += 1
            continue
        caption = build_caption(n_objects, selected, granularity)
        out_records.append({
            "id": rec["id"],
            "image": rec["image"],
            "text": caption,
            "n_objects": n_objects,
            "constraints": selected,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2)

    print(
        f"Compiled {len(out_records)} records to {out_path} "
        f"({n_no_constraints} images skipped: no groundable constraint found)."
    )
    return out_records


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile finetune-dataset.json from CLEVR scene records.")
    p.add_argument("--scenes", default=str(_REPO_DATA_DIR / "finetune-dataset" / "original-clevr-train-scenes.json"))
   #We have no images
    #p.add_argument("--images", default=str(_REPO_DATA_DIR / "finetune-dataset" / "images"))
    p.add_argument("--out", default=str(_REPO_DATA_DIR / "finetune-dataset" / "finetune-dataset.json"))
    p.add_argument(
        "--constraints_per_image", default="random",
        help="1 | <int> | 'random' | 'all' -- how many grounded constraints go into each caption.",
    )
    p.add_argument("--granularity", choices=["short", "medium", "long"], default="medium")
    p.add_argument(
        "--classes", nargs="*", default=None,
        help="Restrict to specific constraint classes, e.g. --classes C1 C8. Default: all C1-C9.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Only compile the first N matching images (for smoke tests).")
    p.add_argument(
        "--max_constraints", type=int, default=3,
        help="Upper bound on how many constraint sentences 'random' mode may pick (default: 3). "
             "Ignored by 'all' (exhaustive) and explicit-int modes.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    compile_dataset(
        Path(args.scenes),
        Path(args.out),
        constraints_per_image=_parse_constraints_per_image(args.constraints_per_image),
        granularity=args.granularity,
        classes=args.classes,
        seed=args.seed,
        limit=args.limit,
        max_constraints=args.max_constraints,
    )



if __name__ == "__main__":
    main()


#--scenes /users/sbsh670/data/clevr/CLEVR_v1.0/scenes/CLEVR_train_scenes.json
#--out /users/sbsh670/data/ccig_finetune