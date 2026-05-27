from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path


COLOR_VALUES = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
MATERIAL_VALUES = ["rubber", "metal"]
SHAPE_VALUES = ["cube", "cylinder", "sphere", "cone"]
SIZE_VALUES = ["small", "large"]
REL_VALUES = ["left", "right", "front", "behind"]
PROP_VALUES = ["color", "material", "shape", "size"]
REGION_VALUES = ["0", "1", "2", "3"]
COUNT_VALUES = ["1", "2", "3"]


PLACEHOLDER_PATTERN = re.compile(r"([RPDNV][0-9]')")


def value_space_for_placeholder(ph: str) -> list[str]:
    if ph.startswith("R"):
        return REGION_VALUES
    if ph.startswith("P"):
        return PROP_VALUES
    if ph.startswith("D"):
        return REL_VALUES
    if ph.startswith("N"):
        return COUNT_VALUES
    # V* value depends on P* if available
    return COLOR_VALUES + MATERIAL_VALUES + SHAPE_VALUES + SIZE_VALUES


def sample_assignment(placeholders: set[str], rng: random.Random) -> dict[str, str]:
    assignment: dict[str, str] = {}
    # Sample non-V first
    ordered = sorted(placeholders, key=lambda x: (x.startswith("V"), x))
    for ph in ordered:
        if ph.startswith("V"):
            p_key = "P" + ph[1] + "'"
            p_val = assignment.get(p_key)
            if p_val == "color":
                assignment[ph] = rng.choice(COLOR_VALUES)
            elif p_val == "material":
                assignment[ph] = rng.choice(MATERIAL_VALUES)
            elif p_val == "shape":
                assignment[ph] = rng.choice(SHAPE_VALUES)
            elif p_val == "size":
                assignment[ph] = rng.choice(SIZE_VALUES)
            else:
                assignment[ph] = rng.choice(value_space_for_placeholder(ph))
        else:
            assignment[ph] = rng.choice(value_space_for_placeholder(ph))
    return assignment


def apply_assignment(s: str, assignment: dict[str, str]) -> str:
    out = s
    # Replace longer keys first to avoid partial collisions
    for ph in sorted(assignment.keys(), key=len, reverse=True):
        out = out.replace(ph, assignment[ph])
        if ph.endswith("'"):
            out = out.replace(ph[:-1], assignment[ph])  # also support unprimed placeholders in comments
    return out


def parse_constraint_file(path: Path) -> list[tuple[str, str]]:
    """
    Returns list of (comment_text, asp_rule_line) tuples.
    """
    entries: list[tuple[str, str]] = []
    pending_comment = ""
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("%"):
                pending_comment = line.lstrip("%").strip()
                continue
            if line.startswith(":-"):
                entries.append((pending_comment, line))
                pending_comment = ""
    return entries


def build_textual_description(*, comment: str, assignment: dict[str, str], asp_rule: str) -> str:
    if comment:
        t = apply_assignment(comment, assignment)
        t = " ".join(t.split())
        if not t.endswith("."):
            t += "."
        return t
    # Fallback if no comment exists in template file
    return "Instantiated constraint: " + asp_rule


def sample_scene_objects(rng: random.Random, n_objects: int) -> list[dict[str, str | int]]:
    objects = []
    for i in range(n_objects):
        objects.append(
            {
                "id": i,
                "region": int(rng.choice(REGION_VALUES)),
                "color": rng.choice(COLOR_VALUES),
                "material": rng.choice(MATERIAL_VALUES),
                "shape": rng.choice(SHAPE_VALUES),
                "size": rng.choice(SIZE_VALUES),
            }
        )
    return objects


def build_scene_facts(objects: list[dict[str, str | int]]) -> str:
    if not objects:
        return ""
    max_id = max(int(o["id"]) for o in objects)
    lines = [f"object(0..{max_id})."]
    for o in objects:
        i = int(o["id"])
        lines.append(f"at({i}, {o['region']}).")
        lines.append(f"hasProperty({i}, color, {o['color']}).")
        lines.append(f"hasProperty({i}, material, {o['material']}).")
        lines.append(f"hasProperty({i}, shape, {o['shape']}).")
        lines.append(f"hasProperty({i}, size, {o['size']}).")
    return "\n".join(lines)


def build_prompt_text(*, objects: list[dict[str, str | int]], constraint_text: str) -> str:
    obj_parts = []
    for o in objects:
        obj_parts.append(
            f"o{o['id']}(region={o['region']}, size={o['size']}, color={o['color']}, material={o['material']}, shape={o['shape']})"
        )
    intro = (
        f"Create a scene with 4 regions and {len(objects)} objects. "
        f"The objects are: {', '.join(obj_parts)}."
    )
    return f"{intro} Satisfy this constraint: {constraint_text}"


def validate_with_clingo(asp_code: str, clingo_bin: str) -> tuple[bool, str]:
    """
    Return (is_satisfiable, raw_stdout) by running clingo on a temp file.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".lp", delete=True, encoding="utf-8") as tf:
        tf.write(asp_code + "\n")
        tf.flush()
        proc = subprocess.run(
            [clingo_bin, "0", tf.name],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if "UNSATISFIABLE" in out:
            return False, out
        if "SATISFIABLE" in out:
            return True, out
        # Unknown output format -> treat as failure for safety
        return False, out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate minimal CCIG ASP dataset JSONL.")
    p.add_argument(
        "--index_json",
        default="question_generation/CLEVR_CCIG_templates/index.json",
        help="Path to index.json containing complexity classes (L0..L8).",
    )
    p.add_argument(
        "--constraint_templates_dir",
        default="image_generation/ConstraintTemplates/CCIG_constraint_templates",
        help="Directory containing constraint_templates_L*.txt files.",
    )
    p.add_argument(
        "--output_jsonl",
        default="ccig_asp_dataset.jsonl",
        help="Output JSONL path.",
    )
    p.add_argument(
        "--instances_per_class",
        type=int,
        default=10,
        help="Default number of instances per complexity class.",
    )
    p.add_argument(
        "--instances_map_json",
        default=None,
        help='Optional JSON string to override per class, e.g. \'{"L0":20,"L1":15}\'.',
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--class_filter",
        nargs="*",
        default=None,
        help="Optional subset of classes to generate, e.g. L0 L1 L2.",
    )
    p.add_argument("--min_objects", type=int, default=5, help="Minimum number of objects per scene.")
    p.add_argument("--max_objects", type=int, default=9, help="Maximum number of objects per scene.")
    p.add_argument(
        "--validate_with_clingo",
        action="store_true",
        help="If set, validate each generated ASP program with clingo and keep only satisfiable instances.",
    )
    p.add_argument(
        "--clingo_bin",
        default="clingo",
        help="Path to clingo binary (default: clingo). Used with --validate_with_clingo.",
    )
    p.add_argument(
        "--max_attempts_per_instance",
        type=int,
        default=50,
        help="Max resampling attempts per instance when --validate_with_clingo is enabled.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    index_path = Path(args.index_json)
    with index_path.open("r", encoding="utf-8") as f:
        index_data = json.load(f)
    constraint_dir = Path(args.constraint_templates_dir)

    override_map: dict[str, int] = {}
    if args.instances_map_json:
        override_map = json.loads(args.instances_map_json)

    class_filter = set(args.class_filter) if args.class_filter else None

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec_id = 0
    with out_path.open("w", encoding="utf-8") as out:
        for item in index_data["complexity_levels"]:
            level = item["level"]
            if class_filter and level not in class_filter:
                continue
            n_instances = int(override_map.get(level, args.instances_per_class))
            template_file = constraint_dir / f"constraint_templates_{level}.txt"
            if not template_file.exists():
                raise FileNotFoundError(f"Missing constraint template file: {template_file}")
            entries = parse_constraint_file(template_file)
            if not entries:
                raise RuntimeError(f"No ASP rules found in template file: {template_file}")

            for i in range(n_instances):
                rec_id += 1
                accepted = False
                last_error = ""
                full_asp = ""
                text = ""
                for _attempt in range(args.max_attempts_per_instance):
                    comment, asp_rule = entries[rng.randrange(len(entries))]
                    placeholders = set(PLACEHOLDER_PATTERN.findall(asp_rule + " " + comment))
                    assignment = sample_assignment(placeholders, rng)
                    asp_instantiated = apply_assignment(asp_rule, assignment)
                    constraint_text = build_textual_description(
                        comment=comment, assignment=assignment, asp_rule=asp_instantiated
                    )
                    n_objects = rng.randint(args.min_objects, args.max_objects)
                    objects = sample_scene_objects(rng, n_objects)
                    scene_facts = build_scene_facts(objects)
                    full_asp = scene_facts + "\n" + asp_instantiated if scene_facts else asp_instantiated
                    text = build_prompt_text(objects=objects, constraint_text=constraint_text)

                    if not args.validate_with_clingo:
                        accepted = True
                        break

                    ok, out = validate_with_clingo(full_asp, args.clingo_bin)
                    if ok:
                        accepted = True
                        break
                    last_error = out

                if not accepted:
                    raise RuntimeError(
                        f"Could not generate satisfiable instance for class {level} "
                        f"after {args.max_attempts_per_instance} attempts.\n"
                        f"Last clingo output:\n{last_error}"
                    )
                record = {
                    "id": f"scene_{rec_id:06d}",
                    "complexity_class": level,
                    "asp_code": full_asp,
                    "textual_description": text,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote dataset to: {out_path}")


if __name__ == "__main__":
    main()
