from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from ccig_template_lib import (
    AspRuleCache,
    ComboComponentFilter,
    ComboSpec,
    ComponentInstance,
    build_asp_program,
    instantiate_component,
    load_background,
    load_combo_specs,
    load_index,
    load_level_templates,
    make_combo_key,
    merge_text_variants,
    pick_template_entry,
    validate_with_clingo,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CCIG constraint-only dataset JSONL.")
    p.add_argument(
        "--index_json",
        default="question_generation/CLEVR_CCIG_templates/index.json",
        help="Path to index.json containing complexity classes (L0..L8).",
    )
    p.add_argument(
        "--templates_dir",
        default="question_generation/CLEVR_CCIG_templates",
        help="Directory containing L*.json prompt templates.",
    )
    p.add_argument(
        "--constraint_templates_dir",
        default="image_generation/ConstraintTemplates/CCIG_constraint_templates",
        help="Directory containing constraint_templates_L*.txt ASP rules.",
    )
    p.add_argument(
        "--background_asp",
        default="data/general_constraints.txt",
        help="Background ASP ontology file.",
    )
    p.add_argument(
        "--relationship_axioms",
        default="data/ccig_relationship_axioms.txt",
        help="ASP axioms defining hasRelationship/3.",
    )
    p.add_argument("--output_jsonl", default="ccig_asp_dataset.jsonl", help="Output JSONL path.")
    p.add_argument(
        "--mode",
        choices=["single", "combo"],
        default="single",
        help="Generate single-level or combined-constraint instances.",
    )
    p.add_argument(
        "--instances_per_class",
        type=int,
        default=10,
        help="Instances per complexity class in single mode.",
    )
    p.add_argument(
        "--instances_per_combo",
        type=int,
        default=10,
        help="Instances per combo pair in combo mode.",
    )
    p.add_argument(
        "--instances_map_json",
        default=None,
        help='Optional JSON override counts, e.g. \'{"L0":20,"L1":15}\'.',
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--class_filter",
        nargs="*",
        default=None,
        help="Optional subset of classes for single mode, e.g. L0 L1.",
    )
    p.add_argument(
        "--constraint_family",
        default=None,
        help=(
            "Single mode: only use templates with this logic variant "
            "(e.g. exist, forbid). Valid values per level are in index.json → constraint_families. "
            "Use with --class_filter when the family is level-specific."
        ),
    )
    p.add_argument(
        "--constraint_family_map_json",
        default=None,
        help=(
            'Single mode: per-level family override, e.g. \'{"L0":"exist","L1":"exist_pair"}\'. '
            "Overrides --constraint_family for listed levels."
        ),
    )
    p.add_argument(
        "--combo_size",
        type=int,
        default=2,
        help="Number of constraints to combine in combo mode (when not using pairs file).",
    )
    p.add_argument(
        "--combo_levels",
        nargs="*",
        default=None,
        help="Pool of levels for random combos, e.g. L0 L1 L2. Defaults to all levels.",
    )
    p.add_argument(
        "--combo_pairs_json",
        default="question_generation/CLEVR_CCIG_templates/combo_pairs.json",
        help="Optional fixed list of level pairs/groups for combo mode.",
    )
    p.add_argument(
        "--text_joiner",
        default=" And ",
        help="Joiner between constraint sentences in combo mode.",
    )
    p.add_argument("--min_objects", type=int, default=5, help="Minimum object count for ASP domain.")
    p.add_argument("--max_objects", type=int, default=9, help="Maximum object count for ASP domain.")
    p.add_argument(
        "--validate_with_clingo",
        action="store_true",
        help="Validate satisfiability with clingo before writing a record.",
    )
    p.add_argument("--clingo_bin", default="clingo", help="Path to clingo binary.")
    p.add_argument(
        "--clingo_time_limit",
        type=int,
        default=10,
        help="Seconds per clingo call (--time-limit); prevents hangs on hard programs.",
    )
    p.add_argument(
        "--max_attempts_per_instance",
        type=int,
        default=50,
        help="Max resampling attempts per instance when validating with clingo.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print level summaries and final stats (no per-instance lines).",
    )
    p.add_argument(
        "--log_clingo_attempts",
        action="store_true",
        help="Log every clingo UNSAT retry (verbose; use with small runs).",
    )
    p.add_argument(
        "--unsat_output_jsonl",
        default=None,
        help=(
            "Path for UNSAT attempts (default: <output_stem>_unsat.jsonl when "
            "--validate_with_clingo is set)."
        ),
    )
    p.add_argument(
        "--no_unsat_sidecar",
        action="store_true",
        help="Do not write a separate JSONL for UNSAT clingo attempts.",
    )
    return p.parse_args()


def unsat_sidecar_path(output_jsonl: Path) -> Path:
    return output_jsonl.parent / f"{output_jsonl.stem}_unsat{output_jsonl.suffix}"


class ProgressLog:
    """stderr progress lines (flushed) so long clingo runs do not look frozen."""

    def __init__(self, *, quiet: bool) -> None:
        self.quiet = quiet
        self.t0 = time.monotonic()
        self.records_written = 0
        self.unsat_records_written = 0
        self.clingo_calls = 0
        self.clingo_unsat = 0

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def say(self, msg: str, *, force: bool = False) -> None:
        if self.quiet and not force:
            return
        print(f"[{self.elapsed():7.1f}s] {msg}", file=sys.stderr, flush=True)

    def step(self, msg: str) -> None:
        self.say(msg, force=True)

    def clingo_attempt(self, *, log_each: bool, label: str, attempt: int, max_attempts: int, sat: bool) -> None:
        self.clingo_calls += 1
        if sat:
            if log_each:
                self.say(f"{label} | clingo {attempt}/{max_attempts} -> SAT")
            return
        self.clingo_unsat += 1
        if log_each:
            self.say(f"{label} | clingo {attempt}/{max_attempts} -> UNSAT (resampling)")

    def summary(
        self,
        out_path: Path,
        *,
        mode: str,
        planned: int,
        unsat_path: Path | None = None,
    ) -> None:
        self.step(
            f"Done ({mode}): wrote {self.records_written}/{planned} SAT records -> {out_path}"
        )
        if unsat_path is not None:
            self.step(
                f"UNSAT sidecar: {self.unsat_records_written} records -> {unsat_path}"
            )
        if self.clingo_calls:
            self.step(
                f"clingo: {self.clingo_calls} calls, {self.clingo_unsat} UNSAT retries, "
                f"{self.clingo_calls - self.clingo_unsat} SAT"
            )


def level_item_by_name(index_data: dict, level: str) -> dict:
    for item in index_data["complexity_levels"]:
        if item["level"] == level:
            return item
    raise KeyError(f"Unknown complexity level: {level}")


def load_level_bundle(
    *,
    item: dict,
    templates_dir: Path,
) -> tuple[str, list[dict]]:
    template_file = item["file"]
    templates = load_level_templates(templates_dir, template_file)
    return template_file, templates


def try_build_instance(
    *,
    components: list[ComponentInstance],
    background: str,
    rng: random.Random,
    min_objects: int,
    max_objects: int,
    text_joiner: str,
) -> tuple[int, list[str], str]:
    n_objects = rng.randint(min_objects, max_objects)
    asp_code = build_asp_program(
        background=background,
        n_objects=n_objects,
        constraint_rules=[c.asp_rule for c in components],
    )
    if len(components) == 1:
        texts = components[0].texts
    else:
        texts = merge_text_variants(components, joiner=text_joiner)
    return n_objects, texts, asp_code


def write_record(
    out,
    *,
    record_id: str,
    mode: str,
    components: list[ComponentInstance],
    n_objects: int,
    texts: list[str],
    asp_code: str,
    combo_spec_id: str | None = None,
    clingo_result: str | None = None,
    extra: dict | None = None,
) -> None:
    """Append one JSONL row and flush so partial output is visible during long runs."""
    classes = sorted({c.complexity_class for c in components})
    families = [c.constraint_family for c in components]
    record: dict = {
        "id": record_id,
        "mode": mode,
        "complexity_classes": classes,
        "constraint_families": families,
        "text": texts,
        "asp_rules": [c.asp_rule for c in components],
        "asp_code": asp_code,
        "n_objects": n_objects,
        "template_files": [c.template_file for c in components],
        "asp_template_files": [c.asp_template_file for c in components],
        "param_assignments": [c.param_assignment for c in components],
    }
    if mode == "single":
        c = components[0]
        if c.property_focus:
            record["property_focus"] = c.property_focus
        if c.relation_focus:
            record["relation_focus"] = c.relation_focus
    else:
        record["combo_size"] = len(components)
        record["combo_key"] = make_combo_key(components)
        if combo_spec_id:
            record["combo_spec_id"] = combo_spec_id
    if clingo_result:
        record["clingo_result"] = clingo_result
    if extra:
        record.update(extra)
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()


def generate_with_validation(
    *,
    build_fn,
    args: argparse.Namespace,
    rng: random.Random,
    progress: ProgressLog,
    label: str,
    mode: str,
    target_record_id: str,
    unsat_out,
    combo_spec_id: str | None,
) -> tuple[list[ComponentInstance], int, list[str], str, int]:
    last_error = ""
    log_attempts = args.log_clingo_attempts or (not args.quiet and args.validate_with_clingo)
    for attempt in range(1, args.max_attempts_per_instance + 1):
        components, n_objects, texts, asp_code = build_fn()
        if not args.validate_with_clingo:
            return components, n_objects, texts, asp_code, 1
        ok, clingo_out = validate_with_clingo(
            asp_code, args.clingo_bin, time_limit_sec=args.clingo_time_limit
        )
        progress.clingo_attempt(
            log_each=log_attempts,
            label=label,
            attempt=attempt,
            max_attempts=args.max_attempts_per_instance,
            sat=ok,
        )
        if ok:
            return components, n_objects, texts, asp_code, attempt
        last_error = clingo_out
        if unsat_out is not None:
            progress.unsat_records_written += 1
            unsat_id = f"unsat_{progress.unsat_records_written:06d}"
            if "ASP_PARSE_ERROR" in clingo_out and not args.quiet:
                progress.say(f"{label} | ASP parse error (bad rule syntax), resampling...")
            write_record(
                unsat_out,
                record_id=unsat_id,
                mode=mode,
                components=components,
                n_objects=n_objects,
                texts=texts,
                asp_code=asp_code,
                combo_spec_id=combo_spec_id,
                clingo_result="UNSAT",
                extra={
                    "for_target_id": target_record_id,
                    "resample_attempt": attempt,
                },
            )
            if not args.quiet:
                progress.say(f"{label} | recorded UNSAT -> {unsat_id} (attempt {attempt})")
    raise RuntimeError(
        f"Could not generate satisfiable instance after {args.max_attempts_per_instance} attempts.\n"
        f"Context: {label}\n"
        f"Last clingo output:\n{last_error}"
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    progress = ProgressLog(quiet=args.quiet)

    repo_root = Path(__file__).resolve().parents[1]
    index_path = repo_root / args.index_json
    templates_dir = repo_root / args.templates_dir
    constraint_dir = repo_root / args.constraint_templates_dir

    progress.step(f"CCIG dataset generator | mode={args.mode} | seed={args.seed}")
    progress.step(f"Output: {repo_root / args.output_jsonl}")

    if args.validate_with_clingo:
        clingo_path = shutil.which(args.clingo_bin)
        if not clingo_path:
            print(
                f"Error: --validate_with_clingo set but '{args.clingo_bin}' not found on PATH.",
                file=sys.stderr,
            )
            sys.exit(2)
        progress.step(
            f"clingo validation ON ({clingo_path}, "
            f"{args.clingo_time_limit}s/time-limit, max {args.max_attempts_per_instance} tries/instance)"
        )
        progress.step(
            "Note: each attempt runs clingo on the full CLEVR program (can take seconds). "
            "Progress lines print to stderr; JSONL rows flush after each write."
        )
    else:
        progress.step("clingo validation OFF (use validate_ccig_asp_dataset.py after)")

    progress.step("Loading ASP background (general_constraints + relationship axioms)...")
    background = load_background(
        repo_root / args.background_asp,
        repo_root / args.relationship_axioms,
    )
    progress.step(f"Background loaded ({len(background)} chars)")

    progress.step(f"Loading index: {index_path}")
    index_data = load_index(index_path)
    override_map: dict[str, int] = {}
    if args.instances_map_json:
        override_map = json.loads(args.instances_map_json)

    family_map: dict[str, str] = {}
    if args.constraint_family_map_json:
        family_map = json.loads(args.constraint_family_map_json)

    class_filter = set(args.class_filter) if args.class_filter else None
    out_path = repo_root / args.output_jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)

    unsat_path: Path | None = None
    if args.validate_with_clingo and not args.no_unsat_sidecar:
        unsat_path = (
            repo_root / args.unsat_output_jsonl
            if args.unsat_output_jsonl
            else unsat_sidecar_path(out_path)
        )
        unsat_path.parent.mkdir(parents=True, exist_ok=True)
        progress.step(f"UNSAT sidecar: {unsat_path}")

    level_items = {item["level"]: item for item in index_data["complexity_levels"]}
    progress.step(f"Loading ASP rule templates from {constraint_dir}...")
    asp_cache = AspRuleCache(constraint_dir)
    bundles: dict[str, tuple[str, list[dict]]] = {}
    progress.step(f"Loading NL templates from {templates_dir}...")
    for level, item in level_items.items():
        bundles[level] = load_level_bundle(item=item, templates_dir=templates_dir)
        if not args.quiet:
            progress.say(f"  {level}: {item['file']} ({len(bundles[level][1])} entries)")

    if args.mode == "single":
        levels_to_run = [
            lv
            for lv in level_items
            if not class_filter or lv in class_filter
        ]
        planned_total = sum(
            int(override_map.get(lv, args.instances_per_class)) for lv in levels_to_run
        )
    else:
        planned_total = 0  # set below for combo

    rec_id = 0
    with ExitStack() as stack:
        out = stack.enter_context(out_path.open("w", encoding="utf-8"))
        unsat_out = (
            stack.enter_context(unsat_path.open("w", encoding="utf-8"))
            if unsat_path is not None
            else None
        )
        if args.mode == "single":
            progress.step(
                f"Single mode: {len(levels_to_run)} levels, {planned_total} instances planned"
            )
            for level, item in level_items.items():
                if class_filter and level not in class_filter:
                    continue
                template_file, templates = bundles[level]
                n_instances = int(override_map.get(level, args.instances_per_class))
                family_note = family_map.get(level, args.constraint_family) or "any"
                progress.step(
                    f"--- {level}: {n_instances} instances | family={family_note} | file={template_file} ---"
                )
                for inst_idx in range(1, n_instances + 1):
                    rec_id += 1
                    target_id = f"scene_{rec_id:06d}"
                    label = f"{level} [{inst_idx}/{n_instances}] {target_id}"

                    def build_single(
                        level=level,
                        template_file=template_file,
                        templates=templates,
                    ):
                        family = family_map.get(level, args.constraint_family)
                        filt = (
                            ComboComponentFilter(level=level, constraint_family=family)
                            if family
                            else None
                        )
                        t_idx, entry = pick_template_entry(
                            templates, rng, component_filter=filt
                        )
                        comp = instantiate_component(
                            level=level,
                            template_file=template_file,
                            template_index=t_idx,
                            template_entry=entry,
                            asp_cache=asp_cache,
                            rng=rng,
                        )
                        n_objects, texts, asp_code = try_build_instance(
                            components=[comp],
                            background=background,
                            rng=rng,
                            min_objects=args.min_objects,
                            max_objects=args.max_objects,
                            text_joiner=args.text_joiner,
                        )
                        return [comp], n_objects, texts, asp_code

                    components, n_objects, texts, asp_code, attempts = generate_with_validation(
                        build_fn=build_single,
                        args=args,
                        rng=rng,
                        progress=progress,
                        label=label,
                        mode="single",
                        target_record_id=target_id,
                        unsat_out=unsat_out,
                        combo_spec_id=None,
                    )
                    write_record(
                        out,
                        record_id=target_id,
                        mode="single",
                        components=components,
                        n_objects=n_objects,
                        texts=texts,
                        asp_code=asp_code,
                        clingo_result="SAT" if args.validate_with_clingo else None,
                    )
                    progress.records_written += 1
                    c0 = components[0]
                    if not args.quiet:
                        progress.say(
                            f"{label} | wrote | family={c0.constraint_family} | "
                            f"n_objects={n_objects} | paraphrases={len(texts)} | "
                            f"clingo_attempts={attempts if args.validate_with_clingo else 'n/a'}"
                        )
                    elif inst_idx == n_instances or inst_idx % 10 == 0:
                        progress.say(f"{level}: {inst_idx}/{n_instances} instances written")
        else:
            combo_pairs_path = repo_root / args.combo_pairs_json
            combo_specs = load_combo_specs(combo_pairs_path)
            if combo_specs:
                combo_jobs = [
                    (spec, spec.instances if spec.instances is not None else args.instances_per_combo)
                    for spec in combo_specs
                ]
            else:
                pool = args.combo_levels or list(level_items.keys())
                if len(pool) < args.combo_size:
                    raise ValueError("combo_levels pool smaller than combo_size")
                combo_jobs = [
                    (
                        ComboSpec(
                            components=[
                                ComboComponentFilter(level=level)
                                for level in sorted(rng.sample(pool, args.combo_size))
                            ]
                        ),
                        1,
                    )
                    for _ in range(args.instances_per_combo)
                ]

            planned_total = sum(count for _, count in combo_jobs)
            progress.step(f"Combo mode: {len(combo_jobs)} job(s), {planned_total} instances planned")

            for job_idx, (spec, count) in enumerate(combo_jobs, start=1):
                spec_label = spec.id or f"combo_{job_idx}"
                levels_str = "+".join(c.level for c in spec.components)
                progress.step(
                    f"--- combo {job_idx}/{len(combo_jobs)}: {spec_label} | "
                    f"levels={levels_str} | count={count} ---"
                )
                for inst_idx in range(1, count + 1):
                    rec_id += 1
                    target_id = f"scene_{rec_id:06d}"
                    label = f"{spec_label} [{inst_idx}/{count}] {target_id}"

                    def build_combo(spec=spec):
                        components: list[ComponentInstance] = []
                        for filt in spec.components:
                            level = filt.level
                            template_file, templates = bundles[level]
                            t_idx, entry = pick_template_entry(
                                templates, rng, component_filter=filt
                            )
                            components.append(
                                instantiate_component(
                                    level=level,
                                    template_file=template_file,
                                    template_index=t_idx,
                                    template_entry=entry,
                                    asp_cache=asp_cache,
                                    rng=rng,
                                )
                            )
                        n_objects, texts, asp_code = try_build_instance(
                            components=components,
                            background=background,
                            rng=rng,
                            min_objects=args.min_objects,
                            max_objects=args.max_objects,
                            text_joiner=args.text_joiner,
                        )
                        return components, n_objects, texts, asp_code

                    components, n_objects, texts, asp_code, attempts = generate_with_validation(
                        build_fn=build_combo,
                        args=args,
                        rng=rng,
                        progress=progress,
                        label=label,
                        mode="combo",
                        target_record_id=target_id,
                        unsat_out=unsat_out,
                        combo_spec_id=spec.id,
                    )
                    write_record(
                        out,
                        record_id=target_id,
                        mode="combo",
                        components=components,
                        n_objects=n_objects,
                        texts=texts,
                        asp_code=asp_code,
                        combo_spec_id=spec.id,
                        clingo_result="SAT" if args.validate_with_clingo else None,
                    )
                    progress.records_written += 1
                    if not args.quiet:
                        progress.say(
                            f"{label} | wrote | families={[c.constraint_family for c in components]} | "
                            f"n_objects={n_objects} | paraphrases={len(texts)} | "
                            f"clingo_attempts={attempts if args.validate_with_clingo else 'n/a'}"
                        )
                    elif inst_idx == count or inst_idx % 10 == 0:
                        progress.say(f"{spec_label}: {inst_idx}/{count} instances written")

    progress.summary(
        out_path, mode=args.mode, planned=planned_total, unsat_path=unsat_path
    )
    print(f"Wrote SAT dataset to: {out_path}", flush=True)
    if unsat_path is not None:
        print(f"Wrote UNSAT dataset to: {unsat_path}", flush=True)


if __name__ == "__main__":
    main()
