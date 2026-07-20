"""
Per-class (C1-C9) grounders: find assignments that are TRUE of a given reconstructed scene,
then render them via common/verbalize.py's existing phrase functions -- the same module
eval_dataset_gen uses for its randomly-sampled constraint instances, imported directly here (not
duplicated) so finetune captions and eval prompts never drift apart in vocabulary/phrasing.

Each `_ground_cN(scene, rng)` yields (variant, assignment) pairs for every variant of class N that
is satisfiable in that scene (skipping variants that don't apply — e.g. C2 universal-property only
fires if some property is actually uniform). `ground_constraints()` is the public entry point used
by compile_dataset.py.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Dict, Iterator, List, Optional, Tuple

from ..common import domain_clevr as domain
from ..common import verbalize
from . import scene_queries as q
from .scene_queries import Scene

PROPS = list(domain.PROPERTIES.keys())
_COUNT_CAP = 5  # verbalize._count_word only spells 1-5; larger counts render as bare digits, still valid.


def _n_str(n: int) -> str:
    return str(n)


# ── C1: existential property (conjunctions + negated variants) ─────────────

def _ground_c1(scene: Scene, rng: random.Random) -> Iterator[Tuple[str, dict]]:
    present = q.present_prop_values(scene)
    if not present:
        return

    prop1, val1 = rng.choice(present)
    yield "1prop", {"P1'": prop1, "V1'": val1}

    # 1prop_neg: some object's value for prop1 differs from val1 (i.e. prop1 isn't uniformly val1)
    if q.count_without(scene, prop1, val1) > 0:
        yield "1prop_neg", {"P1'": prop1, "V1'": val1}

    # 1prop_2val_neg / 1prop_3val_neg: an object whose prop value is none of k excluded values
    for k, variant in ((2, "1prop_2val_neg"), (3, "1prop_3val_neg")):
        for prop in rng.sample(PROPS, len(PROPS)):
            dom = domain.PROPERTIES[prop]
            if len(dom) < k + 1:
                continue
            oid = rng.choice(q.object_ids(scene))
            actual = q.object_value(scene, oid, prop)
            excluded = [v for v in dom if v != actual][:k]
            witness = q.object_matching_none_of(scene, prop, excluded)
            if witness is not None:
                a = {"P1'": prop}
                for i, v in enumerate(excluded, start=1):
                    a[f"V{i}'"] = v
                yield variant, a
                break

    # multi-prop conjunctions: pick one object, use k of its own distinct properties
    oid = rng.choice(q.object_ids(scene))
    attrs = scene["objects"][oid]  # type: ignore[index]
    props_shuffled = rng.sample(PROPS, len(PROPS))

    if len(props_shuffled) >= 2:
        p1, p2 = props_shuffled[0], props_shuffled[1]
        yield "2prop", {"P1'": p1, "V1'": attrs[p1], "P2'": p2, "V2'": attrs[p2]}

        v2_other = q.other_value(p2, attrs[p2])
        v1_other = q.other_value(p1, attrs[p1])
        if v1_other and v2_other:
            yield "2prop_neg", {"P1'": p1, "V1'": v1_other, "P2'": p2, "V2'": v2_other}
        if v2_other:
            yield "2prop_mix_neg", {"P1'": p1, "V1'": attrs[p1], "P2'": p2, "V2'": v2_other}

    if len(props_shuffled) >= 3:
        p1, p2, p3 = props_shuffled[:3]
        yield "3prop", {
            "P1'": p1, "V1'": attrs[p1], "P2'": p2, "V2'": attrs[p2], "P3'": p3, "V3'": attrs[p3],
        }
        dom3 = domain.PROPERTIES[p3]
        if len(dom3) >= 3:
            excluded = [v for v in dom3 if v != attrs[p3]][:2]
            if len(excluded) == 2:
                yield "3prop_val_mix_neg", {
                    "P1'": p1, "V1'": attrs[p1], "P2'": p2, "V2'": attrs[p2],
                    "P3'": p3, "V3'": excluded[0], "V4'": excluded[1],
                }

    if len(props_shuffled) >= 4:
        p1, p2, p3, p4 = props_shuffled[:4]
        yield "4prop", {
            "P1'": p1, "V1'": attrs[p1], "P2'": p2, "V2'": attrs[p2],
            "P3'": p3, "V3'": attrs[p3], "P4'": p4, "V4'": attrs[p4],
        }
        v3_other = q.other_value(p3, attrs[p3])
        v4_other = q.other_value(p4, attrs[p4])
        if v3_other and v4_other:
            yield "4prop_mix_neg", {
                "P1'": p1, "V1'": attrs[p1], "P2'": p2, "V2'": attrs[p2],
                "P3'": p3, "V3'": v3_other, "P4'": p4, "V4'": v4_other,
            }


# ── C2: universal property ──────────────────────────────────────────────────

def _ground_c2(scene: Scene, rng: random.Random) -> Iterator[Tuple[str, dict]]:
    uniform_pairs = [(p, v) for p, v in q.present_prop_values(scene) if q.all_share(scene, p, v)]

    if uniform_pairs:
        p1, v1 = rng.choice(uniform_pairs)
        yield "1prop", {"P1'": p1, "V1'": v1}

    for prop in rng.sample(PROPS, len(PROPS)):
        absent = [v for v in domain.PROPERTIES[prop] if q.count_with(scene, prop, v) == 0]
        if absent:
            yield "1prop_neg", {"P1'": prop, "V1'": rng.choice(absent)}
            if len(absent) >= 2:
                v1, v2 = rng.sample(absent, 2)
                yield "1prop_2val_neg", {"P1'": prop, "V1'": v1, "V2'": v2}
            break

    if len(uniform_pairs) >= 2:
        (p1, v1), (p2, v2) = rng.sample(uniform_pairs, 2)
        yield "2prop", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2}
    if len(uniform_pairs) >= 3:
        (p1, v1), (p2, v2), (p3, v3) = rng.sample(uniform_pairs, 3)
        yield "3prop", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "P3'": p3, "V3'": v3}

    # 2prop_neg: no object simultaneously has (p1=v1) and (p2=v2)
    present = q.present_prop_values(scene)
    for (p1, v1), (p2, v2) in combinations(rng.sample(present, min(len(present), 6)), 2):
        if p1 == p2:
            continue
        if not q.subset_matching(scene, [(p1, v1), (p2, v2)]):
            yield "2prop_neg", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2}
            break

    # 2prop_mix_neg: no object has (p1=v1) and NOT(p2=v2) simultaneously, i.e. every p1=v1 object is p2=v2
    for p1, v1 in rng.sample(present, min(len(present), 6)):
        antecedent = q.objects_with(scene, p1, v1)
        if not antecedent:
            continue
        for p2 in PROPS:
            if p2 == p1:
                continue
            v2 = q.uniform_value_in_subset(scene, antecedent, p2)
            if v2 is not None:
                yield "2prop_mix_neg", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2}
                break
        else:
            continue
        break


# ── C3: conditional (antecedent nonempty -> universal consequent) ──────────

def _ground_c3(scene: Scene, rng: random.Random) -> Iterator[Tuple[str, dict]]:
    present = q.present_prop_values(scene)
    if not present:
        return

    for p1, v1 in rng.sample(present, len(present)):
        antecedent = q.objects_with(scene, p1, v1)
        neg_antecedent = q.objects_without(scene, p1, v1)
        other_props = [p for p in PROPS if p != p1]

        for p2 in rng.sample(other_props, len(other_props)):
            v2 = q.uniform_value_in_subset(scene, antecedent, p2)
            if v2 is not None:
                yield "1propA_1propC", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2}
            v2_neg = q.absent_value_in_subset(scene, antecedent, p2)
            if v2_neg is not None and antecedent:
                yield "1propA_1prop_neg", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2_neg}
            v2c = q.uniform_value_in_subset(scene, neg_antecedent, p2)
            if v2c is not None and neg_antecedent:
                yield "1propA_neg_1propC", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2c}

            remaining = [p for p in other_props if p != p2]
            if remaining and antecedent:
                p3 = remaining[0]
                v3 = q.uniform_value_in_subset(scene, antecedent, p3)
                if v2 is not None and v3 is not None:
                    yield "1propA_2propC", {
                        "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "P3'": p3, "V3'": v3,
                    }
                    if len(remaining) >= 2:
                        p4 = remaining[1]
                        v4 = q.uniform_value_in_subset(scene, antecedent, p4)
                        if v4 is not None:
                            yield "1propA_3propC", {
                                "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2,
                                "P3'": p3, "V3'": v3, "P4'": p4, "V4'": v4,
                            }
            break  # one p2 is enough to try the multi-prop extensions per (p1,v1)
        if any(True for _ in []):  # pragma: no cover - unreachable, keeps structure explicit
            pass

    # two-property antecedents
    for (p1, v1), (p2, v2) in combinations(rng.sample(present, min(len(present), 6)), 2):
        if p1 == p2:
            continue
        antecedent = q.subset_matching(scene, [(p1, v1), (p2, v2)])
        if not antecedent:
            continue
        other_props = [p for p in PROPS if p not in (p1, p2)]
        for p3 in rng.sample(other_props, len(other_props)):
            v3 = q.uniform_value_in_subset(scene, antecedent, p3)
            if v3 is not None:
                yield "2propA_1propC", {
                    "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "P3'": p3, "V3'": v3,
                }
                remaining = [p for p in other_props if p != p3]
                if remaining:
                    p4 = remaining[0]
                    v4 = q.uniform_value_in_subset(scene, antecedent, p4)
                    if v4 is not None:
                        yield "2propA_2propC", {
                            "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2,
                            "P3'": p3, "V3'": v3, "P4'": p4, "V4'": v4,
                        }
                break

        v2_other = q.other_value(p2, v2)
        if v2_other:
            neg_antecedent = q.subset_matching(scene, [(p1, v1)])
            neg_antecedent = [o for o in neg_antecedent if q.object_value(scene, o, p2) != v2]
            if neg_antecedent:
                for p3 in other_props:
                    v3 = q.uniform_value_in_subset(scene, neg_antecedent, p3)
                    if v3 is None:
                        continue
                    remaining = [p for p in other_props if p != p3]
                    if not remaining:
                        continue
                    p4 = remaining[0]
                    v4 = q.uniform_value_in_subset(scene, neg_antecedent, p4)
                    if v4 is not None:
                        yield "2propA_neg_2propC", {
                            "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2_other,
                            "P3'": p3, "V3'": v3, "P4'": p4, "V4'": v4,
                        }
                    break
        break  # one pair is enough

    # three-property antecedent
    if len(present) >= 3:
        for p1, v1, p2, v2, p3, v3 in [
            (a[0], a[1], b[0], b[1], c[0], c[1])
            for a, b, c in combinations(rng.sample(present, min(len(present), 5)), 3)
        ]:
            if len({p1, p2, p3}) < 3:
                continue
            antecedent = q.subset_matching(scene, [(p1, v1), (p2, v2), (p3, v3)])
            if not antecedent:
                continue
            other_props = [p for p in PROPS if p not in (p1, p2, p3)]
            for p4 in other_props:
                v4 = q.uniform_value_in_subset(scene, antecedent, p4)
                if v4 is not None:
                    yield "3propA_1propC", {
                        "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2,
                        "P3'": p3, "V3'": v3, "P4'": p4, "V4'": v4,
                    }
                v4_neg = q.absent_value_in_subset(scene, antecedent, p4)
                if v4_neg is not None:
                    yield "3propA_1propC_neg", {
                        "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2,
                        "P3'": p3, "V3'": v3, "P4'": p4, "V4'": v4_neg,
                    }
                break
            break


# ── C4: 2-hop / shared-subject relational chains ────────────────────────────

def _ground_c4(scene: Scene, rng: random.Random) -> Iterator[Tuple[str, dict]]:
    present = q.present_prop_values(scene)
    directions = domain.DIRECTIONS

    for d1 in rng.sample(directions, len(directions)):
        for d2 in rng.sample(directions, len(directions)):
            for (p1, v1), (p2, v2), (p3, v3) in _sample_triples(present, rng):
                chains = q.find_2hop_chains(scene, p1, v1, d1, p2, v2, d2, p3, v3)
                if chains:
                    yield "1prop_2hop", {
                        "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "P3'": p3, "V3'": v3,
                        "D1'": d1, "D2'": d2,
                    }
                    break

                v3_other = q.other_value(p3, v3)
                if v3_other:
                    for o1, o2 in q.find_relation_pairs(scene, p1, v1, d1, p2, v2):
                        for o3 in q.neighbors(scene, o2, d2):
                            if o3 not in (o1, o2) and q.object_value(scene, o3, p3) != v3_other:
                                yield "1prop_2hop_mix", {
                                    "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2,
                                    "P3'": p3, "V3'": v3_other, "D1'": d1, "D2'": d2,
                                }
                                break
                        else:
                            continue
                        break

                for o3 in q.object_ids(scene):
                    n1_candidates = [o for o in q.neighbors(scene, o3, d1) if q.object_value(scene, o, p1) == v1]
                    if not n1_candidates:
                        continue
                    o1 = n1_candidates[0]
                    n2_candidates = [
                        o for o in q.neighbors(scene, o3, d2)
                        if q.object_value(scene, o, p2) == v2 and o != o1
                    ]
                    if n2_candidates:
                        yield "1prop_shared", {
                            "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2,
                            "P3'": p3, "V3'": q.object_value(scene, o3, p3), "D1'": d1, "D2'": d2,
                        }
                        break
                break
            break
        break


def _sample_triples(present: List[Tuple[str, str]], rng: random.Random, k: int = 4):
    sample = rng.sample(present, min(len(present), k))
    for a, b, c in combinations(sample, 3):
        yield a, b, c
        yield b, a, c
        yield c, a, b


# ── C5: pairwise / triple relational, existential + universal consequent ───

def _ground_c5(scene: Scene, rng: random.Random) -> Iterator[Tuple[str, dict]]:
    present = q.present_prop_values(scene)
    directions = domain.DIRECTIONS

    for (p1, v1), (p2, v2) in combinations(rng.sample(present, min(len(present), 6)), 2):
        for d1 in directions:
            pairs = q.find_relation_pairs(scene, p1, v1, d1, p2, v2)
            if not pairs:
                continue

            all_pairs = [
                (o1, o2)
                for o1 in q.objects_with(scene, p1, v1)
                for o2 in q.objects_with(scene, p2, v2)
                if o1 != o2
            ]
            if all_pairs and len(pairs) == len(all_pairs):
                yield "pair_propA_relC", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1}

            o2_side = [o2 for _, o2 in pairs]
            for p3 in PROPS:
                v3 = q.uniform_value_in_subset(scene, o2_side, p3)
                if v3 is not None:
                    yield "pair_propRelA_propC", {
                        "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1, "P3'": p3, "V3'": v3,
                    }
                    break

            for d2 in directions:
                if d2 == d1:
                    continue
                if all(o1 in q.neighbors(scene, o1, d2) or o2 in q.neighbors(scene, o1, d2) for o1, o2 in pairs):
                    pass  # placeholder to keep structure; real check below
                if all(o2 in q.neighbors(scene, o1, d2) for o1, o2 in pairs):
                    yield "pair_propRelA_RelC", {
                        "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1, "D2'": d2,
                    }
                    break
            break
        break

    for d1 in directions:
        sources = [o1 for o1, _ in q.related_pairs(scene, d1)]
        if not sources:
            continue
        for p1 in PROPS:
            v1 = q.uniform_value_in_subset(scene, sources, p1)
            if v1 is not None:
                yield "pair_relA_propC", {"D1'": d1, "P1'": p1, "V1'": v1}
                break
        break

    for (p1, v1), (p2, v2), (p3, v3) in _sample_triples(present, rng):
        for d1 in directions:
            for d2 in directions:
                triples = [
                    (o1, o2, o3)
                    for o1 in q.objects_with(scene, p1, v1)
                    for o2 in q.objects_with(scene, p2, v2)
                    for o3 in q.objects_with(scene, p3, v3)
                    if len({o1, o2, o3}) == 3
                ]
                if not triples:
                    continue
                if all(o2 in q.neighbors(scene, o1, d1) and o3 in q.neighbors(scene, o2, d2) for o1, o2, o3 in triples):
                    yield "triple_propA_RelC", {
                        "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "P3'": p3, "V3'": v3,
                        "D1'": d1, "D2'": d2,
                    }
                    break
                first_leg = [(o1, o2, o3) for o1, o2, o3 in triples if o2 in q.neighbors(scene, o1, d1)]
                if first_leg and all(o3 in q.neighbors(scene, o2, d2) for _, o2, o3 in first_leg):
                    yield "triple_propRelA_relC", {
                        "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "P3'": p3, "V3'": v3,
                        "D1'": d1, "D2'": d2,
                    }
                    break
            break
        break


# ── C6: existential subject with universal relation to a target class ──────

def _ground_c6(scene: Scene, rng: random.Random) -> Iterator[Tuple[str, dict]]:
    present = q.present_prop_values(scene)
    directions = domain.DIRECTIONS

    for p1, v1 in rng.sample(present, len(present)):
        for o1 in q.objects_with(scene, p1, v1):
            for d1 in directions:
                for p2, v2 in present:
                    if q.universal_relation_holds(scene, o1, d1, p2, v2):
                        yield "1prop", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1}
                        break
                else:
                    continue
                break
            else:
                continue
            break
        break

    for p1, v1 in present:
        for o1 in q.objects_with(scene, p1, v1):
            for d1 in directions:
                stands_to = set(q.neighbors(scene, o1, d1))
                for p2 in PROPS:
                    for v2 in domain.PROPERTIES[p2]:
                        targets = [oid for oid in q.objects_without(scene, p2, v2) if oid != o1]
                        if targets and all(t in stands_to for t in targets):
                            yield "1prop_neg", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1}
                            return
        break


# ── C7: witness relations (existential/counting) ────────────────────────────

def _ground_c7(scene: Scene, rng: random.Random) -> Iterator[Tuple[str, dict]]:
    present = q.present_prop_values(scene)
    directions = domain.DIRECTIONS

    for p1, v1 in rng.sample(present, len(present)):
        antecedent = q.objects_with(scene, p1, v1)
        if not antecedent:
            continue
        for d1 in directions:
            for p2, v2 in present:
                if all(
                    any(q.object_value(scene, n, p2) == v2 for n in q.neighbors(scene, o1, d1))
                    for o1 in antecedent
                ):
                    yield "1prop_propRel", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1}
                    break

                if all(
                    any(q.object_value(scene, n, p2) != v2 for n in q.neighbors(scene, o1, d1))
                    or not q.neighbors(scene, o1, d1)
                    for o1 in antecedent
                ) and any(q.neighbors(scene, o1, d1) for o1 in antecedent):
                    yield "1prop_propRel_neg", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1}
                    break
            break
        break

    for p1, v1 in present:
        antecedent = q.objects_with(scene, p1, v1)
        if not antecedent:
            continue
        for d1 in directions:
            counts = {o1: len(q.neighbors(scene, o1, d1)) for o1 in antecedent}
            for p2, v2 in present:
                exact_counts = {
                    o1: sum(1 for n in q.neighbors(scene, o1, d1) if q.object_value(scene, n, p2) == v2)
                    for o1 in antecedent
                }
                distinct = set(exact_counts.values())
                if len(distinct) == 1 and next(iter(distinct)) >= 1:
                    n = next(iter(distinct))
                    yield "1prop_exact", {
                        "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1, "N'": _n_str(n),
                    }
                    break
            break
        break


# ── C8: counting (no relation + relational witness counts) ─────────────────

def _ground_c8(scene: Scene, rng: random.Random) -> Iterator[Tuple[str, dict]]:
    present = q.present_prop_values(scene)

    for p1, v1 in rng.sample(present, len(present)):
        n = q.count_with(scene, p1, v1)
        if n > 0:
            yield "1prop_exact", {"P1'": p1, "V1'": v1, "N'": _n_str(n)}
            yield "1prop_atleast", {"P1'": p1, "V1'": v1, "N'": _n_str(n)}
            yield "1prop_atmost", {"P1'": p1, "V1'": v1, "N'": _n_str(n)}
        n_without = q.count_without(scene, p1, v1)
        if n_without > 0:
            yield "1prop_exact_neg", {"P1'": p1, "V1'": v1, "N'": _n_str(n_without)}
            yield "1prop_atleast_neg", {"P1'": p1, "V1'": v1, "N'": _n_str(n_without)}
            yield "1prop_atmost_neg", {"P1'": p1, "V1'": v1, "N'": _n_str(n_without)}
        break

    for (p1, v1), (p2, v2) in combinations(rng.sample(present, min(len(present), 6)), 2):
        if p1 == p2:
            continue
        n_both = len(q.subset_matching(scene, [(p1, v1), (p2, v2)]))
        if n_both > 0:
            yield "2prop_exact", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "N'": _n_str(n_both)}
            yield "2prop_atleast", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "N'": _n_str(n_both)}
            yield "2prop_atmost", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "N'": _n_str(n_both)}
        n_mix = len([o for o in q.objects_with(scene, p1, v1) if q.object_value(scene, o, p2) != v2])
        if n_mix > 0:
            yield "2prop_exact_neg", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "N'": _n_str(n_mix)}
            yield "2prop_atleast_neg", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "N'": _n_str(n_mix)}
            yield "2prop_atmost_neg", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "N'": _n_str(n_mix)}
        break

    for (p1, v1), (p2, v2) in combinations(rng.sample(present, min(len(present), 6)), 2):
        for d1 in domain.DIRECTIONS:
            group = q.objects_with(scene, p1, v1)
            n_rel = sum(
                1 for o1 in group
                if any(q.object_value(scene, n, p2) == v2 for n in q.neighbors(scene, o1, d1))
            )
            if n_rel > 0:
                yield "1prop_exact_relational", {
                    "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1, "N'": _n_str(n_rel),
                }
                yield "1prop_atleast_relational", {
                    "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1, "N'": _n_str(n_rel),
                }
                yield "1prop_atmost_relational", {
                    "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1, "N'": _n_str(n_rel),
                }
            n_not_rel = len(group) - n_rel
            if n_not_rel > 0:
                yield "1prop_exact_relational_neg", {
                    "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "D1'": d1, "N'": _n_str(n_not_rel),
                }
            break
        break


# ── C9: cardinality balance between two groups ──────────────────────────────

def _ground_c9(scene: Scene, rng: random.Random) -> Iterator[Tuple[str, dict]]:
    present = q.present_prop_values(scene)

    for (p1, v1), (p2, v2) in combinations(rng.sample(present, min(len(present), 6)), 2):
        if p1 == p2 or q.count_with(scene, p1, v1) == 0:
            continue
        if q.count_with(scene, p1, v1) == q.count_with(scene, p2, v2):
            yield "1prop", {"P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2}
            break

    for combo in combinations(rng.sample(present, min(len(present), 6)), 4):
        (p1, v1), (p2, v2), (p3, v3), (p4, v4) = combo
        if len({p1, p2, p3, p4}) < 4:
            continue
        group_a = q.subset_matching(scene, [(p1, v1), (p2, v2)])
        group_b = q.subset_matching(scene, [(p3, v3), (p4, v4)])
        if group_a and len(group_a) == len(group_b):
            yield "2prop", {
                "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "P3'": p3, "V3'": v3, "P4'": p4, "V4'": v4,
            }
            break

    for combo in combinations(rng.sample(present, min(len(present), 6)), 4):
        (p1, v1), (p2, v2), (p3, v3), (p4, v4) = combo
        if len({p1, p2, p3, p4}) < 4:
            continue
        group_a = [o for o in q.objects_without(scene, p1, v1) if q.object_value(scene, o, p2) == v2]
        group_b = [o for o in q.objects_without(scene, p3, v3) if q.object_value(scene, o, p4) == v4]
        if group_a and len(group_a) == len(group_b):
            yield "2prop_mix", {
                "P1'": p1, "V1'": v1, "P2'": p2, "V2'": v2, "P3'": p3, "V3'": v3, "P4'": p4, "V4'": v4,
            }
            break


_GROUNDERS = {
    "C1": _ground_c1, "C2": _ground_c2, "C3": _ground_c3, "C4": _ground_c4,
    "C5": _ground_c5, "C6": _ground_c6, "C7": _ground_c7, "C8": _ground_c8, "C9": _ground_c9,
}


def ground_constraints(
    scene: Scene,
    classes: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, str]]:
    """
    Find all satisfiable groundings for the requested classes (default: all C1-C9) and render
    them via verbalize.py. Returns a list of {"class", "variant", "short", "medium", "long"}.
    """
    rng = rng or random.Random()
    classes = classes or list(_GROUNDERS.keys())
    if len(q.object_ids(scene)) < 2:
        classes = [c for c in classes if c not in ("C4", "C5", "C6", "C7")]

    results: List[Dict[str, str]] = []
    for cls in classes:
        grounder = _GROUNDERS[cls]
        seen_variants = set()
        try:
            for variant, assignment in grounder(scene, rng):
                if variant in seen_variants:
                    continue
                seen_variants.add(variant)
                try:
                    text = verbalize.verbalize(f"{cls}_{variant}", assignment)
                except (IndexError, ValueError, UnboundLocalError, TypeError, KeyError):
                    # Defensive: skip any grounder/verbalize.py assignment mismatch instead of
                    # crashing the whole compile run. (The one bug we hit this way -- missing
                    # "N'" unpack on *_relational C8 variants -- is already fixed in verbalize.py;
                    # this stays as a safety net against future template/grounder drift.)
                    continue
                if text is None:
                    continue
                results.append({"class": cls, "variant": variant, **text})
        except (IndexError, ValueError):
            continue
    return results
