"""
Natural language verbalization of instantiated CCIG constraints.

Given a template filename stem (e.g. 'C1_2prop_neg') and a completed
placeholder assignment, returns three NL descriptions at different granularities:
  short  — one concise sentence
  medium — one clear sentence with full context
  long   — two-to-three sentences with semantic explanation

Region-specific grammar is handled automatically:
  - Non-region properties use adjective phrasing: "a red object", "every cube object"
  - Region values use locative phrasing: "an object in region_1", "every object in region_2"
"""

from __future__ import annotations

_DIR_PHRASE = {
    "left":   "to the left of",
    "right":  "to the right of",
    "front":  "in front of",
    "behind": "behind",
}

# ── Linguistic helpers ──────────────────────────────────────────────────────

def _adj(p: str, v: str) -> str:
    """Adjectival phrase for a property value: 'red', 'in region_1', 'cube', 'small'."""
    return f"in {v}" if p == "region" else v


def _obj(p: str, v: str) -> str:
    """Indefinite singular object phrase: 'a red object', 'an object in region_1'."""
    if p == "region":
        return f"an object in {v}"
    art = "an" if v[0] in "aeiou" else "a"
    return f"{art} {v} object"


def _Obj(p: str, v: str) -> str:
    """Capitalized version of _obj for sentence start."""
    s = _obj(p, v)
    return s[0].upper() + s[1:]


def _objs(p: str, v: str) -> str:
    """Plural object phrase: 'red objects', 'objects in region_1'."""
    return f"objects in {v}" if p == "region" else f"{v} objects"


def _not_adj(p: str, v: str) -> str:
    """Negated property phrase: 'not red', 'not in region_1', 'non-cube'."""
    return f"not in {v}" if p == "region" else f"non-{v}"


def _not_obj(p: str, v: str) -> str:
    """'an object that is not red', 'an object not in region_1'."""
    if p == "region":
        return f"an object not in {v}"
    return f"an object that is not {v}"


def _bare(p: str, v: str) -> str:
    """
    Noun phrase without article: 'gray object', 'object in region_1'.
    Use after quantifiers or 'at least one': 'at least one gray object'.
    """
    return f"object in {v}" if p == "region" else f"{v} object"


def _quant(q: str, p: str, v: str) -> str:
    """
    Quantified noun phrase: 'every gray object', 'every object in region_1'.
    q is the quantifier word (every, each, no, some, any, …).
    """
    if p == "region":
        return f"{q} object in {v}"
    return f"{q} {v} object"


def _dir(d: str) -> str:
    """'to the left of', 'in front of', 'behind', 'to the right of'."""
    return _DIR_PHRASE.get(d, d)


def _multi_adj(pairs: list[tuple[str, str]]) -> str:
    """Comma-joined adjectives for multiple (prop, val) pairs."""
    parts = [_adj(p, v) for p, v in pairs]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _multi_obj(pairs: list[tuple[str, str]]) -> str:
    """
    Object with multiple properties: 'a red cube object', 'a red object in region_1',
    'a cube object in region_1'.
    """
    non_region = [(p, v) for p, v in pairs if p != "region"]
    region_pairs = [(p, v) for p, v in pairs if p == "region"]

    adjs = " ".join(v for _, v in non_region)
    loc = " and ".join(f"in {v}" for _, v in region_pairs)

    if adjs and loc:
        art = "an" if adjs[0] in "aeiou" else "a"
        return f"{art} {adjs} object {loc}"
    if adjs:
        art = "an" if adjs[0] in "aeiou" else "a"
        return f"{art} {adjs} object"
    # region only
    art = "an"
    return f"{art} object {loc}"


def _not_objs(p: str, v: str) -> str:
    """Plural of objects lacking property: 'non-red objects', 'objects not in region_1'."""
    return f"objects not in {v}" if p == "region" else f"non-{v} objects"


def _n_obj(n: str) -> str:
    """'object' for n='1', 'objects' otherwise."""
    return "object" if n == "1" else "objects"


def _count_word(n: str) -> str:
    return {"1": "one", "2": "two", "3": "three"}.get(n, n)


def _at_least(n: str) -> str:
    return f"at least {_count_word(n)}" if n != "1" else "at least one"


def _at_most(n: str) -> str:
    return f"at most {_count_word(n)}"


def _exactly(n: str) -> str:
    w = _count_word(n)
    return f"exactly {w}"


# ── Per-class verbalizers ───────────────────────────────────────────────────

def _g(a: dict, *keys: str) -> tuple:
    """Quick getter: returns tuple of assignment values for given placeholder keys."""
    return tuple(a.get(k, "?") for k in keys)


def _c1(variant: str, a: dict) -> dict[str, str]:
    
    if variant == "1prop":
        p1, v1 = _g(a, "P1'", "V1'")

        return dict(
            short=f"{_Obj(p1, v1)} is in the scene.",
            medium=f"The scene contains at least one {_bare(p1, v1)}.",
            long=(f"Among all objects placed in the scene, at least one must be {_adj(p1, v1)}; "
                  f"no restriction is placed on the remaining objects."),
        )

    if variant == "1prop_neg":
        p1, v1 = _g(a, "P1'", "V1'")

        return dict(
            short=f"Some object is {_not_adj(p1, v1)}.",
            medium=f"The scene contains at least one object that is {_not_adj(p1, v1)}.",
            long=(f"Not all objects need be {_adj(p1, v1)}; the constraint ensures that at least one "
                  f"object in the scene avoids this property value."),
        )

    if variant == "1prop_2val_neg":
        
        p1, v1, v2 = _g(a, "P1'", "V1'", "V2'")
        return dict(
            short=f"Some object is neither {_adj(p1, v1)} nor {_adj(p1, v2)}.",
            medium=f"The scene contains at least one object that is neither {_adj(p1, v1)} nor {_adj(p1, v2)}.",
            long=(f"Among all objects in the scene, at least one must avoid both {_adj(p1, v1)} "
                  f"and {_adj(p1, v2)} for property {p1}."),
        )

    if variant == "1prop_3val_neg":
        p1, v1, v2, v3 = _g(a, "P1'", "V1'", "V2'", "V3'")
        return dict(
            short=f"Some object is neither {_adj(p1, v1)}, {_adj(p1, v2)}, nor {_adj(p1, v3)}.",
            medium=f"The scene contains at least one object that is none of {_adj(p1, v1)}, {_adj(p1, v2)}, or {_adj(p1, v3)}.",
            long = (f"Among all objects in the scene, at least one object must have a value for {p1} "
                    f"that is neither {_adj(p1, v1)}, {_adj(p1, v2)}, nor {_adj(p1, v3)}."),
        )

    if variant == "2prop":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        o = _multi_obj([(p1, v1), (p2, v2)])
        return dict(
            short=f"{o.capitalize()} is in the scene.",
            medium=f"The scene contains at least one object that is {_adj(p1, v1)} and {_adj(p2, v2)}.",
            long=(f"At least one object combining {_adj(p1, v1)} with {_adj(p2, v2)} must appear "
                  f"somewhere in the scene; other objects are unconstrained."),
        )

    if variant == "2prop_neg":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"Some object is {_not_adj(p1, v1)} and {_not_adj(p2, v2)}.",
            medium=f"The scene contains at least one object that is neither {_adj(p1, v1)} nor {_adj(p2, v2)}.",
            long=(f"At least one object must simultaneously avoid being {_adj(p1, v1)} "
                  f"and avoid being {_adj(p2, v2)}."),
        )
    
    if variant == "2prop_mix_neg":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"There exists an object that is {_adj(p1, v1)} and  {_not_adj(p2, v2)}.",
            medium=f"The scene contains at least one object that is {_adj(p1, v1)} but not {_adj(p2, v2)}.",
            long=(f"At least one object must simultaneously be {_adj(p1, v1)} "
                  f"and avoid being {_adj(p2, v2)}."),
        )

    if variant == "3prop":
        p1, v1, p2, v2, p3, v3 = _g(a, "P1'", "V1'",  "P2'", "V2'", "P3'", "V3'")
        o = _multi_obj([(p1, v1), (p2, v2), (p3, v3)])
        return dict(
            short=f"{o.capitalize()} is in the scene.",
            medium=f"The scene contains at least one object that is {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)}.",
            long=(f"Among all objects in the scene, at least one must satisfy all three property "
                  f"conditions: {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)}."),
        )

    if variant == "3prop_val_mix_neg":
        p1, v1, p2, v2, p3, v3, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "V4'")
        return dict(
            short=(f"Some object that is {_adj(p1, v1)} and {_adj(p2, v2)} "
                   f"is neither {_adj(p3, v3)} nor {_adj(p3, v4)}."),
            medium=(f"The scene contains at least one object that is {_adj(p1, v1)} and {_adj(p2, v2)}, "
                    f"while being neither {_adj(p3, v3)} nor {_adj(p3, v4)}."),
            long=(f"At least one object must satisfy the following conditions: {_adj(p1, v1)} and {_adj(p2, v2)}, "
                  f"but must not be {_adj(p3, v3)} and must not be {_adj(p3, v4)}."),
        )

    if variant == "4prop":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=f"Some object is {_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}.",
            long=(f"The scene contains at least one object satisfying all four properties: "
                    f"{_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}."),
            medium=(f"At least one object must simultaneously be {_adj(p1, v1)}, {_adj(p2, v2)}, "
                  f"{_adj(p3, v3)}, and {_adj(p4, v4)}."),
        )

    if variant == "4prop_mix_neg":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"Some object that is {_adj(p1, v1)} and {_adj(p2, v2)} "
                   f"is {_not_adj(p3, v3)} and {_not_adj(p4, v4)}."),
            medium=(f"The scene contains at least one object that is {_adj(p1, v1)} and {_adj(p2, v2)}, "
                    f"while being {_not_adj(p3, v3)} and {_not_adj(p4, v4)}."),
            long=(f"At least one object must satisfy the following requirements: {_adj(p1, v1)} and {_adj(p2, v2)}, "
                  f"while not being {_adj(p3, v3)} and not being {_adj(p4, v4)}."),
        )

    return None

def _c2(variant: str, a: dict) -> dict[str, str]:
    
    if variant == "1prop":
        p1, v1 = _g(a, "P1'", "V1'")

        return dict(
            short=f"All objects are {_adj(p1, v1)}.",
            medium=f"Every single object in the scene must be {_adj(p1, v1)}.",
            long=(f"The scene enforces a uniform {p1} constraint: each and every object must be "
                  f"{_adj(p1, v1)}, with no exceptions permitted."),
        )

    if variant == "1prop_neg":
        p1, v1 = _g(a, "P1'", "V1'")

        return dict(
            short=f"No object is {_adj(p1, v1)}.",
            medium=f"The scene contains no {_objs(p1, v1)} whatsoever.",
            long=(f"Every object in the scene must avoid being {_adj(p1, v1)}; "
                  f"this property value is globally forbidden."),
        )

    if variant == "1prop_2val_neg":
        p1, v1, v2 = _g(a, "P1'", "V1'", "V2'")
        return dict(
            short=f"No object is either {_adj(p1, v1)} or {_adj(p1, v2)}.",
            medium=f"No object in the scene can be {_adj(p1, v1)} and no object in the scene can be {_adj(p1, v2)}.",
            long=(f"The constraint forbids any object from having the following two values for the property {p1} "
                  f": no object can either be {_adj(p1, v1)} or {_adj(p1, v2)}."),
        )

    if variant == "2prop":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"Every object is {_adj(p1, v1)} and {_adj(p2, v2)}.",
            medium=f"Each object in the scene must simultaneously be {_adj(p1, v1)} and {_adj(p2, v2)}.",
            long=(f"The scene enforces two universal requirements: every object must be "
                  f"{_adj(p1, v1)} and also {_adj(p2, v2)}, with no object free to deviate."),
        )

    if variant == "2prop_neg":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"No object is {_adj(p1, v1)} and {_adj(p2, v2)} simultaneously.",
            medium=f"The scene forbids any object from being {_adj(p1, v1)} and {_adj(p2, v2)} simultaneously.",
            long=(f"An object that is simultaneously {_adj(p1, v1)} and {_adj(p2, v2)} must never "
                  f"appear; every object must avoid at least one of these two property values."),
        )

    if variant == "2prop_mix_neg":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"No object is {_adj(p1, v1)} and {_not_adj(p2, v2)} simultaneously.",
            medium=f"The scene forbids any object from being {_adj(p1, v1)} and {_not_adj(p2, v2)} simultaneously.",
            long=(f"An object that is simultaneously {_adj(p1, v1)} and {_not_adj(p2, v2)} must never "
                  f"appear."),
        )
            
            
           

    if variant == "3prop":
        p1, v1, p2, v2, p3, v3 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'")
        return dict(
            short=f"Every object is {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)}.",
            medium=f"Each object in the scene must be {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)}.",
            long=(f"The scene enforces three universal requirements: every object must be "
                  f"{_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)} simultaneously."),
        )

    if variant == "3prop_val_mix_neg":
        p1, v1, p2, v2, p3, v3, v4 = _g(a, "P1'", "V1'","P2'", "V2'", "P3'", "V3'", "V4'")
        return dict(
            short=(f"No object must be simultaneously be {_adj(p1, v1)} and {_adj(p2, v2)} "
                   f"and is neither {_adj(p3, v3)} nor {_adj(p3, v4)}."),
            medium=(f"It is impossible to have an object that is simultaneously {_adj(p1, v1)} and {_adj(p2, v2)} and "
                    f"{_not_adj(p3, v3)} and {_not_adj(p3, v4)}."),
            long=(f"The scene requires that any object with the following combination is not present: {_adj(p1, v1)} with {_adj(p2, v2)} "
                  f"and {_not_adj(p3, v3)} and {_not_adj(p3, v4)}."),
        )

    if variant == "4prop_neg":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'","P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"No object must be simultaneously {_adj(p1, v1)} and {_adj(p2, v2)} "
                   f"and is neither {_adj(p3, v3)} nor {_adj(p4, v4)}."),
            medium=(f"It is impossible to have an object that is simultaneously {_adj(p1, v1)} and {_adj(p2, v2)} and "
                    f"{_not_adj(p3, v3)} and {_not_adj(p4, v4)}."),
            long=(f"The scene requires that any object with the following combination is not present: {_adj(p1, v1)} and {_adj(p2, v2)} "
                  f"and {_not_adj(p3, v3)} and {_not_adj(p4, v4)}."),
        )
        
        
        

    if variant == "4prop":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'","P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        
        return dict(
            short=f"Every object is {_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}.",
            medium=(f"Each object in the scene must simultaneously satisfy all four: "
                    f"{_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)}, {_adj(p4, v4)}."),
            long=(f"The scene enforces four universal requirements at once: every object must be "
                  f"{_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}."),
        )

    return None


def _c3(variant: str, a: dict) -> dict[str, str]:
    p1, v1 = _g(a, "P1'", "V1'")

    if variant == "1propA_1propC":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"{_quant('Every', p1, v1)} is also {_adj(p2, v2)}.",
            medium=f"Each object that is {_adj(p1, v1)} must also be {_adj(p2, v2)}.",
            long=(f"Whenever an object is {_adj(p1, v1)}, it must simultaneously be {_adj(p2, v2)}; "
                  f"objects that are not {_adj(p1, v1)} are unconstrained."),
        )

    if variant == "1propA_1prop_neg":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"{_quant('Every', p1, v1)} is {_not_adj(p2, v2)}.",
            medium=f"Each object that is {_adj(p1, v1)} must not be {_adj(p2, v2)}.",
            long=(f"Whenever an object is {_adj(p1, v1)}, it must avoid being {_adj(p2, v2)}; "
                  f"the combination of {_adj(p1, v1)} with {_adj(p2, v2)} is forbidden."),
        )

    if variant == "1propA_neg_1propC":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"Every object that is {_not_adj(p1, v1)} must be {_adj(p2, v2)}.",
            medium=f"Each object that is not {_adj(p1, v1)} is required to be {_adj(p2, v2)}.",
            long=(f"Among objects that are not {_adj(p1, v1)}, all of them must be {_adj(p2, v2)}; "
                  f"only {_adj(p1, v1)} objects escape this requirement."),
        )

    if variant == "1propA_2propC":
        p1, v1, p2, v2, p3, v3 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'")
        return dict(
            short=f"{_quant('Every', p1, v1)} must be {_adj(p2, v2)} and {_adj(p3, v3)}.",
            medium=(f"Each object that is {_adj(p1, v1)} must simultaneously satisfy "
                    f"{_adj(p2, v2)} and {_adj(p3, v3)}."),
            long=(f"Whenever an object is {_adj(p1, v1)}, it must be both {_adj(p2, v2)} "
                  f"and {_adj(p3, v3)}; violating either consequent property is forbidden."),
        )

    if variant == "1propA_3propC":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=f"{_quant('Every', p1, v1)} must be {_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}.",
            medium=(f"Each {_bare(p1, v1)} must satisfy all three consequent properties: "
                    f"{_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}."),
            long=(f"The condition {_adj(p1, v1)} triggers three simultaneous requirements: "
                  f"{_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}; any {_bare(p1, v1)} "
                  f"missing any of these is forbidden."),
        )


    if variant == "2propA_1propC":
        p1, v1, p2, v2, p3, v3 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'")
        return dict(
            short=f"Every object that is {_adj(p1, v1)} and {_adj(p2, v2)} is also {_adj(p3, v3)}.",
            medium=(f"Each object that is both {_adj(p1, v1)} and {_adj(p2, v2)} "
                    f"must also be {_adj(p3, v3)}."),
            long=(f"Whenever there is a combination of the following properties, that is, an object is {_adj(p1, v1)} and {_adj(p2, v2)}, "
                  f"it must additionally be {_adj(p3, v3)}."),
        )

    
    if variant == "2propA_2propC":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'","P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"Every object that is {_adj(p1, v1)} and {_adj(p2, v2)} must be "
                   f"{_adj(p3, v3)} and {_adj(p4, v4)}."),
            medium=(f"Each object that is {_adj(p1, v1)} and {_adj(p2, v2)} must also be "
                    f"{_adj(p3, v3)} and {_adj(p4, v4)}."),
            long=(f"Whenever there is a combination of the following properties, that is, the object is {_adj(p1, v1)} and {_adj(p2, v2)}, then it "
                  f"requires that the object is also {_adj(p3, v3)} and {_adj(p4, v4)}."),
        )

    if variant == "2propA_neg_2propC":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"Every object that is {_adj(p1, v1)} and {_not_adj(p2, v2)} must be "
                   f"{_adj(p3, v3)} and {_adj(p4, v4)}."),
            medium=(f"Each object that is {_adj(p1, v1)} but not {_adj(p2, v2)} must be "
                    f"{_adj(p3, v3)} and {_adj(p4, v4)}."),
            long=(f"An object that is {_adj(p1, v1)} while not being {_adj(p2, v2)} must "
                  f"satisfy being {_adj(p3, v3)} and {_adj(p4, v4)} simultaneously."),
        )

    if variant == "3propA_1propC":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"Every object that is {_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)} must be {_adj(p4, v4)}."),
            medium=(f"Each object that is {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)} "
                    f"must also be {_adj(p4, v4)}."),
            long=(f"An object that is {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)} "
                  f"is required to additionally be {_adj(p4, v4)}."),
        )

    if variant == "3propA_1propC_neg":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"Every object that is {_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)} must not be {_adj(p4, v4)}."),
            medium=(f"Each object that is {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)} "
                    f"must not be {_adj(p4, v4)}."),
            long=(f"An object satisfying the following three antecedent conditions of being ({_adj(p1, v1)}, "
                  f"{_adj(p2, v2)}, and {_adj(p3, v3)}) is forbidden from being {_adj(p4, v4)}."),
        )

    return None


def _c4(variant: str, a: dict) -> dict[str, str]:
    
    if variant == "1prop_2hop":

        p1, v1, p2, v2, p3, v3, d1, d2 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D1'", "D2'")
        return dict(
            short=(f"There exists {_obj(p1, v1)} that is {_dir(d1)} {_obj(p2, v2)}, "
                   f"which is {_dir(d2)} {_obj(p3, v3)}."),
            medium=(f"The scene contains a 2-hop chain: {_obj(p1, v1)} is {_dir(d1)} "
                    f"{_obj(p2, v2)}, and the latter is {_dir(d2)} {_obj(p3, v3)}."),
            long=(f"At least one triple of distinct objects must form a 2-hop chain with specific "
                  f"properties: {_obj(p1, v1)} {_dir(d1)} {_obj(p2, v2)} that is {_dir(d2)} {_obj(p3, v3)}."),
        )

    if variant == "1prop_2hop_mix":

        p1, v1, p2, v2, p3, v3, d1, d2 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D1'", "D2'")
        return dict(
            short=(f"There exists {_obj(p1, v1)} that is {_dir(d1)} {_obj(p2, v2)}, "
                   f"which is {_dir(d2)} {_not_obj(p3, v3)}."),
            medium=(f"The scene contains a 2-hop chain: {_obj(p1, v1)} is {_dir(d1)} "
                    f"{_obj(p2, v2)}, and the latter is {_dir(d2)} {_not_obj(p3, v3)}."),
            long=(f"At least one triple of distinct objects must form a 2-hop chain with specific "
                  f"properties: {_obj(p1, v1)} {_dir(d1)} {_obj(p2, v2)} that is {_dir(d2)} {_not_obj(p3, v3)}."),
        )

        
    if variant == "1prop_shared":
        p1, v1, p2, v2, p3, v3, d1, d2 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D1'", "D2'")
        
        return dict(
            short=(f"There exists {_obj(p3, v3)} that is {_dir(d1)} {_obj(p1, v1)}, "
                   f"and {_dir(d2)} {_obj(p2, v2)}."),
            medium=(f"The scene contains an object that is {_adj(p3, v3)} and it is {_dir(d1)} "
                    f"{_obj(p1, v1)}, and is {_dir(d2)} {_obj(p2, v2)}."),
            long=(f"There exists at least one triple of distinct objects such that the following constraints are satisfied: "
                  f" {_obj(p3, v3)} {_dir(d1)} {_obj(p1, v1)}, and is {_dir(d2)} {_obj(p2, v2)}."),
        )
        
    if variant == "1prop_shared_mix":
        p1, v1, p2, v2, p3, v3, d1, d2 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D1'", "D2'")
        
        return dict(
            short=(f"There exists {_obj(p3, v3)} that is {_dir(d1)} {_obj(p1, v1)}, "
                   f"and {_dir(d2)} {_not_obj(p2, v2)}."),
            medium=(f"The scene contains an object that is {_adj(p3, v3)} and it is {_dir(d1)} "
                    f"{_obj(p1, v1)}, and is {_dir(d2)} {_not_obj(p2, v2)}."),
            long=(f"There exists at least one triple of distinct objects such that the following constraints are satisfied: "
                  f" {_obj(p3, v3)} {_dir(d1)} {_obj(p1, v1)}, and is {_dir(d2)} {_not_obj(p2, v2)}."),
        )
        

    
    return None


def _c5(variant: str, a: dict) -> dict[str, str]:
    
    if variant == "pair_propA_relC":
        p1, v1, p2, v2, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'")
        return dict(
            short=(f"If there is {_obj(p1, v1)} and {_obj(p2, v2)}, then the former must be {_dir(d1)} the latter."),
            long=(f"Whenever there is a pair of objects (X1, X2), such that X1 is {_adj(p1, v1)} and X2 is "
                    f"{_adj(p2, v2)}, it must be ensured that X1 is always {_dir(d1)} X2."),
            medium=(f"For every pair of distinct objects where one is {_adj(p1, v1)} and the other "
                  f"is {_adj(p2, v2)}, the first must stand {_dir(d2)} the second."),
        )

    if variant == "pair_propRelA_propC":
        p1, v1, p2, v2, d1, p3, v3 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'", "P3'", "V3'")
        return dict(
            short=(f"If there is {_obj(p1, v1)} {_dir(d1)} {_obj(p2, v2)}, "
            f"then the latter must also be {_adj(p3, v3)}."),
            medium=(f"For any pair of objects (X1, X2), if X1 is {_adj(p1, v1)}, X2 is {_adj(p2, v2)}, and X1 is {_dir(d1)} X2, "
                    f"then X2 must be {_adj(p3, v3)}."),
            long=(f"For every distinct pair of objects where one is {_adj(p1, v1)}, and the other is {_adj(p2, v2)}, "
                  f"and the former is {_dir(d1)} latter, it must be additionally ensured that the latter is also {_adj(p3, v3)}."),
        )

    if variant == "pair_propRelA_RelC":
        p1, v1, p2, v2, d1, d2 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'",  "D2'")
        
        return dict(
            short=(f"If there is {_obj(p1, v1)} {_dir(d1)} {_obj(p2, v2)}, "
            f"then it must also be {_dir(d2)} the latter."),
            medium=(f"If X1 is {_obj(p1, v1)} and it is {_dir(d1)} X2, which is {_obj(p2, v2)}, "
                    f"then it is also the case that X1 is {_dir(d2)} X2."),
            long=(f"For every pair (X1, X2) where X1 is {_adj(p1, v1)}, X2 is {_adj(p2, v2)}, "
                  f"and X1 is {_dir(d1)} X2, the spatial relation {_dir(d2)} must also hold between X1 and X2."),
        )

    

    if variant == "pair_relA_propC":
        # Template: :- object(X1), object(X2), X1!=X2, hasRelationship(X1,X2,D2'), not hasProperty(X1,P1',V1').
        # Meaning: for every pair (X1, X2) with X1 related D2' to X2, X1 must be P1'=V1'.
        d1, p1, v1 = a.get("D1'", "P1'", "V1'")
        return dict(
            short=(f"All objects that are {_dir(d1)} some object, "
                   f"must be {_adj(p1, v1)}."),
            medium=(f"Whenever X1 stands {_dir(d1)} X2, X1 is required to be {_adj(p1, v1)}."),
            long=(f"For every distinct pair of objects where the first stands {_dir(d1)} the second, "
                  f"the first is required to be {_adj(p1, v1)}."),
        )

    if variant == "triple_propA_RelC":
        # Template: :- object(X1), object(X2), X1!=X2, hasRelationship(X1,X2,D2'), not hasProperty(X1,P1',V1').
        # Meaning: for every pair (X1, X2) with X1 related D2' to X2, X1 must be P1'=V1'.
        d1, p1, v1, p2, v2, p3, v3, d2 = a.get("D1'", "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D2'")
        return dict(
            short =(f"If X1 is {_obj(p1,v1)}, X2 is {_obj(p2,v2)}, and X3 is {_obj(p3,v3)}, "
            f"then X1 is {_dir(d1)} X2 and X2 is {_dir(d2)} X3.")
            long=(f"If there are three objects, the first is {_adj(p1,v1)}, the second is {_adj(p2,v2)} and the third is {_adj(p3,v3)}, then, "
                   f"it must be that the first is {_dir(d1)} the second and the second is {_dir(d2)} the third."),
            medium=(f"For every triplet (X1, X2, X3), where X1 is {_obj(p1,v1)}, X2 is {_obj(p2,v2)}, and X3 is {_obj(p3,v3)}, then "
                  f" X1 must stand {_dir(d1)} X2 and  X2 must stand {_dir(d2)} X3."),
        )

    if variant == "triple_propRelA_relC":
        d1, p1, v1, p2, v2, p3, v3, d2 = a.get("D1'", "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D2'")
        return dict(
            short =(f"If X1 is {_obj(p1,v1)}, X2 is {_obj(p2,v2)},  X3 is {_obj(p3,v3)}, and "
            f"X1 is {_dir(d1)} X2, then X2 must stand {_dir(d2)} X3.")
            long=(f"If there are three objects, the first is {_adj(p1,v1)}, the second is {_adj(p2,v2)} and the third is {_adj(p3,v3)}, and "
                   f"it is observed that the first is {_dir(d1)} the second, then it must be ensured that the second is {_dir(d2)} the third."),
            medium=(f"For every triplet (X1, X2, X3), where X1 is {_obj(p1,v1)}, X2 is {_obj(p2,v2)}, and X3 is {_obj(p3,v3)} and  "
                  f" X1 is {_dir(d1)} X2, then it is required that X2 stands {_dir(d2)} X3."),
        )

    
    return None


def _c6(variant: str, a: dict) -> dict[str, str]:
    p1, v1, p2, v2, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'")

    if variant == "1prop":
        return dict(
            short=(f"{_quant('Some', p1, v1)} is {_dir(d1)} {_quant('every', p2, v2)}."),
            medium=(f"At least one {_bare(p1, v1)} stands {_dir(d1)} every single "
                    f"{_bare(p2, v2)} in the scene."),
            long=(f"There exists at least one {_bare(p1, v1)} that, "
                  f"for every {_bare(p2, v2)} in the scene, stands {_dir(d1)} it."),
        )

    if variant == "1prop_neg":
        return dict(
            short=(f"{_quant('Some', p1, v1)} is not {_dir(d1)} any {_bare(p2, v2)}."),
            medium=(f"At least one {_bare(p1, v1)} exists that does not stand {_dir(d1)} "
                    f"any {_bare(p2, v2)}."),
            long=(f"There exists {_obj(p1, v1)} that fails to stand {_dir(d1)} "
                  f"at least one {_bare(p2, v2)} — the universal coverage does not hold."),
        )

    if variant == "2prop":
        p3, v3, p4, v4 = _g(a, "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"Some object that is {_adj(p1, v1)} and {_adj(p2, v2)} is {_dir(d1)} "
                   f"every object that is {_adj(p3, v3)} and {_adj(p4, v4)}."),
            medium=(f"At least one object that is both {_adj(p1, v1)} and {_adj(p2, v2)} "
                    f"stands {_dir(d1)} every object that is {_adj(p3, v3)} and {_adj(p4, v4)}."),
            long=(f"There exists a candidate object satisfying both {_adj(p1, v1)} and {_adj(p2, v2)} "
                  f"that covers all target objects ({_adj(p3, v3)}, {_adj(p4, v4)}) "
                  f"via the {_dir(d1)} relation."),
        )

    if variant == "2prop_neg":
        p3, v3, p4, v4 = _g(a, "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"No object that is {_adj(p1, v1)} and {_adj(p2, v2)} covers every "
                   f"object that is {_adj(p3, v3)} and {_adj(p4, v4)}."),
            medium=(f"The scene has no object that is {_adj(p1, v1)}, {_adj(p2, v2)} and "
                    f"stands {_dir(d1)} every object that is {_adj(p3, v3)}, {_adj(p4, v4)}."),
            long=(f"For every candidate object ({_adj(p1, v1)}, {_adj(p2, v2)}), there is at "
                  f"least one target object ({_adj(p3, v3)}, {_adj(p4, v4)}) that the candidate "
                  f"does not stand {_dir(d1)}."),
        )

    return _generic("C6", variant, a)


def _c7(variant: str, a: dict) -> dict[str, str]:
    p1, v1, p2, v2, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'")

    if variant == "1prop_propRel":
        return dict(
            short=(f"{_quant('Every', p1, v1)} has {_obj(p2, v2)} {_dir(d1)} it."),
            medium=(f"For each {_bare(p1, v1)} in the scene, there must exist "
                    f"at least one {_bare(p2, v2)} standing {_dir(d1)} it."),
            long=(f"Every object that is {_adj(p1, v1)} is required to have a witness: "
                  f"{_obj(p2, v2)} that stands {_dir(d1)} it."),
        )

    if variant == "1prop_propRel_neg":
        return dict(
            short=(f"{_quant('Every', p1, v1)} has no {_bare(p2, v2)} {_dir(d1)} it."),
            medium=(f"For each {_bare(p1, v1)} in the scene, there must be no "
                    f"{_bare(p2, v2)} standing {_dir(d1)} it."),
            long=(f"Every object that is {_adj(p1, v1)} must be unwitnessed: no {_bare(p2, v2)} "
                  f"may stand {_dir(d1)} any {_bare(p1, v1)}."),
        )

    if variant == "propRel_propRel":
        p3, v3, d2 = _g(a, "P3'", "V3'", "D2'")
        return dict(
            short=(f"{_quant('Every', p1, v1)} related {_dir(d1)} {_obj(p2, v2)} "
                   f"has {_obj(p3, v3)} {_dir(d2)} it."),
            medium=(f"For every {_bare(p1, v1)} X that is {_dir(d1)} some {_bare(p2, v2)} Z, "
                    f"there must exist {_obj(p3, v3)} Y standing {_dir(d2)} X."),
            long=(f"Whenever X is {_adj(p1, v1)} and stands {_dir(d1)} {_obj(p2, v2)} Z, "
                  f"a witness Y must exist that is {_adj(p3, v3)} and stands {_dir(d2)} X."),
        )

    if variant == "propRel_propRel_neg":
        p3, v3, d2 = _g(a, "P3'", "V3'", "D2'")
        return dict(
            short=(f"{_quant('Every', p1, v1)} {_dir(d1)} {_obj(p2, v2)} "
                   f"has no {_bare(p3, v3)} {_dir(d2)} it."),
            medium=(f"For every qualifying pair (X {_adj(p1, v1)}, Z {_adj(p2, v2)}, X {_dir(d1)} Z), "
                    f"no {_bare(p3, v3)} may stand {_dir(d2)} X."),
            long=(f"Whenever X is {_adj(p1, v1)} and stands {_dir(d1)} {_obj(p2, v2)} Z, "
                  f"the scene must contain no {_bare(p3, v3)} standing {_dir(d2)} X."),
        )

    if variant == "1prop_exact":
        n = a.get("N'", "?")
        return dict(
            short=(f"{_quant('Every', p1, v1)} has {_exactly(n)} {_objs(p2, v2)} {_dir(d1)} it."),
            medium=(f"For each {_bare(p1, v1)} in the scene, {_exactly(n)} "
                    f"{_objs(p2, v2)} must stand {_dir(d1)} it."),
            long=(f"Every object that is {_adj(p1, v1)} requires {_exactly(n)} witness(es): "
                  f"{_obj(p2, v2)} standing {_dir(d1)} it — no more, no fewer."),
        )

    if variant == "1prop_exact_neg":
        n = a.get("N'", "?")
        _pl = "object" if n == "1" else "objects"
        _not_phrase = (f"not in {v2}" if p2 == "region" else f"not {v2}")
        return dict(
            short=(f"{_quant('Every', p1, v1)} has {_exactly(n)} {_not_phrase} "
                   f"{_pl} {_dir(d1)} it."),
            medium=(f"For each {_bare(p1, v1)}, {_exactly(n)} {_pl} that are not "
                    f"{_adj(p2, v2)} must stand {_dir(d1)} it."),
            long=(f"Each {_bare(p1, v1)} must have exactly {n} {_pl} that "
                  f"are {_not_adj(p2, v2)} and stand {_dir(d1)} it."),
        )

    return _generic("C7", variant, a)


def _c8(variant: str, a: dict) -> dict[str, str]:
    p1, v1, n = _g(a, "P1'", "V1'", "N'")

    if variant == "1prop_exact":
        return dict(
            short=f"There are {_exactly(n)} {_objs(p1, v1)} in the scene.",
            medium=f"The scene must contain {_exactly(n)} {_n_obj(n)} that {('are' if n != '1' else 'is')} {_adj(p1, v1)}.",
            long=(f"The total count of {_adj(p1, v1)} objects across the entire scene must be "
                  f"{_exactly(n)} — neither more nor fewer."),
        )

    if variant == "1prop_atleast":
        return dict(
            short=f"There are {_at_least(n)} {_objs(p1, v1)} in the scene.",
            medium=f"The scene must contain {_at_least(n)} {_objs(p1, v1)}.",
            long=(f"The count of {_adj(p1, v1)} objects in the scene must reach "
                  f"a minimum of {n}."),
        )

    if variant == "1prop_atmost":
        return dict(
            short=f"There are {_at_most(n)} {_objs(p1, v1)} in the scene.",
            medium=f"The scene contains {_at_most(n)} {_objs(p1, v1)}.",
            long=(f"The count of {_adj(p1, v1)} objects in the scene must not exceed {n}."),
        )

    if variant == "1prop_exact_neg":
        return dict(
            short=f"There are {_exactly(n)} {_not_objs(p1, v1)} in the scene.",
            medium=f"The scene must contain {_exactly(n)} {_n_obj(n)} that {'is' if n == '1' else 'are'} {_not_adj(p1, v1)}.",
            long=(f"The count of objects that are not {_adj(p1, v1)} must equal exactly {n}."),
        )

    if variant == "1prop_atleast_neg":
        return dict(
            short=f"There are {_at_least(n)} {_not_objs(p1, v1)} in the scene.",
            medium=f"The scene must contain {_at_least(n)} {_n_obj(n)} that {'is' if n == '1' else 'are'} {_not_adj(p1, v1)}.",
            long=(f"At least {n} objects in the scene must not be {_adj(p1, v1)}."),
        )

    if variant == "1prop_atmost_neg":
        return dict(
            short=f"There are {_at_most(n)} {_not_objs(p1, v1)} in the scene.",
            medium=f"The scene contains {_at_most(n)} {_n_obj(n)} that {'is' if n == '1' else 'are'} {_not_adj(p1, v1)}.",
            long=(f"At most {n} objects in the scene may not be {_adj(p1, v1)}."),
        )

    if variant == "2prop_exact":
        p2, v2 = _g(a, "P2'", "V2'")
        return dict(
            short=f"There are {_exactly(n)} objects that are {_adj(p1, v1)} and {_adj(p2, v2)}.",
            medium=(f"The scene must contain {_exactly(n)} {_n_obj(n)} satisfying both "
                    f"{_adj(p1, v1)} and {_adj(p2, v2)}."),
            long=(f"The count of objects combining {_adj(p1, v1)} with {_adj(p2, v2)} must be "
                  f"exactly {n}."),
        )

    if variant == "2prop_atleast":
        p2, v2 = _g(a, "P2'", "V2'")
        return dict(
            short=f"There are {_at_least(n)} objects that are {_adj(p1, v1)} and {_adj(p2, v2)}.",
            medium=(f"The scene must contain {_at_least(n)} {_n_obj(n)} that {'is' if n == '1' else 'are'} "
                    f"{_adj(p1, v1)} and {_adj(p2, v2)}."),
            long=(f"At least {n} objects must simultaneously satisfy {_adj(p1, v1)} "
                  f"and {_adj(p2, v2)}."),
        )

    if variant == "2prop_atmost":
        p2, v2 = _g(a, "P2'", "V2'")
        return dict(
            short=f"There are {_at_most(n)} objects that are {_adj(p1, v1)} and {_adj(p2, v2)}.",
            medium=(f"The scene contains {_at_most(n)} {_n_obj(n)} satisfying both "
                    f"{_adj(p1, v1)} and {_adj(p2, v2)}."),
            long=(f"At most {n} objects may combine {_adj(p1, v1)} with {_adj(p2, v2)}."),
        )

    if variant in ("2prop_exact_neg", "2prop_atleast_neg", "2prop_atmost_neg"):
        p2, v2 = _g(a, "P2'", "V2'")
        op = ("exactly" if "exact" in variant
              else "at least" if "atleast" in variant else "at most")
        return dict(
            short=f"There are {op} {n} objects that are {_adj(p1, v1)} but not {_adj(p2, v2)}.",
            medium=(f"The scene must contain {op} {n} {_n_obj(n)} that {'is' if n == '1' else 'are'} "
                    f"{_adj(p1, v1)} while not being {_adj(p2, v2)}."),
            long=(f"The count of {_adj(p1, v1)} objects that lack {_adj(p2, v2)} must be {op} {n}."),
        )

    if variant == "1prop_exact_relational":
        p2, v2, d1 = _g(a, "P2'", "V2'", "D1'")
        return dict(
            short=(f"{_exactly(n).capitalize()} {_objs(p1, v1)} "
                   f"stand {_dir(d1)} some {_bare(p2, v2)}."),
            medium=(f"The scene must have {_exactly(n)} {_objs(p1, v1)} "
                    f"that each stand {_dir(d1)} some {_bare(p2, v2)}."),
            long=(f"The count of {_objs(p1, v1)} standing {_dir(d1)} any {_bare(p2, v2)} "
                  f"must equal exactly {n}."),
        )

    if variant == "1prop_atleast_relational":
        p2, v2, d1 = _g(a, "P2'", "V2'", "D1'")
        return dict(
            short=(f"{_at_least(n).capitalize()} {_objs(p1, v1)} "
                   f"stand {_dir(d1)} some {_bare(p2, v2)}."),
            medium=(f"The scene must have {_at_least(n)} {_objs(p1, v1)} "
                    f"standing {_dir(d1)} some {_bare(p2, v2)}."),
            long=(f"At least {n} {_objs(p1, v1)} must each stand {_dir(d1)} "
                  f"at least one {_bare(p2, v2)}."),
        )

    if variant == "1prop_atmost_relational":
        p2, v2, d1 = _g(a, "P2'", "V2'", "D1'")
        return dict(
            short=(f"{_at_most(n).capitalize()} {_objs(p1, v1)} "
                   f"stand {_dir(d1)} some {_bare(p2, v2)}."),
            medium=(f"The scene contains {_at_most(n)} {_objs(p1, v1)} "
                    f"standing {_dir(d1)} some {_bare(p2, v2)}."),
            long=(f"No more than {n} {_objs(p1, v1)} may stand {_dir(d1)} "
                  f"any {_bare(p2, v2)}."),
        )

    if variant in ("1prop_exact_relational_neg", "1prop_atleast_relational_neg",
                   "1prop_atmost_relational_neg"):
        p2, v2, d1 = _g(a, "P2'", "V2'", "D1'")
        op = ("exactly" if "exact" in variant
              else "at least" if "atleast" in variant else "at most")
        return dict(
            short=(f"{op.capitalize()} {n} {_objs(p1, v1)} do NOT "
                   f"stand {_dir(d1)} any {_bare(p2, v2)}."),
            medium=(f"The scene has {op} {n} {_objs(p1, v1)} that "
                    f"do not stand {_dir(d1)} any {_bare(p2, v2)}."),
            long=(f"The count of {_objs(p1, v1)} that fail to stand {_dir(d1)} "
                  f"any {_bare(p2, v2)} must be {op} {n}."),
        )

    return _generic("C8", variant, a)


def _c9(variant: str, a: dict) -> dict[str, str]:
    p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")

    if variant == "1prop":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")

        return dict(
            short=(f"The count of {_objs(p1, v1)} equals the count of {_objs(p2, v2)}."),
            medium=(f"The number of objects that are {_adj(p1, v1)} in the scene must equal "
                    f"the number of objects that are {_adj(p2, v2)}."),
            long=(f"The scene enforces a balance: the cardinality of objects thar are {_adj(p1, v1)} "
                  f"must exactly match the cardinality of objects that are {_adj(p2, v2)} "),
        )

    if variant == "2prop":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"The count of objects that are both ({_adj(p1, v1)} and {_adj(p2, v2)}) equals "
                   f"the count of objects that are both ({_adj(p3, v3)} and {_adj(p4, v4)})."),
            medium=(f"The number of objects that are {_adj(p1, v1)} and {_adj(p2, v2)} simultaneously "
                    f"must equal the number of objects that are both {_adj(p3, v3)} and {_adj(p4, v4)}."),
            long=(f"The scene requires equal cardinality between two groups: the first group being the collection of objects that are both "
                  f"{_adj(p1, v1)} and {_adj(p2, v2)}, and the second group being the collection of objects that are both "
                  f"{_adj(p3, v3)} and {_adj(p4, v4)}."),
        )

    if variant == "2prop_mix":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"The count of object that are ({_not_adj(p1, v1)} but are {_adj(p2, v2)}) equals "
                   f"the count of objects that are ({_not_adj(p3, v3)} but {_adj(p4, v4)})."),
            medium=(f"The number of objects that are {_not_adj(p1, v1)} but {_adj(p2, v2)} "
                    f"must equal the number that are {_not_adj(p3, v3)} but {_adj(p4, v4)}."),
            long=(f"The scene balances two groups: the first group is the collection of objects that are not "
                  f"{_adj(p1, v1)} but  {_adj(p2, v2)}, and the second group is the collection of objects that are not "
                  f"{_adj(p3, v3)} but {_adj(p4, v4)}."),
        )

    return None



# ── Dispatch table ──────────────────────────────────────────────────────────

_HANDLERS = {
    "C1": _c1, "C2": _c2, "C3": _c3, "C4": _c4,
    "C5": _c5, "C6": _c6, "C7": _c7, "C8": _c8, "C9": _c9,
}


# ── Public API ──────────────────────────────────────────────────────────────

def verbalize(template_stem: str, assignment: dict[str, str]) -> dict[str, str]:
    """
    Generate short, medium, and long NL descriptions for an instantiated constraint.

    Args:
        template_stem: Filename stem without extension, e.g. 'C1_2prop_neg'.
        assignment:    Completed placeholder dict, e.g. {"P1'": "color", "V1'": "red"}.

    Returns:
        {'short': ..., 'medium': ..., 'long': ...}
    """
    parts = template_stem.split("_", 1)
    cls = parts[0]  # 'C1', 'C2', …
    variant = parts[1] if len(parts) > 1 else ""
    handler = _HANDLERS.get(cls)
    if handler:
        return handler(variant, assignment)
    