"""
relations.py — computes spatial relations between objects
and builds ground-truth reasoning traces.
"""

import math
import random
from typing import List, Tuple, Optional
from scripts.scene_builder import ObjectMeta


# ------------------------------------------------------------------
# Spatial predicates (all operate on 2D table plane: x, y)
# ------------------------------------------------------------------

def _xy(obj: ObjectMeta) -> Tuple[float, float]:
    return obj.position[0], obj.position[1]


def dist2d(a: ObjectMeta, b: ObjectMeta) -> float:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def is_left_of(a: ObjectMeta, b: ObjectMeta, tol: float = 0.03) -> bool:
    return _xy(a)[0] < _xy(b)[0] - tol


def is_right_of(a: ObjectMeta, b: ObjectMeta, tol: float = 0.03) -> bool:
    return _xy(a)[0] > _xy(b)[0] + tol


def is_in_front_of(a: ObjectMeta, b: ObjectMeta, tol: float = 0.03) -> bool:
    """'in front' = smaller y (closer to camera at y<0)."""
    return _xy(a)[1] < _xy(b)[1] - tol


def is_behind(a: ObjectMeta, b: ObjectMeta, tol: float = 0.03) -> bool:
    return _xy(a)[1] > _xy(b)[1] + tol


def nearest_to(candidates: List[ObjectMeta], anchor: ObjectMeta) -> Optional[ObjectMeta]:
    if not candidates:
        return None
    return min(candidates, key=lambda o: dist2d(o, anchor))


def farthest_from(candidates: List[ObjectMeta], anchor: ObjectMeta) -> Optional[ObjectMeta]:
    if not candidates:
        return None
    return max(candidates, key=lambda o: dist2d(o, anchor))


def between(obj: ObjectMeta, a: ObjectMeta, b: ObjectMeta, tol: float = 0.05) -> bool:
    """obj is roughly between a and b (midpoint check)."""
    mx = (_xy(a)[0] + _xy(b)[0]) / 2
    my = (_xy(a)[1] + _xy(b)[1]) / 2
    return dist2d(obj, ObjectMeta("_", -1, "", "", [], (mx, my, 0), [])) < tol


# ------------------------------------------------------------------
# Instruction templates
# ------------------------------------------------------------------

TEMPLATES = {
    "left_of": (
        "Pick the {pick_cat} to the left of the {anchor_cat} and place it at the target.",
        "left_of",
    ),
    "right_of": (
        "Pick the {pick_cat} to the right of the {anchor_cat} and place it at the target.",
        "right_of",
    ),
    "in_front_of": (
        "Pick the {pick_cat} in front of the {anchor_cat} and place it at the target.",
        "in_front_of",
    ),
    "behind": (
        "Pick the {pick_cat} behind the {anchor_cat} and place it at the target.",
        "behind",
    ),
    "nearest_to": (
        "Pick the {pick_cat} nearest to the {anchor_cat} and place it at the target.",
        "nearest_to",
    ),
    "farthest_from": (
        "Pick the {pick_cat} farthest from the {anchor_cat} and place it at the target.",
        "farthest_from",
    ),
}

PREDICATE_FN = {
    "left_of":    lambda pick, anchor, others: is_left_of(pick, anchor),
    "right_of":   lambda pick, anchor, others: is_right_of(pick, anchor),
    "in_front_of":lambda pick, anchor, others: is_in_front_of(pick, anchor),
    "behind":     lambda pick, anchor, others: is_behind(pick, anchor),
    "nearest_to": lambda pick, anchor, others: pick == nearest_to(others, anchor),
    "farthest_from": lambda pick, anchor, others: pick == farthest_from(others, anchor),
}


# ------------------------------------------------------------------
# Reasoning trace builder
# ------------------------------------------------------------------

def describe_obj(obj: ObjectMeta) -> str:
    return f"{obj.color_name} {obj.category}"


def build_sample(
    objects: List[ObjectMeta],
    relation_type: str,
    place_target_xy: Tuple[float, float],
) -> Optional[dict]:
    """
    Try to build a valid (instruction, reasoning, answer) sample.
    Returns None if no unambiguous pick object is found.
    """
    template_str, rel = TEMPLATES[relation_type]
    predicate = PREDICATE_FN[rel]

    # Pick a random anchor; pick candidates are all other objects
    anchors = random.sample(objects, len(objects))
    for anchor in anchors:
        candidates = [o for o in objects if o.obj_id != anchor.obj_id]
        satisfying = [o for o in candidates if predicate(o, anchor, candidates)]

        # Need exactly one satisfying object to avoid ambiguity
        if len(satisfying) != 1:
            continue

        pick_obj = satisfying[0]
        instruction = template_str.format(
            pick_cat=describe_obj(pick_obj),
            anchor_cat=describe_obj(anchor),
        )

        reasoning = _build_reasoning(
            pick_obj, anchor, candidates, rel, place_target_xy
        )

        place_pose = (
            round(place_target_xy[0], 4),
            round(place_target_xy[1], 4),
            round(objects[0].position[2], 4),   # same surface height
            0.0,
        )

        return {
            "instruction": instruction,
            "relation_type": rel,
            "pick_id": pick_obj.obj_id,
            "anchor_id": anchor.obj_id,
            "place_pose": list(place_pose),
            "reasoning": reasoning,
            "objects": [
                {
                    "id": o.obj_id,
                    "category": o.category,
                    "color": o.color_name,
                    "position": list(o.position),
                }
                for o in objects
            ],
        }

    return None   # no valid (pick, anchor) pair found


def _build_reasoning(
    pick: ObjectMeta,
    anchor: ObjectMeta,
    candidates: List[ObjectMeta],
    relation: str,
    place_xy: Tuple[float, float],
) -> List[str]:
    steps = []

    # Step 1 — identify candidates
    cand_desc = ", ".join(
        f"{o.obj_id} ({describe_obj(o)}) @ {fmt_pos(o)}" for o in candidates
    )
    steps.append(
        f"Step 1 [Candidate Selection]: objects to consider as pick candidates: {cand_desc}. "
        f"Anchor: {anchor.obj_id} ({describe_obj(anchor)}) @ {fmt_pos(anchor)}."
    )

    # Step 2 — compute spatial relation
    if relation in ("left_of", "right_of", "in_front_of", "behind"):
        checks = []
        for o in candidates:
            val = PREDICATE_FN[relation](o, anchor, candidates)
            ax, ay = _xy(anchor)
            ox, oy = _xy(o)
            checks.append(
                f"{o.obj_id}: Δx={ox-ax:+.3f}, Δy={oy-ay:+.3f} → {'✓' if val else '✗'}"
            )
        steps.append(
            f"Step 2 [Spatial Check ({relation})]: " + "; ".join(checks)
        )
    elif relation in ("nearest_to", "farthest_from"):
        dists = [
            f"{o.obj_id}: dist={dist2d(o, anchor):.3f}m" for o in candidates
        ]
        steps.append(
            f"Step 2 [Distance Computation from {anchor.obj_id}]: " + ", ".join(dists)
        )

    # Step 3 — select pick
    steps.append(
        f"Step 3 [Pick Selection]: {pick.obj_id} ({describe_obj(pick)}) "
        f"uniquely satisfies '{relation}' relative to {anchor.obj_id}."
    )

    # Step 4 — place target
    steps.append(
        f"Step 4 [Place Target]: placing at target zone "
        f"(x={place_xy[0]:.3f}, y={place_xy[1]:.3f}) on table surface."
    )

    return steps


def fmt_pos(obj: ObjectMeta) -> str:
    x, y, _ = obj.position
    return f"({x:.3f}, {y:.3f})"
