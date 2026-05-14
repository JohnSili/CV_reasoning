"""
generate_dataset.py — main entry point.

Usage:
    python scripts/generate_dataset.py [--scenes N] [--out DIR] [--seed S] [--mode MODE]

Modes:
    mixed     — YCB objects when available, fallback to primitives (default)
    ycb_only  — only YCB objects (skip scene if not enough YCB available)
    primitive — only primitive shapes (original behaviour)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.scene_builder import SceneBuilder
from scripts.relations import build_sample, TEMPLATES

# ------------------------------------------------------------------
# Primitive object pool
# ------------------------------------------------------------------

PRIMITIVE_POOL = [
    ("cube",     "red",      [0.90, 0.10, 0.10, 1.0], [0.055, 0.055, 0.055]),
    ("cube",     "blue",     [0.10, 0.25, 0.90, 1.0], [0.055, 0.055, 0.055]),
    ("cube",     "green",    [0.10, 0.75, 0.15, 1.0], [0.055, 0.055, 0.055]),
    ("cube",     "yellow",   [0.95, 0.85, 0.05, 1.0], [0.055, 0.055, 0.055]),
    ("cube",     "white",    [0.92, 0.92, 0.92, 1.0], [0.050, 0.050, 0.050]),
    ("cube",     "dark_red", [0.60, 0.05, 0.05, 1.0], [0.055, 0.055, 0.055]),
    ("cylinder", "orange",   [0.95, 0.45, 0.05, 1.0], [0.032, 0.032, 0.080]),
    ("cylinder", "purple",   [0.55, 0.10, 0.70, 1.0], [0.032, 0.032, 0.080]),
    ("cylinder", "teal",     [0.05, 0.65, 0.65, 1.0], [0.032, 0.032, 0.080]),
    ("cylinder", "pink",     [0.95, 0.35, 0.65, 1.0], [0.032, 0.032, 0.080]),
    ("sphere",   "cyan",     [0.05, 0.85, 0.95, 1.0], [0.042, 0.042, 0.042]),
    ("sphere",   "magenta",  [0.90, 0.10, 0.80, 1.0], [0.042, 0.042, 0.042]),
    ("sphere",   "lime",     [0.60, 0.95, 0.10, 1.0], [0.042, 0.042, 0.042]),
    ("sphere",   "gold",     [0.95, 0.80, 0.10, 1.0], [0.042, 0.042, 0.042]),
]

TABLE_X      = (-0.22, 0.22)
TABLE_Y      = (-0.22, 0.22)
MIN_OBJ_DIST = 0.10
RELATION_TYPES = list(TEMPLATES.keys())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def sample_positions(n, rng):
    positions, attempts = [], 0
    while len(positions) < n and attempts < 1000:
        x = rng.uniform(*TABLE_X)
        y = rng.uniform(*TABLE_Y)
        if all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 >= MIN_OBJ_DIST
               for px, py in positions):
            positions.append((x, y))
        attempts += 1
    return positions


def sample_place_target(objects, rng):
    for _ in range(200):
        x = rng.uniform(*TABLE_X)
        y = rng.uniform(*TABLE_Y)
        if all(((x - o.position[0]) ** 2 + (y - o.position[1]) ** 2) ** 0.5 >= MIN_OBJ_DIST
               for o in objects):
            return (round(x, 4), round(y, 4))
    return (0.0, 0.0)


def make_primitive_spec(rng, positions, exclude_colors=None):
    pool = [p for p in PRIMITIVE_POOL
            if exclude_colors is None or p[1] not in exclude_colors]
    chosen = rng.sample(pool, min(len(positions), len(pool)))
    specs = []
    for (cat, col, rgba, size), xy in zip(chosen, positions):
        specs.append({"source": "primitive", "category": cat,
                       "color_name": col, "color_rgba": rgba,
                       "size": size, "position_xy": list(xy)})
    return specs


def make_ycb_spec(ycb_pool, rng, positions):
    chosen = rng.sample(ycb_pool, min(len(positions), len(ycb_pool)))
    specs = []
    for obj, xy in zip(chosen, positions):
        specs.append({"source": "ycb", "name": obj["name"],
                       "urdf": obj["urdf"],
                       "half_height": obj["half_height"],
                       "half_radius": obj["half_radius"],
                       "position_xy": list(xy)})
    return specs


def make_mixed_spec(ycb_pool, rng, positions):
    """Half YCB (if available) + half primitives, or all primitives."""
    if not ycb_pool or len(positions) < 2:
        return make_primitive_spec(rng, positions)
    n_ycb = max(1, len(positions) // 2)
    n_prim = len(positions) - n_ycb
    ycb_pos  = positions[:n_ycb]
    prim_pos = positions[n_ycb:]
    return (make_ycb_spec(ycb_pool, rng, ycb_pos) +
            make_primitive_spec(rng, prim_pos))


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def generate(num_scenes, out_dir, seed, mode):
    rng = random.Random(seed)

    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(out_dir) / "dataset.jsonl"

    builder   = SceneBuilder(width=640, height=480)
    builder.connect()
    ycb_pool  = builder.available_ycb_objects()

    if ycb_pool:
        print(f"YCB objects available: {[o['name'] for o in ycb_pool]}")
    else:
        print("YCB not found — using primitives only")
        if mode == "ycb_only":
            print("ERROR: ycb_only mode requested but pybullet-object-models not installed.")
            builder.disconnect()
            return

    written = skipped = 0

    with open(jsonl_path, "w") as f_out:
        for _ in tqdm(range(num_scenes * 4), desc="Generating"):
            n_objs    = rng.randint(3, 5)
            positions = sample_positions(n_objs, rng)
            if len(positions) < n_objs:
                skipped += 1
                continue

            # Build object specs by mode
            if mode == "primitive":
                specs = make_primitive_spec(rng, positions)
            elif mode == "ycb_only":
                if len(ycb_pool) < n_objs:
                    skipped += 1
                    continue
                specs = make_ycb_spec(ycb_pool, rng, positions)
            else:  # mixed
                specs = make_mixed_spec(ycb_pool, rng, positions)

            yaw      = rng.uniform(0, 360)
            pitch    = rng.uniform(-55, -25)
            dist     = rng.uniform(0.9, 1.3)

            try:
                rgb, objects = builder.build(specs, camera_yaw=yaw,
                                             camera_pitch=pitch,
                                             camera_distance=dist)
            except Exception as e:
                print(f"[WARN] build failed: {e}")
                skipped += 1
                continue

            if len(objects) < 2:
                skipped += 1
                continue

            relation   = rng.choice(RELATION_TYPES)
            place_xy   = sample_place_target(objects, rng)
            sample     = build_sample(objects, relation, place_xy)

            if sample is None:
                skipped += 1
                continue

            img_name = f"scene_{written:05d}.png"
            Image.fromarray(rgb).save(img_dir / img_name)

            record = {
                "scene_id": written,
                "image": f"images/{img_name}",
                "mode": mode,
                "camera": {"yaw": round(yaw, 1), "pitch": round(pitch, 1),
                            "distance": round(dist, 3)},
                **sample,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            written += 1
            if written >= num_scenes:
                break

    builder.disconnect()
    print(f"\nDone. Written: {written}  Skipped: {skipped}")
    print(f"Dataset: {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int,  default=1000)
    parser.add_argument("--out",    type=str,  default="output")
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--mode",   type=str,  default="mixed",
                        choices=["mixed", "ycb_only", "primitive"])
    args = parser.parse_args()
    generate(args.scenes, args.out, args.seed, args.mode)