"""Layout usability and plausibility metrics beyond legality, all computed by program (no LLM, no model):

  NAV  navigability (SceneEval): the largest connected free floor region / all free floor, per room, on a 5 cm raster
       of the floor polygon minus the footprints of floor-standing obstacles and fixed objects
  ACC  accessibility (rule version of SceneEval's VLM-judged metric): the share of floor-standing objects (>= 0.3 m tall)
       whose functional side keeps a ``STRIP_M`` deep strip at least half free (inside the floor, not covered by other
       obstacles); the side is the front (local +X), any non-back side for beds and any side for tables; a seat in front
       of a table, or a table with seats around it, is not blocked by them (that is how they are used)
  COL  SceneEval's collision rate: the share of objects in at least one collision (V3.1's ``validate.check`` on the
       converted room, ``v3_metrics``; tucked seats under tables and declared supports are not collisions)
  OPC  SceneEval's opening clearance: the share of doors whose clearance zone (``evaluate.DOOR_ZONE_DEPTH_M`` deep) a
       floor-standing obstacle intersects (> 0.01 m2)
  SG   scene-graph recall (InstructScene / SceneNAT's 11 relations and iRecall): every ordered pair of objects gets
       one relation: "above" / "below" when one centre is higher than the other by more than half their summed heights
       and one centre lies over the other's footprint; else "none" beyond 3 m between centres (ground plane), "closely
       <dir>" within 1 m, "<dir>" between, <dir> from the angle atan2(ys - yo, xs - xo) in the room frame: right of
       [-pi/4, pi/4), in front of [pi/4, 3pi/4), behind [-3pi/4, -pi/4), left of otherwise. The ground truth's
       (subject category, relation, object category) triplets other than "none" are the reference; SG is the share of
       them the layout also has (multiset, category level, so exchangeable objects may swap), pooled over rooms.
  REL  relation plausibility: the share of applicable mined relation templates a layout satisfies. ``mine`` reads a
       seeded sample of the TRAINING rows only and, for each ordered category pair (A, B) seen together in at least
       ``--min-rooms`` rooms, the gap from each A to its nearest B (footprint distance) and whether A faces it (front
       within 45 degrees of the direction to B's centre). A "near" template keeps the pair when 80% of the gaps are at
       most 1 m (the instance passes at gap <= that 80th percentile); a "faces" template when at least 70% face. A
       layout is scored on every A instance of a room holding a B, for every template of (A, B).

Obstacles: floor-standing objects (bottom within ``FLOOR_M`` of the floor), at least ``THIN_M`` tall, not flat covers
(rugs, mats), and the room's fixed objects standing on the floor.

    python -m fastfill.v2.ops.layout_quality mine --train data/v4fixed/train.jsonl --output templates.json
    python -m fastfill.v2.ops.layout_quality score --rows ROWS --templates templates.json --output DIR \\
        --method "GPT-5.6=gpt56_predictions.jsonl" --method "M1=.../predictions.jsonl"
``score`` also scores the ground truth and a shuffled anchor (every object takes another object's position and yaw,
seeded) so the scale of each metric is visible; writes DIR/layout_quality.json and prints a table.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

import numpy as np
from scipy import ndimage
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from fastfill.anchors import FLAT
from fastfill.scene import norm_cat
from fastfill.v2.cell_costs import DEFAULTS as CELL_COST_DEFAULTS, door_zones
from fastfill.v2.schema import migrate_legacy_row
from fastfill.validate import SEAT, TABLE

# The checker needs the same 0.8 m rule, without importing Torch inference.
DOOR_ZONE_DEPTH_M = CELL_COST_DEFAULTS["door_depth_m"]
FLOOR_M, THIN_M, ACC_MIN_HEIGHT_M, STRIP_M, CELL_M = .15, .1, .3, .5, .05
FACING_DEG, NEAR_MAX_M, NEAR_QUANTILE, FACES_SHARE = 45., 1., .8, .7
BEDS = {"bed"}


def category(c):
    return norm_cat(c or "object")


def head(c):
    words = category(c).split()
    return words[-1] if words else ""


def boxes(row, layout):
    """Per object: category, footprint polygon, front unit vector, local size, bottom above the floor."""
    floor = row["condition"]["room"].get("floor_z_m") or 0.
    cats = {o["id"]: o.get("category") for o in row["condition"]["objects"]}
    out = []
    for o in layout["objects"]:
        sx, sy, sz = o["target_size_local_m"]
        x, y, z = o["bottom_center_m"]
        c, s = math.cos(o["yaw_rad"]), math.sin(o["yaw_rad"])
        corners = [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((sx / 2, sy / 2), (-sx / 2, sy / 2),
                                                                             (-sx / 2, -sy / 2), (sx / 2, -sy / 2))]
        out.append({"id": o["id"], "cat": category(cats.get(o["id"])), "head": head(cats.get(o["id"])),
                    "poly": Polygon(corners), "front": np.array((c, s)), "side": np.array((-s, c)), "centre": np.array((x, y)),
                    "size": (sx, sy, sz), "bottom": z - floor})
    return out


def obstacle(b):
    return b["bottom"] <= FLOOR_M and b["size"][2] >= THIN_M and b["head"] not in FLAT


def fixed_obstacles(row):
    room = row["condition"]["room"]
    floor = room.get("floor_z_m") or 0.
    polys = []
    for f in room.get("fixed_objects") or []:
        if f["bottom_center_m"][2] - floor > FLOOR_M or f["size_local_m"][2] < THIN_M:
            continue
        sx, sy, _ = f["size_local_m"]
        x, y, _ = f["bottom_center_m"]
        c, s = math.cos(f["yaw_rad"]), math.sin(f["yaw_rad"])
        polys.append(Polygon([(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in
                              ((sx / 2, sy / 2), (-sx / 2, sy / 2), (-sx / 2, -sy / 2), (sx / 2, -sy / 2))]))
    return polys


def nav(row, bs):
    floor = Polygon(row["condition"]["room"]["floor_polygon_xy_m"]).buffer(0)
    x0, y0, x1, y1 = floor.bounds
    xs, ys = np.arange(x0 + CELL_M / 2, x1, CELL_M), np.arange(y0 + CELL_M / 2, y1, CELL_M)
    gx, gy = np.meshgrid(xs, ys)
    free = shapely.contains_xy(floor, gx, gy)
    blocked = unary_union([b["poly"] for b in bs if obstacle(b)] + fixed_obstacles(row))
    if not blocked.is_empty:
        free &= ~shapely.contains_xy(blocked, gx, gy)
    total = free.sum()
    if not total:
        return 0.
    labels, count = ndimage.label(free)
    return float(np.bincount(labels.ravel())[1:].max() / total) if count else 0.


def strip(b, direction, half_width, half_depth):
    """The STRIP_M deep rectangle in front of side ``direction`` (unit vector) of box ``b``."""
    across = np.array((-direction[1], direction[0]))
    base = b["centre"] + direction * half_depth
    return Polygon([tuple(base + across * half_width), tuple(base - across * half_width),
                    tuple(base - across * half_width + direction * STRIP_M), tuple(base + across * half_width + direction * STRIP_M)])


def acc(row, bs):
    floor = Polygon(row["condition"]["room"]["floor_polygon_xy_m"]).buffer(0)
    fixed = fixed_obstacles(row)
    hits = total = 0
    for i, b in enumerate(bs):
        if not obstacle(b) or b["size"][2] < ACC_MIN_HEIGHT_M:
            continue
        others = [o["poly"] for j, o in enumerate(bs) if j != i and obstacle(o)
                  and not (b["head"] in SEAT and o["head"] in TABLE) and not (b["head"] in TABLE and o["head"] in SEAT)]
        blocked = unary_union(others + fixed)
        sx, sy, _ = b["size"]
        sides = [(b["front"], sy / 2, sx / 2)]
        if b["head"] in BEDS:
            sides += [(b["side"], sx / 2, sy / 2), (-b["side"], sx / 2, sy / 2)]
        elif b["head"] in TABLE:
            sides += [(-b["front"], sy / 2, sx / 2), (b["side"], sx / 2, sy / 2), (-b["side"], sx / 2, sy / 2)]
        ok = False
        for direction, half_width, half_depth in sides:
            s = strip(b, direction, half_width, half_depth)
            usable = s.intersection(floor).difference(blocked).area
            if usable >= .5 * s.area:
                ok = True
                break
        hits += ok
        total += 1
    return hits, total


def opc(row, bs):
    zones = [Polygon(z) for z in door_zones(row["condition"]["room"], DOOR_ZONE_DEPTH_M)]
    obstacles = [b["poly"] for b in bs if obstacle(b)] + fixed_obstacles(row)
    blocked = sum(any(z.intersection(p).area > .01 for p in obstacles) for z in zones)
    return blocked, len(zones)


def col(row, layout):
    from fastfill.v2.ops.v3_metrics import ir_room, placements
    from fastfill.scene import apply
    from fastfill.validate import check
    room, declared, z0 = ir_room(row)
    got = placements(room, declared, layout, z0)
    if not got:
        return 0, 0
    pl, size = got
    c = check(apply({**room, "objects": [{**o, "size": size[o["id"]]} for o in room["objects"]]}, pl))
    involved = {i for kind in ("main", "contained") for pair in c["collisions"][kind] for i in pair}
    return len(involved), len(pl)


def pair_stats(bs):
    """(A, B) -> list of (gap, faces) for every A instance, B its nearest instance of category B (A != B)."""
    by_cat = defaultdict(list)
    for b in bs:
        if obstacle(b):
            by_cat[b["cat"]].append(b)
    out = defaultdict(list)
    for a_cat, a_list in by_cat.items():
        for b_cat, b_list in by_cat.items():
            if a_cat == b_cat:
                continue
            for a in a_list:
                nearest = min(b_list, key=lambda b: a["poly"].distance(b["poly"]))
                to = nearest["centre"] - a["centre"]
                norm = float(np.linalg.norm(to))
                faces = norm > 1e-6 and float(np.dot(a["front"], to / norm)) >= math.cos(math.radians(FACING_DEG))
                out[(a_cat, b_cat)].append((a["poly"].distance(nearest["poly"]), bool(faces)))
    return out


def mine(a):
    rng = random.Random(a.seed)
    gaps, faces, rooms = defaultdict(list), defaultdict(list), defaultdict(int)
    seen = 0
    with open(a.train) as stream:
        for line in stream:
            if rng.random() > a.fraction:
                continue
            row = migrate_legacy_row(json.loads(line))
            seen += 1
            for pair, values in pair_stats(boxes(row, row["target"])).items():
                rooms[pair] += 1
                gaps[pair] += [g for g, _ in values]
                faces[pair] += [f for _, f in values]
    templates = []
    for pair, n in rooms.items():
        if n < a.min_rooms:
            continue
        q = float(np.quantile(gaps[pair], NEAR_QUANTILE))
        if q <= NEAR_MAX_M:
            templates.append({"a": pair[0], "b": pair[1], "kind": "near", "max_gap_m": round(q, 3), "rooms": n})
        share = float(np.mean(faces[pair]))
        if share >= FACES_SHARE:
            templates.append({"a": pair[0], "b": pair[1], "kind": "faces", "share": round(share, 3), "rooms": n})
    Path(a.output).write_text(json.dumps({"train": str(a.train), "rooms_sampled": seen, "seed": a.seed, "fraction": a.fraction,
                                          "min_rooms": a.min_rooms, "templates": sorted(templates, key=lambda t: -t["rooms"])}, indent=1) + "\n")
    print(f"{seen} training rooms, {len(templates)} templates")
    for t in sorted(templates, key=lambda t: -t["rooms"])[:25]:
        print(" ", t)


def rel(bs, templates):
    stats = pair_stats(bs)
    hits = total = 0
    for t in templates:
        for gap, faces in stats.get((t["a"], t["b"]), ()):
            hits += (gap <= t["max_gap_m"]) if t["kind"] == "near" else faces
            total += 1
    return hits, total


SG_FAR_M, SG_CLOSE_M = 3., 1.


def scene_graph(bs):
    """Counter of (subject category, relation, object category) over ordered pairs, "none" left out."""
    from collections import Counter
    graph = Counter()
    for s in bs:
        for o in bs:
            if s is o:
                continue
            zs, zo = s["bottom"] + s["size"][2] / 2, o["bottom"] + o["size"][2] / 2
            over = o["poly"].covers(shapely.Point(*s["centre"])) or s["poly"].covers(shapely.Point(*o["centre"]))
            half = (s["size"][2] + o["size"][2]) / 2
            if over and zs - zo > half:
                rel = "above"
            elif over and zo - zs > half:
                rel = "below"
            else:
                d = s["centre"] - o["centre"]
                distance = float(np.hypot(*d))
                if distance > SG_FAR_M:
                    continue
                theta = math.atan2(d[1], d[0])
                rel = ("right of" if -math.pi / 4 <= theta < math.pi / 4 else "in front of" if math.pi / 4 <= theta < 3 * math.pi / 4
                       else "behind" if -3 * math.pi / 4 <= theta < -math.pi / 4 else "left of")
                if distance <= SG_CLOSE_M:
                    rel = "closely " + rel
            graph[(s["cat"], rel, o["cat"])] += 1
    return graph


def sg(row, bs):
    reference = scene_graph(boxes(row, row["target"]))
    got = scene_graph(bs)
    return sum(min(n, got[t]) for t, n in reference.items()), sum(reference.values())


def shuffled(row, seed):
    """Every object takes another object's bottom centre and yaw (one random cycle), sizes kept."""
    objs = row["target"]["objects"]
    order = list(range(len(objs)))
    random.Random(seed).shuffle(order)
    moved = [{**objs[i], "bottom_center_m": objs[order[(k + 1) % len(order)]]["bottom_center_m"],
              "yaw_rad": objs[order[(k + 1) % len(order)]]["yaw_rad"]} for k, i in enumerate(order)]
    by_id = {o["id"]: o for o in moved}
    return {"schema_version": "fastfill.v2", "objects": [by_id[o["id"]] for o in objs]}


def score_method(rows, layouts, templates):
    t = defaultdict(float)
    navs = []
    for row, layout in zip(rows, layouts):
        if layout is None:
            continue
        bs = boxes(row, layout)
        navs.append(nav(row, bs))
        for key, (h, n) in (("acc", acc(row, bs)), ("opc", opc(row, bs)), ("col", col(row, layout)), ("rel", rel(bs, templates)),
                            ("sg", sg(row, bs))):
            t[key + "_hits"] += h
            t[key + "_n"] += n
    rate = lambda k: t[k + "_hits"] / t[k + "_n"] if t[k + "_n"] else None
    return {"rooms": len(navs), "NAV": float(np.mean(navs)) if navs else None, "ACC": rate("acc"), "COL": rate("col"),
            "OPC": rate("opc"), "REL": rate("rel"), "SG": rate("sg"), "acc_objects": int(t["acc_n"]), "doors": int(t["opc_n"]),
            "rel_checks": int(t["rel_n"])}


def score(a):
    rows = [migrate_legacy_row(json.loads(l)) for l in Path(a.rows).read_text().splitlines() if l.strip()]
    templates = json.loads(Path(a.templates).read_text())["templates"]
    methods = {"Ground truth": [r["target"] for r in rows], "Shuffled ground truth": [shuffled(r, a.seed + k) for k, r in enumerate(rows)]}
    for spec in a.method:
        name, path = spec.split("=", 1)
        layouts = [json.loads(l).get("layout") for l in Path(path).read_text().splitlines() if l.strip()]
        if len(layouts) != len(rows):
            raise SystemExit(f"{path}: {len(layouts)} predictions for {len(rows)} rows")
        methods[name] = layouts
    results = {name: score_method(rows, layouts, templates) for name, layouts in methods.items()}
    Path(a.output).mkdir(parents=True, exist_ok=True)
    (Path(a.output) / "layout_quality.json").write_text(json.dumps({"templates": a.templates, "results": results}, indent=1) + "\n")
    f = lambda v: "–" if v is None else f"{100 * v:.1f}"
    print("| Method | NAV ↑ | ACC ↑ | REL ↑ | SG ↑ | COL ↓ | OPC ↓ | rooms |\n|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {f(r['NAV'])} | {f(r['ACC'])} | {f(r['REL'])} | {f(r['SG'])} | {f(r['COL'])} | {f(r['OPC'])} | {r['rooms']} |")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--train", required=True)
    m.add_argument("--output", required=True)
    m.add_argument("--fraction", type=float, default=.15, help="share of training rows sampled (seeded)")
    m.add_argument("--min-rooms", type=int, default=150)
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--rows", required=True)
    s.add_argument("--templates", required=True)
    s.add_argument("--method", action="append", default=[], help="NAME=predictions.jsonl")
    s.add_argument("--output", required=True)
    s.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    {"mine": mine, "score": score}[a.command](a)


if __name__ == "__main__":
    main()
