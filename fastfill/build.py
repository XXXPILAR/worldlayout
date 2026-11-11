"""IR rooms -> filtered, deduplicated, leakage-safe SFT files (spec: FastFill I/O design review, 2026-09-24).

    python -m fastfill.build --ir /Volumes/harddisk/fastfill_ir --out data/fastfill_v1 [--sources ...] [--constraint_frac 0.3]

Per room: prep (reject or clean, see `prep`; structure and unlabelled floor boxes become `fixed` context, never
placed) -> cross-source dedup by group -> split by group/content
-> train: rot90 augmentation -> canonical order/ids -> optional constraints (S2) -> field dropout -> messages
   dev/test: canonical -> messages (no augmentation), IR written to {dev,test}_rooms.jsonl for fastfill.evaluate,
   test_constrained_rooms.jsonl adds 1-4 GT-true constraints per test room (rooms without any are skipped).
Eval-only sources (meta.eval_only) go to test with their whole split component (split.assign_splits).
stats.json counts every rejection reason, quality flag, dropped object and inferred vs evidence support per source.
Rejections are only what makes a training target impossible (no verified front, missing boxes, bad numbers, no
boundary, no placeable object); everything else (reference objects through walls, same-category overlaps, tiny rooms,
unexplained dropped floor area, single-object rooms, snapped or tilted boxes) is a FLAG written next to the sample.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from fastfill.anchors import (DIVIDERS, FINISH, FLAT, FLOOR_Z, FURNITURE_SUNK_Z, SOFT_COVERS, SOFT_HANGINGS, WALL_ART,
                              annotate, closed, fixed_geometry, hanging, is_structure)
from fastfill.scene import (GENERIC, canonical, clean_boundary, footprint, head, messages, norm_cat, num, rot90,
                            shown_size, slug)
from fastfill.split import assign_splits
from fastfill.validate import check, holds, support_ok

# dropped boxes that do not block the floor: openings and wall/floor finishes (plant_floor -> plant is not exempt)
FLUSH = re.compile(r"^(doors?|doorframes?|windows?|curtains?|drapes?|blinds?|frames?|openings?|baseboards?|rails?|"
                   r"railings?|radiators?|beams?|shades?|screening|designs?|molds?|moldings?|panels?|paneling|combination|"
                   r"walls?|floors?|ceilings?|skirting|trim|tiles?|boards?)$")


WALL_BAND_M = 0.10       # a dropped box within this band of the boundary keeps to the walls
THICK_WALL_M = 0.30      # structure (doors, windows) may stand this far outside the floor polygon, inside the wall
MIN_AREA_M2 = 0.5        # a floor smaller than this holds no furniture: a broken (mm-scale) boundary, rejected
INTRUSION_M2 = 0.05      # ... unless more than this much of it stands further inside the room
# GT duplicate rule: soft/decor heads, lamps and composites are exempt
SOFT_HEADS = {"plant", "plants", "flower", "flowers", "pillow", "pillows", "cushion", "towel", "book", "books",
              "ornament", "decoration", "vase", "toy", "toys", "clothes", "bag", "lamp", "light"}
COMPOSITE = re.compile(r"\b(combination|l shaped|composite|kids)\b")


def _words(category):
    return re.sub(r"[^a-z]+", " ", (category or "").lower()).split()


def _flush(o, inner, polygon):
    """A dropped floor box that leaves no hole: a flat covering, a soft cover or curtain, or a structural
    opening/finish ('bed frame' is not). In a polygon room (the boundary is the walls) such a box must also keep to
    the walls, under INTRUSION_M2 inside `inner` (the room shrunk by WALL_BAND_M): an open door leaf, a shower
    enclosure or a 'wall' box standing in the room is an obstacle whatever its label. Source-flagged architecture
    (stairs/pillars are shown as fixed boxes when they qualify) needs only the geometry there. In a hull room (a scan's
    floor hull, walls stand inside it) the name decides, flagged boxes may also pass by geometry."""
    w = _words(o["category"])
    if head(o) in FLAT or head(o) in SOFT_COVERS or SOFT_HANGINGS & set(w) or \
            ((head(o) in FINISH or head(o).rstrip("s") in FINISH) and not DIVIDERS & set(w)):
        return True
    stays = Polygon(footprint(o)).intersection(inner).area < INTRUSION_M2
    if o.get("structure"):
        return stays if polygon else \
            (any(FLUSH.match(x) for x in w) and not {"partition", "screen", "clothes"} & set(w)) or stays
    if not (polygon is False or stays):
        return False
    while len(w) > 1 and w[-1] in ("floor", "wall", "ceiling"):   # placement suffix: plant_floor, bookshelf_wall
        w.pop()
    return bool(w) and is_structure(o["category"]) and not {"partition", "screen", "clothes"} & set(w) \
        and bool(FLUSH.match(w[-1]))


FRONT = {"chair", "armchair", "seat", "sofa", "couch", "loveseat", "bed", "desk", "tv", "television", "monitor", "toilet",
         "piano"}
PRIORITY = ["on", "between", "faces", "against_wall", "near"]   # availability in rooms, rarest first


def _drop_with_dependants(objs, bad):
    bad = set(bad)
    while True:
        more = {o["id"] for o in objs if o.get("parent") in bad} - bad
        if not more:
            return [o for o in objs if o["id"] not in bad]
        bad |= more


FLAT_Z = 0.15            # a box lower than this that is not placed is a floor covering, not an obstacle


def _finite(o):
    return all(isinstance(o.get(k), (list, tuple)) and len(o[k]) == 3 for k in ("size", "pos")) and \
        all(_finite_number(v) for v in list(o["size"]) + list(o["pos"]) + [o.get("yaw")])


def _finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _room_geometry_error(room):
    boundary = room.get("boundary")
    if not isinstance(boundary, (list, tuple)) or len(boundary) < 3 or \
            any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in boundary):
        return "boundary_shape"
    if not all(_finite_number(v) for p in boundary for v in p):
        return "bad_numbers"
    if len({tuple(p) for p in boundary}) < 3:
        return "boundary_shape"
    height = room.get("height")
    if height is not None and (not _finite_number(height) or height <= 0):
        return "bad_numbers"
    return None


def _keep_fixed(fixed, objs, shell, cover=0.5, slab=0.5):
    """Fixed boxes worth showing next to the reference layout: touching the room, not a slab; a box whose top is
    below FLAT_Z is a covering or trim, not an obstacle; a box sunk below the floor is cut at the floor. A NON-structure
    box (unlabelled, wall-anchored, wall-hung, no-front) mostly covered by a placed object is that object's second
    annotation and is dropped; a structure box (column, stair, door, window...) is real whatever stands on it, so it
    is always kept (a reference layout through it shows up as a GT fixed collision)."""
    kept = [(Polygon(footprint(o)), o["pos"][2], o["pos"][2] + o["size"][2]) for o in objs]
    near = shell.buffer(WALL_BAND_M)
    wall = shell.buffer(THICK_WALL_M)      # a door or window sits in the wall: up to half a thick wall outside the floor
    out = []
    for f in fixed:
        z0, z1 = f["pos"][2], f["pos"][2] + f["size"][2]
        if z1 < FLAT_Z:
            continue
        if z0 < -FLOOR_Z:
            f, z0 = {**f, "z_src": z0, "pos": [f["pos"][0], f["pos"][1], 0.0], "size": [f["size"][0], f["size"][1], z1]}, 0.0
        fp = Polygon(footprint(f))
        if fp.area <= 0 or fp.area >= slab * shell.area or \
                not fp.intersects(wall if f.get("fixed_kind") == "structure" else near):
            continue
        over = [p for p, lo, hi in kept if min(z1, hi) - max(z0, lo) > 0.05]
        if f.get("fixed_kind") != "structure" and over and unary_union(over).intersection(fp).area >= cover * fp.area:
            continue
        out.append(f)
    return out


def prep(room, a):
    """IR room -> (clean room, None) or (None, reject reason). The clean room carries `fixed`: boxes the model is
    told about with their pose and never places (each tagged fixed_kind for stats):
      structure   doors, windows, columns, stairs... (anchors.fixed_geometry)
      generic     unlabelled boxes ('unknown', 'otherprop', 'object') standing on the floor
      wall        wall-anchored items whose box stands on the floor (a wall cabinet run from z=0)
      floating    wall-hung floor furniture in designer rooms (anchor "fixed" in anchors.infer_anchors)
      no_front    placeable boxes whose front the source does not give (InteriorGS custom cabinetry): the answer needs
                  every object's yaw, so they are shown as built-in instead of rejecting the room (flag no_front_fixed)
    all of them filtered by _keep_fixed. Tilted boxes are kept as upright boxes with their yaw; floor boxes sunk
    up to 30 cm and floor furniture floating up to 50 cm with nothing below (scans, or anything that cannot hang)
    are snapped to the floor; wall / ceiling / inside-cabinet items are dropped without rejecting the room."""
    meta = room.get("meta", {})
    if meta.get("front_known") is not True:
        return None, "front_unknown"
    if meta.get("n_incomplete"):                     # objects without a box would be invisible obstacles
        return None, "incomplete_objects"
    if len({o["id"] for o in room["objects"]}) != len(room["objects"]):
        return None, "duplicate_source_ids"
    if room.get("boundary_type") not in a.boundary_types or not room.get("boundary"):
        return None, "boundary"
    if reason := _room_geometry_error(room):
        return None, reason
    if not all(_finite(o) for o in room["objects"]):
        return None, "bad_numbers"
    fixed = [{**o, "fixed_kind": "structure"} for o in fixed_geometry(room)]
    allobj = annotate(room)                          # structure and soft covers out, anchors inferred
    objs = closed(allobj, a.source_anchors.get(room["source"], a.anchors))
    generic = {o["id"] for o in objs if GENERIC.match(norm_cat(o["category"])) or not norm_cat(o["category"])}
    sunk = {o["id"] for o in objs if not o.get("parent") and o["pos"][2] < -FURNITURE_SUNK_Z}
    objs = _drop_with_dependants(objs, generic | sunk)
    byid = {o["id"]: o for o in objs}                # keep only top-surface support
    objs = _drop_with_dependants(objs, [
        o["id"] for o in objs if o.get("parent") and (
            (p := byid.get(o["parent"])) is None or not support_ok(o, p))])
    nofront = {o["id"] for o in objs if o.get("front_known") is False}
    objs = _drop_with_dependants(objs, nofront)      # shown as fixed below (with what stands on them dropped)
    if any(min(o["size"]) < 0 for o in objs):        # only what is placed must have a size (0 = flat, shown as 1 cm)
        return None, "bad_numbers"
    kept = {o["id"] for o in objs}
    children = {o["id"] for o in allobj if o.get("parent")}
    # every other box that stands on the floor (or is a wall-hung unit, anchor "fixed") is told as fixed
    for o in allobj:
        nc = norm_cat(o["category"])
        if o["id"] in kept or o["id"] in children or o.get("parent") or o["size"][2] < FLAT_Z:
            continue
        if not (-FURNITURE_SUNK_Z <= o["pos"][2] <= FLOOR_Z or o["anchor"] == "fixed" or o["id"] in nofront):
            continue
        if hanging(o["category"]) or WALL_ART.search(nc) or head(o) in SOFT_COVERS:
            continue
        kind = "generic" if o["id"] in generic else "no_front" if o["id"] in nofront else \
            "floating" if o["anchor"] == "fixed" else "wall"
        fixed.append({**{k: v for k, v in o.items() if k not in ("anchor", "parent", "tilted", "anchor_inferred")},
                      "fixed_kind": kind})
    shell = Polygon(room["boundary"]).buffer(0)
    fixed = _keep_fixed(fixed, objs, shell)
    shown = kept | {f["id"] for f in fixed}
    # a dropped box that stood on the floor and is neither shown nor an opening / flush finish leaves an unexplained
    # hole in the reference layout; counted: its part on this room's floor that no placed object covers
    inner = shell.buffer(-WALL_BAND_M)
    # what explains a dropped box: a placed object or a fixed box standing on it (a rug does not)
    covered = unary_union([Polygon(footprint(o)) for o in objs if not o.get("parent") and o["size"][2] >= FLAT_Z]
                          + [Polygon(footprint(f)) for f in fixed if f["size"][2] >= FLAT_Z])
    polygon = room.get("boundary_type") == "polygon"          # (a rug does not explain a hole in the layout)
    low = lambda o: o["pos"][2] <= FLOOR_Z and o["pos"][2] + o["size"][2] >= FLAT_Z   # stands on / sticks out of it
    hidden = sum(Polygon(footprint(o)).intersection(shell).difference(covered).area for o in room["objects"]
                 if o["id"] not in shown and o["id"] not in children and not o.get("parent") and low(o)
                 and not hanging(o["category"])
                 and not WALL_ART.search(norm_cat(o["category"])) and not _flush(o, inner, polygon))
    if len(objs) < a.min_objects:
        return None, "object_count"
    # snap to the DISPLAYED numbers (cm, integer degrees) so checks, constraints and eval see what the model sees;
    # z: floor 0, child = parent top; fixed boxes keep their own z (a window sill), floor-level ones go to 0
    rounded = lambda o: {**o, "size": shown_size(o), "yaw": math.radians(round(math.degrees(o["yaw"])) % 360)}
    objs = [{**rounded(o), **({"z_src": o["pos"][2]} if not o.get("parent") and abs(o["pos"][2]) > 1e-9 else {})}
            for o in objs]                            # provenance: the source z of a floor box snapped to 0
    fixed = [{**rounded(o), "pos": [num(o["pos"][0]), num(o["pos"][1]),
                                    0 if -FLOOR_Z <= o["pos"][2] <= FLOOR_Z else num(o["pos"][2])]} for o in fixed]
    byid = {o["id"]: o for o in objs}
    done = set()

    def z_of(o):
        if o["id"] not in done:
            p = byid.get(o.get("parent"))
            o["pos"] = [num(o["pos"][0]), num(o["pos"][1]), num(z_of(p) + p["size"][2]) if p else 0]
            done.add(o["id"])
        return o["pos"][2]
    for o in objs:
        z_of(o)
    b = clean_boundary(room["boundary"])
    if b is None or (a.max_vertices and len(b) > a.max_vertices) or Polygon(b).area < MIN_AREA_M2:
        return None, "boundary_shape"
    out = {**room, "boundary": b, "objects": objs, "fixed": fixed}
    # quality FLAGS (the room is kept; every flag is counted in stats.json and written next to the sample so a
    # training sampler can weight or exclude it; --reject_flagged turns chosen flags back into rejections)
    flags = {}
    area = Polygon(b).area
    geometry = check(out, oob_tol=a.oob_tol)
    oob = geometry["oob"]
    if oob:                                          # reference objects crossing the walls by more than oob_tol
        flags["oob_objects"] = len(oob)
    if hidden > 0:                                   # unexplained floor area of dropped boxes (see above)
        flags["hidden_m2"] = round(hidden, 2)
    if hidden >= a.hidden_max:
        flags["hidden_obstacle"] = True
    if area < 1.0:
        flags["small_area_m2"] = round(area, 2)
    if any(not o.get("parent") and head(o) not in FLAT and o["size"][2] >= 0.5
           and o["size"][0] * o["size"][1] >= 0.8 * area for o in objs):   # composite built-in boxing the room
        flags["room_filling_object"] = True
    if _bad_overlap(objs):                           # copies of one wardrobe through each other
        flags["overlapping_furniture"] = True
    through = geometry["fixed_blocking"]            # reuse the same box intersections for fixed obstacles
    if through:
        flags["fixed_collision"] = len(through)
    if len(objs) == 1:
        flags["single_object"] = True
    snapped = sum(abs(o.get("z_src") or 0) > FLOOR_Z for o in objs)   # beyond ordinary floor noise
    if snapped:
        flags["z_snapped"] = snapped
    if any(o.get("tilted") for o in objs):
        flags["tilted"] = sum(bool(o.get("tilted")) for o in objs)
    if nofront:
        flags["no_front_fixed"] = len(nofront)
    hit = [k for k in a.reject_flagged if k in flags]
    if hit:
        return None, "flagged:" + hit[0]
    out["meta"] = {**out.get("meta", {}), "flags": flags}
    h = out.get("height")
    if h and any(o["pos"][2] + o["size"][2] > h + 0.05 for o in objs):   # height contradicts the boxes: unknown
        out = {**out, "height": None, "meta": {**out["meta"], "height_dropped": True}}
    elif h:
        out = {**out, "height": num(h)}              # the value the model is shown
    return out, None


def _bad_overlap(objs):
    """Copies of the same furniture that pass through each other (3D box overlap over 30% of the smaller footprint):
    a wardrobe or bookcase entered twice in the source. Different categories are not judged here: at box level an
    appliance built into a cabinet run, a loft bed over its desk or an L-sofa's box over its table look the same as a
    real interpenetration, so those stay collisions reported by evaluate against the GT rate."""
    fl = [o for o in objs if not o.get("parent") and head(o) not in FLAT | SOFT_HEADS
          and not COMPOSITE.search(norm_cat(o["category"]))]
    fps = [Polygon(footprint(o)) for o in fl]
    for i, a in enumerate(fl):
        for j in range(i + 1, len(fl)):
            b = fl[j]
            if norm_cat(a["category"]) != norm_cat(b["category"]):
                continue
            if min(a["pos"][2] + a["size"][2], b["pos"][2] + b["size"][2]) - max(a["pos"][2], b["pos"][2]) <= 0.05:
                continue
            small, big = sorted((fps[i].area, fps[j].area))
            if fps[i].intersection(fps[j]).area > 0.3 * small and small >= big / 3:
                return True
    return False


def exact_key(room):
    """Same boundary and the same objects and fixed boxes (category, size, pose) up to a 90-degree turn."""
    keys = []
    for k in range(4):
        r = rot90(room, k)
        pose = lambda o, c: (c, tuple(o["size"]), num(o["pos"][0]), num(o["pos"][1]), num(o["pos"][2]),
                             round(math.degrees(o["yaw"])) % 360)
        keys.append(repr((r["boundary"], sorted(pose(o, norm_cat(o["category"])) for o in r["objects"]),
                          sorted(pose(o, slug(o["category"])) for o in r.get("fixed") or []))))
    return hashlib.sha1(min(keys).encode()).hexdigest()


def dedup(rooms):
    """Same group in several sources (e.g. a ScanNet scan in Scan2CAD and InternScenes): keep the source whose
    rooms of that group keep more objects; ties go to a polygon boundary, then source name."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rooms:
        by[r["group"]][r["source"]].append(r)
    keep, dropped = [], collections.Counter()
    for g, srcs in by.items():
        best = max(srcs, key=lambda s: (sum(len(r["objects"]) for r in srcs[s]),
                                        any(r["boundary_type"] == "polygon" for r in srcs[s]), s))
        keep += srcs[best]
        for s, rs in srcs.items():
            if s != best:
                dropped[s] += len(rs)
    return keep, dropped


def rng_for(uid, salt):
    return random.Random(int(hashlib.sha1(f"{salt}:{uid}".encode()).hexdigest(), 16))


def extract_constraints(room, rng, max_k=4):
    """1..max_k relations that hold on the reference layout, one per type first (round robin, rarest type first).
    Only a few relations are given, never the full relation graph: the model must still lay out everything else."""
    b = room["boundary"]
    objs = room["objects"]
    fps = {o["id"]: Polygon(footprint(o)) for o in objs}
    cand = collections.defaultdict(list)
    objs = [o for o in objs if head(o) not in FLAT and o["size"][2] >= 0.05]  # rugs/flat boxes: not meaningful arguments
    idx = {o["id"]: o for o in objs}                  # built once, not in every holds() call
    for o in objs:
        if o.get("parent"):
            cand["on"].append(["on", o["id"], o["parent"]])
        if room["boundary_type"] == "polygon" and not o.get("parent") and holds(["against_wall", o["id"]], objs, b, idx):
            cand["against_wall"].append(["against_wall", o["id"]])
    for x in objs:
        for y in objs:
            if x is y or x.get("parent") or y.get("parent"):
                continue
            if head(x) in FRONT and not fps[x["id"]].intersects(fps[y["id"]]) and \
                    holds(["faces", x["id"], y["id"]], objs, b, idx):
                cand["faces"].append(["faces", x["id"], y["id"]])
            if x["id"] < y["id"]:
                gap = fps[x["id"]].distance(fps[y["id"]])
                d = next((d for d in (0.1, 0.2, 0.3, 0.5) if d >= gap), None)
                small = min(fps[x["id"]].area, fps[y["id"]].area)
                if small > 0 and fps[x["id"]].intersection(fps[y["id"]]).area > 0.3 * small:
                    continue                   # a collision or a duplicate annotation, not a "near" relation
                if d is not None and holds(["near", x["id"], y["id"], d], objs, b, idx):   # the one checker decides
                    cand["near"].append(["near", x["id"], y["id"], d])
    floor = [o for o in objs if not o.get("parent")]
    cat = [norm_cat(o["category"]) for o in floor]
    # only footprints the p-q segment can touch are tested (STRtree bbox query, then the same checks in floor order):
    # the same candidates as testing every m, without the O(n^3) sweep (a 965-book library room took hours)
    # the checks of holds(["between", m, p, q]) (segment p-q crosses m's footprint, not only touching it) and of the
    # trivial-endpoint skip, evaluated vectorized on those candidates; same results, same floor order
    geo = np.array([fps[m["id"]] for m in floor], dtype=object)
    tree = STRtree(list(geo))
    for i, p in enumerate(floor):
        for j in range(i + 1, len(floor)):
            if cat[i] != cat[j]:
                continue
            q = floor[j]
            seg = LineString([p["pos"][:2], q["pos"][:2]])
            ks = np.sort(tree.query(seg))
            if not len(ks):
                continue
            g = geo[ks]
            ok = ~shapely.intersects(g, Point(p["pos"][:2])) & ~shapely.intersects(g, Point(q["pos"][:2])) \
                & shapely.intersects(g, seg) & ~shapely.touches(g, seg) & (ks != i) & (ks != j)
            for k in ks[ok]:
                cand["between"].append(["between", floor[k]["id"], p["id"], q["id"]])
    types = [t for t in PRIORITY if cand[t]]         # rarest types first, so all five are covered
    for t in types:
        rng.shuffle(cand[t])
    k, out = rng.randint(1, max_k), []
    while len(out) < k and any(cand[t] for t in types):
        for t in types:
            if cand[t] and len(out) < k:
                out.append(cand[t].pop())
    return out


def count_constraints(st, split, room, cons):
    """stats: constraints per type; 'on' split by evidence (support given by the source vs box-contact inference)."""
    objs = {o["id"]: o for o in room["objects"]}
    for c in cons or []:
        t = c[0]
        if t == "on":
            t += "(inferred)" if objs[c[1]].get("anchor_inferred") else "(source)"
        st[f"{split}_constraints:{t}"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True, help="dir of <source>.jsonl from fastfill.adapters")
    ap.add_argument("--out", required=True, help="new dataset directory (must be absent or empty)")
    ap.add_argument("--sources", nargs="*", default=None, help="default: every file in --ir")
    ap.add_argument("--min_objects", type=int, default=1, help="rooms with fewer placeable objects have no target")
    ap.add_argument("--max_vertices", type=int, default=0, help="reject rooms with more boundary vertices (0: no limit)")
    ap.add_argument("--reject_flagged", nargs="*", default=[],
                    help="quality flags that reject a room instead of being recorded (oob_objects hidden_obstacle "
                         "small_area_m2 room_filling_object overlapping_furniture single_object)")
    ap.add_argument("--keep_duplicates", action="store_true", help="keep exact duplicate layouts inside a split")
    ap.add_argument("--boundary_types", nargs="*", default=["polygon", "hull"])
    ap.add_argument("--anchors", nargs="*", default=["floor", "object"], help="object anchors kept (see anchors.py)")
    ap.add_argument("--oob_tol", type=float, default=0.10, help="flag oob_objects: reference objects outside boundary+tol")
    ap.add_argument("--hidden_max", type=float, default=0.3, help="flag hidden_obstacle: m2 of unexplained dropped floor boxes")
    ap.add_argument("--cap", nargs="*", default=[], help="SOURCE=N: keep N train rooms of SOURCE (hash-sampled); "
                                                          "default none: balance sources at sampling time, not here")
    ap.add_argument("--source_anchors", nargs="*", default=[], help="SOURCE=a,b: anchors kept for that source")
    ap.add_argument("--dev", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--constraint_frac", type=float, default=0.0, help="share of train rooms given as S2 (v1.1: 0.3)")
    ap.add_argument("--no_rot90", action="store_true")
    ap.add_argument("--field_dropout", type=float, default=0.2, help="train: drop room_type / height each with p")
    ap.add_argument("--desc_dropout", type=float, default=0.5, help="train: drop all desc of a room with p")
    a = ap.parse_args()
    a.source_anchors = {k: v.split(",") for k, v in (x.split("=") for x in a.source_anchors)}
    caps = {k: int(v) for k, v in (x.split("=") for x in a.cap)}

    files = _input_files(a, ap)
    _validate_destination(a.out)
    # Hash only implementation modules, excluding local audit/test artifacts.
    here = Path(__file__).resolve().parent
    code_files = _code_files(here)
    code_sha = {str(p.relative_to(here)): _sha256(p) for p in code_files}
    ir_sha = {p.name: _sha256(p) for p in files}
    rooms, stats = [], collections.defaultdict(collections.Counter)
    for f in files:
        for line in _json_lines(f):
            r0 = line
            st = stats[r0["source"]]
            r, why = prep(r0, a)
            st["scanned"] += 1
            if why:
                st["rejected:" + why] += 1
                continue
            rooms.append({**r, "meta": {**r["meta"], "n_src_objects": len(r0["objects"])}})
        print(f"prepared {f.name}: {len(rooms)} retained rooms so far", flush=True)
    if not rooms:
        ap.error("no usable rooms; output was not changed")
    rooms, dropped = dedup(rooms)
    for s, n in dropped.items():
        stats[s]["dedup_dropped"] += n

    splits = assign_splits(rooms, a.dev, a.test)
    # exact duplicates (same boundary, and the same objects and fixed boxes at the same poses, up to a 90-degree
    # turn) stay one room per split: 3D-FRONT/SpatialLM repeat whole designs up to 26 times
    first = {}
    for i in sorted(range(len(rooms)), key=lambda i: hashlib.sha1(rooms[i]["uid"].encode()).hexdigest()):
        key = (splits[i], exact_key(rooms[i]))
        if key in first and not a.keep_duplicates:
            stats[rooms[i]["source"]]["duplicate_layout"] += 1
            splits[i] = "duplicate"
        else:
            first[key] = i
    for src, n in caps.items():                    # hash order, not file order (files are sorted by type)
        idx = sorted((i for i, r in enumerate(rooms) if r["source"] == src and splits[i] == "train"),
                     key=lambda i: hashlib.sha1(rooms[i]["uid"].encode()).hexdigest())
        for i in idx[n:]:
            splits[i] = "capped"
    if not any(s in ("train", "dev", "test") for s in splits):
        ap.error("no rooms remain after filtering; output was not changed")
    destination = Path(a.out).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as staging:
        _write_dataset(rooms, splits, stats, a, Path(staging), code_sha, ir_sha)
        if any(_sha256(p) != ir_sha[p.name] for p in files):
            raise RuntimeError("input IR changed during build; output was not published")
        dependencies = {"__init__.py", "build.py", "anchors.py", "scene.py", "split.py", "validate.py"}
        current_code = {str(p.relative_to(here)): _sha256(p) for p in _code_files(here)}
        if any(current_code.get(name) != code_sha.get(name) for name in dependencies):
            raise RuntimeError("implementation changed during build; output was not published")
        # Serving/training tools do not participate in the build. Record their release-time versions,
        # retaining the start snapshot so concurrent unrelated edits are not attributed to data generation.
        manifest_path = Path(staging) / "MANIFEST.json"
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump({**manifest, "code_sha256": current_code, "build_start_code_sha256": code_sha},
                      f, indent=1, allow_nan=False)
        _validate_destination(destination)
        os.rename(staging, destination)
    for k, v in sorted(stats.items()):
        print(f"{k:24s} {dict(v)}")


def _write_dataset(rooms, splits, stats, a, directory, code_sha, ir_sha):
    with ExitStack() as stack:
        open_output = lambda name: stack.enter_context(open(directory / f"{name}.jsonl", "w", encoding="utf-8"))
        outs = {s: open_output(s) for s in ("train", "dev", "test")}
        ir_outs = {s: open_output(s) for s in
                   ("dev_rooms", "test_rooms", "dev_constrained_rooms", "test_constrained_rooms")}
        for r, s in zip(rooms, splits):
            st = stats[r["source"]]
            if s in ("capped", "duplicate"):
                st["capped"] += s == "capped"
                continue
            # composition of what is WRITTEN (after dedup, duplicates and caps)
            st["height_dropped"] += bool(r.get("meta", {}).get("height_dropped"))
            st["objects_dropped"] += r["meta"]["n_src_objects"] - len(r["objects"]) - len(r["fixed"])
            st["fixed_items"] += len(r["fixed"])
            for f in r["fixed"]:
                st["fixed:" + f["fixed_kind"]] += 1
            st["rooms_with_fixed"] += bool(r["fixed"])
            st["objects_max"] = max(st["objects_max"], len(r["objects"]))
            for k in r["meta"]["flags"]:
                st["flag:" + k] += 1
            st["supports_inferred"] += sum(bool(o.get("anchor_inferred")) for o in r["objects"] if o.get("parent"))
            st["supports_evidence"] += sum(not o.get("anchor_inferred") for o in r["objects"] if o.get("parent"))
            if s == "train":
                # one random stream per purpose: --constraint_frac changes nothing but the constraints (clean ablation:
                # v1.1 minus its constraints is v1.0 byte for byte)
                aug, s2, drop = (rng_for(r["uid"], k) for k in ("aug", "s2", "dropout"))
                c = canonical(r if a.no_rot90 else rot90(r, aug.randrange(4)))
                cons = extract_constraints(c, s2) if s2.random() < a.constraint_frac else None
                st["train_S2" if cons else "train_S1"] += 1
                count_constraints(st, "train", c, cons)
                if drop.random() < a.field_dropout:
                    c = {**c, "room_type": None}
                if drop.random() < a.field_dropout:
                    c = {**c, "height": None}
                m = messages(c, cons, with_desc=drop.random() >= a.desc_dropout)
            else:
                c = canonical(r)
                st[s] += 1
                m = messages(c)
                ir_outs[f"{s}_rooms"].write(json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n")
                cons = extract_constraints(c, rng_for(r["uid"], "eval"))
                if cons:
                    count_constraints(st, s, c, cons)
                    ir_outs[f"{s}_constrained_rooms"].write(json.dumps({**c, "constraints": cons}, ensure_ascii=False, allow_nan=False) + "\n")
            outs[s].write(json.dumps({"uid": r["uid"], "source": r["source"], "flags": r["meta"]["flags"], "messages": m},
                                     ensure_ascii=False, allow_nan=False) + "\n")
    with open(directory / "stats.json", "w", encoding="utf-8") as f:
        json.dump({k: dict(v) for k, v in sorted(stats.items())}, f, indent=1, allow_nan=False)
    manifest = {"args": vars(a),
                "files": {p.name: {"sha256": _sha256(p), "lines": _line_count(p)}
                          for p in sorted(directory.iterdir()) if p.suffix == ".jsonl" or p.name == "stats.json"},
                "code_sha256": code_sha, "ir_sha256": ir_sha}
    with open(directory / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _code_files(here):
    return sorted(p for p in here.rglob("*.py")
                  if not any(part.startswith(("review", ".", "__pycache__")) or part == "tests"
                             for part in p.relative_to(here).parts[:-1]))


def _json_lines(path):
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc


def _input_files(a, ap):
    if not Path(a.ir).is_dir():
        ap.error(f"IR directory does not exist: {a.ir}")
    files = sorted(Path(a.ir).glob("*.jsonl"))
    if a.sources is not None:
        missing = set(a.sources) - {p.stem for p in files}
        if missing:
            ap.error("missing IR sources: " + ", ".join(sorted(missing)))
        files = [p for p in files if p.stem in a.sources]
    if not files or any(p.stat().st_size == 0 for p in files):
        ap.error("selected IR files must exist and be nonempty")
    if any(not 0 <= x <= 1 for x in (a.dev, a.test, a.constraint_frac, a.field_dropout, a.desc_dropout)) or \
            a.dev + a.test > 1:
        ap.error("split/dropout/constraint fractions must be in [0, 1], with dev + test <= 1")
    return files


def _validate_destination(path):
    path = Path(path)
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise FileExistsError(f"refusing to replace existing dataset: {path}; choose a new version directory")


if __name__ == "__main__":
    main()
