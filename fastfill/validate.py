"""Geometric checks for a placed room (IR format, see scene.py). Box-level, upright objects.

valid = no out-of-bounds object, no support failure, nothing through the ceiling. Collisions are reported separately in buckets
because reference layouts themselves collide at box level (chairs tucked under tables, nightstands against
beds, boxes inside cabinets): 8-18% of GT rooms still have large "main" overlaps, so collisions are only
meaningful relative to the GT value on the same rooms.
  oob           footprint leaves boundary.buffer(oob_tol) (only for boundary_type polygon/hull)
  support_fail  parent missing, bottom not within tol of the parent top, center off the parent footprint,
                or no parent and |z| > tol (floating / sunk)
                Support distances are inclusive: <= tol + SUPPORT_EPS, in XY and Z, including the floor.
  collisions    pairs with footprint overlap > min_area and height overlap > tol, supports excluded:
                rugs / mats / boxes under 6 cm are ignored;
                tuck (seat x table, bed x nightstand), contained (a smaller box inside another), main (the
                rest, including a duplicate: same category over half their union, or near-equal footprints),
                severe (main pairs whose overlap covers > 30% of the smaller footprint)
  oor           OptiScene's object overlap rate (axis-aligned w x d boxes, no rotation), for comparison only
  fixed_collisions  (object, fixed box) pairs overlapping like a collision: placements on a column, in a doorway
                or through a stair; reported next to the GT value, not part of valid (reference layouts have some)
  fixed_blocking    the fixed_collisions whose fixed box is not a window: furniture in front of a window (its box
                sticks 5 cm into the room) is normal, a table through a column or a doorway is not. serve.py fails a
                layout on these, build flags reference layouts with them (flag fixed_collision)
"""
import math

from shapely.geometry import LineString, Point, Polygon

from fastfill.anchors import FLAT
from fastfill.scene import footprint, head, norm_cat

SEAT = {"chair", "armchair", "stool", "bench", "seat", "ottoman", "pouf"}
TABLE = {"table", "desk", "counter", "island", "bar", "workstation", "vanity", "workbench"}
SUPPORT_EPS = 1e-9  # meters: floating-point allowance, not an additional geometric tolerance


def support_contains(parent, xy, tol=0.05):
    """Whether XY is within tol + SUPPORT_EPS of the parent's footprint, inclusive.

    Use Euclidean distance, not a polygonal buffer approximation, so corners and edges share the same rule.
    Accepts IR parent geometry (pos, size, yaw) and an XY coordinate; also used during runtime repair.
    """
    return Polygon(footprint(parent)).distance(Point(xy[:2])) <= tol + SUPPORT_EPS


def support_ok(child, parent=None, tol=0.05):
    """Inclusive support check for IR objects; parent=None means the floor.

    Callers must reject an unresolved parent reference before passing None. Both the absolute bottom/top
    height error and Euclidean XY distance to the parent footprint must be <= tol + SUPPORT_EPS.
    """
    top = parent["pos"][2] + parent["size"][2] if parent is not None else 0.0
    return abs(child["pos"][2] - top) <= tol + SUPPORT_EPS and \
        (parent is None or support_contains(parent, child["pos"], tol))


def collision_kind(a, b, fa, fb, tol=0.05):
    ha, hb = head(a), head(b)
    if (ha in SEAT and hb in TABLE) or (hb in SEAT and ha in TABLE):
        return "tuck"
    na, nb = norm_cat(a["category"]), norm_cat(b["category"])
    if (ha == "bed" and ("night" in nb or "bedside" in nb)) or (hb == "bed" and ("night" in na or "bedside" in na)):
        return "tuck"
    # a copy of the same furniture on (almost) the same spot is a duplicate, not a box inside a cabinet
    small, big = sorted((fa.area, fb.area))
    if (na == nb and fa.intersection(fb).area >= 0.5 * fa.union(fb).area) or small >= 0.8 * big:
        return "main"
    for x, y, fx, fy in ((a, b, fa, fb), (b, a, fb, fa)):
        if fy.buffer(tol).contains(fx) and x["pos"][2] + x["size"][2] <= y["pos"][2] + y["size"][2] + tol:
            return "contained"
    return "main"


def check(room, tol=0.05, oob_tol=0.10, min_area=0.01, severe=0.3):
    objs = room["objects"]
    fps = [Polygon(footprint(o)) for o in objs]
    res = {"n": len(objs), "oob": [], "support_fail": [],
           "collisions": {"main": [], "severe": [], "tuck": [], "contained": []}}
    if room.get("boundary") and room.get("boundary_type") in ("polygon", "hull"):
        inner = Polygon(room["boundary"]).buffer(oob_tol + 1e-9)
        res["oob"] = [o["id"] for o, fp in zip(objs, fps) if not inner.contains(fp)]
    h = room.get("height")     # build drops a height that its own boxes exceed by > tol, so GT always passes
    res["ceiling"] = [o["id"] for o in objs if h and o["pos"][2] + o["size"][2] > h + tol]

    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            if a.get("parent") == b["id"] or b.get("parent") == a["id"]:
                continue
            if min(a["pos"][2] + a["size"][2], b["pos"][2] + b["size"][2]) - max(a["pos"][2], b["pos"][2]) <= tol:
                continue
            if FLAT.intersection((head(a), head(b))) or min(a["size"][2], b["size"][2]) < 0.06:
                continue
            area = fps[i].intersection(fps[j]).area
            if area > min_area:
                kind = collision_kind(a, b, fps[i], fps[j], tol)
                res["collisions"][kind].append((a["id"], b["id"]))
                if kind == "main" and area > severe * min(fps[i].area, fps[j].area):
                    res["collisions"]["severe"].append((a["id"], b["id"]))

    res["fixed_collisions"] = []
    for o, fp in zip(objs, fps):
        if head(o) in FLAT or o["size"][2] < 0.06:
            continue
        for f in room.get("fixed") or []:
            if min(o["pos"][2] + o["size"][2], f["pos"][2] + f["size"][2]) - max(o["pos"][2], f["pos"][2]) <= tol:
                continue
            if fp.intersection(Polygon(footprint(f))).area > min_area:
                res["fixed_collisions"].append((o["id"], f["id"]))
    windows = {f["id"] for f in room.get("fixed") or []
               if {"window", "windows"}.intersection(norm_cat(f.get("category")).split())}
    res["fixed_blocking"] = [p for p in res["fixed_collisions"] if p[1] not in windows]

    ids = {o["id"]: k for k, o in enumerate(objs)}
    for o in objs:
        k = ids.get(o.get("parent"))
        if not o.get("parent"):
            bad = not support_ok(o, tol=tol)
        elif k is None:
            bad = True
        else:
            bad = not support_ok(o, objs[k], tol)
        if bad:
            res["support_fail"].append(o["id"])

    boxes = []
    for o in objs:
        x, y, hx, hy = o["pos"][0], o["pos"][1], o["size"][0] / 2, o["size"][1] / 2
        boxes.append(Polygon([(x - hx, y - hy), (x + hx, y - hy), (x + hx, y + hy), (x - hx, y + hy)]))
    total = sum(b.area for b in boxes)
    inter = sum(boxes[i].intersection(boxes[j]).area for i in range(len(boxes)) for j in range(i + 1, len(boxes)))
    res["oor"] = inter / total if total > 0 else 0.0
    res["valid"] = not (res["oob"] or res["support_fail"] or res["ceiling"])
    return res


def holds(c, objs, boundary, index=None):
    """Constraint checker; semantics word for word as in scene.SYSTEM_PROMPT. index: an id -> object dict of objs,
    passed by callers that test many constraints on one room (build.extract_constraints)."""
    o = index if index is not None else {x["id"]: x for x in objs}
    fp = lambda i: Polygon(footprint(o[i]))
    t = c[0]
    if t == "on":
        return o[c[1]].get("parent") == c[2]
    if t == "faces":
        a, b = o[c[1]], o[c[2]]
        ang = math.atan2(b["pos"][1] - a["pos"][1], b["pos"][0] - a["pos"][0]) - a["yaw"]
        return abs((ang + math.pi) % (2 * math.pi) - math.pi) <= math.radians(30) + 1e-9
    if t == "near":
        return fp(c[1]).distance(fp(c[2])) <= c[3] + 1e-9
    if t == "against_wall":     # every point of the side, not samples of it
        q, zone = footprint(o[c[1]]), Polygon(boundary).exterior.buffer(0.1 + 1e-9)
        return any(zone.contains(LineString([q[i], q[(i + 1) % 4]])) for i in range(4))
    if t == "between":                # the segment runs through a's footprint, not just touching its edge
        seg, f = LineString([o[c[2]]["pos"][:2], o[c[3]]["pos"][:2]]), fp(c[1])
        return seg.intersects(f) and not seg.touches(f)
    raise ValueError(t)


if __name__ == "__main__":
    room = {"boundary": [[0, 0], [4, 0], [4, 3], [0, 3]], "boundary_type": "polygon",
            "objects": [{"id": "bed", "category": "bed", "size": [2.0, 1.6, 0.5], "pos": [1.1, 1.5, 0.0], "yaw": 0.0},
                        {"id": "desk", "category": "desk", "size": [1.2, 0.6, 0.75], "pos": [3.6, 0.8, 0.0], "yaw": math.pi / 2},
                        {"id": "lamp", "category": "lamp", "size": [0.2, 0.2, 0.4], "pos": [3.6, 0.8, 0.75], "yaw": 0.0, "parent": "desk"},
                        {"id": "chair", "category": "chair", "size": [.5, .5, .9], "pos": [3.2, 0.8, 0.0], "yaw": 0.0}]}
    r = check(room)
    assert r["valid"] and r["collisions"]["tuck"] == [("desk", "chair")] and not r["collisions"]["main"], r
    # a column through the bed and a window on the south wall behind the desk: only the column is a fixed collision
    fx = [{"id": "col", "category": "column", "size": [.3, .3, 2.8], "pos": [1.5, 1.5, 0], "yaw": 0},
          {"id": "win", "category": "window", "size": [1, .1, 1.2], "pos": [3.6, 0.05, 0.9], "yaw": 0}]
    assert check({**room, "fixed": fx})["fixed_collisions"] == [("bed", "col")] and check({**room, "fixed": fx})["valid"]
    tall = [{"id": "win2", "category": "window", "size": [1.5, .1, 1.2], "pos": [1.1, 2.3, 0.4], "yaw": 0}]   # behind the bed
    r2 = check({**room, "fixed": fx + tall})
    assert ("bed", "win2") in r2["fixed_collisions"] and r2["fixed_blocking"] == [("bed", "col")], r2
    room["objects"][1]["yaw"] = 0.0                    # unrotated desk spans x 3.0..4.2: out of bounds (tol 0.1)
    r = check(room)
    assert r["oob"] == ["desk"] and not r["valid"], r
    room["objects"][0]["pos"] = [2.9, 1.0, 0.0]        # bed moved onto the desk area: main + severe collision
    r = check(room)
    assert ("bed", "desk") in r["collisions"]["main"] and ("bed", "desk") in r["collisions"]["severe"], r
    room["objects"][2]["pos"][2] = 0.3                 # lamp sunk into the desk
    assert check(room)["support_fail"] == ["lamp"]
    assert check({"objects": [{"id": "l", "category": "lamp", "size": [.2, .2, .4], "pos": [1, 1, 0.3], "yaw": 0}]})["support_fail"] == ["l"]
    stack = {"height": 2.5, "objects": [{"id": "w", "category": "wardrobe", "size": [1, .6, 2.2], "pos": [1, 1, 0], "yaw": 0},
                                        {"id": "b", "category": "box", "size": [.4, .4, .8], "pos": [1, 1, 2.2], "yaw": 0, "parent": "w"}]}
    r = check(stack)                                   # top at 3.0 m under a 2.5 m ceiling
    assert r["ceiling"] == ["b"] and not r["valid"] and check({**stack, "height": None})["valid"], r
    b = [[0, 0], [4, 0], [4, 3], [0, 3]]
    a = [{"id": "a", "size": [1, .5, 1], "pos": [3.75, 1.5, 0], "yaw": math.pi / 2}]
    assert holds(["against_wall", "a"], a, b) and not holds(["against_wall", "a"], [{**a[0], "pos": [3.5, 1.5, 0]}], b)
    two = [{"id": "c1", "size": [.5, .5, .9], "pos": [1, 1, 0], "yaw": 0}, {"id": "c2", "size": [.5, .5, .9], "pos": [3, 1, 0], "yaw": math.pi}]
    assert holds(["faces", "c1", "c2"], two, b) and holds(["faces", "c2", "c1"], two, b) and holds(["near", "c1", "c2", 1.5], two, b)
    print("validate.py self-check ok")
