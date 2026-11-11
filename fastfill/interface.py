"""WorldEdge request (scope doc, Appendix B) <-> FastFill model text.

    room, ctx = request_to_room(request)
    prompt = prompt_text(tok, messages(room, room["constraints"], with_target=False))
    ... one generation call ...
    result = placements_from_text(text, room, ctx)

Model side (scene.py): room frame with the boundary AABB min corner at (0, 0), object local +X = semantic front,
integer-degree yaw inside the text. Interface side: the request's own frame (boundary_xy, floor_z), position_m of
the box bottom center, yaw_rad about +Z of the object's own local frame, support_parent = floor_id or the id of the
object it stands on, support_surface = "top". Objects anchored to walls or ceilings ("anchor": "wall" | "ceiling")
and fronts that are not along a horizontal local axis are returned as unsupported instead of being sent to the model,
and so is any constraint that names one of them.
Floor/object anchors are enforced after generation and reported in validation.constraints as ["anchor", id, kind],
with hard=true, trained=false, model_conditioned=false: the existing model protocol cannot express them.
room.fixed_geometry (optional): boxes already in the room that nothing may be placed on or through (doors, windows,
columns, stairs, furniture that stays), each {"id", "category"?, "size_xyz_m", "position_m" (bottom center),
"yaw_rad"?, "fixed_kind"?}; shown to the model as `fixed`, never placed, and overlaps with them are reported in validation
(fixed_collisions), and any such overlap with a non-window box fails the layout (fixed_blocking, serve.py 422). Normalized as build.prep shaped training: |z| <= FLOOR_Z is snapped to
0; a box starting at or above FIXED_MAX_Z, covering half the room or more, or standing clear of the walls is not shown
and is reported in unsupported as "fixed_ignored". The wall band is 0.30 m for fixed_kind="structure", otherwise
0.10 m; if fixed_kind is omitted, anchors.is_structure(category) selects the band. An explicit kind takes precedence.
A constraint may not name a fixed box (v1 relates objects to place).

layout_constraints (Appendix B names). Relational constraints are shown to the model as must-hold; after repair a failed
"hard" one (the default; "hard": false marks a soft one, which is only reported) fails the layout:
    {"type": "faces", "subject", "target"}         {"type": "between", "subject", "anchors": [a, b]}
    {"type": "supported_by", "subject", "parent", "surface": "top"}      {"type": "against_wall", "subject"}
    {"type": "near", "subject", "target", "max_gap_m"}   footprint gap, NOT centre distance (trained on 0.1-0.5)
    {"type": "keepout", "polygon_xy": [[x, y], ...]}      hard: cut out of the floor, padded by KEEPOUT_PAD
Soft keepouts leave the floor unchanged. They are reported as ["keepout", polygon_xy] in request coordinates,
with hard=false, trained=false, model_conditioned=false; the existing model protocol has no advisory region field.
A soft keepout holds when no placed footprint overlaps it by more than 1e-9 square meters (touching is allowed).
against_wall is accepted against the real walls, not keepout cuts, although the model sees a cut as wall.
A padded keepout that stops less than WALL_SNAP short of a wall (or of another keepout) is snapped to it: the strip
left between them is dropped from the floor the model sees (cm rounding folded one under 5 mm into a self-intersection,
a bad_boundary 422, for 35 of 20,000 EmbodiedGen-shaped requests); floor cut off behind such a gap is a pocket, and a
gap that alone would cut off POCKET_M2 or more is left open. A cut that still folds under cm rounding (a gap under 5 mm
to a large corner, a keepout crossing a wall by under 5 mm) is keepout_not_representable, not bad_boundary.
Forms training never contains (build.extract_constraints) are still shown and checked as asked, but reported with
"trained": false in validation: against_wall in a hull room, faces whose subject's category head is not in
build.FRONT, against_wall / near / faces / between naming an object that stands on another, between with anchors of
two categories, an argument (of supported_by: the subject) that is a FLAT covering or under 5 cm tall.
Rejected with ValueError (HTTP 422 in serve.py): malformed or non-finite numbers, a coordinate_system other than
m / right / Z / bbox_bottom_center / rad, duplicate ids, unknown anchors, non-positive sizes or height, constraints
naming unknown objects, fixed boxes or themselves, 'on' cycles or two parents, a keepout that splits the room or leaves an
island (enclosed pockets under POCKET_M2 are treated as unusable floor).
"""
import copy
import math

from shapely.affinity import translate
from shapely.errors import ShapelyError
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points, unary_union

from fastfill.anchors import FIXED_MAX_Z, FLAT, FLOOR_Z, is_structure
from fastfill.build import FRONT, THICK_WALL_M, WALL_BAND_M

FLAT_Z = 0.15        # = build.FLAT_Z: a sunk fixed box whose top stays under this is not shown (checked in __main__)
from fastfill.scene import apply, canonical, clean_boundary, footprint, head, norm_cat, num, parse
from fastfill.validate import check, holds, support_contains

HULL_SOURCES = {"convex_hull", "hull", "scan_hull", "floor_hull"}
KEEPOUT_PAD = 0.08   # < validate.check's oob tolerance (0.10) minus cm rounding: repair onto the keepout edge stays valid
POCKET_M2 = 0.5      # free floor enclosed by keepouts and walls smaller than this is dropped, not a reason to refuse
WALL_SNAP = 0.01     # a strip of floor narrower than this left by a padded keepout (at a wall or keepout) is dropped
ANCHORS = {None, "floor", "object", "wall", "ceiling"}
COORDS = {"units": "m", "handedness": "right", "up_axis": "Z", "position_reference": "bbox_bottom_center",
          "yaw_units": "rad"}
SNAP_TOL = 0.10      # training layouts cross walls by up to this much (40% of test rooms by > 1 mm); EmbodiedGen allows 1 mm
SNAP_GRID = sorted(((dx / 100, dy / 100) for dx in range(-10, 11) for dy in range(-10, 11) if dx * dx + dy * dy <= 100),
                   key=lambda d: d[0] ** 2 + d[1] ** 2)


def request_to_room(req):
    """-> (canonical model room with constraints, ctx for mapping the answer back). Raises ValueError on bad input."""
    try:
        return _request_to_room(req)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, ShapelyError) as e:
        raise ValueError(f"bad_request: {e!r}") from e


def _finite(v, lo=None, above=None):
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) < 1e4
            and (lo is None or v >= lo) and (above is None or v > above)):
        raise ValueError(f"bad_number: {v!r}")
    return float(v)


def _text(v, optional=True):
    if not (isinstance(v, str) or (optional and v is None)):
        raise ValueError(f"bad_text: {v!r}")
    return v


def _request_to_room(req):
    rm = req["room"]
    for k, v in (req.get("coordinate_system") or {}).items():
        if k in COORDS and v != COORDS[k]:
            raise ValueError(f"unsupported coordinate_system {k}={v!r} (FastFill speaks {COORDS})")
    cons_in = req.get("layout_constraints") or []
    pts = rm["boundary_xy"]
    keepout_constraints = [c for c in cons_in if c["type"] == "keepout"]
    for c in keepout_constraints:
        if not isinstance(c.get("hard", True), bool):
            raise ValueError(f"malformed constraint: {c}")
    for p in [q for poly in [pts, *(c["polygon_xy"] for c in keepout_constraints)] for q in poly]:
        _finite(p[0]), _finite(p[1])
    for c in keepout_constraints:
        poly = Polygon(c["polygon_xy"])
        if not poly.is_valid or poly.is_empty or poly.area <= 0:
            raise ValueError("bad_keepout")
    keepouts = [c["polygon_xy"] for c in keepout_constraints if c.get("hard", True)]
    if rm.get("height_m") is not None:
        _finite(rm["height_m"], above=0)
    _text(rm.get("room_type"))
    walls = Polygon(pts)                    # against_wall is checked against these, not against keepout cuts
    floor = walls                           # what answers are snapped into: the walls minus the keepouts themselves
    if keepouts:        # cut out, padded, from what the model sees: it treats a keepout as wall
        if not floor.is_valid:
            raise ValueError("bad_boundary")
        cut = unary_union([Polygon(k) for k in keepouts])
        seen = floor.difference(cut.buffer(KEEPOUT_PAD, join_style="mitre"))
        # opening by WALL_SNAP / 2: strips under WALL_SNAP wide vanish, everything wider comes back edge for edge; the
        # intersection clips the mitre corners that re-grow past a slanted wall where a strip vanished. A gap whose
        # closing alone would cut off POCKET_M2 or more stays open (9 of 20,000 EmbodiedGen-shaped requests: two
        # padded keepouts 5-10 mm apart in front of a corner): the corner behind it stays floor, as without the snap
        snap = seen.intersection(seen.buffer(-WALL_SNAP / 2, join_style="mitre").buffer(WALL_SNAP / 2, join_style="mitre"))
        big = lambda g: sum(q.area >= POCKET_M2 for q in getattr(g, "geoms", [g]))
        seen = seen if big(snap) > 1 and big(seen) == 1 else snap
        parts = sorted(getattr(seen, "geoms", [seen]), key=lambda g: -g.area)
        pockets = [g for g in parts[1:] if g.area < POCKET_M2]
        if seen.is_empty or len(parts) - len(pockets) != 1 or parts[0].interiors:
            raise ValueError("keepout_not_representable")      # fills or splits the room, or leaves an island
        seen = parts[0]
        floor = floor.difference(unary_union([cut, *pockets]))
        pts = seen.exterior.coords[:-1]
    x0, y0 = (min(p[i] for p in pts) for i in (0, 1))
    boundary = clean_boundary([[x - x0, y - y0] for x, y in pts])
    if boundary is None:    # walls that round cleanly: the cut folds under cm rounding (a sub-5 mm gap or wall mouth)
        wx, wy = walls.bounds[:2]
        ok = keepouts and clean_boundary([[x - wx, y - wy] for x, y in walls.exterior.coords[:-1]])
        raise ValueError("keepout_not_representable" if ok else "bad_boundary")
    ctx = {"x0": x0, "y0": y0, "floor_z": _finite(rm.get("floor_z", 0.0)),
           "floor_id": _text(rm.get("floor_id", "floor"), optional=False),
           "floor": translate(floor, -x0, -y0), "walls": [[x - x0, y - y0] for x, y in walls.exterior.coords[:-1]],
           "soft_keepouts": [(copy.deepcopy(c["polygon_xy"]), translate(Polygon(c["polygon_xy"]), -x0, -y0))
                             for c in keepout_constraints if not c.get("hard", True)],
           "front_k": {}, "order": [], "unsupported": []}
    fixed, fixed_ids = [], set()
    near, wall = walls.buffer(WALL_BAND_M), walls.buffer(THICK_WALL_M)
    for f in rm.get("fixed_geometry") or []:
        _text(f["id"], optional=False), _text(f.get("category"))
        kind = _text(f.get("fixed_kind"))
        structural = kind == "structure" if kind is not None else is_structure(f.get("category"))
        sx, sy, sz = (_finite(v, above=0) for v in f["size_xyz_m"])
        x, y, z = (_finite(v) for v in f["position_m"])
        yaw = _finite(f.get("yaw_rad", 0.0)) % (2 * math.pi)
        fixed_ids.add(f["id"])
        z -= ctx["floor_z"]
        fp = Polygon(footprint({"pos": [x, y], "size": [sx, sy], "yaw": yaw}))
        # as build.prep / anchors.fixed_geometry shaped training: nothing starting at FIXED_MAX_Z or above, no slab
        # covering half the room, nothing clear of the walls; a floor-level bottom is z=0
        if z + sz < FLAT_Z:        # a covering or trim (or wholly under the floor): not shown, as in build._keep_fixed
            ctx["unsupported"].append({"id": f["id"], "reason": "fixed_ignored"})
            continue
        if z < -FLOOR_Z:           # sunk below the floor: cut at the floor
            sz, z = z + sz, 0.0
        if z >= FIXED_MAX_Z or fp.area >= 0.5 * walls.area or not fp.intersects(wall if structural else near):
            ctx["unsupported"].append({"id": f["id"], "reason": "fixed_ignored"})
            continue
        fixed.append({"id": f["id"], "ext_id": f["id"], "category": f.get("category") or "structure",
                      "size": [sx, sy, sz], "pos": [x - x0, y - y0, 0.0 if abs(z) <= FLOOR_Z else z], "yaw": yaw})
    ids = [o["id"] for o in req["objects_to_place"]] + [f["id"] for f in rm.get("fixed_geometry") or []]
    if len(set(ids)) < len(ids) or ctx["floor_id"] in ids:
        raise ValueError("duplicate_object_id")
    objs = []
    for o in req["objects_to_place"]:
        _text(o["id"], optional=False), _text(o["category"], optional=False), _text(o.get("desc"))
        sx, sy, sz = (_finite(v, above=0) for v in o["size_xyz_m"])
        if o.get("anchor") not in ANCHORS:
            raise ValueError(f"unknown anchor: {o.get('anchor')!r}")
        fx, fy, fz = (_finite(v) for v in (o.get("semantic_front_local") or [1, 0, 0]))
        phi, h = math.atan2(fy, fx), math.hypot(fx, fy)
        k = round(phi / (math.pi / 2)) % 4
        if o.get("anchor") in ("wall", "ceiling"):
            ctx["unsupported"].append({"id": o["id"], "reason": "unsupported_anchor"})
            continue
        if h < 1e-9 or abs(fz) > 1e-3 * h or abs(math.remainder(phi - k * math.pi / 2, 2 * math.pi)) > 1e-3:
            ctx["unsupported"].append({"id": o["id"], "reason": "front_not_axis_aligned"})
            continue
        if k % 2:                                   # front along local Y: the model's local X is that axis
            sx, sy = sy, sx
        ctx["front_k"][o["id"]] = k
        ctx["order"].append(o["id"])
        objs.append({"id": o["id"], "ext_id": o["id"], "category": o["category"], "size": [sx, sy, sz],
                     "desc": o.get("desc"), "anchor": o.get("anchor")})
    room = canonical({"room_type": rm.get("room_type"), "boundary": boundary, "height": rm.get("height_m"),
                      "boundary_type": "hull" if str(rm.get("boundary_source")).lower() in HULL_SOURCES else "polygon",
                      "fixed": fixed, "objects": objs})
    m = {o["ext_id"]: o["id"] for o in room["objects"]}
    skipped = {u["id"] for u in ctx["unsupported"]}
    cons, ctx["checks"] = [], []            # what the model is shown / what the answer is accepted against
    for c in cons_in:
        t = c["type"]
        if t == "keepout":
            continue
        s = c["subject"]
        anchors = c.get("anchors", [])
        if not isinstance(anchors, list) or not isinstance(c.get("hard", True), bool):
            raise ValueError(f"malformed constraint: {c}")
        refs = [s] + [c[k] for k in ("target", "parent") if k in c] + anchors
        if any(r in fixed_ids for r in refs):
            raise ValueError(f"constraint names a fixed box (v1 constraints only relate objects to place): {c}")
        if any(r not in m and r not in skipped for r in refs):
            raise ValueError(f"constraint on an unknown object: {c}")
        if len(set(refs)) < len(refs):
            raise ValueError(f"constraint names the same object twice: {c}")
        if any(r in skipped for r in refs):         # the rest of the room is still placed
            ctx["unsupported"].append({"constraint": c, "reason": "references_unsupported_object"})
            continue
        if t == "faces":
            k = ["faces", m[s], m[c["target"]]]
        elif t == "between" and len(c["anchors"]) == 2:
            k = ["between", m[s]] + [m[a] for a in c["anchors"]]
        elif t == "supported_by" and c.get("surface", "top") == "top":
            k = ["on", m[s], m[c["parent"]]]
        elif t == "near":
            k = ["near", m[s], m[c["target"]], _finite(c["max_gap_m"], lo=0)]
        elif t == "against_wall":
            k = ["against_wall", m[s]]
        else:
            raise ValueError(f"unsupported constraint: {c}")
        # the model sees cm (near rounded DOWN, so meeting what it sees meets the request); acceptance uses the
        # request's own number; hard=false constraints are reported but do not fail the layout
        cons.append(k[:3] + [num(math.floor(k[3] * 100 + 1e-9) / 100)] if t == "near" else k)
        ctx["checks"].append((k, c.get("hard", True)))
    parent = {}
    for k in cons:
        if k[0] == "on":
            if parent.setdefault(k[1], k[2]) != k[2]:
                raise ValueError(f"{k[1]} is asked to stand on two objects")
    for x in parent:                        # an 'on' cycle can never hold
        seen, cur = {x}, parent[x]
        while cur in parent:
            if cur in seen:
                raise ValueError(f"'on' cycle through {cur}")
            seen.add(cur)
            cur = parent[cur]
    room["constraints"] = cons
    return room, ctx


def snap_inside(pl, room, floor, cons=(), walls=None):
    """Move each object that crosses `floor` (model frame) by at most SNAP_TOL back inside, with what stands on it.
    Lower objects first, so a child is judged after its parent moved; a child is only moved as far as its centre
    stays over its parent (validate.check's support rule), else it is left for validation to reject. A move is never
    taken if it would deepen the overlap of the object OR of anything standing on it (the whole support subtree
    moves with it) with a fixed box of the room (room["fixed"], at that object's height).
    The hard constraints `cons` (validate.holds, against_wall against `walls`) that held before the repair hold after
    it: if the plain pass breaks one, the repair is redone refusing every move that breaks one of them. The plain pass
    goes first because a constraint may fail only between two moves (three chairs pushed off one wall in turn: the
    between through the middle one fails until it moves too). Without this, pushing chairs off walls broke 7 GT-true
    constraints in 8,254 test_constrained rooms (faces 4, against_wall 2, near 1)."""
    size = {o["id"]: o["size"] for o in room["objects"]}
    lay = lambda q: {j: {"pos": p["pos"], "yaw": p["yaw"], "size": size[j], "parent": p["on"]} for j, p in q.items()}
    held = [c for c in cons if holds(c, None, walls, lay(pl))]
    raw = copy.deepcopy(pl)
    _push(pl, room, floor, size, [], walls)
    if not all(holds(c, None, walls, lay(pl)) for c in held):
        pl.update(raw)
        _push(pl, room, floor, size, held, walls)


def _push(pl, room, floor, size, held, walls):
    inside = floor.buffer(1e-3)             # validation()'s tolerance
    fixed = room.get("fixed") or []
    kids = {}
    for i, p in pl.items():
        if p["on"]:
            kids.setdefault(p["on"], []).append(i)
    for i in sorted(pl, key=lambda i: pl[i]["pos"][2]):
        p = pl[i]
        box = lambda dx, dy: Polygon(footprint({"pos": [p["pos"][0] + dx, p["pos"][1] + dy], "size": size[i], "yaw": p["yaw"]}))
        if inside.covers(box(0, 0)):
            continue
        q = pl.get(p["on"])
        seat = {**q, "size": size[p["on"]]} if q else None
        push = [0.0, 0.0]              # per axis, the largest move any outside corner needs to reach the boundary
        for c in box(0, 0).exterior.coords[:-1]:
            if not floor.covers(Point(c)):
                q = nearest_points(floor.boundary, Point(c))[0]
                for k, d in enumerate((q.x - c[0], q.y - c[1])):
                    if abs(d) > abs(push[k]):
                        push[k] = d
        # the object and everything standing on it move together (parse() rejects support cycles); fixed boxes at
        # each one's height (validate.check's rule): no move may deepen any of their overlaps
        tree, todo = [], [i]
        while todo:
            j = todo.pop()
            tree.append(j)
            todo += kids.get(j, [])
        at = lambda j, dx, dy: Polygon(footprint({"pos": [pl[j]["pos"][0] + dx, pl[j]["pos"][1] + dy],
                                                  "size": size[j], "yaw": pl[j]["yaw"]}))
        blocks = {j: unary_union([Polygon(footprint(f)) for f in fixed if min(
            pl[j]["pos"][2] + size[j][2], f["pos"][2] + f["size"][2]) - max(pl[j]["pos"][2], f["pos"][2]) > 0.05])
            for j in tree}
        worst = {j: at(j, 0, 0).intersection(blocks[j]).area + 1e-6 for j in tree}
        mine = [c for c in held if set(tree) & set(c[1:])]

        def keeps(dx, dy):              # every held constraint on the subtree still holds after the move
            ix = {j: {"pos": [q["pos"][0] + dx, q["pos"][1] + dy, q["pos"][2]] if j in tree else q["pos"], "yaw": q["yaw"],
                      "size": size[j], "parent": q["on"]} for j, q in pl.items()} if mine else {}
            return all(holds(c, None, walls, ix) for c in mine)
        # the nearest-boundary push, its axis parts, then any move within SNAP_TOL (1 cm grid, shortest first)
        # (a nearest-boundary push over SNAP_TOL is dropped, the grid still tries every move within SNAP_TOL: a box
        # stuck at a re-entrant corner may need a long diagonal push but a short move along one axis)
        for dx, dy in [c for c in (tuple(push), (push[0], 0), (0, push[1])) if math.hypot(*c) <= SNAP_TOL] + SNAP_GRID:
            if inside.covers(box(dx, dy)) and all(at(j, dx, dy).intersection(blocks[j]).area <= worst[j] for j in tree) \
                    and (seat is None or support_contains(seat, [p["pos"][0] + dx, p["pos"][1] + dy])) and keeps(dx, dy):
                for j in tree:
                    pl[j]["pos"] = [pl[j]["pos"][0] + dx, pl[j]["pos"][1] + dy, pl[j]["pos"][2]]
                break


def placements_from_text(text, room, ctx):
    """Model answer -> {"placements": after repair, "raw_placements": as the model proposed, "validation":
    {"raw": ..., "repaired": ...}, "error": code | None, "unsupported": [...]}, placements in request order.
    Repair = floor objects crossing a wall or a keepout by at most SNAP_TOL moved back inside (snap_inside); it is
    deterministic post-processing, so the raw proposal and its validation are always returned next to it."""
    ids = [o["id"] for o in room["objects"]]
    pl, err = parse(text, ids)
    if err is not None:
        return {"placements": [], "raw_placements": [], "validation": None, "error": err,
                "unsupported": ctx["unsupported"]}
    raw = copy.deepcopy(pl)
    snap_inside(pl, room, ctx["floor"], [k for k, hard in ctx["checks"] if hard], ctx["walls"])
    return {"placements": _to_request(pl, room, ctx), "raw_placements": _to_request(raw, room, ctx),
            "validation": {"raw": validation(raw, room, ctx), "repaired": validation(pl, room, ctx)},
            "error": None, "unsupported": ctx["unsupported"]}


def _to_request(pl, room, ctx):
    model2ext = {o["id"]: o["ext_id"] for o in room["objects"]}
    by_ext = {}
    for mid, p in pl.items():
        ext = model2ext[mid]
        yaw = (p["yaw"] - ctx["front_k"][ext] * math.pi / 2) % (2 * math.pi)   # model +X (front) -> asset frame
        item = {"id": ext, "position_m": [p["pos"][0] + ctx["x0"], p["pos"][1] + ctx["y0"], p["pos"][2] + ctx["floor_z"]],
                "yaw_rad": yaw, "support_parent": model2ext[p["on"]] if p["on"] else ctx["floor_id"]}
        if p["on"]:
            item["support_surface"] = "top"
        by_ext[ext] = item
    return [by_ext[i] for i in ctx["order"]]


def trained(k, lay):
    """False for a constraint form build.extract_constraints never makes, so the model never saw it (v2.1 train: 0
    against_wall in 5,117 constrained hull rooms; no faces with a non-FRONT subject; no against_wall / near / faces /
    between with an argument on another object; no between with anchors of two categories; no argument (of 'on': the
    subject) that is FLAT or under 5 cm tall). Reported, not refused: the caller decides. Thresholds (near 0.1-0.5) and
    whether the relation held on a reference layout are not forms and are not checked here."""
    o = {x["id"]: x for x in lay["objects"]}
    args = [o[i] for i in k[1:2 if k[0] == "on" else None] if i in o]
    return not ((k[0] == "against_wall" and lay["boundary_type"] != "polygon")
                or (k[0] == "faces" and head(o[k[1]]) not in FRONT)
                or (k[0] != "on" and any(x.get("parent") for x in args))
                or any(head(x) in FLAT or x["size"][2] < 0.05 for x in args)
                or (k[0] == "between" and norm_cat(o[k[2]]["category"]) != norm_cat(o[k[3]]["category"])))


def validation(pl, room, ctx):
    """The canonical checks (validate.check / validate.holds) on a placement, with request ids. ok = valid (inside
    boundary + 0.1, supported, under the ceiling), every object (on others too) within 1 mm of the walls minus the
    keepouts, and every hard constraint holding with the request's own numbers against the real walls.
    An overlap with a fixed box other than a window (validate.check fixed_blocking: a column, a door, a stair) also
    fails ok: those are physical obstacles; furniture in front of a window is reported in fixed_collisions only.
    Collisions between placed objects are reported, not gated (a chair tucked under a table is a
    box overlap; the reference layouts themselves have ~10% rooms with a severe one)."""
    ext = {o["id"]: o["ext_id"] for o in room["objects"] + (room.get("fixed") or [])}
    lay = apply(room, pl)
    c = check(lay)
    grown = ctx["floor"].buffer(1e-3)
    outside = [ext[o["id"]] for o in lay["objects"] if not grown.covers(Polygon(footprint(o)))]
    cons = [{"constraint": [k[0]] + [ext.get(v, v) for v in k[1:]], "hard": hard,
             "holds": holds(k, lay["objects"], ctx["walls"]), "trained": trained(k, lay)} for k, hard in ctx["checks"]]
    cons += [{"constraint": ["anchor", ext[o["id"]], o["anchor"]], "hard": True,
              "holds": bool(o.get("parent")) == (o["anchor"] == "object"),
              "trained": False, "model_conditioned": False}
             for o in lay["objects"] if o.get("anchor") in ("floor", "object")]
    cons += [{"constraint": ["keepout", pts], "hard": False,
              "holds": all(Polygon(footprint(o)).intersection(zone).area <= 1e-9 for o in lay["objects"]),
              "trained": False, "model_conditioned": False} for pts, zone in ctx["soft_keepouts"]]
    return {"ok": c["valid"] and not outside and not c["fixed_blocking"] and all(x["holds"] for x in cons if x["hard"]),
            "oob": [ext[i] for i in c["oob"]],
            "support_fail": [ext[i] for i in c["support_fail"]], "ceiling": [ext[i] for i in c["ceiling"]],
            "outside_1mm": outside, "collisions": {k: [[ext[a], ext[b]] for a, b in v] for k, v in c["collisions"].items()},
            "fixed_collisions": [[ext[a], ext[b]] for a, b in c["fixed_collisions"]],
            "fixed_blocking": [[ext[a], ext[b]] for a, b in c["fixed_blocking"]], "constraints": cons}


if __name__ == "__main__":
    import json

    from fastfill.scene import messages, target_json
    from fastfill.validate import holds

    req = {  # scope doc Appendix B, room moved to (10, 20) and one chair modelled with its front along local +Y
        "room": {"id": "new_room", "room_type": "meeting room", "boundary_xy": [[10, 20], [14, 20], [14, 24], [10, 24]],
                 "boundary_source": "specified_polygon", "floor_id": "room_floor", "floor_z": 0.0, "height_m": 2.8},
        "objects_to_place": [
            {"id": "table_1", "category": "table", "size_xyz_m": [1.4, 0.8, 0.75]},
            {"id": "chair_1", "category": "chair", "size_xyz_m": [0.55, 0.5, 0.9], "semantic_front_local": [1, 0, 0]},
            {"id": "chair_2", "category": "chair", "size_xyz_m": [0.5, 0.55, 0.9], "semantic_front_local": [0, 1, 0]},
            {"id": "cup_1", "category": "cup", "size_xyz_m": [0.08, 0.08, 0.1]},
            {"id": "clock_1", "category": "clock", "size_xyz_m": [0.3, 0.05, 0.3], "anchor": "wall"}],
        "layout_constraints": [
            {"type": "faces", "subject": "chair_1", "target": "chair_2", "hard": True},
            {"type": "faces", "subject": "chair_2", "target": "chair_1", "hard": True},
            {"type": "between", "subject": "table_1", "anchors": ["chair_1", "chair_2"], "hard": True},
            {"type": "supported_by", "subject": "cup_1", "parent": "table_1", "surface": "top", "hard": True}]}
    room, ctx = request_to_room(req)
    assert [o["id"] for o in room["objects"]] == ["table_1", "chair_1", "chair_2", "cup_1"], room["objects"]
    assert room["objects"][2]["size"] == [0.55, 0.5, 0.9]            # chair_2 re-framed: front axis first
    assert ctx["unsupported"] == [{"id": "clock_1", "reason": "unsupported_anchor"}]
    assert room["constraints"][3] == ["on", "cup_1", "table_1"]
    # the model answers in its own frame: chair_1 at x=0.9 facing +X, chair_2 at x=3.1 facing -X, cup on the table
    ans = {"table_1": ([2, 2, 0], 0), "chair_1": ([0.9, 2, 0], 0), "chair_2": ([3.1, 2, 0], math.pi),
           "cup_1": ([1.7, 2, 0.75], 0)}
    placed = {**room, "objects": [{**o, "pos": ans[o["id"]][0], "yaw": ans[o["id"]][1],
                                   "parent": "table_1" if o["id"] == "cup_1" else None} for o in room["objects"]]}
    assert all(holds(c, placed["objects"], placed["boundary"]) for c in room["constraints"])
    text = target_json(placed)
    assert "constraints" in messages(room, room["constraints"], with_target=False)[1]["content"]
    out = placements_from_text(text, room, ctx)
    got = {p["id"]: p for p in out["placements"]}
    assert out["error"] is None and [p["id"] for p in out["placements"]] == ["table_1", "chair_1", "chair_2", "cup_1"]
    assert got["chair_1"]["position_m"] == [10.9, 22, 0] and got["chair_1"]["yaw_rad"] == 0
    assert abs(got["chair_2"]["yaw_rad"] - math.pi / 2) < 1e-9           # front on local +Y: asset turned by 90 deg
    assert got["cup_1"]["support_parent"] == "table_1" and got["cup_1"]["support_surface"] == "top"
    assert got["table_1"]["support_parent"] == "room_floor" and "support_surface" not in got["table_1"]
    assert placements_from_text("not json", room, ctx)["error"] == "bad_json"
    # table 3 cm through the east wall (x up to 4.03): moved back inside, the cup on it moves along; 20 cm is left alone
    over = {**placed, "objects": [{**o, "pos": [3.33 if o["id"] == "table_1" else 3.0 if o["id"] == "cup_1" else o["pos"][0],
                                                *o["pos"][1:]]} for o in placed["objects"]]}
    got = {p["id"]: p["position_m"] for p in placements_from_text(target_json(over), room, ctx)["placements"]}
    assert abs(got["table_1"][0] - 13.3) < 1e-9 and abs(got["cup_1"][0] - 12.97) < 1e-9, got
    far = {**over, "objects": [{**o, "pos": [3.5, *o["pos"][1:]]} if o["id"] == "table_1" else o for o in over["objects"]]}
    assert placements_from_text(target_json(far), room, ctx)["placements"][0]["position_m"][0] == 13.5

    # a doorway keepout on the west wall becomes a notch (padded) in the floor the model sees; the frame is unchanged
    door = {"type": "keepout", "polygon_xy": [[10, 21.5], [11, 21.5], [11, 22.5], [10, 22.5]]}
    room2, ctx2 = request_to_room({**req, "layout_constraints": req["layout_constraints"] + [door]})
    assert (ctx2["x0"], ctx2["y0"]) == (10, 20) and [0, 1.42] in room2["boundary"] and [1.08, 2.58] in room2["boundary"]
    assert room2["constraints"] == room["constraints"]
    # a constraint naming an unsupported object is skipped and reported; the rest of the room is still placed
    hang = {"type": "near", "subject": "clock_1", "target": "table_1", "max_gap_m": 0.5}
    room3, ctx3 = request_to_room({**req, "layout_constraints": [hang]})
    assert room3["constraints"] == [] and ctx3["unsupported"][1] == {"constraint": hang, "reason": "references_unsupported_object"}
    up = {**req, "objects_to_place": [{**req["objects_to_place"][0], "semantic_front_local": [0, 0, 1]}], "layout_constraints": []}
    assert request_to_room(up)[1]["unsupported"] == [{"id": "table_1", "reason": "front_not_axis_aligned"}]
    bad = [{**req, "objects_to_place": req["objects_to_place"] * 2},                                  # duplicate ids
           {**req, "layout_constraints": [{"type": "near", "id": "cup_1", "target": "table_1", "max_distance_m": 1.5}]},
           {**req, "layout_constraints": [{"type": "between", "subject": "table_1", "anchors": ["chair_1"]}]},
           {**req, "room": {**req["room"], "floor_z": None}},
           {**req, "objects_to_place": [{**req["objects_to_place"][0], "size_xyz_m": [1.4, -0.8, 0.75]}]},
           {**req, "objects_to_place": [{**req["objects_to_place"][0], "size_xyz_m": [1.4, float("nan"), 0.75]}]},
           {**req, "layout_constraints": [{**door, "polygon_xy": [[9, 19], [15, 19], [15, 25], [9, 25]]}]}]  # fills the room
    obj0 = req["objects_to_place"][0]
    bad += [{**req, "layout_constraints": [{"type": "supported_by", "subject": "cup_1", "parent": "cup_1"}]},   # self
            {**req, "layout_constraints": [{"type": "supported_by", "subject": "cup_1", "parent": "table_1"},
                                           {"type": "supported_by", "subject": "table_1", "parent": "cup_1"}]},  # cycle
            {**req, "layout_constraints": [{"type": "faces", "subject": "chair_1", "target": "chair_2", "hard": "false"}]},
            {**req, "layout_constraints": [{"type": "between", "subject": "table_1", "anchors": "cc"}]},
            {**req, "coordinate_system": {"units": "cm"}},
            {**req, "objects_to_place": [{**obj0, "anchor": "banana"}]},
            {**req, "objects_to_place": req["objects_to_place"] + [{"id": "x", "category": "clock", "size_xyz_m": [-1, 1, 1],
                                                                  "anchor": "wall"}]},
            {**req, "room": {**req["room"], "room_type": 7}},
            {**req, "room": {**req["room"], "height_m": 0}},
            {**req, "objects_to_place": [{**obj0, "size_xyz_m": [1e200, 1, 1]}]},
            {**req, "room": {**req["room"], "floor_z": float("inf")}},
            {**req, "room": {**req["room"], "height_m": float("nan")}},
            {**req, "layout_constraints": [{"type": "near", "subject": "cup_1", "target": "table_1", "max_gap_m": -0.1}]}]
    for b in bad:
        try:
            request_to_room(b)
            raise AssertionError(b)
        except ValueError:
            pass

    # near: the model is shown the threshold rounded down to cm, the answer is accepted against the request's own
    # number; a failed hard constraint fails the layout (after repair), a failed soft one is only reported
    near = lambda gap, hard=True: {"type": "near", "subject": "chair_1", "target": "chair_2", "max_gap_m": gap, "hard": hard}
    r5, c5 = request_to_room({**req, "layout_constraints": [near(0.126)]})
    assert r5["constraints"] == [["near", "chair_1", "chair_2", 0.12]] and c5["checks"][0][0][3] == 0.126
    apart = target_json(placed)                        # chairs 1.7 m apart: fails any near <= 0.5
    v = placements_from_text(apart, *request_to_room({**req, "layout_constraints": [near(0.1)]}))["validation"]
    assert not v["repaired"]["ok"] and v["repaired"]["constraints"][0]["holds"] is False, v
    v = placements_from_text(apart, *request_to_room({**req, "layout_constraints": [near(0.1, hard=False)]}))["validation"]
    assert v["repaired"]["ok"], v
    # a cup hanging 3 cm over the east wall from a table flush with it: the cup is pushed in, staying on the table
    edge = {**placed, "objects": [{**o, "pos": [3.3, 2, 0] if o["id"] == "table_1" else [3.99, 2, 0.75] if o["id"] == "cup_1"
                                   else o["pos"]} for o in placed["objects"]]}
    out = placements_from_text(target_json(edge), room, ctx)
    cup = {p["id"]: p for p in out["placements"]}["cup_1"]["position_m"]
    assert out["validation"]["raw"]["outside_1mm"] == ["cup_1"] and out["validation"]["repaired"]["ok"], out["validation"]
    assert abs(cup[0] - 13.96) < 1e-9, cup
    # fixed geometry: a column at (12, 22) is shown to the model in the room frame, never placed, and a table on it is
    # reported (not refused); a fixed id that repeats an object id is a duplicate
    col = {"id": "col_1", "category": "column", "size_xyz_m": [0.3, 0.3, 2.8], "position_m": [12, 22, 0]}
    room6, ctx6 = request_to_room({**req, "room": {**req["room"], "fixed_geometry": [col]}, "layout_constraints": []})
    assert room6["fixed"][0]["id"] == "column_1" and room6["fixed"][0]["pos"] == [2, 2, 0], room6["fixed"]
    assert json.loads(messages(room6, [], with_target=False)[1]["content"])["fixed"] == [
        {"id": "column_1", "size": [0.3, 0.3, 2.8], "pos": [2, 2, 0], "yaw": 0}]
    out6 = placements_from_text(target_json(placed), room6, ctx6)       # table_1 centred on the column
    assert not out6["validation"]["repaired"]["ok"] and out6["validation"]["repaired"]["fixed_collisions"] == [["table_1", "col_1"]]
    # the repair moves a table with the cup on it; a move that would push the CUP into a fixed upper cabinet (above the
    # table top, at the cup's height) is not taken, so the table stays 3 cm through the wall and the layout fails
    cab = {"id": "cab_1", "category": "cabinet", "size_xyz_m": [0.4, 0.6, 1.0], "position_m": [12.75, 22, 0.78]}
    room9, ctx9 = request_to_room({**req, "room": {**req["room"], "fixed_geometry": [cab]}, "layout_constraints": []})
    moved = {"table_1": [3.33, 2, 0], "cup_1": [3.0, 2, 0.75], "chair_2": [1.5, 3.5, 0]}
    lay9 = {**placed, "objects": [{**o, "pos": moved.get(o["id"], o["pos"])} for o in placed["objects"]]}
    out9 = placements_from_text(target_json(lay9), room9, ctx9)
    got = {p["id"]: p["position_m"] for p in out9["placements"]}
    assert abs(got["table_1"][0] - 13.33) < 1e-9 and abs(got["cup_1"][0] - 13.0) < 1e-9, got
    assert not out9["validation"]["repaired"]["fixed_collisions"] and "table_1" in out9["validation"]["repaired"]["outside_1mm"]
    for b in ({**col, "id": "table_1"}, {**col, "size_xyz_m": [0, 0.3, 2.8]}):        # duplicate id; zero size
        try:
            request_to_room({**req, "room": {**req["room"], "fixed_geometry": [b]}})
            raise AssertionError(b)
        except ValueError:
            pass
    # a constraint may not name a fixed box: refused with its own message
    try:
        request_to_room({**req, "room": {**req["room"], "fixed_geometry": [col]},
                         "layout_constraints": [{"type": "near", "subject": "table_1", "target": "col_1", "max_gap_m": 0.5}]})
        raise AssertionError("constraint on a fixed box accepted")
    except ValueError as e:
        assert "fixed box" in str(e), e
    # training-side normalization: |z| <= FLOOR_Z snapped to 0; a box starting at FIXED_MAX_Z or above, covering half
    # the room, or standing clear of it is not shown and reported as fixed_ignored
    fx = lambda **kw: {**col, **kw}
    room8, ctx8 = request_to_room({**req, "layout_constraints": [], "room": {**req["room"], "floor_z": 1.0, "fixed_geometry": [
        fx(id="door", position_m=[12, 20.5, 1.06]), fx(id="beam", size_xyz_m=[4, 0.3, 0.3], position_m=[12, 22, 2.5]),
        fx(id="slab", size_xyz_m=[3, 3, 0.1], position_m=[12, 22, 1]), fx(id="far", position_m=[30, 30, 1])]}})
    assert [f["ext_id"] for f in room8["fixed"]] == ["door"] and room8["fixed"][0]["pos"] == [2, 0.5, 0], room8["fixed"]
    assert [u["id"] for u in ctx8["unsupported"] if u["reason"] == "fixed_ignored"] == ["beam", "slab", "far"], ctx8["unsupported"]
    # repair never deepens the overlap with a fixed box: a table lying along the east wall 3 cm out (yaw 90, x 3.23..4.03)
    # with its west edge 8 cm over a column at (3.3, 2) is left where it is (every move within SNAP_TOL pushes it further
    # in, so it stays outside_1mm and serve.py refuses the layout); the same table clear of the column is moved back
    col2 = {**col, "position_m": [13.3, 22, 0]}
    room7, ctx7 = request_to_room({**req, "room": {**req["room"], "fixed_geometry": [col2]}, "layout_constraints": []})
    side = lambda y: {**placed, "objects": [{**o, "pos": [3.63, y, o["pos"][2]], "yaw": math.pi / 2 if o["id"] == "table_1" else o["yaw"]}
                                            if o["id"] in ("table_1", "cup_1") else o for o in placed["objects"]]}
    out7 = placements_from_text(target_json(side(2.0)), room7, ctx7)
    got = {p["id"]: p["position_m"] for p in out7["placements"]}
    assert abs(got["table_1"][0] - 13.63) < 1e-9 and out7["validation"]["repaired"]["outside_1mm"] == ["table_1"], out7["validation"]
    assert ["table_1", "col_1"] in out7["validation"]["repaired"]["fixed_collisions"]
    got = {p["id"]: p["position_m"] for p in placements_from_text(target_json(side(2.9)), room7, ctx7)["placements"]}
    assert abs(got["table_1"][0] - 13.6) < 1e-9 and abs(got["cup_1"][0] - 13.6) < 1e-9, got
    # a column sunk 0.5 m below the floor is cut at the floor (height 2.3); a box sunk under the floor is not shown
    from fastfill.build import FLAT_Z as BUILD_FLAT_Z
    assert FLAT_Z == BUILD_FLAT_Z
    sunk = [{"id": "c9", "category": "column", "size_xyz_m": [0.3, 0.3, 2.8], "position_m": [12, 22, -0.5]},
            {"id": "u9", "category": "unknown", "size_xyz_m": [1, 1, 0.4], "position_m": [12.5, 22.5, -0.6]}]
    r7, c7 = request_to_room({**req, "room": {**req["room"], "fixed_geometry": sunk}, "layout_constraints": []})
    assert [(f["ext_id"], f["pos"][2], f["size"][2]) for f in r7["fixed"]] == [("c9", 0.0, 2.3)], r7["fixed"]
    assert {"id": "u9", "reason": "fixed_ignored"} in c7["unsupported"]
    # objects are ordered by cm-rounded sizes clamped at 1 cm, as in training (a 3 mm side sorts as 1 cm)
    thin = {**req, "objects_to_place": [{"id": "mat", "category": "mat", "size_xyz_m": [1.0, 0.003, 0.9]},
                                        {"id": "pad", "category": "mat", "size_xyz_m": [0.5, 0.01, 0.9]}],
            "layout_constraints": []}
    assert [o["ext_id"] for o in request_to_room(thin)[0]["objects"]] == ["mat", "pad"]
    # re-entrant corner of an L-shaped room: the nearest-boundary push is diagonal and long, a 6 cm move along -x works
    ell = Polygon([(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
    plc = {"b": {"pos": [1.76, 2.22, 0], "yaw": 0.0, "on": None}}
    snap_inside(plc, {"objects": [{"id": "b", "size": [0.6, 0.6, 0.8]}]}, ell)
    assert plc["b"]["pos"][:2] == [1.7, 2.22] or abs(plc["b"]["pos"][0] - 1.7) < 1e-9, plc
    # the repair never breaks a hard constraint that held: a chair 3 cm through the south wall faces a box 28.5 deg off
    # its front; the shortest push (0, +3 cm) turns that to 32 deg (> 30), so (-4, +3) cm is taken instead
    sq, two = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)]), {"objects": [{"id": "c", "size": [.5, .5, .9]}, {"id": "b", "size": [.06, .06, .3]}]}
    lay = lambda q: [{"id": i, "pos": q[i]["pos"], "yaw": 0.0, "size": s["size"]} for i, s in zip("cb", two["objects"])]
    for cons, want in (([], [1, .25]), ([["faces", "c", "b"]], [.96, .25])):
        plc = {"c": {"pos": [1, .22, 0], "yaw": 0.0, "on": None}, "b": {"pos": [1.35, .03, 0], "yaw": 0.0, "on": None}}
        assert holds(["faces", "c", "b"], lay(plc), sq.exterior.coords[:-1])
        snap_inside(plc, two, sq, cons, sq.exterior.coords[:-1])
        assert all(abs(u - v) < 1e-9 for u, v in zip(plc["c"]["pos"], want)), plc
        assert holds(["faces", "c", "b"], lay(plc), sq.exterior.coords[:-1]) == bool(cons)
    # two chairs and a side table between them, all 5 cm through the east wall: the between fails after the second chair
    # moves and holds again once the table moves, so the plain pass is kept (a move-by-move guard left chair b outside)
    row = {"objects": [{"id": "a", "size": [.7, .7, .9]}, {"id": "b", "size": [.7, .7, .9]}, {"id": "s", "size": [.38, .38, .6]}]}
    plc = {i: {"pos": [x, y, 0], "yaw": 0.0, "on": None} for i, x, y in (("a", 3.7, 1), ("b", 3.7, 3), ("s", 3.86, 2))}
    sq4 = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    snap_inside(plc, row, sq4, [["between", "s", "a", "b"]], sq4.exterior.coords[:-1])
    assert all(abs(plc[i]["pos"][0] - x) < 1e-9 for i, x in (("a", 3.65), ("b", 3.65), ("s", 3.81))), plc
    # a padded keepout ending 3 mm short of the east wall (the finding's 1.48 m room) snaps to it; 2 cm short stays
    ko = lambda x1: {"room": {"boundary_xy": [[0, 0], [1.48, 0], [1.48, 3.48], [0, 3.48]]},
                     "objects_to_place": [{"id": "a", "category": "chair", "size_xyz_m": [0.5, 0.5, 0.9]}],
                     "layout_constraints": [{"type": "keepout", "polygon_xy": [[.495, -.2], [x1, -.2], [x1, 1], [.495, 1]]}]}
    assert [1.48, 1.08] in request_to_room(ko(1.48 - KEEPOUT_PAD - 0.003))[0]["boundary"]
    assert [1.46, 1.08] in request_to_room(ko(1.48 - KEEPOUT_PAD - 0.02))[0]["boundary"]
    # two padded keepouts 7.5 mm apart in front of a 1.08 x 0.62 m corner (EmbodiedGen-shaped eg#173): the gap stays open
    # and the corner stays floor; 2.9 mm apart the gap folds under cm rounding, as does a keepout crossing a 45-degree
    # wall by 3 mm: keepout_not_representable, not bad_boundary (the walls themselves round cleanly)
    pair = lambda x1: {"room": {"boundary_xy": [[0, 0], [4.5, 0], [4.5, 3], [0, 3]]}, "objects_to_place": ko(0)["objects_to_place"],
                       "layout_constraints": [{"type": "keepout", "polygon_xy": [[3.5, 1.1], [4.5, 1.1], [4.5, 2.3], [3.5, 2.3]]},
                                              {"type": "keepout", "polygon_xy": [[2.1, 2.4], [x1, 2.4], [x1, 3], [2.1, 3]]}]}
    assert [4.5, 3] in request_to_room(pair(3.3325))[0]["boundary"]
    mouth = {"room": {"boundary_xy": [[0, 0], [4, 0], [4, 4], [1, 4], [0, 3]]}, "objects_to_place": ko(0)["objects_to_place"],
             "layout_constraints": [{"type": "keepout", "polygon_xy": [[.58, 2.08], [1.42, 2.08], [1.42, 3.423], [.58, 3.423]]}]}
    for b in (pair(3.3371), mouth):
        try:
            request_to_room(b)
            raise AssertionError(b)
        except ValueError as e:
            assert str(e) == "keepout_not_representable", e
    # forms training never has are checked as asked but flagged: faces from a round table (no FRONT head), against_wall
    # in a hull room, near from a cup that stands on the table; faces / between / on of the Appendix-B request are trained
    assert all(c["trained"] for c in placements_from_text(text, room, ctx)["validation"]["repaired"]["constraints"])
    hull = {"room": {**req["room"], "boundary_source": "scan_hull"}, "objects_to_place": [
        {"id": "rt", "category": "round table", "size_xyz_m": [1, 1, .75]}, {"id": "sh", "category": "shelf", "size_xyz_m": [1, .4, 1.8]},
        {"id": "cup", "category": "cup", "size_xyz_m": [.08, .08, .1]}], "layout_constraints": [
        {"type": "faces", "subject": "rt", "target": "sh"}, {"type": "against_wall", "subject": "sh"},
        {"type": "near", "subject": "cup", "target": "sh", "max_gap_m": 0.5}]}
    rh, ch = request_to_room(hull)
    ans = '{"placements":[{"id":"round_table_1","pos":[2,2,0],"yaw":270},{"id":"shelf_1","pos":[2,0.2,0],"yaw":0},' \
          '{"id":"cup_1","on":"round_table_1","pos":[2,2,0.75],"yaw":0}]}'       # the cup 1.56 m from the shelf
    v = placements_from_text(ans, rh, ch)["validation"]["repaired"]
    assert [(c["trained"], c["holds"]) for c in v["constraints"]] == [(False, True), (False, True), (False, False)], v
    assert not v["ok"] and not v["outside_1mm"] and not v["oob"], v
    # build never relates a rug, a sub-5 cm box or (except by 'on') an object standing on another
    lay = {"boundary_type": "polygon", "objects": [
        {"id": "s", "category": "sofa", "size": [2, .9, .8]}, {"id": "r", "category": "rug", "size": [2, 1.5, .01]},
        {"id": "l", "category": "lamp", "size": [.3, .3, .5], "parent": "s"},
        {"id": "k", "category": "coaster", "size": [.1, .1, .01], "parent": "s"}]}
    assert [trained(k, lay) for k in (["near", "s", "r", .5], ["faces", "s", "r"], ["against_wall", "l"], ["on", "k", "s"],
                                      ["on", "l", "s"], ["against_wall", "s"])] == [False] * 4 + [True] * 2
    print("interface.py self-check ok")
    print(json.dumps(out["placements"][2]))
