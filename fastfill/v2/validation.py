"""Conservative upright-OBB checks for the v2 protocol, independent of v1.

Every check reports one of ``CHECK_STATUSES``: ``pass``, ``violation`` (the
geometry contradicts the requirement) or ``unknown`` (the requirement cannot be
verified from boxes, e.g. undeclared support). Unknown required geometry is never
a pass. This module only verifies boxes; mesh, physics and Solver checks require
separate evidence from external tools.
"""
from copy import deepcopy
import math

from shapely.geometry import LineString, Polygon


SUPPORT_BBOX_TOLERANCE_M = 1e-6
CHECK_STATUSES = ("pass", "violation", "unknown")
# A wall-supported box may stand this far off the boundary line (the against_wall default): source wall objects keep
# a gap (SAGE-10k 8 cm, SceneSmith <= 5 cm, MansionWorld <= 6 cm; the RoomGenBench scenes are SAGE).
WALL_GAP_M = .1


def validate_required_levels(required_levels):
    """Reject misspelled/ambiguous verification requirements at the API boundary."""
    known = {"bbox", "mesh", "physics", "solver"}
    if (not isinstance(required_levels, (list, tuple)) or not required_levels or
            any(not isinstance(level, str) or level not in known for level in required_levels) or
            len(set(required_levels)) != len(required_levels)):
        raise ValueError("required_levels must be a nonempty list/tuple of distinct bbox/mesh/physics/solver names")
    return tuple(required_levels)


def _capability_names(value, label="capabilities"):
    if not isinstance(value, (list, tuple)) or any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError(f"{label} must be a list/tuple of nonempty capability names")
    return tuple(value)


def effective_support_requests(condition):
    """Return new request dictionaries with hard support declarations reconciled.

    A hard/default-hard `on` is the same support requirement as support_parent.
    Soft relations remain preferences and never establish mandatory support.
    Conflicting declarations, invalid references and cycles are rejected before
    asset resolution. Original conditions/token graphs remain untouched.
    """
    objects = tuple(condition["objects"])
    fixed = tuple(condition["room"].get("fixed_objects", ()))
    rows = {obj["id"]: deepcopy(obj) for obj in objects + fixed}
    if len(rows) != len(objects) + len(fixed):
        raise ValueError("support object IDs must be unique")
    for constraint in condition.get("constraints", ()):
        if constraint.get("type") != "on":
            continue
        hard = constraint.get("hard", True)
        if not isinstance(hard, bool):
            raise ValueError("on constraint hard flag must be boolean")
        if not hard:
            continue
        ident = constraint.get("object_id")
        parent = constraint.get("parent_id", constraint.get("target_id"))
        if "parent_id" in constraint and "target_id" in constraint and constraint["parent_id"] != constraint["target_id"]:
            raise ValueError("conflicting on constraint parent references")
        if ident not in rows or parent not in set(rows) | {"floor", "wall"} or ident == parent:
            raise ValueError("invalid hard support reference")
        row = rows[ident]
        if row.get("support_parent") is not None and row["support_parent"] != parent:
            raise ValueError(f"conflicting support parents for {ident}")
        surface = constraint.get("surface_id")
        if surface is not None and row.get("support_surface_id") is not None and row["support_surface_id"] != surface:
            raise ValueError(f"conflicting support surfaces for {ident}")
        rows = {**rows, ident: {**row, "support_parent": parent,
                               **({"support_surface_id": surface} if surface is not None else {})}}
    for ident, row in rows.items():
        parent = row.get("support_parent")
        if parent is not None and (parent not in set(rows) | {"floor", "wall"} or parent == ident):
            raise ValueError("invalid support parent reference")
        current, seen = ident, frozenset()
        while current in rows:
            if current in seen:
                raise ValueError("support graph contains a cycle")
            seen = seen.union((current,))
            current = rows[current].get("support_parent")
    return tuple(rows[obj["id"]] for obj in objects)


def _check(code, status="pass", ids=(), *, hard=True, **evidence):
    return {"code": code, "status": status, "hard": hard,
            "object_ids": list(ids), **evidence}


def _vector(value, length, *, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"expected {length} coordinates")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or (positive and x <= 0) for x in value):
        raise ValueError("nonfinite or invalid coordinate")
    return tuple(float(x) for x in value)


def _geometry(obj, stage):
    key = "actual_size_local_m" if stage == "actual" else "target_size_local_m"
    size = _vector(obj.get(key, obj.get("size_local_m")), 3, positive=True)
    pos = _vector(obj.get("bottom_center_m"), 3)
    yaw = obj.get("yaw_rad")
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
        raise ValueError("invalid yaw")
    if stage == "actual" and obj.get("capabilities") is not None:
        _capability_names(obj["capabilities"])
    _validate_support_bbox(obj.get("support_surfaces"), size)
    return {**obj, "_size": size, "_pos": pos, "_yaw": float(yaw)}


def _world_xy(points, obj):
    c, s = math.cos(obj["_yaw"]), math.sin(obj["_yaw"])
    x, y = obj["_pos"][:2]
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in points]


def footprint(obj):
    """Canonical local full size rotated in XY; it never replaces local size."""
    w, d = obj["_size"][:2]
    return Polygon(_world_xy(((-w/2, -d/2), (w/2, -d/2), (w/2, d/2), (-w/2, d/2)), obj))


def _polygon(points):
    if not isinstance(points, (tuple, list)) or len(points) < 3:
        raise ValueError("polygon needs at least three vertices")
    p = Polygon([_vector(point, 2) for point in points])
    if not p.is_valid or p.area <= 0:
        raise ValueError("invalid polygon")
    return p


def _validate_support_bbox(surfaces, size):
    """Verified surfaces must lie inside their canonical, bottom-centred bbox.

    The 1e-6 metre tolerance permits small metadata roundoff; evidence is never
    clipped or moved. This necessary consistency check does not verify a mesh.
    """
    if surfaces is None:
        return
    if not isinstance(surfaces, (list, tuple)):
        raise ValueError("support surfaces must be a list/tuple or unknown")
    ids = []
    for surface in surfaces:
        if not isinstance(surface, dict) or not isinstance(surface.get("surface_id"), str) or not surface["surface_id"].strip():
            raise ValueError("support surface needs a stable ID")
        ids.append(surface["surface_id"])
        polygon = _polygon(surface["local_polygon_xy_m"])
        z = _vector((surface["local_z_m"],), 1)[0]
        xmin, ymin, xmax, ymax = polygon.bounds
        tol = SUPPORT_BBOX_TOLERANCE_M
        if (z < -tol or z > size[2] + tol or xmin < -size[0]/2 - tol
                or xmax > size[0]/2 + tol or ymin < -size[1]/2 - tol or ymax > size[1]/2 + tol):
            raise ValueError("verified support surface lies outside canonical bbox")
    if len(ids) != len(set(ids)):
        raise ValueError("support surface IDs must be unique per bbox")


def _floor_checks(room, objects, tolerance_m):
    """Check newly requested objects; existing structure may span the slab."""
    floor = room.get("floor_z_m")
    if floor is None or room.get("floor_known") is False:
        return (_check("floor_boundary_unknown", "unknown", reason="coordinate reference is not a verified floor"),)
    floor = _vector((floor,), 1)[0]
    return tuple(_check("floor_lower_bound", "pass" if obj["_pos"][2] >= floor-tolerance_m else "violation",
                        (obj["id"],), bottom_z_m=obj["_pos"][2], floor_z_m=floor, tolerance_m=tolerance_m)
                 for obj in objects)


def _capability_checks(request, obj, stage, *, fixed=False):
    """Fixed assets already exist; generated target envelopes have no asset yet."""
    required = _capability_names(request.get("required_capabilities", ()), "required_capabilities")
    if not required or (stage != "actual" and not fixed):
        return ()
    available = obj.get("capabilities")
    if available is not None:
        available = _capability_names(available)
    status = "unknown" if available is None else ("pass" if set(required).issubset(available) else "violation")
    return (_check("capabilities", status, (obj["id"],), required=list(required),
                   available=list(available) if available is not None else None,
                   geometry_role="fixed" if fixed else "requested"),)


def _bounds_checks(request, obj, tol):
    size = obj["_size"]
    checks = ()
    fixed = request.get("fixed_size_local_m")
    if fixed is not None:
        checks += (_check("fixed_size", "pass" if all(b is None or abs(a-b) <= tol for a, b in
                   zip(size, fixed)) else "violation", (obj["id"],)),)
    bounds = request.get("size_bounds_local_m")
    if bounds is not None:
        lower, upper = _vector(bounds["min"], 3, positive=True), _vector(bounds["max"], 3, positive=True)
        checks += (_check("size_bounds", "pass" if all(lo-tol <= x <= hi+tol for x, lo, hi in
                   zip(size, lower, upper)) else "violation", (obj["id"],)),)
    return checks


def _on_boundary(obj, room, tolerance):
    """A side of the footprint lies within tolerance of the room boundary."""
    zone = _polygon(room["floor_polygon_xy_m"]).exterior.buffer(tolerance+1e-9)
    corners = tuple(footprint(obj).exterior.coords)
    return any(zone.contains(LineString((a, b))) for a, b in zip(corners, corners[1:]))


def on_wall(box, room, tolerance=WALL_GAP_M):
    """Does a target box (bottom_center_m, target_size_local_m, yaw_rad) show a footprint side within ``tolerance``
    of the room's floor polygon (``wall_support``'s geometry)?"""
    return _on_boundary(_geometry(box, "target"), room, tolerance)


def _support(request, obj, index, room, stage, tol):
    parent = request.get("support_parent")
    ids = (obj["id"],)
    if parent is None:
        return _check("support_unknown", "unknown", ids)
    if parent in ("floor", room.get("floor_id", "floor")):
        z = room.get("floor_z_m")
        if room.get("floor_known") is False or z is None:
            return _check("floor_unknown", "unknown", ids)
        return _check("floor_support", "pass" if abs(obj["_pos"][2] - z) <= tol else "violation", ids)
    if parent == "wall":  # a box shows a footprint side within WALL_GAP_M of the room boundary, not the mounting itself
        if room.get("boundary_known") is False or room.get("floor_polygon_xy_m") is None:
            return _check("wall_unknown", "unknown", ids)
        gap = max(tol, WALL_GAP_M)
        return _check("wall_support", "pass" if _on_boundary(obj, room, gap) else "violation", ids, tolerance_m=gap)
    if parent not in index:
        return _check("support_parent_missing", "violation", ids, parent_id=parent)
    ids = (obj["id"], parent)
    parent_obj = index[parent]
    surfaces = parent_obj.get("support_surfaces", ())
    if not surfaces:
        # A target proxy top may support analysis, but never claims an actual tabletop.
        return _check("support_surface_unknown", "unknown", ids, parent_id=parent,
                      stage=stage, reason="bbox top is not verified support geometry")
    requested_surface = request.get("support_surface_id")
    eligible = [sf for sf in surfaces if requested_surface is None or sf["surface_id"] == requested_surface]
    for surface in eligible:
        polygon = _polygon(_world_xy(surface["local_polygon_xy_m"], parent_obj))
        world_z = parent_obj["_pos"][2] + surface["local_z_m"]
        if abs(obj["_pos"][2] - world_z) <= tol and polygon.buffer(tol).covers(footprint(obj)):
            return _check("support", "pass", ids, parent_id=parent, surface_id=surface["surface_id"])
    return _check("support", "violation", ids, parent_id=parent)


def _face_angle(obj, stage):
    if stage != "actual":
        return obj["_yaw"]
    front = obj.get("semantic_front_local")
    if front is None:
        return None
    vector = _vector(front, 3)
    if math.hypot(vector[0], vector[1]) < 1e-12 or abs(vector[2]) > 1e-6:
        return None
    return obj["_yaw"] + math.atan2(vector[1], vector[0])


def _constraint(c, index, room, stage, tol):
    kind, hard = c.get("type"), c.get("hard", True)
    if not isinstance(hard, bool):
        raise ValueError("constraint hard flag must be boolean")
    ident = c.get("object_id")
    ids = (ident,) if ident else ()
    if kind in ("faces_direction", "faces", "near", "clearance", "on", "against_wall", "between") and ident not in index:
        return _check("constraint_reference", "violation", ids, hard=hard, constraint_type=kind)
    if ident is not None and ident not in index:
        return _check("constraint_reference", "violation", ids, hard=hard)
    obj = index.get(ident)
    target_id = c.get("target_id", c.get("parent_id"))
    target = index.get(target_id)
    if target_id is not None and target is None and target_id not in ("floor", "wall"):
        return _check("constraint_reference", "violation", ids, hard=hard, target_id=target_id)
    if kind in ("faces_direction", "faces"):
        angle = _face_angle(obj, stage)
        if angle is None:
            return _check("semantic_front_unknown", "unknown", ids, hard=hard)
        direction = (c["direction_xy"] if kind == "faces_direction" else
                     [target["_pos"][q] - obj["_pos"][q] for q in (0, 1)])
        dx, dy = _vector(direction, 2)
        if math.hypot(dx, dy) < 1e-12:
            return _check("direction_undefined", "unknown", ids, hard=hard)
        error = abs((angle - math.atan2(dy, dx) + math.pi) % (2 * math.pi) - math.pi)
        holds = error <= c.get("tolerance_rad", math.pi / 6) + tol
        return _check("constraint", "pass" if holds else "violation", ids, hard=hard,
                      constraint_type=kind, yaw_error_rad=error)
    if kind in ("near", "clearance"):
        distance = footprint(obj).distance(footprint(target))
        holds = (distance <= c["max_distance_m"] + tol if kind == "near" else
                 distance + tol >= c["min_distance_m"])
        return _check("constraint", "pass" if holds else "violation", (ident, target_id), hard=hard,
                      constraint_type=kind, distance_m=distance)
    if kind == "keepout":
        poly = _polygon(c["polygon_xy_m"])
        checked = (obj,) if obj is not None else tuple(index.values())
        failed = tuple(x["id"] for x in checked if footprint(x).intersection(poly).area > tol * tol)
        return _check("constraint", "violation" if failed else "pass", failed, hard=hard, constraint_type=kind)
    if kind == "on":
        request = {"support_parent": target_id,
                   **({"support_surface_id": c["surface_id"]} if "surface_id" in c else {})}
        result = _support(request, obj, index, room, stage, tol)
        return {**result, "hard": hard, "constraint_type": kind}
    if kind == "against_wall":
        if room.get("boundary_known") is False or room.get("floor_polygon_xy_m") is None:
            return _check("boundary_unknown", "unknown", ids, hard=hard, constraint_type=kind)
        tolerance = _vector((c.get("tolerance_m", .1),), 1)[0]
        if tolerance < 0:
            raise ValueError("wall tolerance must be nonnegative")
        return _check("constraint", "pass" if _on_boundary(obj, room, tolerance) else "violation", ids, hard=hard,
                      constraint_type=kind, tolerance_m=tolerance)
    if kind == "between":
        targets = c.get("target_ids")
        if not isinstance(targets, (list, tuple)) or len(targets) != 2 or any(
                not isinstance(target, str) or target not in index for target in targets):
            return _check("constraint_reference", "violation", ids, hard=hard, constraint_type=kind)
        ends = tuple(index[target]["_pos"][:2] for target in targets)
        if ends[0] == ends[1]:
            return _check("between_segment_undefined", "unknown", ids, hard=hard, constraint_type=kind)
        segment, polygon = LineString(ends), footprint(obj)
        holds = segment.intersects(polygon) and not segment.touches(polygon)
        return _check("constraint", "pass" if holds else "violation", (ident, *targets), hard=hard, constraint_type=kind)
    return _check("constraint_unknown", "unknown", ids, hard=hard, constraint_type=kind)


def _collision_checks(objects, fixed, tol):
    pairs = tuple((a, b, "collision") for i, a in enumerate(objects) for b in objects[i+1:])
    pairs += tuple((a, b, "fixed_collision") for a in objects for b in fixed)
    return tuple(_check(code, "violation", (a["id"], b["id"]), overlap_area_m2=footprint(a).intersection(footprint(b)).area)
                 for a, b, code in pairs
                 if min(a["_pos"][2]+a["_size"][2], b["_pos"][2]+b["_size"][2]) -
                 max(a["_pos"][2], b["_pos"][2]) > tol and
                 footprint(a).intersection(footprint(b)).area > tol*tol)


def validate_scene(condition, objects, *, stage="target", tolerance_m=1e-4, required_levels=("bbox",)):
    """Validate request completeness and upright geometry, keeping unknown explicit.

    Strict defaults require trustworthy boundary and support. Unknown mesh/physics/
    Solver are informational unless named in required_levels; they never become
    reported passes merely because the boxes passed.
    """
    if stage not in ("target", "actual"):
        raise ValueError("stage must be target or actual")
    required_levels = validate_required_levels(required_levels)
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("invalid tolerance")
    checks = ()
    try:
        normalized = tuple(_geometry(obj, stage) for obj in objects)
        requested = {req["id"]: req for req in effective_support_requests(condition)}
        ids = [obj["id"] for obj in normalized]
        if len(ids) != len(set(ids)) or set(ids) != set(requested):
            raise ValueError("all requested IDs must occur exactly once")
        room = condition["room"]
        fixed = tuple(_geometry(obj, stage) for obj in room.get("fixed_objects", ()))
        index = {obj["id"]: obj for obj in normalized + fixed}
        if len(index) != len(normalized) + len(fixed):
            raise ValueError("fixed and requested IDs must be distinct")
        checks += (_check("schema"),)
        checks += _floor_checks(room, normalized, tolerance_m)
        polygon = room.get("floor_polygon_xy_m")
        if room.get("boundary_known") is False or polygon is None:
            checks += (_check("boundary_unknown", "unknown"),)
        else:
            boundary = _polygon(polygon)
            checks += tuple(_check("boundary", "pass" if boundary.buffer(tolerance_m).covers(footprint(obj))
                                   else "violation", (obj["id"],)) for obj in normalized)
        h = room.get("height_m")
        floor_z = room.get("floor_z_m")
        if h is None or floor_z is None or room.get("floor_known") is False:
            checks += (_check("ceiling_unknown", "unknown"),)
        else:
            if not math.isfinite(h) or h <= 0 or not math.isfinite(floor_z):
                raise ValueError("invalid room height or floor")
            checks += tuple(_check("ceiling", "pass" if obj["_pos"][2]+obj["_size"][2] <=
                                   floor_z+h+tolerance_m else "violation", (obj["id"],)) for obj in normalized)
        for obj in normalized:
            checks += _bounds_checks(requested[obj["id"]], obj, tolerance_m)
            checks += (_support(requested[obj["id"]], obj, index, room, stage, tolerance_m),)
            if stage == "actual" and requested[obj["id"]].get("attributes"):
                checks += (_check("attributes_unverified", "unknown", (obj["id"],),
                                  reason="actual asset metadata has no verified typed-attribute contract; external evidence checker required"),)
            if stage == "actual" and requested[obj["id"]].get("semantic_front_required"):
                checks += (_check("semantic_front_required", "pass" if _face_angle(obj, stage) is not None else "unknown",
                                  (obj["id"],), reason="requires a finite nonzero horizontal semantic front"),)
            checks += _capability_checks(requested[obj["id"]], obj, stage)
            if requested[obj["id"]].get("capability_requirements"):
                checks += (_check("capability_requirements_unknown", "unknown", (obj["id"],),
                                  reason="use typed required_capabilities or an external capability checker"),)
        for obj in fixed:
            checks += _capability_checks(obj, obj, stage, fixed=True)
        checks += _collision_checks(normalized, fixed, tolerance_m)
        if room.get("openings"):
            checks += (_check("openings_unchecked", "unknown", reason="opening clearance conversion/validator evidence required"),)
        checks += tuple(_constraint(c, index, room, stage, tolerance_m) for c in condition.get("constraints", ()))
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        checks = (_check("invalid_geometry", "violation", message=str(exc)),)
    checks += tuple(_check(f"{level}_unchecked", "unknown", hard=level in required_levels)
                    for level in ("mesh", "physics", "solver"))
    return {"ok": not any(c["hard"] and c["status"] != "pass" for c in checks),
            "stage": stage, "geometry_level": "bbox", "checks": list(checks),
            "counts": {status: sum(c["status"] == status for c in checks) for status in CHECK_STATUSES},
            "unknown_checks": [c["code"] for c in checks if c["status"] == "unknown"]}
