"""Strict canonical protocol. Validation never repairs or mutates inputs."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from fastfill.v2 import SCHEMA_VERSION


def _keys(value, allowed, required, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(value) - set(allowed):
        raise ValueError(f"{label}: unknown fields {sorted(set(value) - set(allowed))}")
    if set(required) - set(value):
        raise ValueError(f"{label}: missing fields {sorted(set(required) - set(value))}")


def finite(value, label="value", positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{label} must be strictly positive")
    return float(value)


def vector(value, n=3, label="vector", positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{label} must have length {n}")
    return [finite(v, label, positive) for v in value]


def _id(value, label="id"):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{label} must be a nonempty string of at most 256 characters")
    return value


def _polygon(value, label):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        raise ValueError(f"{label} needs at least three XY points")
    points = [vector(p, 2, label) for p in value]
    area = abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, points[1:] + points[:1]))) / 2
    if area <= 0:
        raise ValueError(f"{label} must have positive area")
    # Exact shape validity is checked once at the input boundary, not repaired by buffer(0).
    from shapely.geometry import Polygon
    if not Polygon(points).is_valid:
        raise ValueError(f"{label} is self-intersecting or otherwise invalid")


ROOM_FIELDS = {"frame", "floor_polygon_xy_m", "floor_z_m", "height_m", "boundary_quality",
               "boundary_known", "floor_known", "fixed_objects", "room_type", "openings", "description"}
# exchangeable_group is supervision bookkeeping: it lives in validity (target order), never in the condition.
OBJECT_FIELDS = {"id", "category", "description", "support_parent", "support_surface_id",
                 "size_bounds_local_m", "fixed_size_local_m", "required_capabilities",
                 "retrieval_tolerance", "retrieval_tolerance_log", "attributes",
                 "constraint_role", "semantic_front_required"}


def migrate_legacy_row(row):
    """Move pre-C1 ``condition.objects[i].exchangeable_group`` into ``validity.exchangeable_group``
    and give pre-K1 rows an all-false ``validity.size_axis_swap_allowed``.

    The returned row never renders groups into the condition. Labels follow
    target order (like validity.position); rows without targets keep request order.
    """
    objects = row.get("condition", {}).get("objects", [])
    if not isinstance(objects, list):
        return row
    order = row.get("target", {}).get("objects") or objects
    if any(isinstance(o, Mapping) and "exchangeable_group" in o for o in objects):
        validity = row.get("validity", {})
        if "exchangeable_group" in validity:
            raise ValueError("exchangeable_group declared in both condition objects and validity")
        groups = {o.get("id"): o.get("exchangeable_group") for o in objects}
        condition = {**row["condition"], "objects": [{k: v for k, v in o.items() if k != "exchangeable_group"} for o in objects]}
        row = {**row, "condition": condition,
               "validity": {**validity, "exchangeable_group": [groups.get(o.get("id")) for o in order]}}
    validity = row.get("validity")
    if isinstance(validity, Mapping) and "size_axis_swap_allowed" not in validity:
        row = {**row, "validity": {**validity, "size_axis_swap_allowed": [False] * len(order)}}
    return row


def _metadata(value, label, depth=0):
    """Allow JSON metadata, with explicit nulls and finite numeric values."""
    if depth > 32:
        raise ValueError(f"{label} metadata exceeds nesting limit")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        finite(value, label)
    elif isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError(f"{label} metadata keys must be strings")
        for child in value.values():
            _metadata(child, label, depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _metadata(child, label, depth + 1)
    else:
        raise ValueError(f"{label} must contain JSON-compatible metadata")


def _capabilities(value, label, unknown_allowed=False):
    if unknown_allowed and value is None:
        return
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(f"{label} must contain nonempty capability names")


def _surfaces(value):
    if value is None:
        return
    if not isinstance(value, (list, tuple)):
        raise ValueError("verified support surfaces must be a list or null")
    ids = []
    for surface in value:
        _keys(surface, {"surface_id", "local_polygon_xy_m", "local_z_m"},
              {"surface_id", "local_polygon_xy_m", "local_z_m"}, "support surface")
        ids.append(_id(surface["surface_id"], "surface id"))
        _polygon(surface["local_polygon_xy_m"], "support surface polygon")
        finite(surface["local_z_m"], "support height")
    if len(ids) != len(set(ids)):
        raise ValueError("support surface IDs must be unique per asset")


def _support_graph(objects):
    parents = {obj["id"]: obj.get("support_parent") for obj in objects}
    for obj in objects:
        seen, current = set(), obj["id"]
        while current in parents:
            if current in seen:
                raise ValueError("support graph contains a cycle")
            seen = seen | {current}
            current = parents[current]


def _object(obj):
    _keys(obj, OBJECT_FIELDS, {"id", "category", "description"}, "request object")
    _id(obj["id"])
    _id(obj["category"], "category")
    if not isinstance(obj["description"], str):
        raise ValueError("description must be a string")
    for key in ("support_parent", "support_surface_id"):
        if key in obj and obj[key] is not None:
            _id(obj[key], key)
    if "constraint_role" in obj:
        _id(obj["constraint_role"], "constraint role")
    if "semantic_front_required" in obj and not isinstance(obj["semantic_front_required"], bool):
        raise ValueError("semantic_front_required must be a boolean")
    if "attributes" in obj:
        if not isinstance(obj["attributes"], Mapping):
            raise ValueError("attributes must be an object")
        _metadata(obj["attributes"], "attributes")
    if "fixed_size_local_m" in obj:
        fixed = obj["fixed_size_local_m"]
        if not isinstance(fixed, (list, tuple)) or len(fixed) != 3:
            raise ValueError("fixed size must contain three numbers or nulls")
        for value in fixed:
            if value is not None:
                finite(value, "fixed size", positive=True)
    if "size_bounds_local_m" in obj:
        bounds = obj["size_bounds_local_m"]
        _keys(bounds, {"min", "max"}, {"min", "max"}, "size bounds")
        lo, hi = vector(bounds["min"], positive=True), vector(bounds["max"], positive=True)
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError("size bounds min exceeds max")
        if "fixed_size_local_m" in obj:
            if any(v is not None and not a <= v <= b for v, a, b in zip(obj["fixed_size_local_m"], lo, hi)):
                raise ValueError("fixed size conflicts with size bounds")
    if "required_capabilities" in obj:
        _capabilities(obj["required_capabilities"], "required_capabilities")
    for key in ("retrieval_tolerance", "retrieval_tolerance_log"):
        if key in obj:
            value = obj[key]
            values = vector(value) if isinstance(value, (list, tuple)) else [finite(value)]
            if any(v < 0 for v in values):
                raise ValueError("retrieval tolerance must be nonnegative")


def _constraints(constraints, refs):
    if not isinstance(constraints, list):
        raise ValueError("constraints must be a list")
    for c in constraints:
        if not isinstance(c, Mapping) or not isinstance(c.get("type"), str):
            raise ValueError("constraint needs a type")
        kind = _id(c["type"], "constraint type")
        _metadata(c, "constraint")
        if "hard" in c and not isinstance(c["hard"], bool):
            raise ValueError("constraint hard flag must be a boolean")
        if kind in {"faces_direction", "faces", "near", "clearance", "on", "between", "against_wall"}:
            if c.get("object_id") not in refs:
                raise ValueError("known constraint requires an object reference")
        if kind in {"faces", "near", "clearance"}:
            if c.get("target_id") not in refs or c["target_id"] == c["object_id"]:
                raise ValueError("binary constraint requires a distinct object target")
        if kind == "between":
            targets = c.get("target_ids")
            if (not isinstance(targets, list) or len(targets) != 2
                    or any(not isinstance(target, str) for target in targets) or len(set(targets)) != 2 or any(
                    target not in refs or target == c["object_id"] for target in targets)):
                raise ValueError("between requires two distinct valid target objects")
        if kind == "on":
            target = c.get("target_id", c.get("parent_id"))
            if target not in refs | {"floor", "wall"} or target == c["object_id"]:
                raise ValueError("on constraint requires a distinct support parent")
            if "target_id" in c and "parent_id" in c and c["target_id"] != c["parent_id"]:
                raise ValueError("on constraint has conflicting parent references")
        for kind_name, field in (("near", "max_distance_m"), ("clearance", "min_distance_m")):
            if kind == kind_name and field not in c:
                raise ValueError(f"{kind} constraint requires {field}")
        for key in ("object_id", "target_id", "parent_id"):
            if key in c and c[key] not in refs | {"floor", "wall"}:
                raise ValueError(f"constraint references missing {key}")
        if c["type"] == "faces_direction":
            direction = vector(c.get("direction_xy"), 2)
            if math.hypot(*direction) == 0:
                raise ValueError("facing direction must be nonzero")
        if c["type"] == "keepout":
            _polygon(c.get("polygon_xy_m"), "keepout polygon")
        for key in ("tolerance_rad", "tolerance_m", "max_distance_m", "min_distance_m"):
            if key in c and finite(c[key], key) < 0:
                raise ValueError(f"{key} must be nonnegative")


def validate_condition(condition):
    _keys(condition, {"schema_version", "room", "objects", "constraints"},
          {"schema_version", "room", "objects", "constraints"}, "condition")
    if condition["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    room = condition["room"]
    _keys(room, ROOM_FIELDS, {"frame", "floor_polygon_xy_m"}, "room")
    if room["frame"] != "right_handed_z_up":
        raise ValueError("room frame must be right_handed_z_up")
    _polygon(room["floor_polygon_xy_m"], "floor polygon")
    for key in ("floor_z_m", "height_m"):
        if room.get(key) is not None:
            finite(room[key], key, positive=key == "height_m")
    for key in ("floor_known", "boundary_known"):
        if key in room and not isinstance(room[key], bool):
            raise ValueError(f"{key} must be a boolean")
    if room.get("floor_known") is True and room.get("floor_z_m") is None:
        raise ValueError("known floor requires a floor height")
    fixed = room.get("fixed_objects", [])
    if not isinstance(fixed, list):
        raise ValueError("fixed objects must be a list")
    fixed_ids = []
    for obj in fixed:
        _keys(obj, {"id", "size_local_m", "bottom_center_m", "yaw_rad", "category", "support_surfaces",
                    "capabilities", "required_capabilities", "semantic_front_local"},
              {"id", "size_local_m", "bottom_center_m", "yaw_rad"}, "fixed object")
        fixed_ids.append(_id(obj["id"]))
        vector(obj["size_local_m"], positive=True)
        vector(obj["bottom_center_m"])
        finite(obj["yaw_rad"], "fixed yaw")
        if "category" in obj:
            _id(obj["category"], "fixed category")
        for key in ("capabilities", "required_capabilities"):
            if key in obj:
                _capabilities(obj[key], key, unknown_allowed=key == "capabilities")
        if "support_surfaces" in obj:
            _surfaces(obj["support_surfaces"])
        if obj.get("semantic_front_local") is not None:
            front = vector(obj["semantic_front_local"])
            if math.hypot(*front[:2]) == 0 or abs(front[2]) > 1e-6:
                raise ValueError("semantic front must be a nonzero horizontal vector")
    if not isinstance(condition["objects"], list):
        raise ValueError("objects must be a list")
    for obj in condition["objects"]:
        _object(obj)
    ids = [obj["id"] for obj in condition["objects"]] + fixed_ids
    if len(set(ids)) != len(ids) or any(i in {"floor", "wall"} for i in ids):
        raise ValueError("object IDs must be unique and not reserved")
    for obj in condition["objects"]:
        parent = obj.get("support_parent")
        if parent is not None and (parent not in set(ids) | {"floor", "wall"} or parent == obj["id"]):
            raise ValueError("invalid support parent")
    _support_graph(condition["objects"])
    for key in ("room_type", "description", "boundary_quality"):
        if room.get(key) is not None and not isinstance(room[key], str):
            raise ValueError(f"room {key} must be a string or null")
    if "openings" in room:
        if not isinstance(room["openings"], list):
            raise ValueError("room openings must be a list")
        _metadata(room["openings"], "room openings")
    _constraints(condition["constraints"], set(ids))
    from fastfill.v2.validation import effective_support_requests
    effective_support_requests(condition)
    return condition


def validate_layout(layout, condition=None):
    _keys(layout, {"schema_version", "objects"}, {"schema_version", "objects"}, "layout")
    if layout["schema_version"] != SCHEMA_VERSION or not isinstance(layout["objects"], list):
        raise ValueError("invalid layout schema")
    ids = []
    for obj in layout["objects"]:
        _keys(obj, {"id", "target_size_local_m", "bottom_center_m", "yaw_rad", "support_parent"},
              {"id", "target_size_local_m", "bottom_center_m", "yaw_rad"}, "output object")
        if "support_parent" in obj:
            _id(obj["support_parent"], "output support_parent")
        ids.append(_id(obj["id"]))
        vector(obj["target_size_local_m"], positive=True)
        vector(obj["bottom_center_m"])
        yaw = finite(obj["yaw_rad"], "yaw")
        if not -math.pi <= yaw < math.pi:
            raise ValueError("yaw must be wrapped to [-pi, pi)")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate output IDs")
    if condition is not None and set(ids) != {o["id"] for o in condition["objects"]}:
        raise ValueError("output IDs must equal all requested IDs exactly once")
    # optional predicted support: floor, wall, another output object or a fixed object; never itself, no cycle
    parents = {obj["id"]: obj["support_parent"] for obj in layout["objects"] if "support_parent" in obj}
    known = {"floor", "wall", *ids, *(f["id"] for f in (condition or {}).get("room", {}).get("fixed_objects", ()))}
    for child, parent in parents.items():
        if parent == child or (condition is not None and parent not in known):
            raise ValueError(f"output support_parent of {child!r} must name the floor, a wall, another object or a fixed object")
        seen = {child}
        while parent in parents:
            if parent in seen:
                raise ValueError("output support_parent references form a cycle")
            seen.add(parent)
            parent = parents[parent]
    return layout


def normalize_room(room, reference_height=3.0):
    """Origin/scales depend only on room input; missing height uses fixed 3m."""
    points = room["floor_polygon_xy_m"]
    xmin, ymin = [min(p[q] for p in points) for q in range(2)]
    xmax, ymax = [max(p[q] for p in points) for q in range(2)]
    height = room.get("height_m") or finite(reference_height, positive=True)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("room XY normalization needs positive extents")
    return [float(xmin), float(ymin), float(room.get("floor_z_m") or 0.)], [float(xmax - xmin), float(ymax - ymin), float(height)]
