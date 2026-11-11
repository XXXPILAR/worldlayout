"""Input boundary: reuse the project's canonical protocol without modifying it."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path

# The FastFill package: the copy bundled in this folder (``fastfill/``, the files the workflow and FastFill inference
# need, fastfill/SOURCE.json), else a sibling ``fastfill-main/`` or the checkout this folder sits in.
_here = Path(__file__).resolve().parents[1]
_parent = _here.parent
_main = next((p for p in (_here, _parent / "fastfill-main", _parent) if (p / "fastfill" / "v2").is_dir()), None)
if _main is None:
    raise RuntimeError(f"Expected the FastFill package at {_here}, {_parent / 'fastfill-main'} or {_parent}")
sys.path.insert(0, str(_main))

from fastfill.v2.schema import validate_condition, validate_layout  # noqa: E402

SUPPORTED_CONSTRAINTS = {
    "faces_direction", "faces", "near", "clearance", "on", "between",
    "against_wall", "keepout",
}


DEFAULT_HEIGHT_M = 2.8


def fill_room_defaults(condition: dict) -> tuple[dict, list[str]]:
    """A copy of ``condition`` whose room has a floor height and a ceiling height, plus notes on what was assumed.

    An unknown boundary keeps the given (hull) polygon; a missing floor height is the lowest fixed-object bottom (else
    0); a missing room height is the default, raised above the tallest fixed object."""
    condition = deepcopy(condition)
    room, notes = condition["room"], []
    fixed = room.get("fixed_objects", [])
    if room.get("boundary_known") is False:
        notes.append(f"boundary unknown: using the given {room.get('boundary_quality', 'polygon')} as the room")
    if room.get("floor_z_m") is None or room.get("floor_known") is False:
        floor = min((obj["bottom_center_m"][2] for obj in fixed), default=0.0) if room.get("floor_z_m") is None \
            else room["floor_z_m"]
        notes.append(f"floor height unknown: assumed {floor:.3f} m")
        room["floor_z_m"] = floor
    if room.get("height_m") is None:
        top = max((obj["bottom_center_m"][2] + obj["size_local_m"][2] for obj in fixed), default=0.0)
        room["height_m"] = max(DEFAULT_HEIGHT_M, top - room["floor_z_m"] + .1)
        notes.append(f"room height unknown: assumed {room['height_m']:.2f} m")
    return condition, notes


def _bare(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def missing_descriptions(condition: dict) -> list[str]:
    """Ids of objects to place whose description is empty or only repeats the category name."""
    return [obj["id"] for obj in condition["objects"]
            if not _bare(obj.get("description") or "") or _bare(obj["description"]) == _bare(obj["category"])]


def check_request(condition: dict, *, require_descriptions: bool | None = None) -> dict:
    """Validates the request and returns it with room defaults filled in (see ``fill_room_defaults``).

    Every object to place must carry its own description (more than its category name), as real requests do. The
    older category-only test sets (test300 / exp150) are replayed with ``require_descriptions=False`` or, for whole
    runs, the environment variable WORKFLOW_ALLOW_BARE_DESCRIPTIONS=1."""
    validate_condition(condition)
    if require_descriptions is None:
        require_descriptions = os.environ.get("WORKFLOW_ALLOW_BARE_DESCRIPTIONS") != "1"
    if require_descriptions and (missing := missing_descriptions(condition)):
        raise ValueError(f"{len(missing)} of {len(condition['objects'])} objects have no description beyond their "
                         f"category (every object needs one): {missing[:10]}")
    condition, _ = fill_room_defaults(condition)
    room = condition["room"]
    forbidden = {"fixed_size_local_m", "size_bounds_local_m"}
    for obj in condition["objects"]:
        if forbidden.intersection(obj):
            raise ValueError(f"{obj['id']}: requested furniture size must not be an input")
    for relation in condition["constraints"]:
        if relation["type"] not in SUPPORTED_CONSTRAINTS:
            raise ValueError(f"unsupported constraint type: {relation['type']}")
    for opening in room.get("openings", []):
        if opening.get("type") not in {"door", "window"}:
            raise ValueError("only wall-relative door/window openings are supported")
        wall = opening.get("wall_index")
        if isinstance(wall, bool) or not isinstance(wall, int) or not 0 <= wall < len(room["floor_polygon_xy_m"]):
            raise ValueError("opening.wall_index is invalid")
        position = opening.get("position_on_wall")
        width, height = opening.get("width_m"), opening.get("height_m")
        if (isinstance(position, bool) or not isinstance(position, (int, float)) or
                not 0 <= position <= 1 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0
                                               for v in (width, height))):
            raise ValueError("opening position, width, and height must be valid")
    return condition
