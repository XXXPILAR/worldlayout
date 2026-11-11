"""Step ③ of the draft workflow: one LLM call places every object at its fixed size.

Sizes come from step ② (one per object, e.g. FastFill's prediction); support, priority, groups and relations from
step ① (``make_plan``). The LLM answers positions and yaws only; heights follow from the
support (the floor, a wall-mounted default, or the top of the parent), and sizes stay the given ones whatever the
answer says. With ``free_sizes`` (the workflow without FastFill) the request gives no sizes and the LLM answers each
object's size with its pose (``FREE_SIZE_INSTRUCTIONS``); an object it gives no valid size keeps the given one. An object the answer misses or gets wrong is still placed (on its parent, else at the room's
representative point) and counted in ``missing``, so the draft is always complete.
"""

from __future__ import annotations

import json
import math

from . import llm as llm_client
from .geometry import KEEPOUT_DEPTH_M, KEEPOUT_SIDE_M, Geometry
from .planning import Plan, _number
from .solver import _pose

WALL_BOTTOM_M = 1.15

DRAFT_INSTRUCTIONS = """You place furniture in a room. The user message is JSON: the room (floor polygon in metres, \
world x/y, z up; openings on its walls; fixed objects that cannot move, with bottom centre, size and yaw), the \
objects to place and the constraints, plus a plan: each object's support, placement priority, functional groups and \
soft relations. Every object's size is already fixed (size_m: width = left-right as seen from its front, depth = \
front-back, height); do not change sizes.

Answer with one JSON object only, no prose:
{"objects": {"<id>": {"x": x, "y": y, "yaw_deg": a}}}
optionally with "z" (bottom height above the floor, metres) for objects supported by "wall".

Rules:
- Include every object id exactly once; do not add or remove objects.
- x, y: the object's bottom-centre position in world metres. yaw_deg: the direction its front faces, counter-clockwise \
degrees from the world +x axis; the depth axis points along it.
- An object supported by another object sits on top of it: give x, y within the parent's top.
- Hard constraints must hold. Keep objects inside the floor polygon, off each other and off fixed objects, and keep \
doors clear: nothing at any height (no furniture, nothing hung on the wall, nothing on top of something) within 1.0 m \
in front of a door, from 0.15 m left of it to 0.15 m right of it; only a rug or mat may lie there. Follow the plan's \
groups and relations where you can.
- Wall-supported objects hang with their back against a wall, facing into the room, never over a window."""
# the keepout the critic enforces (geometry's door rule "keepout"), as the rule above words it
assert f"{KEEPOUT_DEPTH_M:.1f} m in front of a door, from {KEEPOUT_SIDE_M:.2f} m" in DRAFT_INSTRUCTIONS

# the draft without FastFill: the LLM gives every size with the pose (the request has no size_m)
FIXED_SIZE_TEXT = """Every object's size is already fixed (size_m: width = left-right as seen from its front, depth = \
front-back, height); do not change sizes."""
FIXED_ANSWER = """{"objects": {"<id>": {"x": x, "y": y, "yaw_deg": a}}}"""
assert FIXED_SIZE_TEXT in DRAFT_INSTRUCTIONS and FIXED_ANSWER in DRAFT_INSTRUCTIONS
FREE_SIZE_INSTRUCTIONS = DRAFT_INSTRUCTIONS.replace(FIXED_SIZE_TEXT, """No object has a size yet: give each object the \
real-world size it should have in this room, following its description (a "large", "king size" or "three-seat" one \
is sized so), with width_m = left-right as seen from its front, depth_m = front-back, height_m = bottom to top, in \
metres.""").replace(FIXED_ANSWER, """{"objects": {"<id>": {"width_m": w, "depth_m": d, "height_m": h, "x": x, "y": y, \
"yaw_deg": a}}}""")


# appended to DRAFT_INSTRUCTIONS when the objects carry a learned facing (``suggested_yaw_deg``)
SUGGESTED_YAW_RULE = """
- suggested_yaw_deg, when given, is the facing a model trained on real rooms predicts for the object (same convention \
as yaw_deg). Use it as the object's yaw and choose the position to suit it (e.g. the wall its back goes against). \
Depart from it only where it cannot work: it would push the object through a wall, block a door, or collide; then \
choose a yaw that fits."""


# appended when the request says which objects the task needs (the others complete the room and may give way)
REQUIRED_RULE = """
- Objects marked "required": true are what the task needs. Place each of them legally (inside the room, off every \
other object, doors clear) and where the robot can reach it (and open it, when it has doors or drawers). Place the \
other objects the same way, legally and within the robot's reach; where both cannot fit, the required ones come \
first and the others give way."""


# appended when a robot uses the room (``robot.Robot``); the objects it concerns are marked in the request
ROBOT_RULE = """
- A mobile robot {width} m wide works in this room. Leave it free floor at least {width} m wide from the doors to \
every object marked "robot_works_here" (it holds objects the robot picks up or puts down) or "front_clearance_m", and \
room to stand beside it: in front of it (also a side, for beds, tables and seats), at least {width} m deep.
- "front_clearance_m": the object has doors or drawers. Keep at least that much free floor in front of it: do not \
face it to a wall or put a bed, sofa, table, chair or other furniture there.
- An object on furniture marked "robot_works_here": keep it within {reach} m of an edge the robot can stand at (not \
in the middle of a large table or at the back of a deep counter)."""


def robot_marks(condition: dict, plan: Plan, sizes: dict[str, list[float]], robot) -> dict[str, dict]:
    """{id: request fields} for ``ROBOT_RULE``: furniture the robot works at, openable furniture's front clearance."""
    from fastfill.v2.ops import layout_quality as lq
    from .robot import openable, sweep_depth
    floor = float(condition["room"].get("floor_z_m") or 0.)
    boxes = {b["id"]: b for b in lq.boxes({"condition": condition}, {"objects": [
        _pose(o["id"], sizes[o["id"]], 0., 0., floor, 0., "floor") for o in condition["objects"]
        if plan.parents.get(o["id"]) == "floor"]})}
    marks: dict[str, dict] = {}
    for ident, b in boxes.items():
        if openable(b):
            marks.setdefault(ident, {})["front_clearance_m"] = round(max(sweep_depth(b), robot.width_m), 2)
    for child, parent in plan.parents.items():
        while parent in plan.parents and parent not in boxes:  # an object on an object: the furniture under both
            parent = plan.parents[parent]
        if parent in boxes and child in sizes and child != parent:
            marks.setdefault(parent, {})["robot_works_here"] = True
    return marks


def request_text(condition: dict, plan: Plan, sizes: dict[str, list[float]],
                 yaws: dict[str, float] | None = None, robot=None, free_sizes: bool = False,
                 required: set[str] | None = None) -> str:
    """The placement request; ``free_sizes``: without size_m (the LLM gives the sizes; ``sizes`` then only serve
    the robot marks' openable test, which needs a box); ``required``: those objects are marked "required"."""
    marks = robot_marks(condition, plan, sizes, robot) if robot is not None else {}
    objects = []
    for obj in condition["objects"]:
        ident = obj["id"]
        depth, width, height = sizes[ident]
        entry = {"id": ident, "category": obj["category"], "description": obj.get("description", ""),
                 "support": plan.parents[ident]}
        if not free_sizes:
            entry["size_m"] = {"width": round(width, 3), "depth": round(depth, 3), "height": round(height, 3)}
        if required and ident in required:
            entry["required"] = True
        if ident in plan.priority:
            entry["priority"] = plan.priority[ident]
        if yaws and ident in yaws:
            entry["suggested_yaw_deg"] = round(math.degrees(math.remainder(float(yaws[ident]), 2 * math.pi)), 1)
        entry.update(marks.get(ident, {}))
        objects.append(entry)
    relations = [{k: v for k, v in r.items() if k not in ("origin", "hard")}
                 for r in plan.relations if r.get("origin") != "group"]
    payload = {"room": condition["room"], "objects": objects, "constraints": condition["constraints"],
               "plan": {"groups": plan.groups, "relations": relations}}
    return json.dumps(payload, ensure_ascii=False)


def robot_rule(robot) -> str:
    if robot is None:
        return ""
    return ROBOT_RULE.format(width=round(robot.width_m, 2), reach=round(robot.arm_reach_m - robot.half_width, 2))


def parse_poses(text: str, ids: set[str]) -> tuple[dict[str, dict], dict]:
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("LLM response does not contain a JSON object")
    answer = json.loads(text[first:last + 1])
    entries = answer.get("objects") if isinstance(answer, dict) else None
    if not isinstance(entries, dict):
        raise ValueError("LLM response has no objects map")
    poses, dropped = {}, {"unknown_object": 0, "pose_invalid": 0}
    for ident, entry in entries.items():
        if ident not in ids:
            dropped["unknown_object"] += 1
            continue
        if not isinstance(entry, dict):
            dropped["pose_invalid"] += 1
            continue
        x, y = _number(entry.get("x"), -1e4, 1e4), _number(entry.get("y"), -1e4, 1e4)
        yaw = _number(entry.get("yaw_deg"), -1e4, 1e4)
        if None in (x, y, yaw):
            dropped["pose_invalid"] += 1
            continue
        poses[ident] = {"x": x, "y": y, "yaw_rad": math.remainder(math.radians(yaw), 2 * math.pi),
                        "z": _number(entry.get("z"), -1e3, 1e3)}
    return poses, {k: v for k, v in dropped.items() if v}


def parse_sizes(text: str, ids: set[str]) -> dict[str, list[float]]:
    """{id: [depth, width, height]} from the answer's width_m / depth_m / height_m (``FREE_SIZE_INSTRUCTIONS``);
    an entry without three valid sizes is left out."""
    answer = json.loads(text[text.find("{"):text.rfind("}") + 1])
    sizes = {}
    for ident, entry in (answer.get("objects") or {}).items():
        if ident in ids and isinstance(entry, dict):
            values = [_number(entry.get(k), 1e-3, 50.) for k in ("depth_m", "width_m", "height_m")]
            if None not in values:
                sizes[ident] = values
    return sizes


def assemble(geometry: Geometry, plan: Plan, sizes: dict[str, list[float]],
             poses: dict[str, dict], *, ids: list[str] | None = None,
             placed: dict[str, dict] | None = None) -> tuple[dict[str, dict], list[str]]:
    """Boxes for every request (or only ``ids``, beside the boxes already ``placed``), parents before children;
    returns (placed, ids without a usable pose)."""
    placed, missing = dict(placed or {}), []
    wanted = set(geometry.requests) if ids is None else set(ids)
    anchor = geometry.polygon.representative_point()

    def put(ident: str) -> None:
        if ident in placed:
            return
        parent = plan.parents.get(ident, "floor")
        if parent in wanted:
            put(parent)
        parent_box = placed.get(parent, geometry.fixed.get(parent))
        if parent not in ("floor", "wall") and parent_box is None:
            parent = "floor"
        size, pose = sizes[ident], poses.get(ident)
        if pose is None:
            missing.append(ident)
            if parent_box is not None:
                x, y = parent_box["bottom_center_m"][:2]
                yaw = parent_box["yaw_rad"]
            else:
                x, y, yaw = anchor.x, anchor.y, 0.0
        else:
            x, y, yaw = pose["x"], pose["y"], pose["yaw_rad"]
        if parent == "floor":
            x, y = geometry.push_inside(size, x, y, yaw)  # flush with (or a few cm through) a wall: just inside it
            z = geometry.floor
        elif parent == "wall":
            x, y, yaw = geometry.snap_to_wall(size, x, y, yaw)  # hung on the nearest wall, back against it
            top = geometry.ceiling - size[2] - .03
            given = pose.get("z") if pose else None
            z = geometry.floor + given if given is not None and 0 <= given <= top - geometry.floor else \
                min(geometry.floor + WALL_BOTTOM_M, top)
            z = max(z, geometry.floor)
        else:
            z = geometry._support_height(_pose(ident, size, x, y, 0, yaw, parent), parent, parent_box)
        placed[ident] = _pose(ident, size, x, y, z, yaw, parent)

    for ident in (geometry.requests if ids is None else ids):
        put(ident)
    return placed, missing


def make_draft(condition: dict, plan: Plan, sizes: dict[str, list[float]], geometry: Geometry, *,
               yaws: dict[str, float] | None = None, robot=None,
               model: str | None = None, base_url: str | None = None,
               api_key: str | None = None, free_sizes: bool = False,
               required: set[str] | None = None) -> tuple[dict[str, dict], dict]:
    """Returns (placed, info) with info = {calls, dropped, missing, error}. ``yaws`` (rad, e.g. FastFill's) go to
    the LLM as ``suggested_yaw_deg``: it decides each position for that facing instead of having it swapped in.
    ``robot`` (``robot.Robot``): the request marks what the robot needs (``ROBOT_RULE``). ``free_sizes``: the LLM
    gives the sizes too; info then also holds ``sizes`` (the ones placed: the LLM's, else the given ones),
    ``size_source`` ({"draft": n, "given": n}) and ``sized_by_draft`` (the ids the LLM sized). ``required``: the
    task's objects, marked in the request (``REQUIRED_RULE``)."""
    info: dict = {"calls": [], "dropped": {}, "missing": [], "error": None}
    poses: dict[str, dict] = {}
    drafted: dict[str, list[float]] = {}
    try:
        instructions = (FREE_SIZE_INSTRUCTIONS if free_sizes else DRAFT_INSTRUCTIONS) + \
            (SUGGESTED_YAW_RULE if yaws else "") + (REQUIRED_RULE if required else "") + robot_rule(robot)
        text, call = llm_client.ask(instructions, request_text(condition, plan, sizes, yaws, robot, free_sizes, required),
                                    model=model, base_url=base_url, api_key=api_key)
        info["calls"].append(call)
        poses, info["dropped"] = parse_poses(text, set(geometry.requests))
        if free_sizes:
            drafted = parse_sizes(text, set(geometry.requests))
    except (RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and "requires --model" in str(exc):
            raise
        info["error"] = f"{type(exc).__name__}: {exc}"[:300]
    if free_sizes:
        sizes = {**sizes, **drafted}
        info.update(sizes=sizes, size_source={"draft": len(drafted), "given": len(sizes) - len(drafted)},
                    sized_by_draft=sorted(drafted))
    placed, info["missing"] = assemble(geometry, plan, sizes, poses)
    return placed, info
