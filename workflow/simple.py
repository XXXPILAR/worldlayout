"""Workflow 2 (``placement="simple"``, the default here): one placement, one program check, one LLM check.

  ① plan     one LLM call for supports, groups and relations (``planning.make_plan``, semantic only); ``plan_llm=False``
             uses the rules
  ② sizes    one per object: FastFill's when given, else the catalog's
  ③ place    ONE LLM call places every object of the room (``drafting.make_draft``; FastFill gives the sizes only)
  ④ check    the program checks once (``critic.Critic``). If there are hard violations (collision, out of bounds, no
             support, blocked door, broken hard constraint), one repair pass takes them up, each once, by the program
             search at fixed sizes; soft costs are not chased and there is no polish, so wherever the placement is legal
             it is left as the LLM made it
  ④b give way with the task's required objects (``required``). Every object, the expansion too, is checked and
             repaired alike in step ④ (legal, and for a robot reachable and free to open), the required objects'
             violations first (the search moves and relocates blockers; no object is ever resized).
             An expansion object is removed only to give way to a required one, and only when nothing else works:
             (1) for each hard violation step ④ leaves on a required object (a collision with it, its reach or
             doors blocked, the room's doors cut off), the expansion object in the most (and largest) of them
             (``give_way_choice``) is searched for anywhere in the room (with what stands on it) and
             moved when the critic prefers that and it is then in no hard violation; only if that fails is it
             removed, with what stands on it; (2) if something was removed and hard violations are left, one more
             repair pass on the smaller room; (3) the expansion objects' own violations are tried the same way, by
             moving only (what cannot be fixed is left out in ④d); (4) if that left a required object a violation,
             step (1) once more. Required objects are never removed or shrunk
  ④c required first: if a required object still has a hard violation (they conflict among themselves, or the
             room's search ran out of time), the required objects alone (the expansion taken out) are repaired again
             and, if that does not clear them, searched for from scratch (``solver.beam_search``, then a repair);
             the expansion objects, those ④b removed included, then come back one at a time (furniture first, the
             larger first, what stands on an object after it), each where it was (an object on a required piece of furniture moved with it,
             ``carry``) if the required objects keep their violations no higher, else searched for anywhere in the
             room under that condition, else removed (with what stands on it). Kept only if the required objects end
             with fewer violations; else the layout before ④c stays
  ④d left out (task mode, since 2026-11-06; after ⑤): an expansion object still in a PHYSICAL violation (it
             collides, sticks out of the room, stands on nothing, blocks a door or a window), one at a time, the one
             in the most such violations first, then the smaller (``give_way_choice``): one standing on another
             object (with nothing on it) goes to the nearest free place on that top (``free_slot``) if there is
             one; else it is removed, with what stands on it, until none is left: as Holodeck drops the small objects
             that still collide and infinite-task-world's placer packs a top and leaves out what fits nowhere. An
             overfull shelf keeps what fits. Never a required object or one a required object stands on; relation
             and robot violations are not reasons
  ④e wall fronts clear (task mode, since 2026-11-07; after ④d): a wall object whose front strip, as SceneEval's
             accessibility scores it, is mostly not free floor (it hangs over furniture) slides along the walls to
             the nearest place where it is, with what stands on it, if that makes no violation and breaks no
             relation (``clear_wall_fronts``); nothing is removed and nothing on the floor moves
  ⑦ fallback (task mode): if a required object is still not legal and reachable with its constraints met, the
             required objects alone are laid out by the program (``required_only``: no LLM; from where they stand and
             searched for from scratch, each repaired) and that room is returned instead when it leaves the required
             objects fewer violations (``status`` "fallback_required_only", the full layout kept in the diagnostics).
             The pipeline also returns it when the workflow fails on an error (``pipeline.run_draft_workflow``), so a
             room is always delivered
  ⑤ review   the LLM reviews the whole room once (``semantic.SemanticPass``, one round): only severe issues are fixed;
             each fix is repaired by the program and kept only if the program gate passes (no new hard violation;
             reach, access and doors no worse) and a blind pairwise comparison prefers it in both orders
  ⑥ result   the layout, plus the layout after each of ③ ④ ⑤ (``diagnostics.stages``), so a bad result can be traced

LLM calls: 1 plan + 1 placement (``drafts`` of them) + 1 review + 3 per severe issue fixed (proposal, two pairwise
votes).
"""

from __future__ import annotations

import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box as shapely_box
from shapely.ops import unary_union

from .catalog import SizeCatalog
from .contract import check_request, fill_room_defaults, validate_condition, validate_layout
from .credentials import FatalAPIError
from .critic import Critic, Violation, critic_geometry, identity, load_templates
from .drafting import assemble, make_draft
from .loop import RELOCATE_DRAFT_WEIGHT, FixedSizes, Loop, LoopBudget, make_ask
from .geometry import WALL_GAP_M, Geometry, footprint
from .solver import _wrap, beam_search
from .planning import Plan, _required_parents, make_plan
from .robot import Robot, RobotChecker, RobotZones
from .semantic import SOFT_TOLERANCE, SemanticPass


def draft_rank(a) -> tuple:
    """Which of several drafts to repair: the fewest hard violations of any tier, then the smallest total. Not the
    critic's tiered key: a draft's collisions and boundary overlaps are mostly small and the repair fixes them,
    while its robot violations mostly stay (20 rooms, two drafts each: preferring fewer legality violations picked a
    draft with 23 robot violations over one with 3)."""
    return len(a.hard), round(sum(v.magnitude for v in a.hard), 4)


def _layout(condition: dict, placed: dict[str, dict]) -> dict:
    return {"schema_version": condition["schema_version"], "objects": [placed[o["id"]] for o in condition["objects"]]}


def task_robot(robot: Robot, required: set[str], articulation: set[str]) -> Robot:
    """The robot checks for a task: every check of ``robot`` (doors linked, furniture carrying objects and openable
    furniture reachable, openable furniture free to open, items within the arm's reach), for the expansion objects
    too, plus the required objects reached (an object on furniture stands for the furniture under it) and the
    articulation objects free to open whatever their category."""
    return replace(robot, reach_ids=frozenset(robot.reach_ids) | frozenset(required),
                   openable_ids=frozenset(robot.openable_ids) | frozenset(articulation))


def _refs(r: dict) -> set[str]:
    return {r.get("object_id"), r.get("target_id"), r.get("parent_id"), *(r.get("target_ids") or [])} - {None}


def without(condition: dict, plan: Plan, gone: set[str]) -> tuple[dict, Plan]:
    """The room and the plan without the objects in ``gone``: their constraints and plan relations go too, a group
    loses them (and goes when its anchor goes)."""
    keep = {o["id"] for o in condition["objects"]} - gone
    known = keep | {f["id"] for f in condition["room"].get("fixed_objects", [])}
    room = copy.deepcopy(condition)
    room["objects"] = [o for o in room["objects"] if o["id"] in keep]
    room["constraints"] = [r for r in room["constraints"] if _refs(r) <= known]
    groups = [{**g, "members": [m for m in g.get("members", []) if m in keep]} for g in plan.groups
              if g.get("anchor") not in gone]
    return room, replace(plan, parents={k: v for k, v in plan.parents.items() if k in keep},
                         relations=[r for r in plan.relations if _refs(r) <= known], groups=groups,
                         priority={k: v for k, v in plan.priority.items() if k in keep},
                         draft_poses={k: v for k, v in plan.draft_poses.items() if k in keep})


def descendants(plan: Plan, ident: str) -> set[str]:
    out, frontier = set(), [ident]
    while frontier:
        parent = frontier.pop()
        for child, p in plan.parents.items():
            if p == parent and child not in out:
                out.add(child)
                frontier.append(child)
    return out


def on_required(v, required: set[str]) -> bool:
    """A hard violation of the required objects: a legality one holding one of them, a robot check about one of them
    (its first id; the others are what is in the way), or the room's doors cut off."""
    if v.code == "robot_door":
        return True
    if v.code.startswith("robot_"):
        return bool(v.ids) and v.ids[0] in required
    return bool(set(v.ids) & required)


def carry(child: dict, before: dict, after: dict) -> dict:
    """``child`` moved rigidly with the furniture under it, from pose ``before`` to ``after`` (same size)."""
    (x0, y0, z0), (x1, y1, z1) = before["bottom_center_m"], after["bottom_center_m"]
    turn = after["yaw_rad"] - before["yaw_rad"]
    dx, dy = child["bottom_center_m"][0] - x0, child["bottom_center_m"][1] - y0
    c, s = math.cos(turn), math.sin(turn)
    x, y = x1 + c * dx - s * dy, y1 + s * dx + c * dy
    return {**child, "bottom_center_m": [round(x, 4), round(y, 4), child["bottom_center_m"][2] + (z1 - z0)],
            "yaw_rad": _wrap(child["yaw_rad"] + turn)}


NEAR_BLOCKER_M = 2.5  # expansion floor objects this near a required object a robot check fails for are tried
NEAR_BLOCKER_TRIES = 5  # at most this many of them per give-way pass (each one is a search over the room)


# the hard violations ④d leaves an expansion object out for: its own placement, not a relation or a robot check
PHYSICAL = frozenset({"collision", "boundary", "vertical_bounds", "floor_support", "wall_support", "object_support",
                      "parent_not_placed", "door_clearance", "window_clearance", "keepout"})


SLOT_STEP_M = .05


def free_slot(geometry: Geometry, placed: dict[str, dict], ident: str, pack: bool = False) -> dict | None:
    """``ident`` (standing on another object) at the nearest point of a SLOT_STEP_M lattice over its parent's top
    where its whole footprint lies on the top and it is in no PHYSICAL violation, at its yaw or a quarter turn
    from it; None when there is none, or when it stands on the floor or a wall. infinite-task-world's placer packs
    a parent's top the same way (``_top_slot``) and leaves out what has no slot."""
    box = placed[ident]
    parent_id = geometry.parents.get(ident)
    parent = placed.get(parent_id, geometry.fixed.get(parent_id))
    if parent is None:
        return None
    surface = geometry._support_polygon(box, parent_id, parent)
    if surface.is_empty:
        return None
    z = geometry._support_height(box, parent_id, parent)
    others = {k: v for k, v in placed.items() if k != ident}
    on_top = surface.buffer(1e-6)
    x0, y0, x1, y1 = surface.bounds
    points = [(x0 + i * SLOT_STEP_M, y0 + j * SLOT_STEP_M) for i in range(int((x1 - x0) / SLOT_STEP_M) + 1)
              for j in range(int((y1 - y0) / SLOT_STEP_M) + 1)]
    # the nearest place to where it is; packing (``repack``): a corner first, row by row
    points.sort(key=(lambda p: (p[1], p[0])) if pack else (lambda p: math.dist(p, box["bottom_center_m"][:2])))
    for x, y in points:
        for yaw in (box["yaw_rad"], box["yaw_rad"] + math.pi / 2):
            candidate = {**box, "bottom_center_m": [round(x, 4), round(y, 4), z], "yaw_rad": _wrap(yaw)}
            if on_top.covers(footprint(candidate)) and \
                    not any(i.code in PHYSICAL for i in geometry.check_box(candidate, others)):
                return candidate
    return None


def greedy_keep(geometry: Geometry, placed: dict[str, dict], ids: list[str]) -> int:
    """How many of ``ids`` ④d keeps without packing: it removes the one in the most PHYSICAL violations (then the
    smaller) until none of them is in one."""
    placed, ids = dict(placed), list(ids)
    while True:
        count = {i: sum(v.code in PHYSICAL for v in geometry.check_box(placed[i], {k: b for k, b in placed.items()
                                                                                     if k != i})) for i in ids}
        worst = max(ids, key=lambda i: (count[i], -placed[i]["target_size_local_m"][0]
                                        * placed[i]["target_size_local_m"][1], i), default=None)
        if worst is None or not count[worst]:
            return len(ids)
        ids.remove(worst)
        del placed[worst]


def repack(geometry: Geometry, placed: dict[str, dict], ids: list[str]) -> tuple[dict[str, dict], list[str]]:
    """``ids`` (expansion objects on one parent, nothing on them) taken off it and put back one at a time, each at
    the first free lattice place from a corner (``free_slot(..., pack=True)``): the rarer kinds first (the one lamp
    before the eighth book: a full top keeps variety), then the larger: (the layout with those that fit, the ids
    that do not)."""
    rest = {k: v for k, v in placed.items() if k not in ids}
    left = []
    area = lambda i: placed[i]["target_size_local_m"][0] * placed[i]["target_size_local_m"][1]  # noqa: E731
    kind = {i: geometry.requests.get(i, {}).get("category", i) for i in ids}
    count = {k: sum(v == k for v in kind.values()) for k in kind.values()}
    for i in sorted(ids, key=lambda i: (count[kind[i]], -area(i), i)):
        slot = free_slot(geometry, {**rest, i: placed[i]}, i, pack=True)
        if slot is None:
            left.append(i)
        else:
            rest[i] = slot
    return rest, left


def give_way_choice(hard: list, placed: dict[str, dict], plan: Plan, required: set[str],
                    required_only: bool = False, exclude: set[str] = frozenset()) -> str | None:
    """The expansion object to remove: the one in the most hard violations, each counted by its magnitude (at most
    1) plus 1 when the violation also holds a required object (a required object's problem comes first); ties: the
    smaller footprint (less is lost), then the id. ``required_only``: only the violations of the required objects
    (``on_required``) count; ``exclude``: ids not to choose. None when no such violation holds a removable
    expansion object (one with nothing required standing on it)."""
    score: dict[str, float] = {}
    for v in hard:
        if required_only and not on_required(v, required):
            continue
        weight = min(v.magnitude, 1.) + (1. if on_required(v, required) else 0.)
        for ident in set(v.ids):
            if ident in placed and ident not in required and ident not in exclude \
                    and not descendants(plan, ident) & required:
                score[ident] = score.get(ident, 0.) + weight
    if not score:
        return None
    area = {i: placed[i]["target_size_local_m"][0] * placed[i]["target_size_local_m"][1] for i in score}
    return max(score, key=lambda i: (round(score[i], 6), -area[i], i))


# ④e: SceneEval's accessibility (ACC; metrics/accessibility.py, metrics.yaml) scores an object's functional side by a
# strip ACCESS_DEPTH_M deep, ACCESS_OFFSET_M off that side and as wide as the object: the share of it on the floor and
# outside the plan of every object whose bbox centre is at most ACCESS_DRAWN_H_M up; an object passes at 0.5
# (infinite-task-world's Physics Pass needs every object to). A wall object hung over furniture fails it. One whose
# strip is under WALL_FRONT_NEED free goes to the nearest place along the walls (WALL_STEP_M apart; another wall counts
# WALL_CHANGE_M more) where it is at least WALL_FRONT_TARGET free: a margin over 0.5, SceneEval rasterises the plan
# at 256 px. On the 150 rooms of test150doors (applied to finished layouts): ACC 78.0 -> 85.4, Physics
# Pass 17.3 -> 31.3 %, nothing else worse.
ACCESS_OFFSET_M, ACCESS_DEPTH_M, ACCESS_DRAWN_H_M = .05, .2, 2.
WALL_FRONT_NEED, WALL_FRONT_TARGET = .55, .65
WALL_STEP_M, WALL_CHANGE_M = .05, 2.


def front_strip(box: dict) -> Polygon:
    """The floor strip SceneEval's ACC scores in front of ``box`` (its local +x side: a wall object faces the room
    with it)."""
    sx, sy = box["target_size_local_m"][:2]
    x, y = box["bottom_center_m"][:2]
    yaw = box["yaw_rad"]
    d = sx / 2 + ACCESS_OFFSET_M + ACCESS_DEPTH_M / 2
    strip = affinity.rotate(shapely_box(-ACCESS_DEPTH_M / 2, -sy / 2, ACCESS_DEPTH_M / 2, sy / 2), yaw,
                            use_radians=True, origin=(0, 0))
    return affinity.translate(strip, x + math.cos(yaw) * d, y + math.sin(yaw) * d)


def front_free(box: dict, blocked, floor: Polygon) -> float:
    """The share of ``box``'s front strip on ``floor`` and outside ``blocked``."""
    strip = front_strip(box)
    return strip.intersection(floor).difference(blocked).area / strip.area if strip.area else 1.


def clear_wall_fronts(critic: Critic, plan: Plan, placed: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    """④e: every wall object (support parent "wall") whose front strip is under WALL_FRONT_NEED free (``front_free``),
    the least free first, moved along a wall to the nearest place (another wall counts WALL_CHANGE_M more) where
      * its strip is at least WALL_FRONT_TARGET free,
      * it hangs WALL_GAP_M off the wall facing the room, at the same height, with what stands on it (``carry``),
      * it and what moves with it have no geometry issue they did not have before (collision, boundary, support,
        door keepout, a wall object over a window), no hard or plan relation of theirs that was met stops being
        met, and the critic finds no hard violation that was not there (the robot checks included),
      * no other wall object's strip drops under WALL_FRONT_NEED because of it.
    Nothing on the floor moves and nothing is removed; one with no such place stays. (the layout, {wall_objects,
    below_need, moved: [{id, with, from, to, other_wall, distance_m}], stuck})."""
    g = critic.g
    floor = g.polygon
    drawn = lambda b: b["bottom_center_m"][2] + b["target_size_local_m"][2] / 2 - g.floor <= ACCESS_DRAWN_H_M  # noqa: E731
    fixed = [footprint(b) for b in g.fixed.values() if drawn(b)]

    def blocked_by(layout: dict[str, dict], skip=()):
        return unary_union(fixed + [footprint(b) for j, b in layout.items() if j not in skip and drawn(b)])

    def codes(layout: dict[str, dict], j: str) -> set[str]:
        return {i.code for i in g.check_box(layout[j], {k: b for k, b in layout.items() if k != j})}

    def met(layout: dict[str, dict], ids: set[str]) -> set[int]:
        return {n for n, r in enumerate(g.hard + g.soft) if _refs(r) & ids and g.relation_issue(r, layout) is None}

    walls = [i for i in placed if plan.parents.get(i) == "wall"] if g._walls else []
    before = {i: front_free(placed[i], blocked_by(placed, {i}), floor) for i in walls}
    moves, stuck = [], []
    for i in sorted(walls, key=lambda j: (before[j], j)):
        b = placed[i]
        group = [i] + sorted(descendants(plan, i) & set(placed))
        blocked = blocked_by(placed, set(group))
        if front_free(b, blocked, floor) >= WALL_FRONT_NEED:
            continue
        sx, sy = b["target_size_local_m"][:2]
        x0, y0, z0 = b["bottom_center_m"]
        c, s = math.cos(b["yaw_rad"]), math.sin(b["yaw_rad"])
        back = Point(x0 - c * sx / 2, y0 - s * sx / 2)
        home = min(range(len(g._walls)), key=lambda w: (
            c * g._walls[w][3][0] + s * g._walls[w][3][1] <= .5,
            back.distance(LineString([g._walls[w][0], (g._walls[w][0][0] + g._walls[w][1][0] * g._walls[w][2],
                                                       g._walls[w][0][1] + g._walls[w][1][1] * g._walls[w][2])]))))
        places = []
        for w, (a, u, length, n) in enumerate(g._walls):
            depth = sx / 2 + WALL_GAP_M
            for k in range(int((length - sy) / WALL_STEP_M + 1e-9) + 1 if length >= sy else 0):
                t = sy / 2 + k * WALL_STEP_M
                x, y = a[0] + u[0] * t + n[0] * depth, a[1] + u[1] * t + n[1] * depth
                cost = math.dist((x, y), (x0, y0)) + (0. if w == home else WALL_CHANGE_M)
                if cost > 1e-3:
                    places.append((cost, x, y, math.atan2(n[1], n[0]), w != home))
        places.sort()
        old_codes = {j: codes(placed, j) for j in group}
        old_met = met(placed, set(group))
        old_hard = None
        done = None
        for cost, x, y, yaw, other_wall in places:
            nb = {**b, "bottom_center_m": [round(x, 4), round(y, 4), z0], "yaw_rad": _wrap(yaw)}
            if front_free(nb, blocked, floor) < WALL_FRONT_TARGET:
                continue
            trial = {**placed, i: nb, **{j: carry(placed[j], b, nb) for j in group[1:]}}
            if any(not codes(trial, j) <= old_codes[j] for j in group) or not old_met <= met(trial, set(group)):
                continue
            shape = unary_union([footprint(trial[j]) for j in group if drawn(trial[j])])
            if any(w not in group and before[w] >= WALL_FRONT_NEED and front_strip(trial[w]).intersects(shape)
                   and front_free(trial[w], blocked_by(trial, {w}), floor) < WALL_FRONT_NEED for w in walls):
                continue
            if old_hard is None:
                old_hard = {identity(v) for v in critic.hard(placed)}
            if not {identity(v) for v in critic.hard(trial)} <= old_hard:
                continue
            done = trial, nb, other_wall
            break
        if done is None:
            stuck.append(i)
            continue
        placed, nb, other_wall = done
        moves.append({"id": i, "with": group[1:], "from": [round(x0, 3), round(y0, 3)], "to": nb["bottom_center_m"][:2],
                      "other_wall": other_wall, "distance_m": round(math.dist((x0, y0), nb["bottom_center_m"][:2]), 3)})
    return placed, {"wall_objects": len(walls), "below_need": sum(v < WALL_FRONT_NEED for v in before.values()),
                    "moved": moves, "stuck": stuck}


def required_only(condition: dict, sizes: dict[str, list[float]] | None, required: set[str],
                  articulation: set[str] = frozenset(), *, plan: Plan | None = None,
                  start: dict[str, dict] | None = None, robot: Robot | None = None, robot_margin: float = 0.0,
                  hard_zones: bool = False, catalog_path: str | None = None, templates_path: str | None = None,
                  beam_width: int = 24, candidate_limit: int = 48, repair_seconds: float | None = None,
                  repair_rounds: int = 1, reason: str = "") -> dict:
    """The fallback room: only the required objects, legal and reachable first, their constraints met as far as
    that allows (no LLM). Candidates: ``start`` (where they stand in a full layout) and the search from scratch
    (``solver.beam_search``, three modes); each is repaired (relocation on, the required objects' violations first).
    If no candidate is legal and reachable with the constraints as hard ones, the same is tried with the constraints
    soft (met where they can be, never at the cost of a collision, the room's bounds, a door or the robot: an
    upstream constraint that cannot hold with them is reported, ``unmet_constraints``). The layout with the fewest
    legality and robot violations, then the fewest unmet constraints, is returned in ``run_simple``'s result form
    with ``status`` "fallback_required_only". Sizes: ``sizes`` where given, else the catalog's; never shrunk."""
    start_s = time.perf_counter()
    validate_condition(condition)
    condition, notes = fill_room_defaults(condition)
    condition = check_request(condition)
    plan = plan or make_plan(condition, llm=False)
    required, articulation = set(required), set(articulation)
    everything = [o["id"] for o in condition["objects"]]
    room_, plan_ = without(condition, plan, set(everything) - required)
    catalog = SizeCatalog(catalog_path)
    fixed_sizes = {}
    for obj in room_["objects"]:
        size = (sizes or {}).get(obj["id"])
        fixed_sizes[obj["id"]] = [float(v) for v in size] if size and len(size) == 3 and min(size) > 0 \
            else catalog.candidates(obj)[0][0]
    if robot is not None:
        robot = task_robot(robot, required, articulation)
    planned = replace(robot, width_m=robot.width_m + robot_margin) if robot is not None and robot_margin else robot
    catalog_sized = FixedSizes(fixed_sizes)
    templates = load_templates(templates_path)

    def build(cond: dict):
        geometry = critic_geometry(cond, plan_)
        critic = Critic.for_plan(geometry, plan_, templates)
        if robot is not None:
            critic.robot = RobotChecker.for_critic(planned, critic)
            geometry.robot_zones = RobotZones(planned, geometry, hard=hard_zones)
        return geometry, critic

    def solve(cond: dict) -> list:
        geometry, critic = build(cond)
        locked = _required_parents(cond)
        starts = [{i: b for i, b in (start or {}).items() if i in required}] if start and required <= set(start) else []
        for mode in ("global", "wall", "spacious"):
            _state, trace = beam_search(geometry, catalog_sized, plan_, mode=mode, beam_width=max(64, beam_width),
                                        candidate_limit=max(96, candidate_limit), density=2)
            complete = sorted(list(trace.get("complete") or [])[:12], key=lambda c: critic.assess(c).key())
            starts += complete[:2] or ([{**assemble(geometry, plan_, fixed_sizes, {})[0], **trace["partial"]}]
                                       if trace.get("partial") else [])
        if not starts:
            starts = [assemble(geometry, plan_, fixed_sizes, {})[0]]
        out = []
        for candidate in starts:
            loop = Loop(geometry, plan_, fixed_sizes, catalog, critic, locked_parents=locked,
                        budget=LoopBudget(min_soft_cost=math.inf, search_seconds=repair_seconds),
                        beam_width=beam_width, candidate_limit=candidate_limit, relocate=True,
                        rounds=max(repair_rounds, 3), first=frozenset(required))
            repaired, _info = loop.run(candidate)
            out.append(repaired)
            if not critic.assess(repaired).hard:
                break
        return out

    geometry, critic = build(room_)  # the constraints as given (hard): what the result is reported against

    def score(placed: dict[str, dict]) -> tuple:
        hard = critic.assess(placed).hard
        legal_robot = [v for v in hard if not v.code.startswith("relation_")]
        return len(legal_robot), round(sum(v.magnitude for v in legal_robot), 4), len(hard) - len(legal_robot)
    tried = solve(room_)
    constraints_soft = False
    if all(score(p)[0] for p in tried) and room_["constraints"]:
        soft_room = copy.deepcopy(room_)
        soft_room["constraints"] = [{**c, "hard": False} if c["type"] not in ("on", "keepout") else c
                                    for c in soft_room["constraints"]]
        tried += solve(soft_room)
        constraints_soft = True
    placed = min(tried, key=score)
    final = critic.assess(placed)
    tried = [(final, placed)] * len(tried)  # (the count is reported)
    layout = _layout(room_, placed)
    validate_layout(layout, room_)
    issues = [v.as_dict() for v in final.hard if on_required(v, required)]
    category = {o["id"]: o["category"] for o in condition["objects"]}
    end = time.perf_counter()
    return {"status": "fallback_required_only", "layout": layout,
            "diagnostics": {"placement": "fallback: required objects only", "notes": notes, "reason": reason,
                            "support_parents": dict(plan_.parents), "sizes": fixed_sizes,
                            "hard_issues": [v.as_dict() for v in final.hard],
                            "assessments": {"final": final.summary()}, "candidates_tried": len(tried),
                            "constraints_relaxed": constraints_soft,
                            "unmet_constraints": [c for c in room_["constraints"]
                                                  if c["type"] != "keepout" and geometry.relation_issue(c, placed)],
                            "robot": RobotChecker.for_critic(robot, critic).report(layout, dict(plan_.parents))
                            if robot is not None else None,
                            "task": {"required": sorted(required), "articulation": sorted(articulation),
                                     "expansion": len(everything) - len(required), "expansion_kept": 0,
                                     "moved": [], "required_first": {"tried": False},
                                     "removed": [{"id": i, "category": category[i], "with": [], "step": "fallback",
                                                  "because": ["fallback: only the required objects are placed"]}
                                                 for i in everything if i not in required],
                                     "expansion_issues": [], "required_ok": not issues, "required_issues": issues}},
            "timing_s": {"fallback": round(end - start_s, 3), "total": round(end - start_s, 3)}}


def run_simple(condition: dict, sizes: dict[str, list[float]] | None, *, plan: Plan | None = None,
               yaws: dict[str, float] | None = None, catalog_path: str | None = None, llm: bool = True,
               model: str | None = None, base_url: str | None = None, api_key: str | None = None,
               plan_llm: bool = True, repair: bool = True, semantic: bool = True, semantic_fixes: int = 3,
               soft_tolerance: float = SOFT_TOLERANCE, templates_path: str | None = None, beam_width: int = 24,
               candidate_limit: int = 48, semantic_ask=None, robot: Robot | None = None, drafts: int = 1,
               robot_margin: float = 0.0, hard_zones: bool = False, relocate: bool = False,
               repair_rounds: int = 1, repair_seconds: float | None = None, draft_sizes: bool = False,
               required: set[str] | None = None, articulation: set[str] | None = None,
               task_context: dict | None = None) -> dict:
    """{status, layout, diagnostics, timing_s}. ``plan``: one made already (the pipeline's), else made here.
    ``semantic_ask`` replaces the LLM client for the review (tests); the review runs only with ``llm`` or it.
    ``task_context`` adds task intent to review and allows at most one retry with the rejected edit's reason.
    ``robot``: its checks (``robot.RobotChecker``: doors linked, furniture reachable, openable furniture free to open)
    are hard violations in step ④ and the review's gate, and the final layout is reported for it. The placement
    request asks for what the robot needs (``drafting.ROBOT_RULE``), and the repair search ranks its candidates with
    the checks' zones (``robot.RobotZones``). ``drafts``: placement calls made in parallel; the one with the fewest
    hard violations of any kind is repaired (``draft_rank``), and every draft is recorded; each call is paid. Every
    object keeps its given size (FastFill's): no step resizes an object, required or expansion. ``robot_margin``: the placement request, the critic and the
    search use a robot this much wider, so the layout keeps a margin (grid planners keep a clearance; HSM a 1 x 1 m
    door zone); the result is reported for ``robot`` itself. ``hard_zones``: the draft fixes each carrying
    furniture's approach side and the search filters on the robot zones (``RobotZones.reserve``, ``hard``).
    ``relocate`` / ``repair_rounds``: step ④ may move a robot check's blocker anywhere in the room and takes further
    passes while they help (``Loop``). ``repair_seconds``: the program search's time limit for step ④; None: no
    limit. ``draft_sizes`` (the workflow without FastFill; ``sizes`` must be
    None): the placement call gives every object's size with its pose (``drafting.FREE_SIZE_INSTRUCTIONS``), and
    the chosen draft's sizes are the fixed sizes of every later step; an object it gives no size keeps the catalog's
    (``size_source``: {"draft": n, "catalog": n}).

    ``required`` (ids; the upstream task agent's required furniture): the task mode. The placement request marks them,
    the robot checks hold for every object and add the task's (``task_robot``: the required objects reached,
    ``articulation`` free to open), the repair takes up their violations first and never shrinks them, and step ④b
    moves expansion objects (every other one) out of the way, removing one only for a required object when moving
    it fails; an expansion object whose own physical violations cannot be fixed is left out at the end (④d,
    ``removed`` step "does not fit"). The layout holds the objects kept;
    ``diagnostics.task`` lists what was moved, what was removed and why, the expansion objects' violations left
    (``expansion_issues``), and ``required_ok``: every required object kept with no hard violation on it and the
    doors linked. Required objects are never removed."""
    if draft_sizes and sizes:
        raise ValueError("draft_sizes: the placement call gives the sizes, so no sizes are given")
    start = time.perf_counter()
    validate_condition(condition)
    condition, notes = fill_room_defaults(condition)
    condition = check_request(condition)
    if required is not None:
        ids = {o["id"] for o in condition["objects"]}
        required, articulation = set(required), set(articulation or ())
        if not required <= ids or not articulation <= required:
            raise ValueError(f"required ids must be objects of the request and articulation ids required ones: "
                             f"{sorted(required - ids)} {sorted(articulation - required)}")
        if robot is not None:
            robot = task_robot(robot, required, articulation)
    if plan is None:
        plan = make_plan(condition, llm=llm and plan_llm, model=model, base_url=base_url, api_key=api_key)
    planned = time.perf_counter()
    catalog, fixed_sizes, size_source = SizeCatalog(catalog_path), {}, {"given": 0, "catalog": 0}
    for obj in condition["objects"]:
        size = (sizes or {}).get(obj["id"])
        if size is not None and len(size) == 3 and all(isinstance(v, (int, float)) and v > 0 for v in size):
            fixed_sizes[obj["id"]], size_source["given"] = [float(v) for v in size], size_source["given"] + 1
        else:
            fixed_sizes[obj["id"]], size_source["catalog"] = catalog.candidates(obj)[0][0], size_source["catalog"] + 1
    templates = load_templates(templates_path)
    planned_robot = replace(robot, width_m=robot.width_m + robot_margin) if robot is not None and robot_margin \
        else robot  # what placement, critic and search plan for; the report is for ``robot``

    def room(condition: dict, plan: Plan):
        geometry = critic_geometry(condition, plan)
        critic = Critic.for_plan(geometry, plan, templates)
        if robot is not None:
            critic.robot = RobotChecker.for_critic(planned_robot, critic)
            # the search's form of the checks (Geometry.soft_score; with hard_zones also a filter)
            geometry.robot_zones = RobotZones(planned_robot, geometry, hard=hard_zones)
        return geometry, critic
    geometry, critic = room(condition, plan)
    locked = _required_parents(condition)
    required_ids = frozenset(required or ())

    # ③ the placement call(s): with ``drafts`` > 1, that many in parallel, and the one with the fewest violations is kept
    if llm and drafts > 1:
        def one(_k):
            return make_draft(condition, plan, fixed_sizes, geometry, yaws=yaws, robot=planned_robot, model=model,
                              base_url=base_url, api_key=api_key, free_sizes=draft_sizes, required=required)
        with ThreadPoolExecutor(drafts) as pool:
            tried = list(pool.map(one, range(drafts)))
        assessed_drafts = [critic.assess(p) for p, _info in tried]
        keys = [draft_rank(a) for a in assessed_drafts]
        best = min(range(drafts), key=lambda k: keys[k])
        placed, draft = tried[best]
        draft = {**draft, "calls": [c for _p, info in tried for c in info["calls"]], "chosen": best,
                 "candidates": [{"key": list(a.key()), "rank": list(k), "error": info["error"],
                                 "missing": len(info["missing"]), "layout": _layout(condition, p)}
                                for a, k, (p, info) in zip(assessed_drafts, keys, tried)]}
    elif llm:
        placed, draft = make_draft(condition, plan, fixed_sizes, geometry, yaws=yaws, robot=planned_robot, model=model, base_url=base_url,
                                   api_key=api_key, free_sizes=draft_sizes, required=required)
    else:  # offline: every object at its fallback (on its support, else the room's representative point)
        placed, missing = assemble(geometry, plan, fixed_sizes, {})
        draft = {"calls": [], "dropped": {}, "missing": missing, "error": "llm disabled"}
    if draft_sizes and "sizes" in draft:  # the chosen draft's sizes are fixed from here on, as FastFill's would be
        fixed_sizes = draft.pop("sizes")
        size_source = {"draft": draft["size_source"]["draft"], "catalog": draft["size_source"]["given"]}
    placed_s = time.perf_counter()
    stages = {"draft": _layout(condition, placed)}
    reserved = geometry.robot_zones.reserve(placed) if hard_zones and geometry.robot_zones is not None else {}
    assessed = {"draft": critic.assess(placed)}

    # ④ one program check (and one repair pass over the hard violations only)
    check = {"result": "clean" if not assessed["draft"].hard else "not repaired"}
    def repair_pass(placed: dict[str, dict]) -> tuple[dict[str, dict], dict]:
        loop = Loop(geometry, plan, fixed_sizes, catalog, critic, locked_parents=locked,
                    budget=LoopBudget(min_soft_cost=math.inf, search_seconds=repair_seconds),
                    beam_width=beam_width, candidate_limit=candidate_limit, relocate=relocate, rounds=repair_rounds,
                    first=required_ids)
        placed, check = loop.run(placed)
        check["result"] = "repaired"
        return placed, check

    if repair and assessed["draft"].hard:
        placed, check = repair_pass(placed)
    stages["after_check"] = _layout(condition, placed)
    assessed["after_check"] = critic.assess(placed)

    # ④b the expansion objects give way to the required ones (task mode)
    removed: list[dict] = []
    moved: list[dict] = []
    if required is not None:
        whole = (condition, plan, locked)  # the room before anything is removed (④c puts removed objects back)
        last_pose: dict[str, dict] = {}
        def required_issues(critic, placed) -> int:
            return sum(on_required(v, required) for v in critic.assess(placed).hard)

        def move(placed: dict[str, dict], pick: str, hard: list, target=None) -> dict[str, dict] | None:
            """``pick`` (with what stands on it) searched for anywhere in the room; the new layout when the critic
            prefers it and ``pick`` is then in no hard violation, else None. ``target``: the violation the move is
            for (default: the largest one ``pick`` is in)."""
            target = target or max((v for v in hard if pick in v.ids),
                                   key=lambda v: (on_required(v, required), v.magnitude))
            scope = {pick} | (descendants(plan, pick) & set(placed))
            loop = Loop(geometry, plan, fixed_sizes, catalog, critic, locked_parents=locked,
                        budget=LoopBudget(min_soft_cost=math.inf), beam_width=beam_width,
                        candidate_limit=candidate_limit)
            candidate, after = loop._pick(loop.program(scope, target, placed, RELOCATE_DRAFT_WEIGHT), critic)
            if after is None or any(pick in v.ids for v in after.hard):
                return None
            if required_issues(critic, candidate) > required_issues(critic, placed):
                return None  # an expansion object's move never costs a required object
            return candidate if critic.better(after, critic.assess(placed))[0] else None

        def near_blocker(hard: list, placed: dict[str, dict], tried: set[str]):
            """(expansion object, violation) for a robot check of a required object that names no expansion object
            (several block it together, so no single removal shows): the nearest expansion floor object to the
            object, within NEAR_BLOCKER_M, not tried yet; else None."""
            for v in hard:
                if not (on_required(v, required) and v.code in ("robot_reach", "robot_arm", "robot_openable")) \
                        or any(i in placed and i not in required for i in v.ids):
                    continue
                subject = placed.get(v.ids[-1] if v.code == "robot_arm" and len(v.ids) > 1 else v.ids[0])
                if subject is None:
                    continue
                sx, sy = subject["bottom_center_m"][:2]
                near = sorted((math.dist((sx, sy), b["bottom_center_m"][:2]), i) for i, b in placed.items()
                              if i not in required and i not in tried and plan.parents.get(i) == "floor"
                              and not descendants(plan, i) & required)
                if near and near[0][0] <= NEAR_BLOCKER_M:
                    return near[0][1], v
            return None

        def category(ident: str) -> str:
            return next(o["category"] for o in condition["objects"] if o["id"] == ident)

        def remove(placed: dict[str, dict], pick: str, hard: list, step: str = "for the required objects",
                   because: list[str] | None = None) -> dict[str, dict]:
            nonlocal condition, plan, geometry, critic, locked
            gone = {pick} | (descendants(plan, pick) & set(placed))
            last_pose.update({i: placed[i] for i in gone})
            removed.append({"id": pick, "category": category(pick), "with": sorted(gone - {pick}),
                            "step": step,
                            "because": because or [f"{v.code} {list(v.ids)}" + (f" ({v.detail})" if v.detail else "")
                                                   for v in hard if pick in v.ids or v.ids[:1] == (pick,)]})
            condition, plan = without(condition, plan, gone)
            geometry, critic = room(condition, plan)
            locked = {k: v for k, v in locked.items() if k not in gone}
            return {k: v for k, v in placed.items() if k not in gone}

        def give_way(placed: dict[str, dict], for_required: bool) -> dict[str, dict]:
            """``for_required``: the required objects' violations, an expansion object moved or else removed;
            otherwise every violation, expansion objects moved only (each tried once). A required object's robot
            check that names no expansion object tries the expansion objects near it (``near_blocker``): moved when
            that lowers the required objects' violations, else removed only if the removal lowers them."""
            nonlocal condition, plan, geometry, critic, locked
            tried: set[str] = set()
            near_tried: set[str] = set()
            while True:
                hard = critic.assess(placed).hard
                pick = give_way_choice(hard, placed, plan, required, for_required,
                                       exclude=set() if for_required else tried)
                if pick is None and for_required:
                    found = near_blocker(hard, placed, near_tried) if len(near_tried) < NEAR_BLOCKER_TRIES else None
                    if found is None:
                        return placed
                    pick, target = found
                    near_tried.add(pick)
                    before_n = required_issues(critic, placed)
                    category = next(o["category"] for o in condition["objects"] if o["id"] == pick)
                    new = move(placed, pick, hard, target)
                    if new is not None and required_issues(critic, new) < before_n:
                        moved.append({"id": pick, "category": category, "step": "for the required objects"})
                        placed = new
                        continue
                    state = (placed, condition, plan, geometry, critic, locked, len(removed))
                    placed = remove(placed, pick, hard)
                    if required_issues(critic, placed) >= before_n:  # it did not help: put it back
                        placed, condition, plan, geometry, critic, locked, n = state
                        del removed[n:]
                    continue
                if pick is None:
                    return placed
                category = next(o["category"] for o in condition["objects"] if o["id"] == pick)
                if pick not in tried:
                    tried.add(pick)
                    new = move(placed, pick, hard)
                    if new is not None:
                        moved.append({"id": pick, "category": category,
                                      "step": "for the required objects" if for_required else "its own violations"})
                        placed = new
                        continue
                if not for_required:
                    continue  # an expansion object's own problem never removes it
                placed = remove(placed, pick, hard)
        placed = give_way(placed, for_required=True)
        if removed and repair and critic.assess(placed).hard:
            placed, check["after_removal"] = repair_pass(placed)  # the space freed may let the search fix the rest
        placed = give_way(placed, for_required=False)
        if any(on_required(v, required) for v in critic.assess(placed).hard):
            placed = give_way(placed, for_required=True)
        stages["after_give_way"] = _layout(condition, placed)
        assessed["after_give_way"] = critic.assess(placed)

        # ④c required first: the required objects alone, then the expansion back, giving way
        before_n = required_issues(critic, placed)
        first_s = time.perf_counter()
        if before_n:
            snapshot = (placed, condition, plan, geometry, critic, locked, list(removed), len(moved))
            expansion = set(placed) - required
            r_condition, r_plan = without(condition, plan, expansion)
            r_geometry, r_critic = room(r_condition, r_plan)
            r_locked = {k: v for k, v in locked.items() if k not in expansion}
            alone = {i: placed[i] for i in required}
            loop = Loop(r_geometry, r_plan, fixed_sizes, catalog, r_critic, locked_parents=r_locked,
                        budget=LoopBudget(min_soft_cost=math.inf, search_seconds=repair_seconds),
                        beam_width=beam_width, candidate_limit=candidate_limit, relocate=True,
                        rounds=max(repair_rounds, 3), first=required_ids)
            alone, _info = loop.run(alone)
            how, trace_n = "repair", {"alone_after_repair": required_issues(r_critic, alone)}
            if required_issues(r_critic, alone):
                sizes_only = FixedSizes({i: fixed_sizes[i] for i in required})
                _state, trace = beam_search(r_geometry, sizes_only, r_plan, mode="global", beam_width=max(64, beam_width),
                                            candidate_limit=max(96, candidate_limit), density=2)
                fresh = sorted(((r_critic.assess(c), c) for c in list(trace.get("complete") or [])[:24]),
                               key=lambda t: t[0].key())
                if fresh:
                    candidate, _info = Loop(r_geometry, r_plan, fixed_sizes, catalog, r_critic, locked_parents=r_locked,
                                            budget=LoopBudget(min_soft_cost=math.inf,
                                                              search_seconds=repair_seconds),
                                            beam_width=beam_width, candidate_limit=candidate_limit, relocate=True,
                                            rounds=max(repair_rounds, 3), first=required_ids).run(fresh[0][1])
                    trace_n["alone_from_scratch"] = required_issues(r_critic, candidate)
                    if required_issues(r_critic, candidate) < required_issues(r_critic, alone):
                        alone, how = candidate, "search from scratch"

            def root(i: str) -> str | None:  # the required object an expansion object stands on, if any
                seen = set()
                while i in plan.parents and i not in seen:
                    seen.add(i)
                    i = plan.parents[i]
                    if i in required:
                        return i
                return None
            # the expansion back one at a time, those ④b removed too, never raising the required objects'
            # violations above the alone ones; in the whole room again
            condition, plan, locked = whole
            geometry, critic = room(condition, plan)
            original = {**last_pose, **placed}
            expansion = set(original) - required
            current = {i: dict(b) for i, b in alone.items()}
            limit = required_issues(critic, current)
            area = lambda i: fixed_sizes[i][0] * fixed_sizes[i][1]  # noqa: E731

            def depth(i: str) -> int:
                n, seen = 0, set()
                while plan.parents.get(i) in original and i not in seen:
                    seen.add(i)
                    i, n = plan.parents[i], n + 1
                return n
            dropped: list[str] = []
            filler = Loop(geometry, plan, fixed_sizes, catalog, critic, locked_parents=locked,
                          budget=LoopBudget(min_soft_cost=math.inf), beam_width=beam_width,
                          candidate_limit=candidate_limit)
            for i in sorted(expansion, key=lambda i: (depth(i), -area(i), i)):
                parent = plan.parents.get(i)
                if parent in original and parent not in current:
                    dropped.append(i)  # its support was removed
                    continue
                pose = carry(original[i], original[parent], current[parent]) if parent in current else original[i]
                trial = {**current, i: pose}
                if required_issues(critic, trial) <= limit:
                    current = trial
                    continue
                target = Violation("required_first", (i,), 1.)
                found = [c for c in filler.program({i}, target, trial, RELOCATE_DRAFT_WEIGHT)
                         if required_issues(critic, c) <= limit]
                if found:
                    current = min(found, key=lambda c: critic.assess(c).key())
                else:
                    dropped.append(i)
            trace_n.update(put_back=len(expansion) - len(dropped), dropped=len(dropped),
                           seconds_to_put_back=round(time.perf_counter() - first_s, 1))
            removed.clear()  # what ④c removes replaces what ④b had removed (it may have come back)
            if dropped:
                removed.extend({"id": i, "category": next(o["category"] for o in condition["objects"] if o["id"] == i),
                                "with": [], "step": "for the required objects",
                                "because": ["required first: no place in the room that keeps the required objects legal "
                                            "and reachable"]} for i in dropped)
                condition, plan = without(condition, plan, set(dropped))
                geometry, critic = room(condition, plan)
                locked = {k: v for k, v in locked.items() if k not in dropped}
            merged = {k: v for k, v in current.items() if k not in dropped}
            placed = give_way(merged, for_required=True)
            trace_n["after_give_way"] = required_issues(critic, placed)
            if repair and critic.assess(placed).hard:
                placed, check["after_required_first"] = repair_pass(placed)
            placed = give_way(placed, for_required=False)
            if any(on_required(v, required) for v in critic.assess(placed).hard):
                placed = give_way(placed, for_required=True)
            after_n = required_issues(critic, placed)
            used = after_n < before_n
            if not used:  # no better for the required objects: the layout before ④c stays
                placed, condition, plan, geometry, critic, locked, removed_before, n_moved = snapshot
                removed[:] = removed_before
                del moved[n_moved:]
            required_first = {"tried": True, "used": used, "how": how, "required_issues_before": before_n,
                              "required_issues_after": after_n, "steps": trace_n,
                              "seconds": round(time.perf_counter() - first_s, 1)}
            stages["after_required_first"] = _layout(condition, placed)
            assessed["after_required_first"] = critic.assess(placed)
        else:
            required_first = {"tried": False}
    checked_s = time.perf_counter()

    # ⑤ bbox VLM review before RoomGen; task edits may retry once with the rejected move's reason.
    review = {"result": "skipped"}
    review_ask = semantic_ask or (make_ask(model, base_url, api_key) if llm else None)
    if semantic and review_ask is not None:
        current_sizes = {i: list(b["target_size_local_m"]) for i, b in placed.items()}
        reviewer = SemanticPass(geometry, critic, plan, current_sizes, catalog, locked_parents=locked, ask=review_ask,
                                stage="room", soft_tolerance=soft_tolerance, condition=condition,
                                beam_width=beam_width, candidate_limit=candidate_limit, task_context=task_context)
        parents = dict(plan.parents)
        everything = set(placed)
        try:
            placed, review = reviewer.run(placed, everything, everything,
                                          rounds=2 if task_context is not None else 1, max_fixes=semantic_fixes)
        except FatalAPIError:
            raise
        except Exception as exc:  # the review is optional: keep the checked layout
            plan.parents.clear()
            plan.parents.update(parents)
            review = {"result": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        review["llm_calls"] = reviewer.calls
    # ④d what does not fit is left out (see the module doc)
    left_out_s = time.perf_counter()
    if required is not None:
        while True:  # the critic's hard violations of these codes (its geometry's; no robot check needed)
            physical = [Violation(i.code, i.ids, critic._hard_magnitude(i, placed), detail=i.detail)
                        for i in critic.g.validate(placed) if i.code in PHYSICAL]
            pick = give_way_choice(physical, placed, plan, required)
            if pick is None:
                break
            parent = plan.parents.get(pick)
            leaf = lambda i: not descendants(plan, i) & set(placed)  # noqa: E731
            on_object = parent not in (None, "floor", "wall") and leaf(pick)
            slot = free_slot(critic.g, placed, pick) if on_object else None
            if slot is not None:  # a free place on the same top first
                placed = {**placed, pick: slot}
                moved.append({"id": pick, "category": category(pick), "step": "a free place on its parent"})
                continue
            if on_object:  # then that top packed again: kept when it keeps more of its objects than removing does
                siblings = [i for i in placed if plan.parents.get(i) == parent and i not in required and leaf(i)]
                packed, left = repack(critic.g, placed, siblings)
                if len(siblings) - len(left) > greedy_keep(critic.g, placed, siblings):
                    placed = {**packed, **{i: placed[i] for i in left}}
                    moved.extend({"id": i, "category": category(i), "step": "packed on its parent"}
                                 for i in siblings if i not in left)
                    for i in left:
                        placed = remove(placed, i, physical, step="does not fit",
                                        because=[f"no free place on {parent} after packing its top"])
                    continue
            placed = remove(placed, pick, physical, step="does not fit")
        if any(r["step"] == "does not fit" for r in removed):
            stages["after_left_out"] = _layout(condition, placed)
    left_out_seconds = time.perf_counter() - left_out_s
    # ④e the floor in front of wall objects kept clear (see the module doc)
    wall_clear_s = time.perf_counter()
    wall_clear = None
    if required is not None:
        placed, wall_clear = clear_wall_fronts(critic, plan, placed)
        moved.extend({"id": m["id"], "category": category(m["id"]), "step": "a clear wall place"}
                     for m in wall_clear["moved"])
        if wall_clear["moved"]:
            stages["after_wall_clear"] = _layout(condition, placed)
    wall_clear_seconds = time.perf_counter() - wall_clear_s
    stages["final"] = _layout(condition, placed)
    final = assessed["final"] = critic.assess(placed)

    layout = _layout(condition, placed)
    validate_layout(layout, condition)
    task = None
    if required is not None:
        issues = [v.as_dict() for v in final.hard if on_required(v, required)]
        task = {"required": sorted(required), "articulation": sorted(articulation), "required_first": required_first,
                "expansion": len(fixed_sizes) - len(required), "expansion_kept": len(placed) - len(required),
                "expansion_issues": [v.as_dict() for v in final.hard if not on_required(v, required)],
                "moved": moved, "removed": removed, "wall_clear": wall_clear, "required_ok": not issues,
                "required_issues": issues}
    fallback = None
    if task is not None and not task["required_ok"]:
        # ⑦ the fallback: only the required objects, if that leaves them better off
        fallback = required_only(whole[0], fixed_sizes, required, articulation, plan=whole[1], start=placed,
                                 robot=robot, robot_margin=robot_margin, hard_zones=hard_zones,
                                 catalog_path=catalog_path, templates_path=templates_path, beam_width=beam_width,
                                 candidate_limit=candidate_limit, repair_seconds=repair_seconds,
                                 repair_rounds=repair_rounds,
                                 reason=f"{len(issues)} hard violations on required objects in the full layout")
        if len(fallback["diagnostics"]["task"]["required_issues"]) >= len(issues):
            fallback = None  # no better for the required objects: the full layout stays
    end = time.perf_counter()
    full = {"status": "clean" if not final.hard else "best_effort", "layout": layout,
            "diagnostics": {"placement": "simple", "notes": notes, "plan_source": plan.source, "plan_llm": plan.llm,
                            "support_parents": dict(plan.parents), "functional_groups": plan.groups,
                            "plan_relations": plan.relations, "size_source": size_source, "sizes": fixed_sizes,
                            "draft": draft, "check": check, "semantic": review, "stages": stages,
                            "assessments": {k: v.summary() for k, v in assessed.items()},
                            "critic": {"final": final.summary(),
                                       "unresolved": [v.as_dict() for v in final.hard],
                                       "status_detail": "hard_valid" if not final.hard else "violations_remaining"},
                            "hard_issues": [v.as_dict() for v in final.hard],
                            "robot": RobotChecker.for_critic(robot, critic).report(layout, dict(plan.parents))
                            if robot is not None else None,
                            "repair": {"relocate": relocate, "rounds": repair_rounds, "seconds": repair_seconds},
                            "task": task,
                            "robot_plan": {"margin_m": robot_margin, "hard_zones": hard_zones,
                                           "approach_sides": reserved} if robot is not None else None},
            "timing_s": {"plan": round(planned - start, 3), "place": round(placed_s - planned, 3),
                         "check": round(checked_s - placed_s, 3), "review": round(left_out_s - checked_s, 3),
                         "left_out": round(left_out_seconds, 3), "wall_clear": round(wall_clear_seconds, 3),
                         "total": round(end - start, 3)}}
    if fallback is None:
        return full
    for key in ("plan_llm", "draft", "semantic", "critic"):  # the LLM calls made, for the cost
        fallback["diagnostics"][key] = full["diagnostics"].get(key)
    fallback["diagnostics"]["fallback_from"] = {"status": full["status"], "layout": full["layout"],
                                                "task": full["diagnostics"]["task"],
                                                "hard_issues": full["diagnostics"]["hard_issues"]}
    fallback["timing_s"] = {**full["timing_s"], "fallback": fallback["timing_s"]["fallback"],
                            "total": round(time.perf_counter() - start, 3)}
    return fallback
