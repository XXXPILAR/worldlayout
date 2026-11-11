"""The critic: one program-only assessment that every step proposing a change is accepted or rolled back by.

``Critic.assess(placed)`` returns
  H  hard violations (``Geometry.validate``: boundary, height, support, collisions, keep-out, the input's hard
     constraints, and door clearance when the geometry treats doors as hard), each with a continuous magnitude in
     metres (square root of an overlap area, a height or support gap, a relation's distance / angle gap), so a
     candidate that turns a 20 cm overlap into a 1 cm one counts as progress (critic_flowchart_v2 §3);
  S  soft costs, each ``weight * min(magnitude / scale, 1)`` so S is bounded per term:
       relation_user   the input's soft constraints (weight 1)
       relation_plan   step ①'s soft relations and group links, suggestions only (weight 0.5)
       template_near / template_faces   relation templates mined from the training rooms (``critic_templates.json``,
                       the even half of the training rooms): for each floor obstacle of category A
                       and the nearest obstacle of category B, the gap beyond the template's 80th-percentile gap, or
                       the turn needed to face B (``facing_error``: within 45 degrees of B's centre, as REL, or a ray
                       along the front hitting B's footprint) (weight 0.5); REL (fastfill.v2.ops.layout_quality)
                       counts the same pairs as pass / fail with the odd half's templates
       access          a floor obstacle at least 0.3 m tall whose usable sides (front; beds also the long sides,
                       tables every side; seats and tables do not block each other) all have less than half of the
                       0.5 m strip in front free (layout_quality.acc's rule), magnitude the missing share (weight 0.5)
       wall_back       an object of a category that stands back to a wall in at least 80 % of the training rooms
                       (``critic_wall_rules.json``, the even half of the training rooms) whose back face
                       is more than 0.15 m from every wall or whose front is more than 10 degrees off the wall's
                       inward normal: ``wall_back_error`` (weight 0.5)
       reach           an object of a category reached in at least 90 % of the training rooms that a person 0.4 m wide
                       cannot walk up to from a door (``unreachable``), magnitude the distance from the reachable floor
                       (at least 0.1), the ids include the objects blocking the way (weight 1)
       window          a soft window-clearance issue (weight 0.25), door clearance when doors are soft (weight 1)
With a robot (``critic.robot = robot.RobotChecker(...)``), H also holds robot_door / robot_reach / robot_openable /
robot_arm: doors linked, the furniture the robot works at reachable, openable furniture free to open, objects on
furniture within the arm's reach, for that robot's width.
Visual feedback is not part of either score (critic_flowchart_v2 §3: program scores and visual review are kept apart).

``better(candidate, current)`` is the commit rule: with hard violations left, fewer violations, or the same count
with a smaller total magnitude, and no new violation larger than ``PROTECT_M``; at equal H, a lower S; a hard-valid
layout only accepts hard-valid candidates. With a robot the rule runs in tiers: legality (every hard code but the robot
ones) first, then the robot checks the same way, then S, so a legality repair is never kept out by a robot check. Both margins (``EPS_H``, ``EPS_S``) keep noise from counting as progress.

S uses the same definitions the evaluation reports (ACC, REL with another half's templates); a loop that optimises S
can inflate those numbers, so claims of better layouts need metrics the critic does not see (design rules, a VLM or
human comparison). This is recorded, not solved, here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from . import contract  # noqa: F401  (puts the fastfill package on sys.path)
from fastfill.v2.ops import layout_quality as lq
from fastfill.validate import SEAT, TABLE

from .geometry import Geometry, Issue, footprint
from .planning import Plan

TEMPLATES = Path(__file__).resolve().parents[1] / "critic_templates.json"
WALL_RULES = Path(__file__).resolve().parents[1] / "critic_wall_rules.json"
EPS_H, EPS_S, PROTECT_M = .005, .01, .05
WEIGHTS = {"relation_user": 1., "relation_plan": .5, "template_near": .5, "template_faces": .5, "access": .5,
           "window": .25, "door": 1., "wall_back": .5, "reach": 1.}
SCALE = {"distance": 1., "angle": math.pi / 2, "share": .5}
# person width as mined into critic_wall_rules.json (``reach_width_m``); the arm is how near a person's centre must
# come to a side to use it beyond the half width
REACH_HALF_WIDTH_M, REACH_ARM_M = .2, .4
SOFA = {"sofa", "couch", "loveseat", "chaise"}
# no functional front: any side may face the wall (a plant turned 90 degrees is not wrong)
ANY_SIDE = {"lamp", "floorlamp", "plant", "plants", "floorplant", "plantfloor", "bin", "hamper", "basket", "vase"}
WALL_GAP_M, WALL_ANGLE_RAD, WALL_MIN_LENGTH_M = .15, math.radians(10.), .3


@dataclass(frozen=True)
class Violation:
    code: str
    ids: tuple[str, ...]
    magnitude: float
    weight: float = 1.
    detail: str = ""

    @property
    def cost(self) -> float:
        return self.weight * min(self.magnitude, 1.)

    def as_dict(self) -> dict:
        return {"code": self.code, "object_ids": list(self.ids), "magnitude": round(self.magnitude, 4),
                "weight": self.weight, **({"detail": self.detail} if self.detail else {})}


@dataclass
class Assessment:
    hard: list[Violation] = field(default_factory=list)
    soft: list[Violation] = field(default_factory=list)

    @property
    def hard_count(self) -> int:
        return len(self.hard)

    @property
    def hard_magnitude(self) -> float:
        return sum(v.magnitude for v in self.hard)

    @property
    def soft_total(self) -> float:
        return sum(v.cost for v in self.soft)

    @property
    def legality(self) -> list[Violation]:
        """The hard violations a validator rejects a layout for (everything but the robot checks)."""
        return [v for v in self.hard if not v.code.startswith("robot_")]

    @property
    def robot(self) -> list[Violation]:
        return [v for v in self.hard if v.code.startswith("robot_")]

    def key(self) -> tuple:
        """Smaller is better: legality, then the robot checks, then the soft cost (``Critic.better``'s order)."""
        legal, robot = self.legality, self.robot
        return (len(legal), round(sum(v.magnitude for v in legal), 4), len(robot), round(sum(v.magnitude for v in robot), 4),
                round(self.soft_total, 4))

    def summary(self) -> dict:
        by_code: dict[str, list] = {}
        for v in self.soft:
            entry = by_code.setdefault(v.code, [0, 0.])
            entry[0] += 1
            entry[1] += v.cost
        return {"hard_count": self.hard_count, "hard_magnitude": round(self.hard_magnitude, 4),
                "soft_total": round(self.soft_total, 4),
                "soft_by_code": {k: {"count": n, "cost": round(c, 4)} for k, (n, c) in sorted(by_code.items())},
                "hard": [v.as_dict() for v in self.hard]}


def load_templates(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else TEMPLATES
    return json.loads(path.read_text())["templates"] if path.is_file() else []


def load_wall_rules(path: str | Path | None = None) -> dict:
    """{"against_wall": categories, "reach": categories, "reach_half_width_m": float}; empty sets without the file."""
    path = Path(path) if path else WALL_RULES
    if not path.is_file():
        return {"against_wall": set(), "reach": set(), "reach_half_width_m": REACH_HALF_WIDTH_M}
    data = json.loads(path.read_text())
    return {"against_wall": {r["category"] for r in data["against_wall"]}, "reach": set(data["reach_categories"]),
            "reach_half_width_m": data["reach_width_m"] / 2}


def _sqrt_area(area: float) -> float:
    return math.sqrt(max(area, 0.))


# ---------------------------------------------------------------- orientation and reachability (SceneSmith's tools,
# made stricter: oriented boxes instead of axis-aligned ones, a distance limit on "back to the wall", and the pairs to
# check come from templates and wall rules, not from a model's choice)
def facing_error(b: dict, target: Polygon, centre_rad: float | None = math.radians(lq.FACING_DEG)) -> float:
    """Radians the front of ``b`` must turn to face ``target`` (its oriented footprint). Facing means a ray from the
    centre along the front hits the target (SceneSmith's check_facing test, so a large target allows more) or, with
    ``centre_rad``, the front is within that angle of the direction to the target's centre (layout_quality's test,
    which the training rooms pass more often for small targets); 0 when facing."""
    origin = b["centre"]
    to = np.asarray(target.centroid.coords[0]) - origin
    if target.contains(Point(origin)) or float(np.linalg.norm(to)) <= 1e-9:
        return 0.
    base = math.atan2(to[1], to[0])

    def rel(v) -> float:
        return (math.atan2(v[1], v[0]) - base + math.pi) % (2 * math.pi) - math.pi

    spans = [rel(np.asarray(c) - origin) for c in target.exterior.coords[:-1]]
    f = rel(b["front"])
    ray = max(f - max(spans), 0.) if f > 0 else max(min(spans) - f, 0.)
    return ray if centre_rad is None else min(ray, max(abs(f) - centre_rad, 0.))


def walls(floor: Polygon, min_length_m: float = WALL_MIN_LENGTH_M) -> list[tuple[LineString, np.ndarray]]:
    """The floor polygon's edges at least ``min_length_m`` long, each with its unit normal into the room."""
    coords = list(floor.exterior.coords)
    inward = 1. if floor.exterior.is_ccw else -1.  # counter-clockwise: the interior is left of each edge
    out = []
    for a, c in zip(coords[:-1], coords[1:]):
        d = np.subtract(c, a)
        length = float(np.linalg.norm(d))
        if length >= min_length_m:
            u = d / length
            out.append((LineString([a, c]), inward * np.array((-u[1], u[0]))))
    return out


def any_side(b: dict) -> bool:
    return b["head"] in ANY_SIDE or b["cat"].replace(" ", "") in ANY_SIDE


def wall_back_error(b: dict, wall_list: list[tuple[LineString, np.ndarray]], gap_m: float = WALL_GAP_M,
                    angle_rad: float = WALL_ANGLE_RAD) -> tuple[float, float, float]:
    """How far ``b`` is from standing with its back against a wall, facing into the room: over the walls, the least
    (gap of the back face's midpoint beyond ``gap_m``) + (angle between the front and the wall's inward normal beyond
    ``angle_rad``, over pi/2). Returns (error, gap, angle) for the best wall; the gap is limited, unlike SceneSmith's
    unbounded "away" ray, so a wardrobe in the middle of the room facing away from a far wall does not pass. Objects
    without a functional front (``ANY_SIDE``) may stand with any side to the wall."""
    sides = [(b["front"], b["size"][0] / 2)]
    if any_side(b):
        sides += [(-b["front"], b["size"][0] / 2), (b["side"], b["size"][1] / 2), (-b["side"], b["size"][1] / 2)]
    best = (math.inf, math.inf, math.inf)
    for (line, normal), (front, half) in ((w, f) for w in wall_list for f in sides):
        gap = Point(*(b["centre"] - front * half)).distance(line)
        angle = math.acos(max(-1., min(1., float(np.dot(front, normal)))))
        error = max(gap - gap_m, 0.) + max(angle - angle_rad, 0.) / (math.pi / 2)
        if error < best[0]:
            best = (error, gap, angle)
    return best


def _tier(candidate: list[Violation], current: list[Violation], name: str) -> tuple[int, str]:
    """(+1 better / -1 worse / 0 the same, why) for one tier of hard violations: none may appear where there were
    none; otherwise fewer, or as many with a smaller total magnitude, and no new one larger than ``PROTECT_M``."""
    if not current:
        return (-1, f"new {name} violation on a {name}-valid layout") if candidate else (0, "")
    new = {identity(v) for v in candidate} - {identity(v) for v in current}
    worst_new = max((v.magnitude for v in candidate if identity(v) in new), default=0.)
    count, magnitude = len(candidate) - len(current), sum(v.magnitude for v in candidate) - sum(v.magnitude for v in current)
    if worst_new > PROTECT_M:
        return -1, f"new {name} violation of {worst_new:.3f}"
    if count < 0:
        return 1, f"fewer {name} violations"
    if count > 0:
        return -1, f"{name} violations not reduced"
    if magnitude < -EPS_H:
        return 1, f"smaller {name} violations"
    if magnitude > EPS_H:
        return -1, f"{name} violations not reduced"
    return 0, ""


def identity(v: Violation) -> tuple:
    """What makes a hard violation the same one before and after a move (``Critic.better``'s new violations). A robot
    check's ids list the objects in the way after the one it is about, and those change with every move: the robot
    checks are identified by their object (by their door for robot_door)."""
    if v.code == "robot_door":
        return v.code, v.detail.split(" sealed")[0]
    if v.code.startswith("robot_"):
        return v.code, v.ids[:1]
    return v.code, v.ids


def approach_sides(b: dict) -> list[tuple[np.ndarray, float, float]]:
    """The sides one uses ``b`` from, as (outward unit vector, half width along the side, half depth to the side):
    the front; beds also the long sides; tables and seats any side (a chair is pulled out from behind)."""
    sx, sy, _ = b["size"]
    sides = [(b["front"], sy / 2, sx / 2)]
    if b["head"] in lq.BEDS:
        sides += [(b["side"], sx / 2, sy / 2), (-b["side"], sx / 2, sy / 2)]
    elif b["head"] in TABLE or b["head"] in SEAT:
        sides += [(-b["front"], sy / 2, sx / 2), (b["side"], sx / 2, sy / 2), (-b["side"], sx / 2, sy / 2)]
    return sides


def walkable(floor: Polygon, obstacles: list[Polygon], half_width_m: float):
    """Where the centre of a person ``2 * half_width_m`` wide can stand (SceneSmith's reachability: the floor shrunk
    by the half width minus every obstacle grown by it)."""
    free = floor.buffer(-half_width_m)
    if obstacles and not free.is_empty:
        free = free.difference(unary_union([o.buffer(half_width_m) for o in obstacles]))
    return [g for g in getattr(free, "geoms", [free]) if not g.is_empty and g.area > 1e-6]


def _kind(b: dict) -> str:
    return "seat" if b["head"] in SEAT or b["head"] in SOFA else "table" if b["head"] in TABLE else ""


def unreachable(floor: Polygon, bs: list[dict], fixed: list[Polygon], doors: list[Polygon],
                half_width_m: float = REACH_HALF_WIDTH_M, categories: set[str] | None = None,
                blockers: bool = True) -> list[tuple[str, float, tuple]]:
    """Floor obstacles at least ``lq.ACC_MIN_HEIGHT_M`` tall (of ``categories``, when given) that a person
    ``2 * half_width_m`` wide cannot walk up to from a door.

    The free space a person's centre can use splits into regions; the regions touching a door's clearance zone are
    the reachable ones (the largest region when the room has no door). An object is reached when a reachable region
    comes within ``REACH_ARM_M`` of one of its approach sides (``approach_sides``). Seats and tables do not block each
    other (one sits down between a sofa and its coffee table), as in ACC. Returns (id, distance from the reachable
    space to its nearest side, the obstacles whose removal alone makes it reachable) per unreached object; unlike
    SceneSmith's region count, a sliver behind a bed does not count, and the blockers name what to move."""
    obstacles = [b for b in bs if lq.obstacle(b)]
    targets = [b for b in obstacles if b["size"][2] >= lq.ACC_MIN_HEIGHT_M
               and (categories is None or b["cat"] in categories)]
    ignore = {"seat": "table", "table": "seat"}

    def reached_space(kind: str, skip: str | None = None):
        regions = walkable(floor, [o["poly"] for o in obstacles if o["id"] != skip
                                   and not (kind and _kind(o) == ignore[kind])] + fixed, half_width_m)
        if not regions:
            return None
        touching = [g for g in regions if any(g.distance(d) <= 1e-6 for d in doors)] if doors else []
        return unary_union(touching or [max(regions, key=lambda g: g.area)])

    def gap(space, b) -> float:
        if space is None:
            return math.inf
        reach = half_width_m + REACH_ARM_M
        zones = []
        for d, hw, hd in approach_sides(b):
            a = np.array((-d[1], d[0]))
            near, far = b["centre"] + d * hd, b["centre"] + d * (hd + reach)
            zones.append(Polygon([tuple(near + a * hw), tuple(near - a * hw), tuple(far - a * hw), tuple(far + a * hw)]))
        return min(space.distance(z) for z in zones)

    spaces = {k: reached_space(k) for k in {_kind(b) for b in targets}}
    out = []
    for b in targets:
        kind = _kind(b)
        g = gap(spaces[kind], b)
        if g <= 1e-6:
            continue
        found = ()
        if blockers and spaces[kind] is not None:
            near = [o for o in obstacles if o is not b and o["poly"].distance(b["poly"]) <= 2 * half_width_m + REACH_ARM_M
                    and not (kind and _kind(o) == ignore[kind])]
            found = tuple(o["id"] for o in near if gap(reached_space(kind, skip=o["id"]), b) <= 1e-6)
        out.append((b["id"], g, found))
    return out


def critic_geometry(condition: dict, plan: Plan) -> Geometry:
    """The critic's rules: doors kept clear by infinite-task-world's entry keepout (door_rule "keepout", since
    2026-11-05; "evaluator" before), windows soft except for a wall object hung over one."""
    return Geometry(condition, plan.parents, plan.relations, hard_doors=True, hard_windows=False, door_rule="keepout",
                    wall_windows=True)


class Critic:
    def __init__(self, geometry: Geometry, user_soft: list[dict], plan_soft: list[dict], templates: list[dict],
                 wall_rules: dict | None = None):
        self.g = geometry
        self.user_soft, self.plan_soft = user_soft, plan_soft
        self.templates: dict[str, list[dict]] = {}
        for t in templates:
            self.templates.setdefault(t["a"], []).append(t)
        self.row = {"condition": geometry.condition}  # what layout_quality's helpers read
        self.fixed_polys = lq.fixed_obstacles(self.row)
        self.floor_poly = geometry.polygon.buffer(0)
        self.rules = load_wall_rules() if wall_rules is None else wall_rules
        self.walls = walls(self.floor_poly) if self.floor_poly.geom_type == "Polygon" else []
        self.door_zones = [zone for zone, _ in geometry.doors if not zone.is_empty]
        self.robot = None  # a robot.RobotChecker: its checks join the hard violations

    @classmethod
    def for_plan(cls, geometry: Geometry, plan, templates: list[dict] | None = None,
                 wall_rules: dict | None = None) -> "Critic":
        user = [r for r in geometry.explicit if not r.get("hard", True)]
        return cls(geometry, user, list(plan.relations), load_templates() if templates is None else templates, wall_rules)

    # ---------------------------------------------------------------- hard
    def _hard_magnitude(self, issue: Issue, placed: dict[str, dict]) -> float:
        g, code = self.g, issue.code
        box = placed.get(issue.ids[0]) if issue.ids else None
        if box is None:
            return 1.
        shape = footprint(box)
        z, height = box["bottom_center_m"][2], box["target_size_local_m"][2]
        if code == "collision":
            other = placed.get(issue.ids[1], g.fixed.get(issue.ids[1]))
            return max(_sqrt_area(shape.intersection(footprint(other)).area), .01) if other else 1.
        if code == "boundary":
            return max(_sqrt_area(shape.difference(g.polygon).area), .01)
        if code == "vertical_bounds":
            return max(g.floor - z, z + height - g.ceiling, .01)
        if code == "floor_support":
            return abs(z - g.floor)
        if code == "wall_support":  # the back's gap to a wall it faces away from (1 m when it faces none)
            return max(min(g.wall_gap(box), 1.), .01)
        if code == "object_support":
            parent = placed.get(issue.ids[1], g.fixed.get(issue.ids[1]))
            if parent is None:
                return 1.
            surface = g._support_polygon(box, issue.ids[1], parent)
            return max(abs(z - g._support_height(box, issue.ids[1], parent)) + surface.distance(shape.centroid), .01)
        if code == "door_clearance":
            return max((_sqrt_area(g.door_overlap(box, zone, opening)) for zone, opening in g.clearance_doors),
                       default=.01)
        if code == "window_clearance":
            return max((_sqrt_area(shape.intersection(strip).area) for strip, _ in g.windows), default=.01)
        if code == "keepout":
            return max((_sqrt_area(shape.intersection(Polygon(r["polygon_xy_m"])).area) for r in g.hard
                        if r["type"] == "keepout"), default=.01)
        if code.startswith("relation_"):
            relation = next((r for r in g.hard if "relation_" + r["type"] == code and r.get("object_id") == issue.ids[0]),
                            None)
            gap = g.relation_gap(relation, placed) if relation else None
            return max(gap or 0., .01)
        return 1.

    def hard(self, placed: dict[str, dict]) -> list[Violation]:
        out = [Violation(i.code, i.ids, self._hard_magnitude(i, placed), detail=i.detail) for i in self.g.validate(placed)]
        if self.robot is not None:
            out += self.robot.violations(placed, self.g.parents)
        return out

    # ---------------------------------------------------------------- soft
    def _relations(self, relations: list[dict], code: str, placed: dict[str, dict]) -> list[Violation]:
        out = []
        for r in relations:
            gap = self.g.relation_gap(r, placed)
            if gap:
                scale = SCALE["angle"] if r["type"] in ("faces", "faces_direction") else SCALE["distance"]
                ids = (r["object_id"],) + tuple(t for t in [r.get("target_id")] + list(r.get("target_ids") or []) if t)
                out.append(Violation(code, ids, gap / scale, WEIGHTS[code], r["type"]))
        return out

    def _quality_boxes(self, placed: dict[str, dict]) -> list[dict]:
        layout = {"objects": [placed[i] for i in self.g.requests if i in placed]}
        return lq.boxes(self.row, layout)

    def _templates(self, bs: list[dict]) -> list[Violation]:
        by_cat: dict[str, list[dict]] = {}
        for b in bs:
            if lq.obstacle(b):
                by_cat.setdefault(b["cat"], []).append(b)
        out = []
        for a_cat, a_list in by_cat.items():
            for t in self.templates.get(a_cat, ()):
                b_list = by_cat.get(t["b"])
                if not b_list or t["b"] == a_cat:
                    continue
                for a in a_list:
                    nearest = min(b_list, key=lambda b: a["poly"].distance(b["poly"]))
                    if t["kind"] == "near":
                        excess = a["poly"].distance(nearest["poly"]) - t["max_gap_m"]
                        if excess > 0:
                            out.append(Violation("template_near", (a["id"], nearest["id"]), excess / SCALE["distance"],
                                                 WEIGHTS["template_near"], f"{a_cat} near {t['b']}"))
                    else:
                        excess = facing_error(a, nearest["poly"])
                        if excess > 0:
                            out.append(Violation("template_faces", (a["id"], nearest["id"]), excess / SCALE["angle"],
                                                 WEIGHTS["template_faces"], f"{a_cat} faces {t['b']}"))
        return out

    def _access(self, bs: list[dict]) -> list[Violation]:
        out = []
        obstacles = [b for b in bs if lq.obstacle(b)]
        for b in obstacles:
            if b["size"][2] < lq.ACC_MIN_HEIGHT_M:
                continue
            blocked = unary_union([o["poly"] for o in obstacles if o is not b
                                   and not (b["head"] in SEAT and o["head"] in TABLE)
                                   and not (b["head"] in TABLE and o["head"] in SEAT)] + self.fixed_polys)
            sx, sy, _ = b["size"]
            sides = [(b["front"], sy / 2, sx / 2)]
            if b["head"] in lq.BEDS:
                sides += [(b["side"], sx / 2, sy / 2), (-b["side"], sx / 2, sy / 2)]
            elif b["head"] in TABLE:
                sides += [(-b["front"], sy / 2, sx / 2), (b["side"], sx / 2, sy / 2), (-b["side"], sx / 2, sy / 2)]
            best = 0.
            for direction, half_width, half_depth in sides:
                s = lq.strip(b, direction, half_width, half_depth)
                if s.area > 0:
                    best = max(best, s.intersection(self.floor_poly).difference(blocked).area / s.area)
            if best < .5:
                out.append(Violation("access", (b["id"],), (.5 - best) / SCALE["share"], WEIGHTS["access"],
                                     f"best side {best:.0%} free"))
        return out

    def _wall_back(self, bs: list[dict]) -> list[Violation]:
        out = []
        for b in bs:
            if b["cat"] in self.rules["against_wall"] and lq.obstacle(b) and self.walls:
                error, gap, angle = wall_back_error(b, self.walls)
                if error > 0:
                    out.append(Violation("wall_back", (b["id"],), error, WEIGHTS["wall_back"],
                                         f"back {gap:.2f} m from the wall, {math.degrees(angle):.0f} deg off"))
        return out

    def wall_direction(self, box: dict) -> tuple[float, float] | None:
        """The inward normal of the wall ``box`` is nearest to standing back against (the loop's search target)."""
        b = lq.boxes(self.row, {"objects": [box]})[0]
        best = min(self.walls, key=lambda w: wall_back_error(b, [w])[0], default=None)
        return None if best is None else (float(best[1][0]), float(best[1][1]))

    def _reach(self, bs: list[dict]) -> list[Violation]:
        if not self.rules["reach"] or self.floor_poly.geom_type != "Polygon":
            return []
        return [Violation("reach", (ident,) + blockers, max(gap, .1) / SCALE["distance"], WEIGHTS["reach"],
                          "blocked by " + ", ".join(blockers) if blockers else "no way in")
                for ident, gap, blockers in unreachable(self.floor_poly, bs, self.fixed_polys, self.door_zones,
                                                        self.rules["reach_half_width_m"], self.rules["reach"])]

    def soft(self, placed: dict[str, dict]) -> list[Violation]:
        out = self._relations(self.user_soft, "relation_user", placed)
        out += self._relations(self.plan_soft, "relation_plan", placed)
        bs = self._quality_boxes(placed)
        out += self._templates(bs) + self._access(bs) + self._wall_back(bs) + self._reach(bs)
        for box in placed.values():
            for issue in self.g.opening_issues(box, doors=not self.g.hard_doors, windows=self.g.soft_windows_for(box)):
                code = "door" if issue.code == "door_clearance" else "window"
                out.append(Violation(code, issue.ids, self._hard_magnitude(issue, placed), WEIGHTS[code]))
        return out

    def assess(self, placed: dict[str, dict]) -> Assessment:
        return Assessment(self.hard(placed), self.soft(placed))

    # ---------------------------------------------------------------- commit rule
    @staticmethod
    def better(candidate: Assessment, current: Assessment) -> tuple[bool, str]:
        """Legality first, then the robot checks, then the soft cost: a tier decides when it differs (``_tier``),
        the next one only when it is the same. A robot check never keeps a legality repair out, and a soft gain never
        buys a robot violation."""
        for name, a, b in (("hard", candidate.legality, current.legality), ("robot", candidate.robot, current.robot)):
            verdict, reason = _tier(a, b, name)
            if verdict:
                return verdict > 0, reason
        if candidate.soft_total < current.soft_total - EPS_S:
            return True, "soft cost lower" if not current.hard else "same hard violations, soft cost lower"
        return False, "soft cost not lower"
