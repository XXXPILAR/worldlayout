"""Robot checks: can a mobile robot of a given width use the room? (program only, 2D footprints, no rendering)

The layout is handed to the robot pipeline (WorldEdge: asset realization, validation, execution), so besides the
geometric checks (``Geometry.validate``) a layout must leave the robot a way to work. For a robot ``width_m`` wide:

  robot_door      every door is linked to the others through floor the robot's centre can use (the floor shrunk by half
                  the width minus every floor obstacle grown by it); a room with one door must leave that door open to
                  the room. Ids: the objects whose removal alone links the door (what the repair moves).
  robot_reach     a piece of furniture the robot has to work at is reachable from the doors: the robot's centre can
                  come within ``half width + arm_m`` of one of its usable sides (``critic.approach_sides``). Unlike the
                  critic's soft ``reach`` (a person, mined categories, seats and tables let one pass between them),
                  nothing lets the robot through. Which furniture: ``reach_ids`` (the task's objects; an object on
                  furniture stands for the floor furniture under it), furniture that carries another requested object
                  (``reach_supports``: something on it will be picked up or put down) and openable furniture
                  (``reach_openable``). Ids: the object, then the objects whose removal alone reaches it.
  robot_openable  the doors or drawers of openable floor furniture (``OPENABLE``, plus ``openable_ids``: what the
                  request says the task opens) can open: the sweep in front of it (as wide as its front,
                  ``sweep_depth`` deep) is inside the room and at most ``OPEN_BLOCKED_SHARE`` of it is covered by floor
                  obstacles. Ids: the object, then what covers the sweep.
  robot_arm       an object standing on furniture (``items``; and every task object in ``reach_ids``) is within the
                  arm's reach: some point the robot's centre can reach from the doors is at most ``arm_reach_m`` from
                  the object's centre in plan (EmbodiedGen's ``check_reachable``: 0.20-0.85 m from the robot base to a
                  manipulated object, ``bfs_placement``; its heading limit does not apply to a robot that can turn in
                  place). robot_reach says the robot can stand at the furniture; this says the object is not too deep
                  on it (the middle of a large table, the back of a counter). Ids: the object, then its furniture.

Both levels follow EmbodiedGen: navigation (``RoamTrajectoryGenerator``: the floor minus furniture footprints, cells at
least the clearance from every obstacle, the connected component; the same set as the shrunk floor here, computed
exactly instead of on a 5 cm grid) and manipulation (``check_reachable`` from the robot base).

The doors are the door strips ``Geometry`` builds (``DOOR_DEPTH_M`` deep); with no door in the room (65 of test300's
300 rooms) the largest free region stands in for the way in. ``door_narrow`` notes doors narrower than the robot:
nothing in the layout can fix that, so it is reported, not counted.

``RobotChecker.violations(placed, parents)`` gives the critic's hard violations (``Critic(robot=...)``);
``RobotChecker.report(placed, parents)`` the same as a dict for evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from fastfill.v2.ops import layout_quality as lq

from .contract import fill_room_defaults
from .critic import Violation, approach_sides, walkable
from .geometry import Geometry, Issue

# openable floor furniture: the last word of the category (``lq.head``), or the whole category
OPENABLE_HEADS = {"cabinet", "wardrobe", "closet", "armoire", "dresser", "nightstand", "sideboard", "cupboard",
                  "credenza", "credensa", "refrigerator", "fridge", "freezer", "oven", "dishwasher", "washer", "dryer",
                  "locker", "pantry", "chest", "pedestal", "cooler"}
OPENABLE_CATS = {"bedside table", "washing machine", "side drawer unit", "floor locker column", "microwave oven",
                 "tv stand"}
SWEEP_MIN_M, SWEEP_MAX_M = .35, .6  # a door leaf or drawer sweeps about the furniture's depth, within these
OPEN_BLOCKED_SHARE = .1             # share of the sweep that may be covered (a chair's corner)
SUPPORT_TOP_TOL_M = .05             # inferred supports: the child's bottom within this of the parent's top
EPS = 1e-6


@dataclass(frozen=True)
class Robot:
    width_m: float = .5
    arm_m: float = .4                       # beyond the half width: how near the centre must come to a side
    arm_reach_m: float = .85                # robot centre to an object it handles, in plan (EmbodiedGen's max_reach)
    items: bool = True                      # objects on furniture must be within arm_reach_m
    reach_ids: frozenset = field(default_factory=frozenset)
    reach_supports: bool = True
    reach_openable: bool = True
    doors: bool = True
    openable: bool = True
    openable_ids: frozenset = field(default_factory=frozenset)  # the objects the task opens (articulation), besides
                                                                # openable furniture by category (OPENABLE)

    @property
    def half_width(self) -> float:
        return self.width_m / 2


def openable(b: dict) -> bool:
    return lq.obstacle(b) and b["size"][2] >= lq.ACC_MIN_HEIGHT_M and \
        (b["head"] in OPENABLE_HEADS or b["cat"] in OPENABLE_CATS)


def opens(robot: Robot, b: dict) -> bool:
    """Whether the robot checks treat ``b`` as furniture to open: openable furniture by category, and the objects the
    task opens (``openable_ids``; floor furniture only: a sweep is floor in front of it)."""
    return openable(b) or (b["id"] in robot.openable_ids and lq.obstacle(b))


def sweep_depth(b: dict) -> float:
    return min(max(b["size"][0], SWEEP_MIN_M), SWEEP_MAX_M)


def sweep(b: dict) -> Polygon:
    """The rectangle in front of ``b`` its doors or drawers open into."""
    sx, sy, _ = b["size"]
    across, near = b["side"], b["centre"] + b["front"] * sx / 2
    far = near + b["front"] * sweep_depth(b)
    return Polygon([tuple(near + across * sy / 2), tuple(near - across * sy / 2),
                    tuple(far - across * sy / 2), tuple(far + across * sy / 2)])


def infer_parents(bs: list[dict]) -> dict[str, str]:
    """For a layout without supports (another method's output): an object above the floor stands on the floor
    obstacle whose footprint holds its centre and whose top is at its bottom; else on nothing known ("?")."""
    out = {}
    floor = [b for b in bs if lq.obstacle(b)]
    for b in bs:
        if b["bottom"] <= lq.FLOOR_M:
            out[b["id"]] = "floor"
            continue
        centre = b["poly"].centroid
        under = [p for p in floor if p is not b and p["poly"].covers(centre)
                 and abs(p["bottom"] + p["size"][2] - b["bottom"]) <= SUPPORT_TOP_TOL_M]
        out[b["id"]] = min(under, key=lambda p: p["poly"].area)["id"] if under else "?"
    return out


class RobotChecker:
    def __init__(self, robot: Robot, condition: dict, door_zones: list[Polygon] | None = None):
        self.robot = robot
        self.row = {"condition": condition}
        room = condition["room"]
        self.floor = Polygon(room["floor_polygon_xy_m"]).buffer(0)
        self.fixed = lq.fixed_obstacles(self.row)
        if door_zones is None:
            geometry = Geometry(fill_room_defaults(condition)[0], {}, [])
            door_zones = [zone for zone, _ in geometry.doors if not zone.is_empty]
        self.door_zones = door_zones
        widths = [float(o.get("width_m", 0)) for o in room.get("openings", []) if o.get("type") == "door"]
        self.narrow = [round(w, 3) for w in widths if w < robot.width_m]
        self.requests = {o["id"] for o in condition["objects"]}

    @classmethod
    def for_critic(cls, robot: Robot, critic) -> "RobotChecker":
        return cls(robot, critic.g.condition, critic.door_zones)

    # ---------------------------------------------------------------- free space
    def _space(self, obstacles: list[dict], skip: str | None = None):
        """(regions, the main region): the main one is the region touching the most doors (ties: the largest);
        without doors, the largest region."""
        regions = walkable(self.floor, [o["poly"] for o in obstacles if o["id"] != skip] + self.fixed,
                           self.robot.half_width)
        if not regions:
            return regions, None
        touches = [sum(g.distance(d) <= EPS for d in self.door_zones) for g in regions]
        best = max(range(len(regions)), key=lambda k: (touches[k], regions[k].area))
        return regions, regions[best]

    def _door_gaps(self, regions, main) -> list[float]:
        """Per door: 0 when it touches the main region; else the distance to it (inf without free space)."""
        if main is None:
            return [math.inf] * len(self.door_zones)
        return [0. if main.distance(d) <= EPS else main.distance(d) for d in self.door_zones]

    def _reach_gap(self, main, b: dict) -> float:
        if main is None:
            return math.inf
        reach = self.robot.half_width + self.robot.arm_m
        zones = []
        for d, hw, hd in approach_sides(b):
            a = np.array((-d[1], d[0]))
            near, far = b["centre"] + d * hd, b["centre"] + d * (hd + reach)
            zones.append(Polygon([tuple(near + a * hw), tuple(near - a * hw), tuple(far - a * hw), tuple(far + a * hw)]))
        return min(main.distance(z) for z in zones)

    # ---------------------------------------------------------------- which furniture the robot works at
    def required(self, bs: list[dict], parents: dict[str, str]) -> dict[str, str]:
        """{floor obstacle id: why} for the furniture the robot must reach."""
        by_id = {b["id"]: b for b in bs}

        def root(ident: str) -> str | None:
            seen = set()
            while ident in by_id and ident not in seen:
                seen.add(ident)
                if lq.obstacle(by_id[ident]):
                    return ident
                ident = parents.get(ident, "?")
            return None

        out = {}
        for ident in self.robot.reach_ids:
            r = root(ident)
            if r is not None:
                out.setdefault(r, "task object" if r == ident else f"carries task object {ident}")
        if self.robot.reach_supports:
            for child, parent in parents.items():
                r = root(parent) if parent in by_id else None
                if r is not None and child in by_id:
                    out.setdefault(r, f"carries {child}")
        if self.robot.reach_openable:
            for b in bs:
                if opens(self.robot, b):
                    out.setdefault(b["id"], "openable")
        return {k: v for k, v in out.items() if by_id[k]["size"][2] >= lq.ACC_MIN_HEIGHT_M}

    # ---------------------------------------------------------------- checks
    def _check(self, bs: list[dict], parents: dict[str, str], blockers: bool = True) -> dict:
        obstacles = [b for b in bs if lq.obstacle(b)]
        regions, main = self._space(obstacles)
        found: dict[str, list] = {"robot_door": [], "robot_reach": [], "robot_openable": [], "robot_arm": []}

        def removal_fixes(test, near: list[dict]) -> tuple:
            return tuple(o["id"] for o in near if test(self._space(obstacles, skip=o["id"])))

        if self.robot.doors and self.door_zones:
            for k, gap in enumerate(self._door_gaps(regions, main)):
                if gap <= EPS:
                    continue
                ids = ()
                if blockers:
                    zone = self.door_zones[k]
                    near = [o for o in obstacles if o["poly"].distance(zone) <= 2 * self.robot.width_m]
                    ids = removal_fixes(lambda s, k=k: self._door_gaps(*s)[k] <= EPS, near)
                    if not ids:  # nothing alone links it: the objects in front of the door, nearest first
                        ids = tuple(o["id"] for o in sorted(near, key=lambda o: o["poly"].distance(zone)))[:3]
                found["robot_door"].append((ids, gap, f"door {k}" + (" sealed" if math.isinf(gap) else "")))
        by_id = {b["id"]: b for b in bs}
        for ident, why in sorted(self.required(bs, parents).items()):
            b = by_id[ident]
            gap = self._reach_gap(main, b)
            if gap <= EPS:
                continue
            ids = ()
            if blockers and main is not None:
                near = [o for o in obstacles if o is not b
                        and o["poly"].distance(b["poly"]) <= 2 * self.robot.width_m + self.robot.arm_m]
                ids = removal_fixes(lambda s, b=b: self._reach_gap(s[1], b) <= EPS, near)
            found["robot_reach"].append(((ident,) + ids, gap, why + (", blocked by " + ", ".join(ids) if ids else "")))
        if self.robot.openable:
            for b in obstacles:
                if not opens(self.robot, b):
                    continue
                zone = sweep(b)
                outside = zone.difference(self.floor).area
                covering = [o for o in obstacles if o is not b and o["poly"].intersection(zone).area > 1e-4]
                covered = unary_union([o["poly"] for o in covering] + self.fixed).intersection(zone).area \
                    if covering or self.fixed else 0.
                share = (outside + covered) / max(zone.area, EPS)
                if share > OPEN_BLOCKED_SHARE:
                    what = (["the wall"] if outside > 1e-4 else []) + [o["id"] for o in covering]
                    found["robot_openable"].append(((b["id"],) + tuple(o["id"] for o in covering), share,
                                                    f"{share:.0%} of the {sweep_depth(b):.2f} m sweep blocked by "
                                                    + (", ".join(what) or "fixed objects")))
        for ident, root in self.items(bs, parents).items():
            gap = math.inf if main is None else main.distance(Point(by_id[ident]["centre"])) - self.robot.arm_reach_m
            if gap > EPS:
                found["robot_arm"].append(((ident, root), gap, f"on {root}: " + (
                    "no way in" if math.isinf(gap) else f"{gap + self.robot.arm_reach_m:.2f} m from where the robot can stand")))
        return found

    def items(self, bs: list[dict], parents: dict[str, str]) -> dict[str, str]:
        """{object id: the floor furniture under it} for the objects the arm must reach."""
        by_id = {b["id"]: b for b in bs}
        out = {}
        for b in bs:
            if lq.obstacle(b) or not (self.robot.items or b["id"] in self.robot.reach_ids):
                continue
            seen, cur = {b["id"]}, parents.get(b["id"], "?")
            while cur in by_id and not lq.obstacle(by_id[cur]) and cur not in seen:
                seen.add(cur)
                cur = parents.get(cur, "?")
            if cur in by_id:
                out[b["id"]] = cur
        return out

    def violations(self, placed: dict[str, dict], parents: dict[str, str]) -> list[Violation]:
        bs = lq.boxes(self.row, {"objects": list(placed.values())})
        out = []
        for code, items in self._check(bs, parents).items():
            for ids, magnitude, detail in items:
                m = 1. if math.isinf(magnitude) else magnitude
                out.append(Violation(code, tuple(ids), max(m, .05 if code != "robot_openable" else .01), detail=detail))
        return out

    def report(self, layout: dict, parents: dict[str, str] | None = None) -> dict:
        """Per room: each check's failures plus the totals evaluation pools."""
        bs = lq.boxes(self.row, layout)
        parents = parents or infer_parents(bs)
        found = self._check(bs, parents, blockers=False)
        required = self.required(bs, parents)
        n_open = sum(opens(self.robot, b) for b in bs)
        obstacles = [b for b in bs if lq.obstacle(b) and b["size"][2] >= lq.ACC_MIN_HEIGHT_M]
        _regions, main = self._space([b for b in bs if lq.obstacle(b)])
        reached_all = sum(self._reach_gap(main, b) <= EPS for b in obstacles)
        return {"width_m": self.robot.width_m, "doors": len(self.door_zones), "door_narrow": self.narrow,
                "doors_unlinked": len(found["robot_door"]), "required": len(required),
                "required_unreached": len(found["robot_reach"]), "openable": n_open,
                "openable_blocked": len(found["robot_openable"]), "items": len(self.items(bs, parents)),
                "items_unreached": len(found["robot_arm"]), "furniture": len(obstacles),
                "furniture_reached": reached_all,
                "failures": {k: [{"object_ids": list(ids), "detail": d} for ids, _m, d in v] for k, v in found.items()},
                "ok": not any(found.values())}


# ---------------------------------------------------------------- the search's form of the checks
STAND_BLOCKED_SHARE = .5  # a side's standing strip with more than this covered is not a place to stand


class RobotZones:
    """The robot checks' local form for the placement search (``Geometry.soft_score`` counts these issues).

    The checks above need the room's free space, too slow for every beam candidate (milliseconds each, thousands per
    layer), so the search sees zones in front of furniture instead, as the usage / clearance regions of Yu et al.
    2011 ("Make it Home") and Merrell et al. 2011, Infinigen Indoors' accessibility cost and EmbodiedSmith's approach
    sides do:
      robot_openable  openable furniture: the rectangle in front, as wide as its front and max(sweep depth, robot
                      width) deep (the doors open and the robot stands there), inside the room and at most
                      ``OPEN_BLOCKED_SHARE`` covered
      robot_reach     furniture carrying a requested object: one of its approach sides (``critic.approach_sides``) has
                      a standing strip the robot's width deep, inside the room and at most ``STAND_BLOCKED_SHARE``
                      covered
      robot_arm       an object on such furniture: within ``arm_reach_m`` of a robot centre standing in a free strip
    The critic's hard robot checks (free space, doors linked) still decide; these steer the search towards layouts that
    pass them. Shapes are cached per pose.

    ``reserve(placed)`` fixes, for each piece of furniture carrying objects, the approach side the robot will use: the
    side the draft leaves most open (EmbodiedSmith's approach direction, decided before the search instead of
    re-chosen by it). From then on only that side's strip counts for robot_reach and robot_arm. ``hard``: the search
    drops a candidate that adds a zone issue whenever another candidate for that object adds none (MANSION and
    EmbodiedSmith filter on reachability instead of ranking by it); with no such candidate, the ranking decides as
    before, since every object must be placed."""

    def __init__(self, robot: Robot, geometry: Geometry, hard: bool = False):
        self.robot, self.g, self.hard = robot, geometry, hard
        self.row = {"condition": geometry.condition}
        self.floor = geometry.polygon.buffer(0)
        self.fixed = lq.fixed_obstacles(self.row)
        self.cache: dict[tuple, dict] = {}
        self.covered: dict[tuple, tuple] = {}  # (zone, the obstacles near it) -> _covered's answer
        self.reserved: dict[str, int] = {}     # furniture id -> index of its approach side (``approach_sides``)

    def _box(self, box: dict) -> dict:
        key = (box["id"], *box["bottom_center_m"], box["yaw_rad"], *box["target_size_local_m"])
        b = self.cache.get(key)
        if b is None:
            b = lq.boxes(self.row, {"objects": [box]})[0]
            b["key"] = key
            b["obstacle"], b["openable"] = lq.obstacle(b), opens(self.robot, b)
            b["bounds"] = b["poly"].bounds
            if b["openable"]:
                b["zone"] = self._strip(b, b["front"], b["size"][1] / 2, b["size"][0] / 2,
                                        max(sweep_depth(b), self.robot.width_m))
            b["stands"] = [self._strip(b, d, hw, hd, self.robot.width_m) for d, hw, hd in approach_sides(b)] \
                if b["obstacle"] and b["size"][2] >= lq.ACC_MIN_HEIGHT_M else []
            if len(self.cache) > 50000:
                self.cache.clear()
                self.covered.clear()
            self.cache[key] = b
        return b

    @staticmethod
    def _strip(b: dict, d, hw: float, hd: float, depth: float) -> tuple[Polygon, tuple]:
        a = np.array((-d[1], d[0]))
        near, far = b["centre"] + d * hd, b["centre"] + d * (hd + depth)
        poly = Polygon([tuple(near + a * hw), tuple(near - a * hw), tuple(far - a * hw), tuple(far + a * hw)])
        return poly, poly.bounds

    def _covered(self, zone: tuple, owner: str, obstacles: list[dict]) -> tuple[float, list[str]]:
        poly, (x0, y0, x1, y1) = zone
        near = [o for o in obstacles if o["id"] != owner and o["bounds"][0] < x1 and o["bounds"][2] > x0
                and o["bounds"][1] < y1 and o["bounds"][3] > y0]
        key = (zone[1], poly.area, tuple(o["key"] for o in near))
        hit = self.covered.get(key)
        if hit is None:
            hits = [o for o in near if o["poly"].intersection(poly).area > 1e-4]
            shapes = [o["poly"] for o in hits] + [f for f in self.fixed if f.intersects(poly)]
            covered = unary_union(shapes).intersection(poly).area if shapes else 0.
            hit = self.covered[key] = ((poly.difference(self.floor).area + covered) / max(poly.area, EPS),
                                       [o["id"] for o in hits])
        return hit

    def _carried(self, bs: dict[str, dict]) -> dict[str, list[str]]:
        """{furniture id: the objects on it the robot handles}: every object on furniture, or only the task's
        (``reach_ids``) when the robot does not work at every carrying furniture (``reach_supports`` off)."""
        carried: dict[str, list[str]] = {}
        for ident, b in bs.items():
            if not self.robot.reach_supports and ident not in self.robot.reach_ids:
                continue
            parent = self.g.parents.get(ident)
            if parent in bs and bs[parent]["stands"] and not b["obstacle"]:
                carried.setdefault(parent, []).append(ident)
        return carried

    def _stands(self, p: dict) -> list:
        k = self.reserved.get(p["id"])
        return [p["stands"][k]] if k is not None and k < len(p["stands"]) else p["stands"]

    def reserve(self, placed: dict[str, dict]) -> dict[str, int]:
        """Fix each carrying furniture's approach side: the side least covered in ``placed`` (ties: the first,
        ``approach_sides`` lists the front first)."""
        self.reserved.clear()
        bs = {i: self._box(b) for i, b in placed.items()}
        obstacles = [b for b in bs.values() if b["obstacle"]]
        for parent in self._carried(bs):
            shares = [self._covered(zone, parent, obstacles)[0] for zone in bs[parent]["stands"]]
            self.reserved[parent] = min(range(len(shares)), key=lambda k: (round(shares[k], 3), k))
        return dict(self.reserved)

    def issues(self, placed: dict[str, dict]) -> list:
        bs = {i: self._box(b) for i, b in placed.items()}
        obstacles = [b for b in bs.values() if b["obstacle"]]
        out = []
        for b in obstacles:
            if b["openable"]:
                share, hits = self._covered(b["zone"], b["id"], obstacles)
                if share > OPEN_BLOCKED_SHARE:
                    out.append(Issue("robot_openable", (b["id"], *hits)))
        for parent, children in self._carried(bs).items():
            p = bs[parent]
            free, blockers = [], set()
            for zone in self._stands(p):
                share, hits = self._covered(zone, parent, obstacles)
                if share <= STAND_BLOCKED_SHARE:
                    free.append(zone[0])
                blockers.update(hits)
            if not free:
                out.append(Issue("robot_reach", (parent, *sorted(blockers))))
                continue
            reach = self.robot.arm_reach_m
            for child in children:
                centre = Point(bs[child]["centre"])
                # the robot's centre stands half its width into a strip: within reach of that line
                if min(z.distance(centre) for z in free) + self.robot.half_width > reach + EPS:
                    out.append(Issue("robot_arm", (child, parent)))
        return out
