"""Exact 2D footprints plus height intervals for bbox-level hard checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import shapely
from shapely.geometry import LineString, Polygon
from . import contract  # noqa: F401  (puts the fastfill package on sys.path)
from fastfill.anchors import FLAT
from fastfill.scene import norm_cat
from fastfill.v2.validation import on_wall

EPS = 1e-5
# door rule "evaluator" (the critic's): the door clearance OPC scores (fastfill.v2.ops.layout_quality.opc): a door's
# 0.8 m strip (evaluate.DOOR_ZONE_DEPTH_M, cell_costs.door_zones' rectangle) is blocked by a floor-standing obstacle
# (bottom within FLOOR_OBSTACLE_M of the floor, at least THIN_OBSTACLE_M tall, not a rug or mat) overlapping it by
# more than DOOR_OVERLAP_M2; objects on furniture or on the wall above it do not block it
DOOR_DEPTH_M, FLOOR_OBSTACLE_M, THIN_OBSTACLE_M, DOOR_OVERLAP_M2 = .8, .15, .1, .01
# door rule "keepout" (the critic's since 2026-11-05): infinite-task-world's entry keepout, the rule its admission gate
# refuses a room by (embodied_gen/world/admit.py _check_keepout, A2_KEEPOUT_BLOCKED; growth_planner.KEEPOUT_DEPTH_M,
# KEEPOUT_SIDE_MARGIN_M): a rectangle KEEPOUT_DEPTH_M deep in front of every door, KEEPOUT_SIDE_M wider than the door
# on each side, that no object may enter at any height (a wall shelf or a clock above the door counts), a floor
# covering excepted (on the floor and under KEEPOUT_COVERING_H_M tall: a rug or mat; backend.is_floor_covering). An
# object enters it when its axis-aligned bounding box (backend.footprint_xy) overlaps it by more than KEEPOUT_SLACK_M
# along both axes (admit._overlaps, TOL_M).
KEEPOUT_DEPTH_M, KEEPOUT_SIDE_M, KEEPOUT_COVERING_H_M, KEEPOUT_SLACK_M = 1., .15, .1, .01
# the evaluator's tolerance (fastfill.v2.validation.validate_scene tolerance_m): vertical bounds
BOUNDARY_TOL = 1e-4
# Contact as SceneEval checks it (its metric code and metrics.yaml). Its OOB
# fails an object when under 99% of the points sampled on it hit the floor below: a footprint flush with the wall
# and out by float noise loses its whole back face, so a footprint keeps BOUNDARY_INSET_M inside the room. Its
# Support moves a wall object 5 mm off its wall and casts rays backward from its back face, needing a hit within
# 1 cm: a wall object's back (local -x) lies within WALL_CONTACT_M of a wall it faces away from (front within
# WALL_FACING_RAD of the wall's inward normal), and is placed WALL_GAP_M off it (snap_to_wall).
BOUNDARY_INSET_M = .001
WALL_GAP_M, WALL_CONTACT_M, WALL_FACING_RAD = .002, .004, math.radians(10.)
# an object stands on another when its footprint centre lies on the parent's surface, at least this far inside it
# (SceneEval's Support needs the centroid inside the hull of the contact points it samples), and its bottom is at
# the surface height; its footprint may overhang (ground-truth shelves and tables show it)
SUPPORT_CENTRE_MARGIN = -.01
# bottom vs support height (and floor) agree within the evaluator's tolerance, not float noise
SUPPORT_Z_TOL = 1e-4


# the search asks for the same boxes' footprints and wall contacts millions of times (every candidate re-scores the
# room's relations): both are pure functions of the pose, cached here (shapely geometries are immutable)
_FOOTPRINTS: dict[tuple, Polygon] = {}
_ON_WALL: dict[tuple, bool] = {}
CACHE_LIMIT = 200_000


def footprint(box: dict) -> Polygon:
    sx, sy = box["target_size_local_m"][:2]
    x, y = box["bottom_center_m"][:2]
    yaw = box["yaw_rad"]
    key = (x, y, yaw, sx, sy)
    shape = _FOOTPRINTS.get(key)
    if shape is None:
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(-sx / 2, -sy / 2), (sx / 2, -sy / 2),
                   (sx / 2, sy / 2), (-sx / 2, sy / 2)]
        shape = Polygon([(x + c * a - s * b, y + s * a + c * b) for a, b in corners])
        if len(_FOOTPRINTS) > CACHE_LIMIT:
            _FOOTPRINTS.clear()
        _FOOTPRINTS[key] = shape
    return shape


def on_wall_cached(box: dict, room: dict, tolerance: float) -> bool:
    """``validation.on_wall`` for a box, memoised by the room's polygon and the box's pose and size."""
    key = (tuple(map(tuple, room["floor_polygon_xy_m"])), *box["bottom_center_m"], box["yaw_rad"],
           *box["target_size_local_m"], tolerance)
    hit = _ON_WALL.get(key)
    if hit is None:
        hit = bool(on_wall(box, room, tolerance=tolerance))
        if len(_ON_WALL) > CACHE_LIMIT:
            _ON_WALL.clear()
        _ON_WALL[key] = hit
    return hit


def as_box(fixed: dict) -> dict:
    return {"id": fixed["id"], "target_size_local_m": fixed["size_local_m"],
            "bottom_center_m": fixed["bottom_center_m"], "yaw_rad": fixed["yaw_rad"]}


def _vertical_overlap(a: dict, b: dict) -> bool:
    az, bz = a["bottom_center_m"][2], b["bottom_center_m"][2]
    return min(az + a["target_size_local_m"][2], bz + b["target_size_local_m"][2]) - max(az, bz) > EPS


def _angle_error(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _signed_area(points: list) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1])) / 2


def _opening_strip(points: list, opening: dict, depth: float, margin: float = 0.) -> Polygon | None:
    """The strip ``depth`` deep in front of ``opening`` (into the room), as wide as the opening; with ``margin``,
    that much wider on each side and not cut at the wall's ends (the caller intersects it with the room)."""
    index = opening.get("wall_index")
    if not isinstance(index, int) or not 0 <= index < len(points):
        return None
    a, b = points[index], points[(index + 1) % len(points)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length <= EPS:
        return None
    ux, uy = dx / length, dy / length
    sign = 1 if _signed_area(points) > 0 else -1
    nx, ny = -sign * uy, sign * ux
    center = float(opening.get("position_on_wall", .5)) * length
    width = float(opening.get("width_m", 0))
    lo, hi = center - width / 2 - margin, center + width / 2 + margin
    if not margin:
        lo, hi = max(0, lo), min(length, hi)
    if hi <= lo:
        return None
    p, q = (a[0] + lo * ux, a[1] + lo * uy), (a[0] + hi * ux, a[1] + hi * uy)
    return Polygon([p, q, (q[0] + depth * nx, q[1] + depth * ny),
                    (p[0] + depth * nx, p[1] + depth * ny)])


@dataclass(frozen=True)
class Issue:
    code: str
    ids: tuple[str, ...]
    detail: str = ""

    def as_dict(self) -> dict:
        return {"code": self.code, "object_ids": list(self.ids), "detail": self.detail}


class Geometry:
    def __init__(self, condition: dict, parents: dict[str, str], relations: list[dict],
                 *, allow_thin_overlap: bool = False, hard_openings: bool = True, hard_doors: bool | None = None,
                 hard_windows: bool | None = None, door_rule: str = "workflow", wall_windows: bool = False):
        """``hard_openings=False`` (the draft workflow without the critic) scores door and window clearance as soft
        issues: the evaluator does not check openings, and ground-truth rooms often place furniture in those strips.
        ``hard_doors`` / ``hard_windows`` set the two separately (default: ``hard_openings``). ``door_rule``:
        "workflow" (any object in the strip below the door top, a floor object at most 5 cm tall excepted),
        "evaluator" (the OPC rule, see DOOR_DEPTH_M) or "keepout" (infinite-task-world's entry keepout, see
        KEEPOUT_DEPTH_M). ``wall_windows``: with soft windows, a wall object over a window (its back strip, between
        sill and top) is still a hard window_clearance issue (SceneSmith refuses wall objects on openings), while a
        floor object under a window stays soft."""
        if door_rule not in ("workflow", "evaluator", "keepout"):
            raise ValueError(f"unknown door_rule {door_rule!r}")
        self.condition = condition
        self.room = condition["room"]
        self.floor = float(self.room["floor_z_m"])
        self.ceiling = self.floor + float(self.room["height_m"])
        self.polygon = Polygon(self.room["floor_polygon_xy_m"])
        # the boundary check's region, built once (``check_box`` runs for every search candidate)
        self._inside = self.polygon.buffer(-BOUNDARY_INSET_M)
        shapely.prepare(self._inside)
        # the walls a wall object can hang on: (a, unit direction, length, inward unit normal) per polygon edge
        ring = shapely.geometry.polygon.orient(self.polygon).exterior.coords[:-1]  # counter-clockwise
        self._walls = []
        for a, b in zip(ring, ring[1:] + ring[:1]):
            length = math.dist(a, b)
            if length > 1e-6:
                u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
                self._walls.append((a, u, length, (-u[1], u[0])))
        self._wall_gaps: dict[tuple, float] = {}
        self.requests = {o["id"]: o for o in condition["objects"]}
        self.fixed_raw = {o["id"]: o for o in self.room.get("fixed_objects", [])}
        self.fixed = {ident: as_box(obj) for ident, obj in self.fixed_raw.items()}
        self.parents = parents
        self.allow_thin_overlap = allow_thin_overlap
        self.hard_doors = hard_openings if hard_doors is None else hard_doors
        self.hard_windows = hard_openings if hard_windows is None else hard_windows
        self.hard_openings = self.hard_doors and self.hard_windows
        self.door_rule = door_rule
        self.wall_windows = wall_windows
        # robot.RobotZones, when a robot uses the room: its zone issues join the soft score the search ranks by
        self.robot_zones = None
        self.explicit = condition["constraints"]
        self.soft = relations + [r for r in self.explicit if not r.get("hard", True)]
        self.hard = [r for r in self.explicit if r.get("hard", True)]
        # doors: the DOOR_DEPTH_M strips (the robot's doors and the "workflow" / "evaluator" clearance); keepouts:
        # the "keepout" rule's rectangles
        self.doors, self.keepouts, self.windows = [], [], []
        for opening in self.room.get("openings", []):
            if opening.get("type") == "door":
                zone = _opening_strip(self.room["floor_polygon_xy_m"], opening, DOOR_DEPTH_M)
                if zone is not None:
                    self.doors.append((zone.intersection(self.polygon), opening))
                keep = _opening_strip(self.room["floor_polygon_xy_m"], opening, KEEPOUT_DEPTH_M, KEEPOUT_SIDE_M)
                if keep is not None:
                    self.keepouts.append((keep.intersection(self.polygon), opening))
            elif opening.get("type") == "window":
                strip = _opening_strip(self.room["floor_polygon_xy_m"], opening, .12)
                if strip is not None:
                    self.windows.append((strip.intersection(self.polygon), opening))

    def _support_polygon(self, child: dict, parent_id: str, parent: dict) -> Polygon:
        raw = self.fixed_raw.get(parent_id)
        surface_id = self.requests.get(child["id"], {}).get("support_surface_id")
        surfaces = raw.get("support_surfaces") if raw else None
        if surfaces:
            selected = next((s for s in surfaces if surface_id is None or s["surface_id"] == surface_id), None)
            if selected is None:
                return Polygon()
            x, y = parent["bottom_center_m"][:2]
            c, s = math.cos(parent["yaw_rad"]), math.sin(parent["yaw_rad"])
            return Polygon([(x + c * u - s * v, y + s * u + c * v)
                            for u, v in selected["local_polygon_xy_m"]])
        return footprint(parent)

    def _support_height(self, child: dict, parent_id: str, parent: dict) -> float:
        raw = self.fixed_raw.get(parent_id)
        surface_id = self.requests.get(child["id"], {}).get("support_surface_id")
        surfaces = raw.get("support_surfaces") if raw else None
        if surfaces:
            selected = next((s for s in surfaces if surface_id is None or s["surface_id"] == surface_id), None)
            if selected is not None:
                return parent["bottom_center_m"][2] + selected["local_z_m"]
        return parent["bottom_center_m"][2] + parent["target_size_local_m"][2]

    def wall_gap(self, box: dict) -> float:
        """Distance from ``box``'s back face to the nearest wall it faces away from (inf when it faces none)."""
        sx, sy = box["target_size_local_m"][:2]
        x, y = box["bottom_center_m"][:2]
        yaw = box["yaw_rad"]
        key = (x, y, yaw, sx, sy)
        gap = self._wall_gaps.get(key)
        if gap is None:
            c, s = math.cos(yaw), math.sin(yaw)
            bx, by = x - c * sx / 2, y - s * sx / 2
            back = LineString([(bx + s * sy / 2, by - c * sy / 2), (bx - s * sy / 2, by + c * sy / 2)])
            gap = min((back.distance(LineString([a, (a[0] + u[0] * length, a[1] + u[1] * length)]))
                       for a, u, length, n in self._walls
                       if c * n[0] + s * n[1] >= math.cos(WALL_FACING_RAD)), default=math.inf)
            if len(self._wall_gaps) > CACHE_LIMIT:
                self._wall_gaps.clear()
            self._wall_gaps[key] = gap
        return gap

    def snap_to_wall(self, size: list[float], x: float, y: float, yaw: float) -> tuple[float, float, float]:
        """(x, y, yaw) of a wall object of ``size`` near (x, y) hung on the nearest wall: its back WALL_GAP_M off the
        wall, facing into the room, at the same place along the wall (kept on it when it fits)."""
        def along(wall):
            a, u, length, _ = wall
            return min(max((x - a[0]) * u[0] + (y - a[1]) * u[1], 0.), length)

        def distance(wall):
            a, u, _, n = wall
            t = along(wall)
            facing = math.cos(yaw) * n[0] + math.sin(yaw) * n[1]
            return math.dist((x, y), (a[0] + u[0] * t, a[1] + u[1] * t)) + (0. if facing > 0 else 1e-3)

        if not self._walls:
            return x, y, yaw
        wall = min(self._walls, key=distance)
        a, u, length, n = wall
        half = size[1] / 2
        t = length / 2 if length <= 2 * half else min(max(along(wall), half), length - half)
        depth = size[0] / 2 + WALL_GAP_M
        return a[0] + u[0] * t + n[0] * depth, a[1] + u[1] * t + n[1] * depth, math.atan2(n[1], n[0])

    def push_inside(self, size: list[float], x: float, y: float, yaw: float,
                    limit: float = .05) -> tuple[float, float]:
        """(x, y) of a footprint of ``size`` at (x, y, yaw) moved straight off the walls it crosses or touches, by at
        most ``limit``, to lie BOUNDARY_INSET_M inside the room (a draft flush with a wall, or a few cm through it);
        (x, y) unchanged when it is inside already or that does not bring it inside."""
        box = {"target_size_local_m": size, "bottom_center_m": [x, y, 0.], "yaw_rad": yaw}
        shape = footprint(box)
        if self._inside.covers(shape):
            return x, y
        corners = shape.exterior.coords[:-1]
        want = BOUNDARY_INSET_M + 2e-4  # past the pose's rounding to 0.1 mm (solver._pose)
        dx = dy = 0.
        for a, u, length, n in self._walls:
            need = 0.
            for p in corners:
                t = (p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1]
                inside = (p[0] - a[0]) * n[0] + (p[1] - a[1]) * n[1]
                if -limit <= t <= length + limit and -limit <= inside < want:
                    need = max(need, want - inside)
            dx, dy = dx + n[0] * need, dy + n[1] * need
        if not 0 < math.hypot(dx, dy) <= limit:
            return x, y
        box["bottom_center_m"] = [x + dx, y + dy, 0.]
        return (x + dx, y + dy) if self._inside.covers(footprint(box)) else (x, y)

    def check_box(self, box: dict, placed: dict[str, dict]) -> list[Issue]:
        ident = box["id"]
        shape = footprint(box)
        z, height = box["bottom_center_m"][2], box["target_size_local_m"][2]
        issues = []
        if not self._inside.covers(shape):
            issues.append(Issue("boundary", (ident,)))
        if z < self.floor - BOUNDARY_TOL or z + height > self.ceiling + BOUNDARY_TOL:
            issues.append(Issue("vertical_bounds", (ident,)))
        parent_id = self.parents[ident]
        if parent_id == "floor":
            if abs(z - self.floor) > SUPPORT_Z_TOL:
                issues.append(Issue("floor_support", (ident,)))
        elif parent_id == "wall":
            if self.wall_gap(box) > WALL_CONTACT_M:
                issues.append(Issue("wall_support", (ident,)))
        else:
            parent = placed.get(parent_id, self.fixed.get(parent_id))
            if parent is None:
                issues.append(Issue("parent_not_placed", (ident, parent_id)))
            else:
                surface = self._support_polygon(box, parent_id, parent)
                if (abs(z - self._support_height(box, parent_id, parent)) > SUPPORT_Z_TOL or
                        not surface.buffer(SUPPORT_CENTRE_MARGIN).covers(shape.centroid)):
                    issues.append(Issue("object_support", (ident, parent_id)))

        if self.hard_doors or self.hard_windows:
            issues.extend(self.opening_issues(box, doors=self.hard_doors, windows=self.hard_windows))
        if self.wall_windows and not self.hard_windows and parent_id == "wall":
            issues.extend(self.opening_issues(box, doors=False, windows=True))
        issues.extend(self._collisions_and_keepout(box, placed, shape, height))
        return issues

    @property
    def clearance_doors(self) -> list[tuple[Polygon, dict]]:
        """The door zones ``door_rule`` keeps clear: the keepout rectangles, or the DOOR_DEPTH_M strips."""
        return self.keepouts if self.door_rule == "keepout" else self.doors

    def soft_windows_for(self, box: dict) -> bool:
        """Does ``box``'s window clearance count as a soft issue (not already a hard one)?"""
        return not self.hard_windows and not (self.wall_windows and self.parents.get(box["id"]) == "wall")

    def blocks_doors(self, box: dict) -> bool:
        """Can ``box`` block a door under ``door_rule``?"""
        z, height = box["bottom_center_m"][2], box["target_size_local_m"][2]
        if self.door_rule == "keepout":  # all but a floor covering, at any height
            return not (self.parents.get(box["id"], "floor") == "floor" and height < KEEPOUT_COVERING_H_M)
        if self.door_rule == "evaluator":
            words = norm_cat(self.requests.get(box["id"], {}).get("category") or "object").split()
            return z - self.floor <= FLOOR_OBSTACLE_M and height >= THIN_OBSTACLE_M and \
                (words[-1] if words else "") not in FLAT
        return not (height <= .05 and self.parents.get(box["id"], "floor") == "floor")

    def door_overlap(self, box: dict, zone: Polygon, opening: dict) -> float:
        """Area of ``box``'s footprint in the door ``zone`` that counts as blocking (0 when it does not block)."""
        if not self.blocks_doors(box):
            return 0.
        if self.door_rule == "keepout":  # the axis-aligned bounds in, past the slack along both axes
            part = shapely.box(*footprint(box).bounds).intersection(zone)
            if part.is_empty:
                return 0.
            x0, y0, x1, y1 = part.bounds
            return part.area if min(x1 - x0, y1 - y0) > KEEPOUT_SLACK_M else 0.
        z, height = box["bottom_center_m"][2], box["target_size_local_m"][2]
        if self.door_rule == "workflow":
            door_top = self.floor + float(opening.get("height_m", 2.1))
            if not (z < door_top - EPS and z + height > self.floor + EPS):
                return 0.
        area = footprint(box).intersection(zone).area
        return area if area > (DOOR_OVERLAP_M2 if self.door_rule == "evaluator" else EPS) else 0.

    def opening_issues(self, box: dict, *, doors: bool = True, windows: bool = True) -> list[Issue]:
        ident, shape = box["id"], footprint(box)
        z, height = box["bottom_center_m"][2], box["target_size_local_m"][2]
        issues = []
        for zone, opening in self.clearance_doors if doors else ():
            if self.door_overlap(box, zone, opening):
                issues.append(Issue("door_clearance", (ident,)))
                break
        for strip, opening in self.windows if windows else ():
            sill = self.floor + float(opening.get("sill_height_m", 0))
            top = sill + float(opening.get("height_m", 1.2))
            if z < top - EPS and z + height > sill + EPS and shape.intersection(strip).area > EPS:
                issues.append(Issue("window_clearance", (ident,)))
                break
        return issues

    def _collisions_and_keepout(self, box: dict, placed: dict[str, dict], shape, height: float) -> list[Issue]:
        ident, issues = box["id"], []
        x0, y0, x1, y1 = shape.bounds
        for other in list(placed.values()) + list(self.fixed.values()):
            if other["id"] == ident:
                continue
            if self.allow_thin_overlap and (height <= .05 or other["target_size_local_m"][2] <= .05):
                continue
            other_shape = footprint(other)
            a0, b0, a1, b1 = other_shape.bounds  # disjoint bounds: no overlap (exact, only skips the area test)
            if a0 > x1 or a1 < x0 or b0 > y1 or b1 < y0:
                continue
            if _vertical_overlap(box, other) and shape.intersection(other_shape).area > EPS:
                issues.append(Issue("collision", (ident, other["id"])))
        for relation in self.hard:
            if relation["type"] == "keepout":
                target = relation.get("object_id")
                if (target is None or target == ident) and shape.intersection(
                        Polygon(relation["polygon_xy_m"])).area > EPS:
                    issues.append(Issue("keepout", (ident,)))
        return issues

    def relation_issue(self, relation: dict, placed: dict[str, dict]) -> Issue | None:
        kind = relation["type"]
        if kind == "keepout":
            return None
        ident = relation["object_id"]
        obj = placed.get(ident, self.fixed.get(ident))
        target_id = relation.get("target_id", relation.get("parent_id"))
        target = placed.get(target_id, self.fixed.get(target_id)) if target_id else None
        if obj is None or (kind in {"near", "clearance", "faces"} and target is None):
            return None
        if kind == "faces_direction":
            dx, dy = relation["direction_xy"]
            ok = _angle_error(obj["yaw_rad"], math.atan2(dy, dx)) <= relation.get("tolerance_rad", math.pi / 6) + EPS
        elif kind == "faces":
            dx = target["bottom_center_m"][0] - obj["bottom_center_m"][0]
            dy = target["bottom_center_m"][1] - obj["bottom_center_m"][1]
            ok = math.hypot(dx, dy) > EPS and _angle_error(
                obj["yaw_rad"], math.atan2(dy, dx)) <= relation.get("tolerance_rad", math.pi / 6) + EPS
        elif kind in {"near", "clearance"}:
            distance = footprint(obj).distance(footprint(target))
            ok = (distance <= relation.get("max_distance_m", 1.0) + EPS if kind == "near" else
                  distance + EPS >= relation["min_distance_m"])
        elif kind == "against_wall":
            ok = on_wall_cached(obj, self.room, relation.get("tolerance_m", .1))
        elif kind == "between":
            targets = relation["target_ids"]
            if any(t not in placed and t not in self.fixed for t in targets):
                return None
            a, b = [placed.get(t, self.fixed.get(t))["bottom_center_m"][:2] for t in targets]
            segment = LineString([a, b])
            ok = segment.intersects(footprint(obj)) and not segment.touches(footprint(obj))
        elif kind == "on":
            ok = self.parents.get(ident) == target_id
        else:
            raise ValueError(f"unsupported relation: {kind}")
        return None if ok else Issue("relation_" + kind, (ident,) + ((target_id,) if target_id else ()))

    def relation_gap(self, relation: dict, placed: dict[str, dict]) -> float | None:
        """How far ``relation`` is from holding, 0 when ``relation_issue`` passes it (None: not applicable). Metres
        for distances, radians for facings; a failed against_wall / between / on counts at least 0.05."""
        if self.relation_issue(relation, placed) is None:
            kind = relation["type"]
            obj = placed.get(relation.get("object_id"), self.fixed.get(relation.get("object_id")))
            return None if kind == "keepout" or obj is None else 0.
        kind = relation["type"]
        obj = placed.get(relation["object_id"], self.fixed.get(relation["object_id"]))
        target_id = relation.get("target_id", relation.get("parent_id"))
        target = placed.get(target_id, self.fixed.get(target_id)) if target_id else None
        if kind in {"faces", "faces_direction"}:
            if kind == "faces":
                dx = target["bottom_center_m"][0] - obj["bottom_center_m"][0]
                dy = target["bottom_center_m"][1] - obj["bottom_center_m"][1]
            else:
                dx, dy = relation["direction_xy"]
            return max(_angle_error(obj["yaw_rad"], math.atan2(dy, dx)) - relation.get("tolerance_rad", math.pi / 6), 0.)
        if kind in {"near", "clearance"}:
            distance = footprint(obj).distance(footprint(target))
            return max(distance - relation.get("max_distance_m", 1.0), 0.) if kind == "near" else \
                max(relation["min_distance_m"] - distance, 0.)
        if kind == "against_wall":
            return max(footprint(obj).distance(self.polygon.exterior) - relation.get("tolerance_m", .1), .05)
        if kind == "between":
            a, b = [placed.get(t, self.fixed.get(t))["bottom_center_m"][:2] for t in relation["target_ids"]]
            return max(LineString([a, b]).distance(footprint(obj).centroid), .05)
        return 1.

    def new_hard_issues(self, box: dict, placed: dict[str, dict]) -> list[Issue]:
        trial = {**placed, box["id"]: box}
        issues = self.check_box(box, placed)
        for relation in self.hard:
            if relation.get("object_id") == box["id"] or box["id"] in (
                    relation.get("target_id"), relation.get("parent_id"), *(relation.get("target_ids") or [])):
                issue = self.relation_issue(relation, trial)
                if issue is not None:
                    issues.append(issue)
        return issues

    def validate(self, placed: dict[str, dict]) -> list[Issue]:
        issues = []
        if set(placed) != set(self.requests):
            issues.append(Issue("missing_or_extra_ids", tuple(sorted(set(placed) ^ set(self.requests)))))
        for ident, box in placed.items():
            issues.extend(self.check_box(box, {k: v for k, v in placed.items() if k != ident}))
        for relation in self.hard:
            issue = self.relation_issue(relation, placed)
            if issue is not None:
                issues.append(issue)
        return issues

    def soft_score(self, placed: dict[str, dict]) -> tuple[float, list[Issue]]:
        issues = [issue for relation in self.soft
                  if (issue := self.relation_issue(relation, placed)) is not None]
        if not (self.hard_doors and self.hard_windows):
            issues += [issue for box in placed.values() for issue in
                       self.opening_issues(box, doors=not self.hard_doors, windows=not self.hard_windows)]
        if self.robot_zones is not None:
            issues += self.robot_zones.issues(placed)
        return -float(len(issues)), issues
