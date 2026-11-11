"""Multi-proposer candidate search with hard geometric pruning and local repair."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .catalog import SizeCatalog
from .geometry import WALL_GAP_M, Geometry, footprint
from .planning import Plan


def _wrap(yaw: float) -> float:
    return (yaw + math.pi) % (2 * math.pi) - math.pi


def _pose(ident: str, size: list[float], x: float, y: float, z: float,
          yaw: float, parent: str) -> dict:
    return {"id": ident, "target_size_local_m": size,
            # z unrounded: it is a support height that the checks compare exactly (see geometry.SUPPORT_Z_TOL)
            "bottom_center_m": [round(x, 4), round(y, 4), float(z)],
            "yaw_rad": _wrap(yaw), "support_parent": parent}


def _priority(request: dict, size: list[float], geometry: Geometry) -> float:
    ident = request["id"]
    area = size[0] * size[1]
    targets = sum(ident in (r.get("target_id"), r.get("parent_id")) or
                  ident in r.get("target_ids", []) for r in geometry.explicit + geometry.soft)
    subjects = sum(ident == r.get("object_id") for r in geometry.explicit)
    return area + .4 * targets + .22 * subjects


def _order(geometry: Geometry, catalog: SizeCatalog, plan: Plan, subset: set[str]) -> list[str]:
    pending, done, result = set(subset), set(geometry.fixed) | (set(geometry.requests) - subset), []
    relation_dependencies: dict[str, set[str]] = {ident: set() for ident in geometry.requests}
    for relation in geometry.hard:
        ident = relation.get("object_id")
        if ident not in relation_dependencies:
            continue
        if relation["type"] == "between":
            relation_dependencies[ident].update(relation["target_ids"])
        elif relation["type"] in {"near", "faces", "clearance"}:
            relation_dependencies[ident].add(relation["target_id"])
    while pending:
        ready = [ident for ident in pending if plan.parents[ident] in done | {"floor", "wall"} and
                 relation_dependencies[ident] <= done]
        if not ready:
            ready = [ident for ident in pending if plan.parents[ident] in done | {"floor", "wall"}]
        if not ready:
            raise ValueError("support dependencies cannot be ordered")
        # the plan's priority (1 = first) leads; objects it does not rank follow, by size and relation count
        chosen = max(ready, key=lambda ident: (ident in plan.priority, -plan.priority.get(ident, 0), _priority(
                     geometry.requests[ident],
                     catalog.candidates(geometry.requests[ident])[0][0], geometry)))
        result.append(chosen)
        pending.remove(chosen)
        done.add(chosen)
    return result


def _corners_relative(size: list[float], yaw: float, nx: float, ny: float) -> float:
    c, s = math.cos(yaw), math.sin(yaw)
    return abs(nx * c + ny * s) * size[0] / 2 + abs(-nx * s + ny * c) * size[1] / 2


@dataclass
class Candidate:
    box: dict
    unary: float
    origin: str


@dataclass
class State:
    placed: dict[str, dict]
    unary: float
    rank: float


LOCAL_RINGS_M = (.05, .10, .20, .40)  # local repair: offsets around the draft, nearest first
# a draft may carry a second yaw (``alt_yaw_rad``: the hybrid's LLM yaw beside FastFill's): turning to it costs this
# many quarter turns, so the solver keeps the first yaw unless the object only fits with the second
ALT_YAW_PENALTY_TURNS = .25


class CandidateGenerator:
    """Pose proposals per object. Mode "local" (step ④ local repair) keeps sizes as the catalog gives them (one per
    object, see ``FixedSizes``), proposes rings of offsets around the draft, penalises distance from the draft by
    ``draft_weight`` per metre (and per quarter turn of yaw) instead of a flat draft bonus, adds a same-direction yaw
    to relation candidates, moves a child with its parent, and shrinks an object only when nothing else fits."""

    def __init__(self, geometry: Geometry, catalog: SizeCatalog, plan: Plan,
                 mode: str, density: int, draft_weight: float = 0.):
        self.g, self.catalog, self.plan, self.mode, self.density = geometry, catalog, plan, mode, density
        self.draft_weight = draft_weight
        self.local = mode == "local"
        self.shrink: dict[str, int] = {}  # id -> shrink level (local mode, set only while a retry runs)
        self.cache: dict[str, list[Candidate]] = {}

    def _sizes(self, ident: str) -> list[tuple[list[float], float, str]]:
        request = self.g.requests[ident]
        level = self.shrink.get(ident)
        if level:
            return self.catalog.shrunk(request, level)
        return self.catalog.candidates(request)

    def _draft_cost(self, box: dict) -> float:
        draft = self.plan.draft_poses.get(box["id"])
        if not draft or not self.draft_weight:
            return 0.
        x, y = box["bottom_center_m"][:2]
        turn = abs(_wrap(box["yaw_rad"] - draft["yaw_rad"])) / (math.pi / 2)
        if draft.get("alt_yaw_rad") is not None:
            turn = min(turn, abs(_wrap(box["yaw_rad"] - draft["alt_yaw_rad"])) / (math.pi / 2) + ALT_YAW_PENALTY_TURNS)
        return self.draft_weight * (math.hypot(x - draft["x"], y - draft["y"]) + turn)

    def _unary(self, box: dict, size_score: float, origin: str) -> float:
        area = box["target_size_local_m"][0] * box["target_size_local_m"][1]
        edge = min(footprint(box).distance(self.g.polygon.exterior), 1.0)
        wall_weight = .55 if area > .7 else .12
        if self.mode == "spacious":
            wall_weight *= .6
        elif self.mode == "wall":
            wall_weight *= 1.5
        score = size_score - wall_weight * edge
        if self.local:
            return score - self._draft_cost(box)
        if origin == "llm":
            score += .5
        if origin == "relative":
            score += .15
        return score

    def _generic(self, ident: str) -> list[Candidate]:
        if ident in self.cache:
            return self.cache[ident]
        parent = self.plan.parents[ident]
        if parent not in {"floor", "wall"}:
            return []
        poly = self.g.polygon
        points = self.g.room["floor_polygon_xy_m"]
        sign = 1 if sum(a[0] * b[1] - b[0] * a[1] for a, b in
                        zip(points, points[1:] + points[:1])) > 0 else -1
        sizes = self._sizes(ident)
        proposals: list[Candidate] = []

        def add(size: list[float], size_score: float, x: float, y: float, yaw: float,
                z: float, origin: str) -> None:
            if parent == "wall":  # hung on the nearest wall, back against it (geometry.WALL_CONTACT_M)
                x, y, yaw = self.g.snap_to_wall(size, x, y, yaw)
            box = _pose(ident, size, x, y, z, yaw, parent)
            if not self.g.check_box(box, {}):
                proposals.append(Candidate(box, self._unary(box, size_score, origin), origin))

        for size, size_score, _source in sizes:
            for a, b in zip(points, points[1:] + points[:1]):
                dx, dy = b[0] - a[0], b[1] - a[1]
                length = math.hypot(dx, dy)
                if length <= .1:
                    continue
                tx, ty = dx / length, dy / length
                nx, ny = -sign * ty, sign * tx
                yaw = math.atan2(ny, nx)
                extent = _corners_relative(size, yaw, nx, ny)
                count = max(2, math.ceil(length / self.density))
                for j in range(count):
                    t = (j + .5) / count
                    for offset in ((WALL_GAP_M,) if parent == "wall" else (.04, .35)):
                        x = a[0] + t * dx + (extent + offset) * nx
                        y = a[1] + t * dy + (extent + offset) * ny
                        z = (min(self.g.floor + 1.15, self.g.ceiling - size[2] - .03)
                             if parent == "wall" else self.g.floor)
                        if z >= self.g.floor:
                            add(size, size_score, x, y, yaw, z, "wall")
            if parent == "floor":
                xmin, ymin, xmax, ymax = poly.bounds
                step = max(.45, self.density * .6)
                nx = max(2, min(11, math.ceil((xmax - xmin) / step)))
                ny = max(2, min(11, math.ceil((ymax - ymin) / step)))
                directions = [math.atan2(r["direction_xy"][1], r["direction_xy"][0])
                              for r in self.g.explicit if r.get("type") == "faces_direction" and
                              r.get("object_id") == ident]
                for i in range(nx):
                    for j in range(ny):
                        x = xmin + (i + .5) * (xmax - xmin) / nx
                        y = ymin + (j + .5) * (ymax - ymin) / ny
                        for yaw in (0, math.pi / 2, math.pi, -math.pi / 2, *directions):
                            add(size, size_score, x, y, yaw, self.g.floor, "grid")
            draft = self.plan.draft_poses.get(ident)
            if draft:
                z = (min(self.g.floor + 1.15, self.g.ceiling - size[2] - .03)
                     if parent == "wall" else self.g.floor)
                offsets = ((0, 0), (.2, 0), (-.2, 0), (0, .2), (0, -.2)) if not self.local else \
                    ((0, 0), *((r * math.cos(k * math.pi / 4), r * math.sin(k * math.pi / 4))
                               for r in LOCAL_RINGS_M for k in range(8)))
                yaws = [draft["yaw_rad"]]
                if self.local and draft.get("alt_yaw_rad") is not None and \
                        abs(_wrap(draft["alt_yaw_rad"] - draft["yaw_rad"])) > 1e-3:
                    yaws.append(draft["alt_yaw_rad"])
                for dx, dy in offsets:
                    for yaw in yaws:
                        add(size, size_score, draft["x"] + dx, draft["y"] + dy, yaw, z, "llm")
        # Preserve spatial diversity, otherwise every top candidate can come from one wall.
        proposals.sort(key=lambda c: c.unary, reverse=True)
        selected, used = [], set()
        if self.local:  # the draft rings are finer than the diversity grid below: keep them all
            selected = [c for c in proposals if c.origin == "llm"]
            proposals = [c for c in proposals if c.origin != "llm"]
        for candidate in proposals:
            box = candidate.box
            x, y = box["bottom_center_m"][:2]
            key = (round(x / .35), round(y / .35), round(box["yaw_rad"] / .35),
                   tuple(box["target_size_local_m"]))
            if key in used:
                continue
            used.add(key)
            selected.append(candidate)
            if len(selected) >= 110:
                break
        self.cache[ident] = selected
        return selected

    def _on_parent(self, ident: str, placed: dict[str, dict]) -> list[Candidate]:
        parent_id = self.plan.parents[ident]
        parent = placed.get(parent_id, self.g.fixed.get(parent_id))
        if parent is None:
            return []
        px, py = parent["bottom_center_m"][:2]
        psx, psy = parent["target_size_local_m"][:2]
        yaw = parent["yaw_rad"]
        c, s = math.cos(yaw), math.sin(yaw)
        # (local x, local y, yaw relative to the parent, score); local mode first tries the child's drafted offset
        # on its parent, so a moved parent carries its child along
        spots = [(ux * psx, uy * psy, 0., 0.) for ux, uy in ((0, 0), (-.2, 0), (.2, 0), (0, -.2), (0, .2))]
        child, anchor = self.plan.draft_poses.get(ident), self.plan.draft_poses.get(parent_id)
        if anchor is None and parent_id in self.g.fixed:
            box = self.g.fixed[parent_id]
            anchor = {"x": box["bottom_center_m"][0], "y": box["bottom_center_m"][1], "yaw_rad": box["yaw_rad"]}
        if self.local:
            spots = [(u, v, 0., -.2) for u, v, _, _ in spots]
            if child and anchor:
                dx, dy = child["x"] - anchor["x"], child["y"] - anchor["y"]
                ca, sa = math.cos(anchor["yaw_rad"]), math.sin(anchor["yaw_rad"])
                spots.insert(0, (ca * dx + sa * dy, -sa * dx + ca * dy, _wrap(child["yaw_rad"] - anchor["yaw_rad"]), 0.))
        proposals = []
        for size, size_score, _source in self._sizes(ident):
            edge = []
            if self.g.robot_zones is not None:
                # a robot uses the room: also spots along each edge of the top, so an object the arm cannot reach
                # in the middle of a large table can move within reach of a side (EmbodiedGen's bfs_placement samples
                # the support's surface; robot.RobotZones ranks the spots by the arm's reach)
                du, dv = max(psx / 2 - size[0] / 2 - .05, 0.), max(psy / 2 - size[1] / 2 - .05, 0.)
                edge = [(u, v, 0., -.3) for u, v in ((du, 0.), (-du, 0.), (0., dv), (0., -dv),
                                                       (du, dv * .5), (du, -dv * .5), (-du, dv * .5), (-du, -dv * .5))
                        if (u, v) != (0., 0.)]
            for u, v, turn, score in spots + edge:
                x = px + c * u - s * v
                y = py + s * u + c * v
                draft = _pose(ident, size, x, y, 0, yaw + turn, parent_id)
                z = self.g._support_height(draft, parent_id, parent)
                box = _pose(ident, size, x, y, z, yaw + turn, parent_id)
                if not self.g.check_box(box, placed):
                    proposals.append(Candidate(box, size_score + score, "support"))
        return proposals

    def _relative(self, ident: str, placed: dict[str, dict]) -> list[Candidate]:
        relations = self.g.explicit + self.g.soft
        targets = [r["target_id"] for r in relations
                   if r.get("object_id") == ident and r.get("target_id") in placed | self.g.fixed]
        targets += [r["object_id"] for r in relations
                    if r.get("target_id") == ident and r.get("object_id") in placed | self.g.fixed]
        between = [r for r in relations if r.get("type") == "between" and
                   r.get("object_id") == ident and all(t in placed | self.g.fixed for t in r["target_ids"])]
        if (not targets and not between) or self.plan.parents[ident] != "floor":
            return []
        result = []
        for relation in between:
            a, b = [placed.get(t, self.g.fixed.get(t))["bottom_center_m"][:2]
                    for t in relation["target_ids"]]
            face_targets = [placed.get(r["target_id"], self.g.fixed.get(r["target_id"]))
                            for r in relations if r.get("type") == "faces" and
                            r.get("object_id") == ident and r.get("target_id") in placed | self.g.fixed]
            for size, size_score, _source in self._sizes(ident):
                for t in (.2, .35, .5, .65, .8):
                    x, y = a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t
                    angles = ([math.atan2(face_targets[0]["bottom_center_m"][1] - y,
                                         face_targets[0]["bottom_center_m"][0] - x)] if face_targets else
                              [0, math.pi / 2, math.pi, -math.pi / 2])
                    for yaw in angles:
                        box = _pose(ident, size, x, y, self.g.floor, yaw, "floor")
                        if not self.g.check_box(box, {}):
                            result.append(Candidate(box, self._unary(box, size_score, "relative"), "between"))
        for target_id in set(targets):
            target = placed.get(target_id, self.g.fixed.get(target_id))
            tx, ty = target["bottom_center_m"][:2]
            near_limits = [r["max_distance_m"] for r in relations if r.get("type") == "near" and
                           {r.get("object_id"), r.get("target_id")} == {ident, target_id}]
            if near_limits:
                max_gap = min(near_limits)
                gaps = (min(.02, max_gap * .3), min(.07, max_gap * .8))
            else:
                gaps = (.12, .35)
            for size, size_score, _source in self._sizes(ident):
                for angle in (k * math.pi / 4 for k in range(8)):
                    ux, uy = math.cos(angle), math.sin(angle)
                    # face the target; local mode also tries the target's own direction (a nightstand beside a bed)
                    for yaw in (angle + math.pi, target["yaw_rad"]) if self.local else (angle + math.pi,):
                        distance = (_corners_relative(size, yaw, ux, uy) +
                                    _corners_relative(target["target_size_local_m"],
                                                      target["yaw_rad"], ux, uy))
                        for gap in gaps:
                            x, y = tx + (distance + gap) * ux, ty + (distance + gap) * uy
                            box = _pose(ident, size, x, y, self.g.floor, yaw, "floor")
                            if not self.g.check_box(box, {}):
                                result.append(Candidate(box, self._unary(box, size_score, "relative"), "relative"))
        return result

    def _zone_filter(self, ident: str, placed: dict[str, dict], valid: list[Candidate], blocked: Counter) -> list:
        """With hard robot zones: drop the candidates that add a zone issue (by the object it concerns) when some
        candidate adds none; else keep them all (every object must be placed)."""
        zones = self.g.robot_zones
        if zones is None or not zones.hard or not valid:
            return valid
        base = {(i.code, i.ids[0]) for i in zones.issues(placed)}
        keep = [c for c in valid if not {(i.code, i.ids[0]) for i in zones.issues({**placed, ident: c.box})} - base]
        if keep and len(keep) < len(valid):
            blocked["robot_zone"] += len(valid) - len(keep)
            return keep
        return valid

    def candidates(self, ident: str, placed: dict[str, dict], limit: int) -> tuple[list[Candidate], Counter]:
        if self.plan.parents[ident] not in {"floor", "wall"}:
            candidates = self._on_parent(ident, placed)
            blocked = Counter({"support_geometry": 1} if not candidates else {})
            return self._zone_filter(ident, placed, candidates, blocked), blocked
        all_candidates = self._generic(ident) + self._relative(ident, placed)
        valid, blocked = [], Counter()
        shrink = (0, *getattr(self.catalog, "levels", (1, 2))) if self.local and \
            getattr(self.catalog, "allow_shrink", True) else (0,)
        for level in shrink:
            if level:  # local mode: shrink only when no full-size pose fits
                self.shrink[ident] = level
                self.cache.pop(ident, None)
                all_candidates = self._generic(ident) + self._relative(ident, placed)
                self.shrink.pop(ident)
                self.cache.pop(ident, None)
            for candidate in all_candidates:
                issues = self.g.new_hard_issues(candidate.box, placed)
                if not issues:
                    valid.append(candidate)
                else:
                    blocked.update({issue.code for issue in issues})
            if valid:
                if level:
                    blocked[f"shrunk_level_{level}"] += 1
                break
        valid = self._zone_filter(ident, placed, valid, blocked)
        valid.sort(key=lambda c: c.unary + 3 * self.g.soft_score({**placed, ident: c.box})[0], reverse=True)
        if not all_candidates:
            blocked["no_unary_valid_candidate"] += 1
        return valid[:limit], blocked


def _signature(placed: dict[str, dict], order: list[str]) -> tuple:
    return tuple((round(placed[ident]["bottom_center_m"][0] / .25),
                  round(placed[ident]["bottom_center_m"][1] / .25),
                  round(placed[ident]["yaw_rad"] / .4)) for ident in order)


def beam_search(geometry: Geometry, catalog: SizeCatalog, plan: Plan, *, mode: str,
                beam_width: int, candidate_limit: int, density: int,
                initial: dict[str, dict] | None = None, subset: set[str] | None = None,
                draft_weight: float = 0.) -> tuple[State | None, dict]:
    """A failed search leaves its best partial assignment in ``trace["partial"]`` (the last nonempty beam)."""
    initial = initial or {}
    subset = subset or (set(geometry.requests) - set(initial))
    order = _order(geometry, catalog, plan, subset)
    generator = CandidateGenerator(geometry, catalog, plan, mode, density, draft_weight)
    beam = [State(dict(initial), 0.0, 0.0)]
    trace = {"mode": mode, "order": order, "beam_width": beam_width, "layers": [], "blocked": {}}
    for ident in order:
        pool, blocked = [], Counter()
        for state in beam:
            candidates, causes = generator.candidates(ident, state.placed, candidate_limit)
            blocked.update(causes)
            if not candidates:
                blocked["no_valid_candidate"] += 1
            for candidate in candidates:
                placed = {**state.placed, ident: candidate.box}
                unary = state.unary + candidate.unary
                soft, _ = geometry.soft_score(placed)
                pool.append(State(placed, unary, unary + 3 * soft))
        pool.sort(key=lambda s: s.rank, reverse=True)
        next_beam, seen = [], set()
        for state in pool:
            signature = _signature(state.placed, order[:len(state.placed) - len(initial)])
            if signature in seen:
                continue
            seen.add(signature)
            next_beam.append(state)
            if len(next_beam) >= beam_width:
                break
        trace["layers"].append({"id": ident, "states": len(next_beam)})
        if not next_beam:
            trace["blocked"] = dict(blocked)
            trace["partial"] = beam[0].placed
            return None, trace
        beam = next_beam
    complete = [state for state in beam if not geometry.validate(state.placed)]
    if not complete:
        trace["partial"] = max(beam, key=lambda s: s.rank).placed
    # every complete state of the last beam, best rank first (the critic loop picks among them by its own score)
    trace["complete"] = [s.placed for s in sorted(complete, key=lambda s: s.rank, reverse=True)]
    return (max(complete, key=lambda s: s.rank) if complete else None), trace
