"""The program check-and-repair loop (steps ④-④c of ``simple``): every proposal is the program's, at fixed sizes.

Each attempt takes the worst issue not tried yet (hard violations by magnitude while any are left, then soft costs of
at least ``min_soft_cost``) and builds a repair scope (flowchart §2 B):
  level 0  the issue's objects, their collision neighbours and their relation neighbours (step ①'s relations and the
           input's constraints, one hop), with everything standing on them
  level 1  + the functional groups they belong to
  level 2  + every object (hard issues only)
Fixed objects never move; objects outside the scope are obstacles, and every candidate is checked on the whole room.

Proposals (flowchart §2 C):
  program  the solver's local beam search over the scope at fixed sizes (no shrinking), near the current poses; a
           template issue (A far from / not facing B) enters the search as a temporary soft relation so the search
           proposes positions beside B. The critic picks among the complete states of the last beam.
  relocate (``relocate=True``) when the local proposals fail on a robot check (robot_reach, robot_openable,
           robot_door): each blocker the check names (up to ``RELOCATE_MOVERS``), then the furniture the robot cannot
           reach, is searched for over the whole room with what stands on it, its distance from where it stands
           nearly free (``RELOCATE_DRAFT_WEIGHT``): a sofa across four cabinets goes to another wall instead of
           shifting 0.2 m (MANSION and EmbodiedSmith re-place the blocking object rather than nudging it).
Every candidate goes through ``Critic.better`` (flowchart §2 D, §3): accepted, or rolled back with the reason kept.
The loop stops when nothing is left to try: every issue (code and objects) is taken up once, worst first, and an
issue that appears after a change is taken up too. Work, not time, decides how far the repair goes (as SceneSmith's
critique rounds and solver iteration limits, LayoutVLM's fixed iterations): every attempt is bounded by the beam
(``beam_width``, ``candidate_limit``), and the attempts are bounded by ``max_attempts`` (default: 24 plus three per
issue found at the start). ``stale_limit`` (default off) stops after that many attempts in a row without an accepted
change. There is no time limit by default: how long a room takes never decides how well it is repaired.
``search_seconds``, when given (``repair_seconds``: 180 s per pass in the paper's runs), stops the loop once the program search has
used that long; the loop then ends with termination "time limit", so a result cut short by time is never silent.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass

from . import llm as llm_client
from .catalog import SizeCatalog
from .critic import WEIGHTS, Assessment, Critic, Violation  # WEIGHTS' keys are the soft codes
from .drafting import WALL_BOTTOM_M
from .geometry import Geometry
from .planning import Plan, _number
from .solver import _pose, _wrap, beam_search

DRAFT_WEIGHT = 2.0  # per metre away from the draft; one violated soft relation costs 3 in the beam rank


class FixedSizes:
    """Catalog interface for the solver with one size per object: no step resizes an object, so ``shrunk`` offers
    none (``allow_shrink`` False: the solver's local mode tries no shrink level)."""
    allow_shrink = False

    def __init__(self, sizes: dict[str, list[float]]):
        self.sizes = sizes

    def candidates(self, request: dict, llm_hint=None) -> list[tuple[list[float], float, str]]:
        return [(self.sizes[request["id"]], 0., "fixed")]

    def shrunk(self, request: dict, level: int) -> list[tuple[list[float], float, str]]:
        return []


def _with_children(plan: Plan, ids: set[str]) -> set[str]:
    result, changed = set(ids), True
    while changed:
        before = len(result)
        result |= {ident for ident, parent in plan.parents.items() if parent in result}
        changed = len(result) != before
    return result

CANDIDATES_ASSESSED = 8  # complete beam states the critic scores per proposal
# the beam's cost per metre (and quarter turn) away from the current pose: hard issues keep step ④'s weight (stay
# near the draft); a soft issue usually needs a longer move (a member 2 m from its group) than one soft relation (3
# in the beam rank) pays for at that weight. The critic still decides whether the move is accepted.
SOFT_DRAFT_WEIGHT = .5
# relocation (``Loop(relocate=True)``): a robot check's blocker, or the furniture it cannot reach, is searched for
# over the whole room (the search's wall and grid candidates), its distance from where it stands nearly free
RELOCATE_DRAFT_WEIGHT = .1
RELOCATE_MOVERS = 3          # blockers tried per robot issue (then the furniture itself)
RELOCATABLE = {"robot_reach", "robot_openable", "robot_door"}


@dataclass
class LoopBudget:
    max_attempts: int | None = None      # issues taken up; None: 24 + 3 per issue found at the start
    stale_limit: int | None = None       # attempts in a row without an accepted change; None: no early stop
    search_seconds: float | None = None  # program search time limit; None (default): none, the attempts decide
    min_soft_cost: float = .05           # soft violations below this weighted cost are left alone


def _describe(v: Violation) -> str:
    unit = {"template_faces": "of facing error (normalised)", "relation_plan": "(normalised gap)",
            "relation_user": "(normalised gap)", "access": "(missing free share, normalised)",
            "wall_back": "(back-to-wall gap beyond 0.15 m plus angle off, normalised)",
            "reach": "m from the floor a person can reach from a door"}
    return f"{v.code} {list(v.ids)}: {v.magnitude:.3f}{' m' if v.code in HARD_M else ' ' + unit.get(v.code, '')}" + \
        (f" ({v.detail})" if v.detail else "")


HARD_M = {"collision", "boundary", "vertical_bounds", "floor_support", "wall_support", "object_support",
          "door_clearance", "window_clearance", "keepout", "template_near"}


class Loop:
    def __init__(self, geometry: Geometry, plan: Plan, sizes: dict[str, list[float]], catalog: SizeCatalog,
                 critic: Critic, *, locked_parents: dict[str, str], budget: LoopBudget | None = None,
                 beam_width: int = 24, candidate_limit: int = 48, draft_weight: float = DRAFT_WEIGHT,
                 relocate: bool = False, rounds: int = 1, first: frozenset = frozenset()):
        """Every object keeps its size: the search moves and turns objects, never resizes one. ``first``: violations
        holding one of these ids
        (the task's required objects) are taken up before the others of their tier.
        ``relocate``: robot issues the local search cannot fix try moving their blocker anywhere in the room.
        ``rounds``: passes over the issues; a new pass starts only when the last one accepted a change and hard
        violations are left (every issue is taken up once per pass)."""
        self.g, self.plan, self.sizes, self.catalog, self.critic = geometry, plan, sizes, catalog, critic
        self.locked = locked_parents
        self.budget = budget or LoopBudget()
        self.beam_width, self.candidate_limit, self.draft_weight = beam_width, candidate_limit, draft_weight
        self.fixed_sizes = FixedSizes(sizes)
        self.relocate, self.rounds = relocate, max(1, rounds)
        self.first = frozenset(first)
        self.search_s = 0.

    # ---------------------------------------------------------------- scope
    def scope(self, ids: tuple[str, ...], level: int, current: Assessment) -> set[str]:
        requests = set(self.g.requests)
        base = {i for i in ids if i in requests}
        result = set(base)
        for v in current.hard:
            if v.code == "collision" and set(v.ids) & base:
                result |= set(v.ids) & requests
        for r in self.plan.relations + self.g.explicit:
            linked = {r.get("object_id"), r.get("target_id"), r.get("parent_id"), *(r.get("target_ids") or [])}
            if linked & base:
                result |= linked & requests
        if level >= 1:
            for group in self.plan.groups:
                members = {group["anchor"], *group["members"]} & requests
                if members & result:
                    result |= members
        if level >= 2:
            result = set(requests)
        return _with_children(self.plan, result)

    # ---------------------------------------------------------------- program proposal
    def _extra_relations(self, target: Violation, placed: dict[str, dict] | None = None) -> list[dict]:
        if target.code == "template_near":
            return [{"type": "near", "object_id": target.ids[0], "target_id": target.ids[1], "max_distance_m": .25,
                     "hard": False, "origin": "critic"}]
        if target.code == "template_faces":
            return [{"type": "faces", "object_id": target.ids[0], "target_id": target.ids[1], "hard": False,
                     "tolerance_rad": math.radians(40), "origin": "critic"}]
        if target.code == "wall_back" and placed is not None:
            direction = self.critic.wall_direction(placed[target.ids[0]])
            if direction:
                return [{"type": "against_wall", "object_id": target.ids[0], "tolerance_m": .1, "hard": False,
                         "origin": "critic"},
                        {"type": "faces_direction", "object_id": target.ids[0], "direction_xy": list(direction),
                         "tolerance_rad": math.radians(8), "hard": False, "origin": "critic"}]
        return []

    def program(self, scope: set[str], target: Violation, placed: dict[str, dict],
                draft_weight: float | None = None) -> list[dict[str, dict]]:
        geometry = self.g
        extra = self._extra_relations(target, placed)
        if extra:
            geometry = copy.copy(self.g)
            geometry.soft = self.g.soft + extra
        saved = self.plan.draft_poses
        self.plan.draft_poses = {i: {"x": b["bottom_center_m"][0], "y": b["bottom_center_m"][1], "yaw_rad": b["yaw_rad"]}
                                 for i, b in placed.items()}
        start = time.perf_counter()
        try:
            frozen = {i: b for i, b in placed.items() if i not in scope}
            _state, trace = beam_search(geometry, self.fixed_sizes, self.plan, mode="local", beam_width=self.beam_width,
                                        candidate_limit=self.candidate_limit, density=1, initial=frozen, subset=scope,
                                        draft_weight=draft_weight if draft_weight is not None else
                                        SOFT_DRAFT_WEIGHT if target.code in WEIGHTS else self.draft_weight)
        finally:
            self.plan.draft_poses = saved
            self.search_s += time.perf_counter() - start
        candidates = list(trace.get("complete") or [])[:CANDIDATES_ASSESSED]
        if not candidates and trace.get("partial"):
            candidates = [{**placed, **trace["partial"]}]
        return candidates

    # ---------------------------------------------------------------- the review's fix request (semantic.py)
    def _payload(self, scope: set[str], placed: dict[str, dict]) -> str:
        objects = []
        for obj in self.g.condition["objects"]:
            ident, box = obj["id"], placed[obj["id"]]
            depth, width, height = box["target_size_local_m"]
            objects.append({"id": ident, "category": obj["category"], "description": obj.get("description", ""),
                            "support": self.plan.parents[ident],
                            "size_m": {"width": round(width, 3), "depth": round(depth, 3), "height": round(height, 3)},
                            "x": round(box["bottom_center_m"][0], 3), "y": round(box["bottom_center_m"][1], 3),
                            "z": round(box["bottom_center_m"][2], 3),
                            "yaw_deg": round(math.degrees(box["yaw_rad"]), 1), "movable": ident in scope,
                            "support_changeable": ident in scope and ident not in self.locked})
        room = {k: v for k, v in self.g.room.items() if k in ("floor_polygon_xy_m", "floor_z_m", "height_m", "openings",
                                                               "fixed_objects")}
        return json.dumps({"room": room, "objects": objects, "constraints": self.g.condition["constraints"]},
                          ensure_ascii=False)

    def _apply(self, answer: dict, scope: set[str], placed: dict[str, dict]) -> tuple[dict[str, dict], dict[str, str]]:
        """New boxes from the answer: poses for scope objects, supports re-picked where allowed, children carried."""
        parents = dict(self.plan.parents)
        poses: dict[str, dict] = {}
        entries = answer.get("objects")
        if isinstance(entries, list):  # [{"id": ..., "x": ...}] instead of the map asked for
            entries = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
        for ident, entry in (entries if isinstance(entries, dict) else {}).items():
            if ident not in scope or not isinstance(entry, dict):
                continue
            x, y, yaw = (_number(entry.get(k), -1e4, 1e4) for k in ("x", "y", "yaw_deg"))
            if None in (x, y, yaw):
                continue
            poses[ident] = {"x": x, "y": y, "yaw_rad": math.remainder(math.radians(yaw), 2 * math.pi),
                            "z": _number(entry.get("z"), -1e3, 1e3)}
            support = entry.get("support")
            if isinstance(support, str) and ident not in self.locked and support != ident and \
                    (support in ("floor", "wall") or support in self.g.requests or support in self.g.fixed):
                trial = {**parents, ident: support}
                seen, cur = {ident}, support
                while cur in trial and cur not in ("floor", "wall"):
                    if cur in seen:
                        break
                    seen.add(cur)
                    cur = trial[cur]
                else:
                    parents[ident] = support
        new = dict(placed)
        order, done = [], set()

        def visit(ident: str) -> None:
            if ident in done:
                return
            parent = parents.get(ident)
            if parent in self.g.requests:
                visit(parent)
            done.add(ident)
            order.append(ident)

        # children placed so far move along (in step-by-step placement, objects still to come are not in ``placed``)
        moved = _with_children(Plan(parents), set(poses)) & set(placed)
        for ident in moved:
            visit(ident)
        for ident in order:
            if ident not in moved:
                continue
            parent, old, size = parents[ident], placed[ident], self.sizes[ident]
            pose = poses.get(ident)
            if pose is None:  # carried by a moved parent: same offset and turn in the parent's frame
                anchor_old, anchor_new = placed.get(parent), new.get(parent)
                if anchor_old is None or anchor_new is None:
                    continue
                turn = anchor_new["yaw_rad"] - anchor_old["yaw_rad"]
                dx = old["bottom_center_m"][0] - anchor_old["bottom_center_m"][0]
                dy = old["bottom_center_m"][1] - anchor_old["bottom_center_m"][1]
                c, s = math.cos(turn), math.sin(turn)
                pose = {"x": anchor_new["bottom_center_m"][0] + c * dx - s * dy,
                        "y": anchor_new["bottom_center_m"][1] + s * dx + c * dy,
                        "yaw_rad": _wrap(old["yaw_rad"] + turn), "z": None}
            if parent == "floor":
                pose = dict(pose)
                pose["x"], pose["y"] = self.g.push_inside(size, pose["x"], pose["y"], pose["yaw_rad"])
                z = self.g.floor
            elif parent == "wall":
                pose = dict(pose)
                pose["x"], pose["y"], pose["yaw_rad"] = self.g.snap_to_wall(size, pose["x"], pose["y"], pose["yaw_rad"])
                top = self.g.ceiling - size[2] - .03
                given = pose.get("z")
                z = given if given is not None and self.g.floor <= given <= top else \
                    (old["bottom_center_m"][2] if self.plan.parents[ident] == "wall" else min(self.g.floor + WALL_BOTTOM_M, top))
            else:
                parent_box = new.get(parent, self.g.fixed.get(parent))
                if parent_box is None:
                    continue
                z = self.g._support_height(_pose(ident, size, pose["x"], pose["y"], 0, pose["yaw_rad"], parent),
                                           parent, parent_box)
            new[ident] = _pose(ident, size, pose["x"], pose["y"], z, pose["yaw_rad"], parent)
        return new, parents

    def movers(self, target: Violation) -> list[str]:
        """Floor objects to re-place for a robot issue: its blockers (up to RELOCATE_MOVERS), then its own object."""
        if target.code not in RELOCATABLE:
            return []
        ids = [i for i in target.ids if i in self.g.requests]
        own = [] if target.code == "robot_door" else ids[:1]
        blockers = ids if target.code == "robot_door" else ids[1:]
        out = []
        for ident in blockers[:RELOCATE_MOVERS] + own:
            if self.plan.parents.get(ident) == "floor" and ident not in out:
                out.append(ident)
        return out

    # ---------------------------------------------------------------- loop
    def _targets(self, current: Assessment, tried: set) -> list[Violation]:
        # legality before the robot checks (``Critic.better``'s tiers), the task's (``first``) then the largest first
        pool = sorted(current.hard, key=lambda v: (v.code.startswith("robot_"), not set(v.ids) & self.first,
                                                   -v.magnitude)) if current.hard else \
            sorted((v for v in current.soft if v.cost >= self.budget.min_soft_cost), key=lambda v: -v.cost)
        return [v for v in pool if (v.code, v.ids) not in tried]

    def _pick(self, candidates: list[dict[str, dict]], critic: Critic) -> tuple[dict[str, dict] | None, Assessment | None]:
        scored = [(critic.assess(c), c) for c in candidates]
        if not scored:
            return None, None
        best = min(scored, key=lambda t: t[0].key())
        return best[1], best[0]

    def max_attempts(self, current: Assessment) -> int:
        if self.budget.max_attempts is not None:
            return self.budget.max_attempts
        return 24 + 3 * len(self._targets(current, set()))

    def run(self, placed: dict[str, dict]) -> tuple[dict[str, dict], dict]:
        start = time.perf_counter()
        current = self.critic.assess(placed)
        max_attempts, limit = self.max_attempts(current), self.budget.search_seconds
        info = {"initial": current.summary(), "attempts": [], "termination": "max_attempts",
                "max_attempts": max_attempts, "time_limit_s": limit}
        tried, stale, passes, pass_accepted = set(), 0, 1, False
        for _ in range(max_attempts * self.rounds):
            targets = self._targets(current, tried)
            if not targets and passes < self.rounds and pass_accepted and current.hard:
                tried, passes, pass_accepted = set(), passes + 1, False  # another pass over what is left
                targets = self._targets(current, tried)
            if not targets:
                info["termination"] = "no issue left to try"
                break
            if limit is not None and self.search_s >= limit:
                info["termination"] = "time limit"
                break
            target = targets[0]
            tried.add((target.code, target.ids))
            accepted, seen = False, []
            levels = (0, 1, 2) if current.hard else (0, 1)
            for level in levels:
                scope = self.scope(target.ids, level, current)
                if scope in seen or not scope:
                    continue
                seen.append(scope)
                t0 = time.perf_counter()
                candidate, after = self._pick(self.program(scope, target, placed), self.critic)
                ok, reason = self.critic.better(after, current) if after else (False, "no candidate")
                info["attempts"].append({"target": target.as_dict(), "level": level, "scope": len(scope),
                                         "source": "program", "accepted": ok, "reason": reason,
                                         "before": current.key(), "after": after.key() if after else None,
                                         "seconds": round(time.perf_counter() - t0, 3)})
                if ok:
                    placed, current, accepted = candidate, after, True
                    break
            if not accepted and self.relocate:
                for mover in self.movers(target):
                    if limit is not None and self.search_s >= limit:
                        break
                    scope = _with_children(self.plan, {mover})
                    t0 = time.perf_counter()
                    candidate, after = self._pick(self.program(scope, target, placed, RELOCATE_DRAFT_WEIGHT), self.critic)
                    ok, reason = self.critic.better(after, current) if after else (False, "no candidate")
                    info["attempts"].append({"target": target.as_dict(), "level": "relocate", "scope": len(scope),
                                             "source": "relocate", "mover": mover, "accepted": ok, "reason": reason,
                                             "before": current.key(), "after": after.key() if after else None,
                                             "seconds": round(time.perf_counter() - t0, 3)})
                    if ok:
                        placed, current, accepted = candidate, after, True
                        break
            pass_accepted = pass_accepted or accepted
            stale = 0 if accepted else stale + 1
            if self.budget.stale_limit is not None and stale >= self.budget.stale_limit:
                info["termination"] = "no improvement"
                break
        info.update(final=current.summary(), passes=passes, search_seconds=round(self.search_s, 3),
                    seconds=round(time.perf_counter() - start, 3))
        return placed, info


class FocusedLoop(Loop):
    """The loop restricted to ``movable`` objects (the review's repair of a fix): only issues involving them are
    taken up, only they move; everything else is frozen."""

    def __init__(self, *args, movable: set[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.movable = set(movable)

    def _targets(self, current: Assessment, tried: set) -> list[Violation]:
        mine = [v for v in current.hard if set(v.ids) & self.movable and (v.code, v.ids) not in tried]
        if mine:
            return sorted(mine, key=lambda v: (v.code.startswith("robot_"), -v.magnitude))
        return sorted((v for v in current.soft if set(v.ids) & self.movable and (v.code, v.ids) not in tried
                       and v.cost >= self.budget.min_soft_cost), key=lambda v: -v.cost)

    def scope(self, ids: tuple[str, ...], level: int, current: Assessment) -> set[str]:
        return super().scope(ids, level, current) & self.movable & set(self.g.requests)


def make_ask(model: str | None, base_url: str | None, api_key: str | None):
    """An LLM request function (the review's, ``semantic.SemanticPass``) through the workflow's client."""
    def ask(instructions: str, prompt: str, *, images: list[str] | None = None):
        return llm_client.ask(instructions, prompt, model=model, base_url=base_url, api_key=api_key, images=images,
                              **({"reasoning_effort": "xhigh"} if images else {}))
    ask.supports_images = True
    return ask
