"""Step ⑤ of ``simple``: the LLM semantic review. The program checks legality and usability, the LLM checks whether
the arrangement makes sense, and the LLM only ever proposes.

One pass over a set of objects (``SemanticPass.run``):
  review    one request with the room, the user's requirements (each object's description, the input's
            constraints; the plan's groups and relations marked as a planning model's suggestions), every object
            with its status (review, placed as context, or pending: still to be placed) and the program's soft
            findings on the reviewed objects as evidence, each labelled with its kind (usability, user requirement,
            planner suggestion, or typical in training rooms); the answer lists semantic issues, each with a severity
            (severe / moderate / minor). Only severe issues are acted on; the others are recorded.
  propose   per severe issue (at most ``max_fixes``), one request for new poses of the issue's movable objects
            (children move with their parent). The candidate never replaces the current layout directly.
  repair    the program's search fixes the hard violations the proposal made, moving only those objects
            (``FocusedLoop`` restricted to hard issues; soft costs are not chased, so the search does not undo the
            proposal).
  accept    ``admissible``: hard-valid if the current layout is, otherwise no more and no larger hard violations;
            reach, access and door costs no worse; the soft total up by at most ``soft_tolerance``. Then a blind
            pairwise comparison of the changed objects (with the same requirements and pending objects, without the
            reviewer's problem text), asked twice with the two layouts swapped: accepted only if the candidate wins
            both times; a split or "same" keeps the current layout.
A second round re-reviews an accepted change, or retries a task edit with its failure reason. Committed edits pass
the program's nonregression gates. Image-capable clients see bbox diagrams before RoomGen, not meshes.
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy

from .catalog import SizeCatalog
from .credentials import FatalAPIError
from .critic import EPS_H, EPS_S, Assessment, Critic
from .geometry import Geometry
from .loop import FocusedLoop, Loop, LoopBudget, _describe, _with_children
from .planning import Plan
from .visual import layout_image

SEVERITIES = ("severe", "moderate", "minor")
GATED = ("reach", "access", "door")  # usability costs a semantic change may not worsen
SOFT_TOLERANCE = .1                  # how much the soft total may rise for a change the reviewer prefers

STAGE_NOTES = {
    "floor": "Only the floor-standing furniture is placed so far. The objects in \"pending\" (items on furniture and "
             "on walls) are placed afterwards to fit this layout: do not report them as missing, but judge the "
             "furniture with them in mind (e.g. seating and the wall TV still to come).",
    "room": "Every object is placed. Review the whole room.",
    "small": "Review the objects with status \"review\" (items on furniture and on walls). Those with status "
             "\"placed\" are context and were already reviewed; report a problem with them only if a reviewed item "
             "makes it one.",
}

# how the reviewer should weigh each kind of program evidence
EVIDENCE_KIND = {"reach": "usability", "access": "usability", "door": "usability", "window": "usability",
                 "relation_user": "user requirement (soft)", "relation_plan": "planning-model suggestion",
                 "wall_back": "typical in training rooms; exceptions are fine",
                 "template_near": "typical in training rooms; exceptions are fine",
                 "template_faces": "typical in training rooms; exceptions are fine"}

REQUIREMENT_RULES = """What the user asked for comes first. Each object's "description" and the input's \
"constraints" ("requirements.constraints"; hard ones must hold, soft ones are wishes) are the user's. \
"requirements.plan" (functional groups and relations) was suggested by a planning model: it is not a user \
requirement. Use common sense only where the descriptions and constraints say nothing: a description that gives an \
object a particular role (a bookcase that divides the room, a desk facing the window) makes that arrangement right, \
not a problem. Objects with status "pending" are not placed yet (their size and intended support are given); they \
will be placed afterwards to fit the layout."""

REQUIREMENT_RULES += """ When task_context is present (under requirements, or at the top level for a fix), use \
the original task and each required/dependency object's reason to judge task usability. initial_relations describe \
the starting scene; goals are future outcomes for the robot, never placement instructions to complete now. Keep \
the required objects, fixed sizes and locked supports. Images, when provided, are annotated bbox schematics with \
IDs and front arrows, not mesh renders: do not infer grasp success, exact accessibility, or IK from them. \
previous_edit_failures report rejected edits: use their concrete reasons to avoid repeating the same failed move."""

REVIEW_INSTRUCTIONS = """You review an indoor furniture layout for whether it makes sense. The user message is JSON: \
the room (floor polygon in metres, world x/y, z up; openings; fixed objects), the user's "requirements", every \
object with its category, description, support, size (width = left-right as seen from its front, depth = \
front-back, height) and status (review, placed, pending); placed ones also with bottom-centre position, yaw_deg (the \
direction its front faces, counter-clockwise degrees from world +x) and front_xy (the same direction as a unit \
vector); a "note" on what to review; and "evidence": soft findings of a program check, each with its kind. The \
evidence is measurement, not a verdict: "usability" findings are real obstacles to using an object, findings \
"typical in training rooms" only say what most rooms do.

""" + REQUIREMENT_RULES + """

Collisions, objects outside the room and blocked doors are checked by a program: do not report them. Report \
problems of meaning and use only, for example: a seat turned away from the table or desk it belongs to, a TV or \
screen not facing the seating, a bed or wardrobe standing free in the room or with its front to a wall without a \
reason in its description, things that belong together far apart, an item on a support where it makes no sense, \
a layout that blocks how another object is used, a layout that goes against a description or constraint.

Answer with one JSON object only, no prose:
{"issues": [{"ids": ["<id>", ...], "severity": "severe" | "moderate" | "minor", "problem": "...", "fix": "..."}]}
severe: the arrangement breaks how the room is used, goes against what the user asked for, or looks clearly wrong \
to anyone; moderate: noticeably odd; minor: a matter of taste. ids: placed objects that would have to move, the one \
to move first. An empty list if nothing is wrong. At most 6 issues."""

FIX_INSTRUCTIONS = """You fix one problem a reviewer found in how part of a furniture layout is arranged. The user \
message is JSON: the room (floor polygon in metres, world x/y, z up; openings; fixed objects that cannot move), every \
object with its category, description, fixed size (width = left-right as seen from its front, depth = front-back, \
height), current pose and support, the objects you may move ("movable"), objects still to be placed ("pending"), \
the user's constraints and the planning model's suggestions ("plan"), and the problem with a suggested fix. A \
program resolves small overlaps afterwards, and the change is kept only if a second reviewer prefers it.

""" + REQUIREMENT_RULES + """ The reviewer can be wrong: do not break a description or constraint to follow it.

"""
FIX_INSTRUCTIONS += """Answer with one JSON object only, no prose:
{"objects": {"<id>": {"x": x, "y": y, "yaw_deg": a}}}
Only movable objects may appear; leave out those that should stay. x, y: bottom-centre in world metres; yaw_deg: the direction the object's front faces, counter-clockwise degrees from world +x. An object whose "support_changeable" is true may also get "support": "floor", "wall" or another object's id when its current support cannot hold it; objects on a moved parent move with it unless you give them a pose. Do not change sizes.

Fix the listed problems without creating new ones: stay inside the floor polygon, off other objects and fixed objects, keep door strips clear, keep the input's constraints. Prefer small moves; keep related furniture together (seats with their table, nightstands beside the bed, etc.)."""

PAIRWISE_INSTRUCTIONS = """You compare two arrangements of the same room. The user message is JSON: the room, the \
user's "requirements", the objects that are the same in both ("unchanged"), objects still to be placed ("pending"), \
and the objects that differ, as placed in "layout_1" and in "layout_2" (each with category, description, support, \
bottom-centre position in metres, yaw_deg = the direction the front faces, counter-clockwise degrees from world +x, \
front_xy the same as a unit vector; width = left-right as seen from the front, depth = front-back). Judge which \
arrangement better fits what the user asked for, makes more sense for how the room is used, and looks more natural.

""" + REQUIREMENT_RULES + """

Answer with one JSON object only, no prose:
{"better": "1" | "2" | "same", "reason": "..."}"""


def _json_object(text: str) -> dict | None:
    first, last = text.find("{"), text.rfind("}")
    try:
        answer = json.loads(text[first:last + 1]) if 0 <= first < last else None
    except json.JSONDecodeError:
        answer = None
    return answer if isinstance(answer, dict) else None


def _cost(a: Assessment, code: str) -> float:
    return sum(v.cost for v in a.soft if v.code == code)


def admissible(candidate: Assessment, current: Assessment, soft_tolerance: float = SOFT_TOLERANCE) -> tuple[bool, str]:
    """The program half of accepting a semantic change (the reviewer's preference is the other half)."""
    for name, a, b in (("hard", candidate.legality, current.legality), ("robot", candidate.robot, current.robot)):
        if b == [] and a:
            return False, f"new {name} violation on a {name}-valid layout"
        if len(a) > len(b) or sum(v.magnitude for v in a) > sum(v.magnitude for v in b) + EPS_H:
            return False, f"{name} violations not kept down"
    for code in GATED:
        before, after = _cost(current, code), _cost(candidate, code)
        if after > before + EPS_S:
            return False, f"{code} worse ({before:.3f} -> {after:.3f})"
    if candidate.soft_total > current.soft_total + soft_tolerance:
        return False, f"soft cost up {candidate.soft_total - current.soft_total:.3f} (tolerance {soft_tolerance})"
    return True, "program checks pass"


class SemanticPass:
    """One stage's review over ``geometry`` (the room as placed so far) with ``critic``; ``ask(instructions, prompt)``
    returns (text, call) like ``llm.ask``. A callback declaring ``supports_images`` also accepts ``images=``.
    ``task_context`` is review context, separate from the canonical FastFill condition and its initial constraints.
    """

    def __init__(self, geometry: Geometry, critic: Critic, plan: Plan, sizes: dict[str, list[float]],
                 catalog: SizeCatalog, *, locked_parents: dict[str, str], ask, stage: str, condition: dict | None = None,
                 soft_tolerance: float = SOFT_TOLERANCE, repair_seconds: float | None = None,
                 beam_width: int = 24, candidate_limit: int = 48, task_context: dict | None = None):
        self.g, self.critic, self.plan, self.sizes, self.catalog = geometry, critic, plan, sizes, catalog
        self.locked, self.ask, self.stage = locked_parents, ask, stage
        # the whole request (the geometry of stage one holds only the placed objects): descriptions, pending objects
        self.condition = condition or geometry.condition
        self.task_context = deepcopy(task_context)
        self.objects = {o["id"]: o for o in self.condition["objects"]}
        self.soft_tolerance, self.repair_seconds = soft_tolerance, repair_seconds
        self.beam_width, self.candidate_limit = beam_width, candidate_limit
        self.loop = Loop(geometry, plan, sizes, catalog, critic, locked_parents=locked_parents,
                         beam_width=beam_width, candidate_limit=candidate_limit)  # payload and answer parsing
        self.calls: list[dict] = []
        self.search_s = 0.
        self.failures: list[dict] = []

    # ---------------------------------------------------------------- requests
    def _images(self, *layouts: dict[str, dict]) -> list[str]:
        return [layout_image(self._room(), layout) for layout in layouts] if \
            getattr(self.ask, "supports_images", False) else []

    def _ask(self, instructions: str, payload: dict, images: list[str] | None = None) -> dict | None:
        images = images or []
        if images:
            payload = {**payload, "visual_observation": {
                "scope": "annotated bbox schematic before RoomGen; not a mesh render or robot execution proof",
                "image_order": ["layout_1", "layout_2"] if len(images) == 2 else ["current_layout"]}}
        try:
            text, call = self.ask(instructions, json.dumps(payload, ensure_ascii=False),
                                  **({"images": images} if images else {}))
        except (RuntimeError, ValueError) as exc:
            self.calls.append({"error": f"{type(exc).__name__}: {exc}"[:300], "image_count": len(images)})
            return None
        self.calls.append({**call, "image_count": len(images)})
        return _json_object(text)

    def _room(self) -> dict:
        return {k: v for k, v in self.g.room.items() if k in ("floor_polygon_xy_m", "floor_z_m", "height_m", "openings",
                                                               "fixed_objects")}

    def requirements(self) -> dict:
        """The input's constraints (hard or soft) and the plan's groups and relations, marked as suggestions."""
        relations = [{k: v for k, v in r.items() if k not in ("origin", "hard")}
                     for r in self.plan.relations if r.get("origin") != "group"]
        return {"constraints": [{**r, "hard": r.get("hard", True)} for r in self.condition["constraints"]],
                "plan": {"note": "suggestions of a planning model, not user requirements",
                         "groups": self.plan.groups, "relations": relations},
                **({"task_context": deepcopy(self.task_context)} if self.task_context is not None else {})}

    def _entry(self, ident: str, parents: dict[str, str] | None = None) -> dict:
        obj = self.objects[ident]
        depth, width, height = self.sizes[ident]
        return {"id": ident, "category": obj["category"], "description": obj.get("description", ""),
                "support": (parents or self.plan.parents).get(ident),
                "size_m": {"width": round(width, 3), "depth": round(depth, 3), "height": round(height, 3)}}

    def _view(self, placed: dict[str, dict], ident: str, parents: dict[str, str] | None = None) -> dict:
        box = placed[ident]
        yaw = box["yaw_rad"]
        return {**self._entry(ident, parents),
                "x": round(box["bottom_center_m"][0], 3), "y": round(box["bottom_center_m"][1], 3),
                "z": round(box["bottom_center_m"][2], 3), "yaw_deg": round(math.degrees(yaw), 1),
                "front_xy": [round(math.cos(yaw), 3), round(math.sin(yaw), 3)]}

    def pending(self, placed: dict[str, dict]) -> list[dict]:
        return [{**self._entry(i), "status": "pending"} for i in self.objects if i not in placed]

    def review(self, placed: dict[str, dict], review_ids: set[str], current: Assessment) -> list[dict] | None:
        objects = [{**self._view(placed, i), "status": "review" if i in review_ids else "placed"}
                   for i in self.objects if i in placed]
        evidence = [{"kind": EVIDENCE_KIND.get(v.code, "measurement"), "finding": _describe(v)}
                    for v in current.soft if set(v.ids) & review_ids]
        answer = self._ask(REVIEW_INSTRUCTIONS, {"room": self._room(), "requirements": self.requirements(),
                                                 "objects": objects + self.pending(placed),
                                                 "note": STAGE_NOTES[self.stage], "evidence": evidence[:30],
                                                 **({"previous_edit_failures": deepcopy(self.failures)}
                                                    if self.failures else {})}, self._images(placed))
        if answer is None:
            return None
        issues, entries = [], answer.get("issues")
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            ids = entry.get("ids")
            ids = [ids] if isinstance(ids, str) else ids if isinstance(ids, list) else []
            ids = [i for i in ids if isinstance(i, str) and i in placed]
            severity = str(entry.get("severity", "")).strip().lower()
            if ids:
                issues.append({"ids": ids, "severity": severity if severity in SEVERITIES else "minor",
                               "problem": str(entry.get("problem", ""))[:400], "fix": str(entry.get("fix", ""))[:400]})
        return issues

    def propose(self, issue: dict, scope: set[str], placed: dict[str, dict],
                current: Assessment) -> tuple[dict[str, dict], dict[str, str]] | None:
        payload = json.loads(self.loop._payload(scope, placed))
        for entry in payload["objects"]:
            entry["description"] = self.objects[entry["id"]].get("description", "")
        requirements = self.requirements()
        payload["constraints"], payload["plan"] = requirements["constraints"], requirements["plan"]
        if "task_context" in requirements:
            payload["task_context"] = requirements["task_context"]
        if self.failures:
            payload["previous_edit_failures"] = deepcopy(self.failures)
        payload["pending"] = self.pending(placed)
        payload["problem"] = issue["problem"]
        payload["suggested_fix"] = issue["fix"]
        answer = self._ask(FIX_INSTRUCTIONS, payload, self._images(placed))
        if answer is None:
            return None
        new, parents = self.loop._apply(answer, scope, placed)
        if new == placed and parents == self.plan.parents:
            return None
        return new, parents

    def repair(self, candidate: dict[str, dict], scope: set[str]) -> tuple[dict[str, dict], dict]:
        # every hard issue of the moved objects tried; ``repair_seconds`` (default: none) is a time limit
        budget = LoopBudget(search_seconds=self.repair_seconds, min_soft_cost=math.inf)
        loop = FocusedLoop(self.g, self.plan, self.sizes, self.catalog, self.critic, locked_parents=self.locked,
                           budget=budget, movable=scope, beam_width=self.beam_width,
                           candidate_limit=self.candidate_limit)
        fixed, info = loop.run(candidate)
        self.search_s += loop.search_s
        return fixed, {"hard_before": info["initial"]["hard_count"], "hard_after": info["final"]["hard_count"],
                       "termination": info["termination"]}

    def prefer(self, current: dict[str, dict], candidate: dict[str, dict],
               current_parents: dict[str, str]) -> tuple[bool, list]:
        """Blind pairwise comparison, both orders: True only if the candidate wins twice. ``plan.parents`` holds the
        candidate's supports; ``current_parents`` the current layout's."""
        changed = [i for i in self.g.requests if i in candidate and
                   (candidate[i] != current.get(i) or self.plan.parents.get(i) != current_parents.get(i))]
        unchanged = [self._view(current, i) for i in self.objects if i in current and i not in changed]
        old = [self._view(current, i, current_parents) for i in changed]
        new = [self._view(candidate, i) for i in changed]
        votes = []
        for first, second, wanted, first_layout, second_layout in (
                (old, new, "2", current, candidate), (new, old, "1", candidate, current)):
            answer = self._ask(PAIRWISE_INSTRUCTIONS, {"room": self._room(), "requirements": self.requirements(),
                                                       "unchanged": unchanged, "pending": self.pending(current),
                                                       "layout_1": first, "layout_2": second},
                               self._images(first_layout, second_layout))
            better = str((answer or {}).get("better", "")).strip()
            votes.append({"better": better, "candidate_wins": better == wanted,
                          "reason": str((answer or {}).get("reason", ""))[:300]})
            if better != wanted:
                break
        return len(votes) == 2 and all(v["candidate_wins"] for v in votes), votes

    # ---------------------------------------------------------------- the pass
    def run(self, placed: dict[str, dict], review_ids: set[str], movable: set[str], *, rounds: int = 2,
            max_fixes: int = 3, escape: dict | None = None) -> tuple[dict[str, dict], dict]:
        """``escape`` (stage two): {"ids": stage-one objects, "left": moves allowed, "recheck": placed -> (ok,
        reason)}; a severe issue naming stage-one objects may move them while ``left`` lasts, and the change must
        also pass ``recheck`` (stage one's program checks)."""
        start = time.perf_counter()
        current = self.critic.assess(placed)
        info = {"stage": self.stage, "rounds": [], "accepted": 0}
        self.failures = []
        task_context = self.task_context is not None
        for _ in range(min(rounds, 2) if task_context else rounds):
            issues = self.review(placed, review_ids, current)
            record = {"issues": issues, "fixes": []}
            info["rounds"].append(record)
            if issues is None:
                record["error"] = "review request failed"
                break
            accepted = False
            for issue in [i for i in issues if i["severity"] == "severe"][:max_fixes]:
                ids, allowed, escaped = set(issue["ids"]), set(movable), False
                if escape and ids & escape["ids"] and escape["left"] > 0:
                    allowed |= ids & escape["ids"]
                    escape["left"] -= 1
                    escaped = True
                scope = (_with_children(self.plan, ids & allowed) & set(placed)) & (allowed | movable)
                fix = {"ids": issue["ids"], "scope": sorted(scope), "escape": escaped, "accepted": False}
                record["fixes"].append(fix)
                if not scope:
                    fix["reason"] = "no movable object in the issue"
                    continue
                saved, ok = dict(self.plan.parents), False
                try:
                    proposal = self.propose(issue, scope, placed, current)
                    if proposal is None:
                        fix["reason"] = "no usable proposal"
                        continue
                    candidate, parents = proposal
                    self.plan.parents.update(parents)  # the geometry and the critic read this dict
                    candidate, fix["repair"] = self.repair(candidate, scope)
                    after = self.critic.assess(candidate)
                    fix["after"] = after.key()
                    ok, fix["reason"] = admissible(after, current, self.soft_tolerance)
                    if ok and escaped:
                        ok, fix["reason"] = escape["recheck"](candidate)
                    if ok:
                        ok, fix["votes"] = self.prefer(placed, candidate, saved)
                        fix["reason"] = "reviewer preferred the change in both orders" if ok else \
                            "reviewer did not prefer the change in both orders"
                except FatalAPIError:
                    raise
                except Exception as exc:  # a malformed answer or a failure on it: this candidate only is rejected
                    ok, fix["reason"] = False, f"error: {type(exc).__name__}: {exc}"[:300]
                finally:
                    if not ok:
                        self.plan.parents.clear()
                        self.plan.parents.update(saved)
                if ok:
                    fix["accepted"], fix["before"] = True, current.key()
                    placed, current, accepted = candidate, after, True
                    info["accepted"] += 1
            self.failures = [{"operation": "move", "ids": fix["ids"], "scope": fix["scope"],
                              "reason": fix["reason"]} for fix in record["fixes"]
                             if not fix["accepted"] and fix["scope"]]
            if not accepted and not (task_context and self.failures):
                break
        info.update(final=current.summary(), calls=len(self.calls), search_s=round(self.search_s, 3),
                    seconds=round(time.perf_counter() - start, 3))
        return placed, info
