"""Step ⑤, the LLM review (``semantic.SemanticPass``), and the repair loop's budget."""

from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

from workflow.catalog import SizeCatalog
from workflow.contract import check_request, fill_room_defaults
from workflow.critic import Assessment, Critic, Violation, critic_geometry
from workflow.geometry import Geometry
from workflow.loop import LoopBudget
from workflow.planning import make_plan
from workflow.semantic import FIX_INSTRUCTIONS, PAIRWISE_INSTRUCTIONS, REVIEW_INSTRUCTIONS, SemanticPass, admissible
from workflow.solver import _pose

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "size_catalog.json"
SIZES = {"bed": [2.1, 1.6, .5], "nightstand": [.45, .45, .55], "desk": [.6, 1.2, .75], "chair": [.5, .5, .9],
         "lamp": [.25, .25, .45]}


def sub_room(condition: dict, plan, ids, templates: list[dict]):
    """Geometry and critic of the room holding only ``ids`` (plus its fixed objects): the input's constraints and the
    plan's relations whose objects are all there (a review that sees other objects as still to be placed)."""
    ids = set(ids)
    sub = copy.deepcopy(condition)
    sub["objects"] = [o for o in sub["objects"] if o["id"] in ids]
    known = ids | {f["id"] for f in sub["room"].get("fixed_objects", [])}

    def inside(r: dict) -> bool:
        return {r.get("object_id"), r.get("target_id"), r.get("parent_id"), *(r.get("target_ids") or [])} - {None} <= known
    sub["constraints"] = [r for r in sub["constraints"] if inside(r)]
    geometry = critic_geometry(sub, plan)
    geometry.soft = [r for r in geometry.soft if inside(r)]
    critic = Critic(geometry, [r for r in geometry.explicit if not r.get("hard", True)],
                    [r for r in plan.relations if inside(r)], templates)
    return geometry, critic


def _room():
    condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
    return check_request(fill_room_defaults(condition)[0])


def _layout(chair_yaw: float) -> dict:
    placed = {"bed": _pose("bed", SIZES["bed"], 2.5, 2.948, 0., -math.pi / 2, "floor"),  # 2 mm off the top wall
              "nightstand": _pose("nightstand", SIZES["nightstand"], 1.15, 3.7, 0., -math.pi / 2, "floor"),
              "desk": _pose("desk", SIZES["desk"], 4.6, 1.5, 0., math.pi, "floor"),
              "chair": _pose("chair", SIZES["chair"], 3.95, 1.5, 0., chair_yaw, "floor")}
    placed["lamp"] = _pose("lamp", SIZES["lamp"], 4.6, 1.5, .75, math.pi, "desk")
    return placed


class FakeLLM:
    """Answers the three semantic requests: a review naming the chair, a fix, and pairwise votes."""

    def __init__(self, severity="severe", fix=None, vote="faithful"):
        self.severity, self.fix, self.vote = severity, fix or {"x": 3.95, "y": 1.5, "yaw_deg": 0}, vote
        self.seen: list[str] = []
        self.payloads: list[tuple[str, dict]] = []

    def __call__(self, instructions: str, prompt: str):
        payload = json.loads(prompt)
        self.payloads.append((instructions, payload))
        if instructions == REVIEW_INSTRUCTIONS:
            self.seen.append("review")
            chair = next(o for o in payload["objects"] if o["id"] == "chair")
            issues = [] if abs(chair["yaw_deg"]) < 5 else \
                [{"ids": ["chair"], "severity": self.severity, "problem": "the chair faces away from the desk",
                  "fix": "turn it to the desk"}]
            return json.dumps({"issues": issues}), {"usage": {}}
        if instructions == FIX_INSTRUCTIONS:
            self.seen.append("fix")
            return json.dumps({"objects": {"chair": self.fix}}), {"usage": {}}
        assert instructions == PAIRWISE_INSTRUCTIONS
        self.seen.append("pairwise")
        if self.vote == "always_2":
            return json.dumps({"better": "2"}), {"usage": {}}
        yaw = {k: next(o["yaw_deg"] for o in payload[k] if o["id"] == "chair") for k in ("layout_1", "layout_2")}
        return json.dumps({"better": "1" if abs(yaw["layout_1"]) < abs(yaw["layout_2"]) else "2"}), {"usage": {}}


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.condition = _room()
        self.plan = make_plan(self.condition, llm=False)
        self.g = Geometry(self.condition, self.plan.parents, self.plan.relations, hard_doors=True, hard_windows=False,
                          door_rule="evaluator")
        self.critic = Critic.for_plan(self.g, self.plan, templates=[])

    def _pass(self, llm, **kwargs):
        return SemanticPass(self.g, self.critic, self.plan, SIZES, SizeCatalog(CATALOG), locked_parents={}, ask=llm,
                            stage="floor", **kwargs)

    def test_admissible(self):
        clean = Assessment([], [Violation("reach", ("bed",), .2)])
        self.assertFalse(admissible(Assessment([Violation("collision", ("a", "b"), .01)], []), clean)[0])
        self.assertFalse(admissible(Assessment([], [Violation("reach", ("bed",), .5)]), clean)[0])
        self.assertFalse(admissible(Assessment([], [Violation("reach", ("bed",), .2), Violation("template_near", ("a", "b"), .2)]),
                                    clean, soft_tolerance=.1)[0])
        self.assertTrue(admissible(Assessment([], [Violation("reach", ("bed",), .2),
                                                   Violation("template_near", ("a", "b"), .08)]), clean)[0])

    def test_fix_kept_when_preferred_in_both_orders(self):
        llm = FakeLLM()
        placed, info = self._pass(llm).run(_layout(math.pi), {"bed", "nightstand", "desk", "chair"},
                                           {"bed", "nightstand", "desk", "chair"})
        self.assertAlmostEqual(placed["chair"]["yaw_rad"], 0., places=3)
        self.assertEqual(info["accepted"], 1)
        self.assertEqual(llm.seen[:4], ["review", "fix", "pairwise", "pairwise"])
        self.assertEqual(llm.seen[-1], "review")  # the second round re-reviews and finds nothing

    def test_split_vote_keeps_the_layout(self):
        start = _layout(math.pi)
        placed, info = self._pass(FakeLLM(vote="always_2")).run(start, {"chair"}, {"chair"})
        self.assertEqual(placed, start)
        self.assertEqual(info["accepted"], 0)

    def test_minor_issue_is_not_fixed(self):
        llm = FakeLLM(severity="minor")
        start = _layout(math.pi)
        placed, _ = self._pass(llm).run(start, {"chair"}, {"chair"})
        self.assertEqual(placed, start)
        self.assertEqual(llm.seen, ["review"])

    def test_illegal_proposal_is_never_committed(self):
        llm = FakeLLM(fix={"x": 2.5, "y": 2.95, "yaw_deg": 0})  # into the bed
        placed, _ = self._pass(llm).run(_layout(math.pi), {"chair"}, {"chair"})
        self.assertEqual(self.critic.assess(placed).hard, [])

    def test_frozen_objects_do_not_move(self):
        llm = FakeLLM()
        start = _layout(math.pi)
        placed, info = self._pass(llm).run(start, {"chair"}, {"lamp"})  # the chair is not movable here
        self.assertEqual(placed, start)
        self.assertEqual(info["rounds"][0]["fixes"][0]["reason"], "no movable object in the issue")

    def test_requests_carry_the_requirements(self):
        """Stage one's review sees descriptions, the user's constraints, the plan as suggestions and the pending lamp;
        the pairwise judge gets the same context but not the reviewer's problem text."""
        start = _layout(math.pi)
        floor = {i: b for i, b in start.items() if i != "lamp"}
        g1, c1 = sub_room(self.condition, self.plan, floor, [])
        llm = FakeLLM()
        SemanticPass(g1, c1, self.plan, SIZES, SizeCatalog(CATALOG), locked_parents={}, ask=llm, stage="floor",
                     condition=self.condition).run(floor, set(floor), set(floor), rounds=1)
        kinds = [k for k, _ in llm.payloads]
        self.assertEqual(kinds, [REVIEW_INSTRUCTIONS, FIX_INSTRUCTIONS, PAIRWISE_INSTRUCTIONS, PAIRWISE_INSTRUCTIONS])
        review, fix, pairwise = llm.payloads[0][1], llm.payloads[1][1], llm.payloads[2][1]
        objects = {o["id"]: o for o in review["objects"]}
        self.assertEqual(objects["bed"]["description"], "a queen bed with upholstered headboard")
        self.assertEqual(objects["lamp"]["status"], "pending")
        self.assertEqual(objects["chair"]["status"], "review")
        self.assertTrue(all(c["hard"] in (True, False) for c in review["requirements"]["constraints"]))
        self.assertIn("not user requirements", review["requirements"]["plan"]["note"])
        self.assertTrue(all({"kind", "finding"} <= set(e) for e in review["evidence"]))
        self.assertEqual([p["id"] for p in fix["pending"]], ["lamp"])
        self.assertTrue(all("description" in o for o in fix["objects"]))
        self.assertIn("requirements", pairwise)
        self.assertEqual([p["id"] for p in pairwise["pending"]], ["lamp"])
        self.assertNotIn("faces away", json.dumps(pairwise))  # blind: no problem text
        self.assertNotIn("problem", pairwise)

    def test_task_context_and_images_follow_review_fix_and_blind_orders(self):
        context = {"task": "Sit at the desk and switch on the lamp", "kind": "operational",
                   "initial_relations": [], "goals": [{"subject": "lamp", "relation": "powered_on"}],
                   "objects": [{"id": "desk", "role": "required", "reason": "task workspace"}]}
        condition_before = copy.deepcopy(self.condition)

        class Visual(FakeLLM):
            supports_images = True

            def __init__(self):
                super().__init__()
                self.images = []

            def __call__(self, instructions, prompt, *, images=None):
                self.images.append(images)
                return super().__call__(instructions, prompt)

        llm = Visual()
        reviewer = self._pass(llm, task_context=context)
        reviewer.run(_layout(math.pi), set(SIZES), set(SIZES), rounds=1)
        self.assertEqual([len(images) for images in llm.images], [1, 1, 2, 2])
        for (_, payload), images in zip(llm.payloads, llm.images):
            self.assertIn("bbox", payload["visual_observation"]["scope"])
            self.assertTrue(all(image.startswith("data:image/png;base64,") for image in images))
        review, fix, pairwise = [p for _, p in llm.payloads[:3]]
        self.assertEqual(review["requirements"]["task_context"], context)
        self.assertEqual(fix["task_context"], context)
        self.assertEqual(pairwise["requirements"]["task_context"], context)
        self.assertEqual(llm.images[2], list(reversed(llm.images[3])))
        self.assertEqual([call["image_count"] for call in reviewer.calls], [1, 1, 2, 2])
        self.assertEqual(self.condition, condition_before)
        self.assertNotIn("problem", pairwise)
        self.assertNotIn("previous_edit_failures", pairwise)
        self.assertIn("future", REVIEW_INSTRUCTIONS)

    def test_rejected_geometry_is_returned_to_next_task_round(self):
        context = {"task": "Sit at the desk", "kind": "operational"}

        class Reattempt(FakeLLM):
            def __call__(self, instructions, prompt):
                if instructions == FIX_INSTRUCTIONS:
                    self.fix = {"x": 2.5, "y": 2.95, "yaw_deg": 0} if self.seen.count("fix") == 0 else \
                        {"x": 3.95, "y": 1.5, "yaw_deg": 0}
                return super().__call__(instructions, prompt)

        llm = Reattempt()
        reviewer = self._pass(llm, task_context=context)
        # Keep the invalid first proposal unchanged so the actual geometry gate rejects it.
        reviewer.repair = lambda candidate, scope: (candidate, {"termination": "test: no repair"})
        placed, info = reviewer.run(_layout(math.pi), {"chair"}, {"chair"}, rounds=2)
        self.assertEqual(info["accepted"], 1)
        self.assertAlmostEqual(placed["chair"]["yaw_rad"], 0)
        reviews = [p for instructions, p in llm.payloads if instructions == REVIEW_INSTRUCTIONS]
        self.assertEqual(len(reviews), 2)
        failure = reviews[1]["previous_edit_failures"][0]
        self.assertEqual(failure["ids"], ["chair"])
        self.assertEqual(failure["operation"], "move")
        self.assertIn("hard", failure["reason"])
        fixes = [p for instructions, p in llm.payloads if instructions == FIX_INSTRUCTIONS]
        self.assertEqual(fixes[1]["previous_edit_failures"], reviews[1]["previous_edit_failures"])
        for instructions, payload in llm.payloads:
            if instructions == PAIRWISE_INSTRUCTIONS:
                self.assertNotIn("previous_edit_failures", payload)

    def test_malformed_answers_reject_only_the_candidate(self):
        start = _layout(math.pi)

        class Odd(FakeLLM):
            def __init__(self, fix_answer, **kw):
                super().__init__(**kw)
                self.fix_answer = fix_answer

            def __call__(self, instructions, prompt):
                if instructions == FIX_INSTRUCTIONS:
                    self.seen.append("fix")
                    if isinstance(self.fix_answer, Exception):
                        raise self.fix_answer
                    return json.dumps(self.fix_answer), {"usage": {}}
                return super().__call__(instructions, prompt)

        # a list instead of the map: read as the same poses
        listed = Odd({"objects": [{"id": "chair", "x": 3.95, "y": 1.5, "yaw_deg": 0}]})
        placed, info = self._pass(listed).run(start, {"chair"}, {"chair"})
        self.assertAlmostEqual(placed["chair"]["yaw_rad"], 0., places=3)
        for answer in ({"objects": "chair"}, {"objects": [1, 2]}, {"objects": {"chair": [3.95, 1.5]}},
                       TypeError("unexpected failure")):
            placed, info = self._pass(Odd(answer)).run(start, {"chair"}, {"chair"})
            self.assertEqual(placed, start)
            self.assertFalse(info["rounds"][0]["fixes"][0]["accepted"])
        self.assertIn("error: TypeError", info["rounds"][0]["fixes"][0]["reason"])

    def test_malformed_review_is_read_safely(self):
        for issues in ({"chair": "bad"}, "none", [{"ids": "chair", "severity": "minor", "problem": "x"}]):
            llm = lambda ins, prompt, issues=issues: (json.dumps({"issues": issues}), {"usage": {}})  # noqa: E731
            placed, info = self._pass(llm).run(_layout(math.pi), {"chair"}, {"chair"})
            self.assertEqual(placed, _layout(math.pi))
        self.assertEqual(info["rounds"][0]["issues"][0]["ids"], ["chair"])


class BudgetTests(unittest.TestCase):
    """The repair is bounded by work (every issue tried, a number of attempts), not by time."""

    def setUp(self):
        self.condition = _room()
        self.plan = make_plan(self.condition, llm=False)
        self.g = Geometry(self.condition, self.plan.parents, self.plan.relations, hard_doors=True,
                          hard_windows=False, door_rule="evaluator")
        self.critic = Critic.for_plan(self.g, self.plan, templates=[])
        crowded = _layout(math.pi)
        crowded["chair"] = _pose("chair", SIZES["chair"], 4.3, 1.5, 0., math.pi, "floor")  # into the desk
        self.crowded = crowded

    def _loop(self, budget, movable=None):
        from workflow.loop import FocusedLoop, Loop
        kw = {"locked_parents": {}, "budget": budget}
        if movable is None:
            return Loop(self.g, self.plan, SIZES, SizeCatalog(CATALOG), self.critic, **kw)
        return FocusedLoop(self.g, self.plan, SIZES, SizeCatalog(CATALOG), self.critic, movable=movable, **kw)

    def test_default_attempts_scale_with_the_issues_and_no_time_limit(self):
        current = self.critic.assess(self.crowded)
        loop = self._loop(LoopBudget())
        self.assertEqual(loop.max_attempts(current), 24 + 3 * len(loop._targets(current, set())))
        self.assertIsNone(LoopBudget().search_seconds)

    def test_every_issue_is_tried_without_a_time_budget(self):
        placed, info = self._loop(LoopBudget()).run(self.crowded)
        self.assertEqual(info["termination"], "no issue left to try")
        self.assertIsNone(info["time_limit_s"])
        self.assertEqual(self.critic.assess(placed).hard, [])

    def test_an_explicit_time_limit_is_reported(self):
        _, info = self._loop(LoopBudget(search_seconds=0.)).run(self.crowded)
        self.assertEqual(info["termination"], "time limit")


if __name__ == "__main__":
    unittest.main()
