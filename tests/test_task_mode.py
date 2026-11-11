from __future__ import annotations

import json
import unittest
from pathlib import Path

from workflow import simple
from workflow.contract import validate_layout
from workflow.critic import Violation
from workflow.drafting import REQUIRED_RULE, request_text
from workflow.loop import FixedSizes
from workflow.planning import Plan, make_plan
from workflow.robot import Robot, opens

ROOT = Path(__file__).resolve().parents[1]
CATALOG = str(ROOT / "size_catalog.json")


def condition() -> dict:
    """The example bedroom (bed, nightstand, desk, chair, lamp on the desk) with a near constraint on the chair."""
    return json.loads((ROOT / "examples" / "bedroom.json").read_text())


def box(ident: str, w: float = 1., d: float = 1.) -> dict:
    return {"id": ident, "target_size_local_m": [d, w, .5], "bottom_center_m": [0., 0., 0.], "yaw_rad": 0.}


class GiveWayChoiceTests(unittest.TestCase):
    plan = Plan(parents={"bed": "floor", "desk": "floor", "lamp": "desk", "chair": "floor", "rug": "floor"})
    placed = {i: box(i) for i in ("bed", "desk", "lamp", "chair")} | {"rug": box("rug", 2., 2.)}

    def test_a_violation_with_a_required_object_counts_more(self):
        hard = [Violation("collision", ("chair", "rug"), .5), Violation("collision", ("bed", "chair"), .2)]
        # chair: .5 + (.2 + 1) beats the rug's .5; the bed is required and never chosen
        self.assertEqual(simple.give_way_choice(hard, self.placed, self.plan, {"bed"}), "chair")

    def test_required_only_ignores_violations_between_expansion_objects(self):
        hard = [Violation("collision", ("chair", "rug"), .9)]
        self.assertIsNone(simple.give_way_choice(hard, self.placed, self.plan, {"bed"}, required_only=True))
        self.assertEqual(simple.give_way_choice(hard, self.placed, self.plan, {"bed"}, exclude={"chair"}), "rug")
        self.assertEqual(simple.give_way_choice(hard, self.placed, self.plan, {"bed"}), "chair")  # tie: smaller one
        door = [Violation("robot_door", ("rug",), .3)]  # the room's doors cut off: the required objects' problem
        self.assertEqual(simple.give_way_choice(door, self.placed, self.plan, {"bed"}, required_only=True), "rug")

    def test_a_robot_check_is_the_required_objects_only_when_it_is_about_one(self):
        about_chair = Violation("robot_reach", ("chair", "bed", "rug"), .3)  # the chair blocked by the bed and the rug
        self.assertFalse(simple.on_required(about_chair, {"bed"}))
        self.assertTrue(simple.on_required(Violation("robot_reach", ("bed", "rug"), .3), {"bed"}))
        self.assertTrue(simple.on_required(Violation("collision", ("rug", "bed"), .3), {"bed"}))

    def test_nothing_carrying_a_required_object_is_removed(self):
        hard = [Violation("collision", ("desk", "bed"), .9)]
        self.assertIsNone(simple.give_way_choice(hard, self.placed, self.plan, {"bed", "lamp"}))  # the lamp is on it


class CarryTests(unittest.TestCase):
    def test_an_object_on_furniture_moves_and_turns_with_it(self):
        import math
        table0 = {"id": "t", "target_size_local_m": [1, 1, .7], "bottom_center_m": [1., 1., 0.], "yaw_rad": 0.}
        table1 = {**table0, "bottom_center_m": [3., 2., 0.], "yaw_rad": math.pi / 2}
        cup = {"id": "c", "target_size_local_m": [.1, .1, .1], "bottom_center_m": [1.3, 1., .7], "yaw_rad": 0.}
        moved = simple.carry(cup, table0, table1)
        self.assertAlmostEqual(moved["bottom_center_m"][0], 3.)
        self.assertAlmostEqual(moved["bottom_center_m"][1], 2.3)
        self.assertAlmostEqual(moved["bottom_center_m"][2], .7)
        self.assertAlmostEqual(moved["yaw_rad"], math.pi / 2)


class WithoutTests(unittest.TestCase):
    def test_constraints_relations_and_groups_follow_the_removed_objects(self):
        cond = condition()
        plan = Plan(parents={o["id"]: "floor" for o in cond["objects"]} | {"lamp": "desk"},
                    relations=[{"type": "near", "object_id": "chair", "target_id": "desk"},
                               {"type": "near", "object_id": "nightstand", "target_id": "bed"}],
                    groups=[{"anchor": "desk", "members": ["desk", "chair", "lamp"]},
                            {"anchor": "bed", "members": ["bed", "nightstand"]}],
                    priority={"bed": 1, "chair": 3})
        room, sub = simple.without(cond, plan, {"chair"})
        self.assertNotIn("chair", [o["id"] for o in room["objects"]])
        self.assertFalse(any("chair" in simple._refs(r) for r in room["constraints"]))
        self.assertEqual(len(room["constraints"]), len(cond["constraints"]) - 2)  # near and faces on the chair
        self.assertEqual([r["object_id"] for r in sub.relations], ["nightstand"])
        self.assertEqual(sub.groups[0]["members"], ["desk", "lamp"])
        self.assertNotIn("chair", sub.parents)
        _room, sub = simple.without(cond, plan, {"desk", "lamp"})
        self.assertEqual([g["anchor"] for g in sub.groups], ["bed"])  # a group goes with its anchor
        self.assertEqual(len(cond["objects"]), 5)  # the input is not changed


class TaskInputsTests(unittest.TestCase):
    def test_the_task_robot_keeps_every_check_and_adds_the_task_objects(self):
        robot = simple.task_robot(Robot(width_m=.5), {"bed", "nightstand"}, {"nightstand"})
        self.assertEqual((robot.reach_ids, robot.openable_ids), (frozenset({"bed", "nightstand"}), frozenset({"nightstand"})))
        self.assertTrue(robot.reach_supports and robot.reach_openable and robot.items and robot.doors)  # expansion too
        cabinet = {"id": "c1", "cat": "cabinet", "head": "cabinet", "size": (.5, 1., 1.), "bottom": 0.}
        trunk = {"id": "t1", "cat": "trunk", "head": "trunk", "size": (.5, 1., .5), "bottom": 0.}
        self.assertTrue(opens(robot, cabinet))           # openable by its category, required or not
        self.assertFalse(opens(robot, trunk))            # a trunk is not openable by category...
        self.assertTrue(opens(simple.task_robot(Robot(), set(), {"t1"}), trunk))  # ...but is when the task opens it

    def test_required_objects_are_marked_in_the_request_and_never_shrunk(self):
        cond = condition()
        plan = make_plan(cond, llm=False)
        sizes = {o["id"]: [.5, .5, .5] for o in cond["objects"]}
        objects = json.loads(request_text(cond, plan, sizes, required={"bed"}))["objects"]
        self.assertEqual([o["id"] for o in objects if o.get("required")], ["bed"])
        self.assertIn('"required": true', REQUIRED_RULE)
        fixed = FixedSizes(sizes)  # the workflow never shrinks an object
        self.assertEqual(fixed.shrunk({"id": "bed", "category": "bed"}, 1), [])
        self.assertEqual(fixed.shrunk({"id": "desk", "category": "desk"}, 1), [])


class TaskModeRunTests(unittest.TestCase):
    """The task mode end to end, offline (no LLM: every object starts at its fallback, so the expansion has to give way)."""

    @classmethod
    def setUpClass(cls):
        cls.cond = condition()
        cls.sizes = {"bed": [2.1, 1.6, .5], "nightstand": [.45, .45, .55], "desk": [.6, 1.2, .75],
                     "chair": [.5, .5, .9], "lamp": [.25, .25, .45]}
        cls.result = simple.run_simple(cls.cond, cls.sizes, catalog_path=CATALOG, llm=False, robot=Robot(width_m=.5),
                                       required={"bed", "nightstand"}, articulation={"nightstand"},
                                       relocate=True, repair_seconds=20)

    def test_required_objects_stay_whole_and_the_result_says_what_happened(self):
        task = self.result["diagnostics"]["task"]
        boxes = {o["id"]: o for o in self.result["layout"]["objects"]}
        self.assertTrue({"bed", "nightstand"} <= set(boxes))
        self.assertTrue(all(b["target_size_local_m"] == self.sizes[i] for i, b in boxes.items()))  # no resizing
        self.assertEqual((task["required"], task["articulation"], task["expansion"]),
                         (["bed", "nightstand"], ["nightstand"], 3))
        self.assertEqual(task["expansion_kept"], len(boxes) - 2)
        removed = {d["id"] for d in task["removed"]} | {i for d in task["removed"] for i in d["with"]}
        self.assertEqual(removed | set(boxes), set(self.sizes))
        self.assertTrue(task["required_ok"] == (not task["required_issues"]))
        self.assertTrue(task["required_ok"], task["required_issues"])  # this room has room for the task
        self.assertIn("tried", task["required_first"])
        self.assertIsInstance(task["expansion_issues"], list)
        # an expansion object is removed for a required one, or because it fits nowhere (④d)
        self.assertTrue(all(d["step"] in ("for the required objects", "does not fit") for d in task["removed"]))
        self.assertFalse([v for v in self.result["diagnostics"]["hard_issues"] if v["code"] in simple.PHYSICAL])

    def test_the_layout_is_valid_for_the_objects_kept(self):
        kept = {o["id"] for o in self.result["layout"]["objects"]}
        room, _plan = simple.without(self.cond, Plan(parents={}), set(self.sizes) - kept)
        validate_layout(self.result["layout"], room)

    def test_without_required_the_workflow_is_unchanged(self):
        result = simple.run_simple(self.cond, self.sizes, catalog_path=CATALOG, llm=False, robot=Robot(width_m=.5),
                                   repair_seconds=5)
        self.assertIsNone(result["diagnostics"]["task"])
        self.assertEqual(len(result["layout"]["objects"]), 5)

    def test_wrong_ids_are_refused(self):
        with self.assertRaisesRegex(ValueError, "required ids"):
            simple.run_simple(self.cond, self.sizes, catalog_path=CATALOG, llm=False, required={"sofa"})
        with self.assertRaisesRegex(ValueError, "required ids"):
            simple.run_simple(self.cond, self.sizes, catalog_path=CATALOG, llm=False, required={"bed"},
                              articulation={"desk"})


if __name__ == "__main__":
    unittest.main()


class LeftOutTests(unittest.TestCase):
    """④d: twelve expansion books on a 0.6 x 1.2 m desk (0.90 m2 of books on 0.72 m2) cannot all fit; those left
    colliding are left out, the fewest, and nothing else is."""

    @classmethod
    def setUpClass(cls):
        cls.cond = condition()
        cls.books = [f"book{i}" for i in range(12)]
        cls.cond["objects"] += [{"id": b, "category": "book", "description": "a hardcover book", "support_parent": "desk"}
                                for b in cls.books]
        cls.sizes = {"bed": [2.1, 1.6, .5], "nightstand": [.45, .45, .55], "desk": [.6, 1.2, .75],
                     "chair": [.5, .5, .9], "lamp": [.25, .25, .45], **{b: [.25, .3, .04] for b in cls.books}}
        cls.result = simple.run_simple(cls.cond, cls.sizes, catalog_path=CATALOG, llm=False, robot=Robot(width_m=.5),
                                       required={"bed", "nightstand"}, articulation={"nightstand"},
                                       relocate=True, repair_seconds=10)

    def test_what_does_not_fit_is_left_out_and_the_rest_is_legal(self):
        task = self.result["diagnostics"]["task"]
        kept = {o["id"] for o in self.result["layout"]["objects"]}
        left_out = [d for d in task["removed"] if d["step"] == "does not fit"]
        self.assertTrue(left_out)  # the desk cannot hold them all
        self.assertTrue({"bed", "nightstand", "desk"} <= kept)
        self.assertTrue(all(d["because"] and all(r.split()[0] in simple.PHYSICAL or r.startswith("no free place on desk")
                                                 for r in d["because"]) for d in left_out))
        self.assertFalse([v for v in self.result["diagnostics"]["hard_issues"] if v["code"] in simple.PHYSICAL])
        books_kept = [b for b in self.books if b in kept]
        # the top packed again (a 0.6 x 1.2 m desk holds 8 of these books): the lamp, the one of its kind, first
        self.assertIn("lamp", kept)
        self.assertGreaterEqual(len(books_kept), 6)
        self.assertIn("after_left_out", self.result["diagnostics"]["stages"])

    def test_the_wall_fronts_step_runs_after_it(self):
        wall_clear = self.result["diagnostics"]["task"]["wall_clear"]  # ④e (test_wall_clear.py)
        self.assertEqual(set(wall_clear), {"wall_objects", "below_need", "moved", "stuck"})
        self.assertIn("wall_clear", self.result["timing_s"])


class FallbackTests(unittest.TestCase):
    """A room is always delivered in the task mode: the required objects alone, laid out by the program."""
    sizes = {"bed": [2.1, 1.6, .5], "nightstand": [.45, .45, .55], "desk": [.6, 1.2, .75], "chair": [.5, .5, .9],
             "lamp": [.25, .25, .45]}

    def test_the_fallback_holds_only_the_required_objects_legal_and_reachable(self):
        result = simple.required_only(condition(), self.sizes, {"bed", "nightstand"}, {"nightstand"},
                                      robot=Robot(width_m=.5), catalog_path=CATALOG, repair_seconds=10, reason="test")
        self.assertEqual(result["status"], "fallback_required_only")
        self.assertEqual({o["id"] for o in result["layout"]["objects"]}, {"bed", "nightstand"})
        task = result["diagnostics"]["task"]
        self.assertTrue(task["required_ok"], task["required_issues"])
        self.assertEqual(sorted(d["id"] for d in task["removed"]), ["chair", "desk", "lamp"])
        self.assertEqual(result["layout"]["objects"][0]["target_size_local_m"], self.sizes["bed"])
        self.assertEqual(simple.NEAR_BLOCKER_TRIES, 5)

    def test_an_error_in_the_workflow_still_delivers_the_fallback_room(self):
        from unittest import mock
        from workflow.pipeline import run_draft_workflow
        with mock.patch.object(simple, "run_simple", side_effect=RuntimeError("boom")):
            result = run_draft_workflow(condition(), self.sizes, catalog_path=CATALOG, llm=False,
                                        robot=Robot(width_m=.5), required={"bed", "nightstand"},
                                        articulation={"nightstand"}, repair_seconds=10)
        self.assertEqual(result["status"], "fallback_required_only")
        self.assertIn("RuntimeError: boom", result["diagnostics"]["reason"])
        self.assertEqual({o["id"] for o in result["layout"]["objects"]}, {"bed", "nightstand"})
        with mock.patch.object(simple, "run_simple", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):  # outside the task mode an error is an error
                run_draft_workflow(condition(), self.sizes, catalog_path=CATALOG, llm=False)
