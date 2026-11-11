from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from workflow import planning
from workflow.planning import make_plan

ROOT = Path(__file__).resolve().parents[1]


def room(objects, constraints=()):
    return {"schema_version": "fastfill.v2",
            "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [5, 0], [5, 4], [0, 4]],
                     "floor_z_m": 0, "height_m": 2.8, "room_type": "bedroom"},
            "objects": [{"id": i, "category": c, "description": d} for i, c, d in objects],
            "constraints": list(constraints)}


def fake_llm(answer):
    call = {"model": "fake", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "latency_s": 0.1, "failed_attempts": 0}
    return mock.patch.object(planning.llm_client, "ask", return_value=(json.dumps(answer), call))


class RulePlanTests(unittest.TestCase):
    def test_head_noun_matching(self):
        # "table" must not match the lamp itself or a bedside table; "bed" must not match "bedside table"
        plan = make_plan(room([("lamp", "table lamp", "small table lamp"), ("desk", "desk", "desk"),
                               ("pillow", "pillow", "pillow"), ("ns", "bedside table", "x"), ("bed", "bed", "bed")]),
                         llm=False)
        self.assertEqual(plan.parents["lamp"], "ns")
        self.assertEqual(plan.parents["pillow"], "bed")
        self.assertEqual(plan.parents["ns"], "floor")
        self.assertIn({"anchor": "bed", "members": ["ns"]}, plan.groups)

    def test_desk_lamp_prefers_desk_and_lamps_spread(self):
        plan = make_plan(room([("l1", "lamp", "desk lamp"), ("l2", "lamp", "bedside table lamp"),
                               ("l3", "lamp", "bedside table lamp"), ("d", "desk", "d"),
                               ("n1", "nightstand", "n"), ("n2", "nightstand", "n")]), llm=False)
        self.assertEqual((plan.parents["l1"], plan.parents["l2"], plan.parents["l3"]), ("d", "n1", "n2"))

    def test_tv_stand_and_two_tables(self):
        plan = make_plan(room([("s", "sofa", "s"), ("tvs", "tv stand", "x"), ("c", "chair", "c"),
                               ("t1", "dining table", "t"), ("t2", "desk", "d")]), llm=False)
        self.assertFalse([r for r in plan.relations if r["object_id"] == "s"])  # a tv stand is not a tv
        self.assertFalse([r for r in plan.relations if r["object_id"] == "c"])  # two tables: ambiguous

    def test_on_conflicting_with_support_parent(self):
        condition = room([("a", "vase", "v"), ("b", "table", "t"), ("c", "shelf", "s")],
                         [{"type": "on", "object_id": "a", "parent_id": "c"}])
        condition["objects"][0]["support_parent"] = "b"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            make_plan(condition, llm=False)


class LlmPlanTests(unittest.TestCase):
    condition = room([("bed", "bed", "queen bed"), ("ns", "nightstand", "n"), ("lamp", "lamp", "lamp"),
                      ("vase", "vase", "v"), ("art", "painting", "p")])

    def setUp(self):
        self.condition = json.loads(json.dumps(LlmPlanTests.condition))
        self.condition["objects"][3]["support_parent"] = "floor"  # locked

    def test_fields_are_checked_and_mapped(self):
        answer = {"objects": {
            "bed": {"support": "floor", "size_m": {"width": 1.6, "depth": 2.1, "height": .5}, "priority": 1,
                    "draft": {"x": 2.5, "y": 2.9, "yaw_deg": -90}},
            "ns": {"support": "ns", "size_m": {"width": "wide", "depth": .4, "height": .5}, "priority": "x"},
            "lamp": {"support": "ns", "size_m": {"width": .3, "depth": .3, "height": .5}, "priority": 3},
            "vase": {"support": "bed", "priority": 2},
            "ghost": {"support": "floor"}},
            "relations": [{"type": "near", "object_id": "ns", "target_id": "bed", "max_distance_m": "close"},
                          {"type": "faces", "object_id": "bed", "target_id": "bed"},
                          {"type": "against_wall", "object_id": "bed"}],
            "groups": [{"anchor": "bed", "members": ["ns", "lamp", "nope"]}]}
        with fake_llm(answer):
            plan = make_plan(self.condition)
        self.assertEqual(plan.source, "llm+rules")
        self.assertEqual(plan.parents["ns"], "floor")    # self-support rejected
        self.assertEqual(plan.parents["lamp"], "ns")
        self.assertEqual(plan.parents["vase"], "floor")  # locked by the input
        self.assertEqual(plan.parents["art"], "wall")    # rule plan kept where the LLM is silent
        self.assertEqual(plan.priority, {"bed": 1, "vase": 2, "lamp": 3})
        self.assertEqual(plan.draft_poses, {})  # sizes and poses are steps ② and ③'s, whatever the answer says
        near = [r for r in plan.relations if r["origin"] == "llm" and r["type"] == "near"]
        self.assertEqual(near[0]["max_distance_m"], 1.0)  # non-numeric distance -> default, no crash
        self.assertFalse([r for r in plan.relations if r["type"] == "faces" and r["origin"] == "llm"])
        self.assertEqual(plan.groups, [{"anchor": "bed", "members": ["ns", "lamp"]}])
        group_near = [r for r in plan.relations if r["origin"] == "group"]
        self.assertEqual([r["object_id"] for r in group_near], [])  # ns already linked; lamp is on the nightstand
        dropped = plan.llm["dropped"]
        self.assertEqual((dropped["unknown_object"], dropped["support_invalid"], dropped["priority_invalid"],
                          dropped["relation_invalid"], dropped["objects_missing"]), (1, 1, 1, 1, 1))
        self.assertNotIn("size_invalid", dropped)  # sizes in the answer are not read
        self.assertEqual(plan.llm["calls"][0]["usage"]["prompt_tokens"], 10)

    def test_cycle_rejected(self):
        answer = {"objects": {"bed": {"support": "ns"}, "ns": {"support": "bed"}}}
        with fake_llm(answer):
            plan = make_plan(self.condition)
        self.assertFalse(planning._would_cycle(plan.parents))

    def test_failure_falls_back_to_rules(self):
        with mock.patch.object(planning.llm_client, "ask", side_effect=RuntimeError("LLM request failed: HTTP 500")):
            plan = make_plan(self.condition)
        self.assertEqual(plan.source, "rules (llm failed)")
        self.assertIn("HTTP 500", plan.llm["error"])
        with mock.patch.object(planning.llm_client, "ask", return_value=("no json here", {})):
            self.assertEqual(make_plan(self.condition).source, "rules (llm failed)")

    def test_missing_credentials_raise(self):
        # a missing settings file stops the run (FatalAPIError) instead of falling back to the rule plan
        from workflow.credentials import FatalAPIError
        with mock.patch("workflow.credentials.DEFAULT_ENV_FILE", ROOT / "no-such-settings.env"):
            with self.assertRaisesRegex(FatalAPIError, "no OpenAI settings file"):
                make_plan(self.condition)


if __name__ == "__main__":
    unittest.main()
