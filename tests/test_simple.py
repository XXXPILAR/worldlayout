from __future__ import annotations

import json
import unittest
from pathlib import Path

from test_semantic import SIZES, FakeLLM, _room  # shared fixtures
from workflow import credentials
from workflow.pipeline import run_draft_workflow
from workflow.planning import make_plan
from workflow.robot import Robot
from workflow.semantic import REVIEW_INSTRUCTIONS
from workflow.simple import run_simple

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "size_catalog.json"


def _request():
    return json.loads((ROOT / "examples" / "bedroom.json").read_text())


class SimpleTests(unittest.TestCase):
    def test_the_robot_margin_is_planned_for_and_the_robot_reported(self):
        result = run_simple(_request(), None, catalog_path=str(CATALOG), llm=False, robot=Robot(width_m=.5),
                            robot_margin=.1, hard_zones=True)
        d = result["diagnostics"]
        self.assertEqual(d["robot"]["width_m"], .5)
        self.assertEqual(d["robot_plan"]["margin_m"], .1)
        self.assertTrue(d["robot_plan"]["hard_zones"])

    def test_offline_run_keeps_every_stage(self):
        result = run_simple(_request(), None, catalog_path=str(CATALOG), llm=False)
        d = result["diagnostics"]
        self.assertEqual(d["placement"], "simple")
        self.assertEqual(set(d["stages"]), {"draft", "after_check", "final"})
        self.assertEqual({b["id"] for b in result["layout"]["objects"]}, {o["id"] for o in _request()["objects"]})
        self.assertEqual(d["semantic"], {"result": "skipped"})
        # every object starts at one point offline: the one check repairs hard violations only
        self.assertGreater(d["assessments"]["draft"]["hard_count"], 0)
        self.assertLess(d["assessments"]["after_check"]["hard_count"], d["assessments"]["draft"]["hard_count"])
        self.assertEqual(d["check"]["result"], "repaired")
        self.assertEqual(d["check"]["termination"], "no issue left to try")
        self.assertTrue(all(a["target"]["code"] not in ("relation_plan", "template_near", "template_faces",
                                                        "wall_back", "reach", "access")
                            for a in d["check"]["attempts"]))  # soft costs are not chased
        self.assertEqual(result["status"] == "clean", d["critic"]["final"]["hard_count"] == 0)

    def test_one_review_round_of_the_whole_room(self):
        llm = FakeLLM()
        plan = make_plan(_room(), llm=False)
        result = run_simple(_request(), SIZES, plan=plan, catalog_path=str(CATALOG), llm=False, semantic_ask=llm)
        review = result["diagnostics"]["semantic"]
        self.assertEqual(review["stage"], "room")
        self.assertEqual(len(review["rounds"]), 1)
        self.assertEqual(llm.seen.count("review"), 1)
        first = next(p for k, p in llm.payloads if k == REVIEW_INSTRUCTIONS)
        self.assertTrue(all(o["status"] == "review" for o in first["objects"]))

    def test_review_never_breaks_legality(self):
        llm = FakeLLM(fix={"x": 2.5, "y": 2.95, "yaw_deg": 0})  # a fix that drops the chair into the bed
        plan = make_plan(_room(), llm=False)
        result = run_simple(_request(), SIZES, plan=plan, catalog_path=str(CATALOG), llm=False, semantic_ask=llm)
        d = result["diagnostics"]
        self.assertLessEqual(d["assessments"]["final"]["hard_count"], d["assessments"]["after_check"]["hard_count"])

    def test_robot_checks_are_hard_and_reported(self):
        result = run_simple(_request(), None, catalog_path=str(CATALOG), llm=False, robot=Robot(width_m=.5))
        d = result["diagnostics"]
        self.assertEqual(d["robot"]["width_m"], .5)
        robot_hard = [v for v in d["critic"]["final"]["hard"] if v["code"].startswith("robot_")]
        self.assertEqual(d["robot"]["ok"], not robot_hard)
        self.assertIsNone(run_simple(_request(), None, catalog_path=str(CATALOG), llm=False)["diagnostics"]["robot"])

    def test_the_earlier_interface_still_works(self):
        """infinite-task-world's bridge passes placement="simple", critic=True and imports orders.critic_geometry."""
        from workflow import critic, orders
        self.assertIs(orders.critic_geometry, critic.critic_geometry)
        result = run_draft_workflow(_request(), None, catalog_path=str(CATALOG), llm=False, placement="simple",
                                    critic=True)
        self.assertEqual(result["diagnostics"]["placement"], "simple")
        for kwargs in ({"placement": "staged"}, {"critic": False}):
            with self.assertRaises(ValueError):
                run_draft_workflow(_request(), None, catalog_path=str(CATALOG), llm=False, **kwargs)

    def test_pipeline_runs_the_simple_workflow(self):
        result = run_draft_workflow(_request(), None, catalog_path=str(CATALOG), llm=False)
        self.assertEqual(result["diagnostics"]["placement"], "simple")
        self.assertIn("plan", result["timing_s"])

    def test_key_file_is_the_original_workflows(self):
        """workflow2 has no .env of its own: it reads the original workflow's in place (never a copy)."""
        if not credentials._OWN_ENV_FILE.is_file():
            expected = credentials._V1_ENV_FILE if credentials._V1_ENV_FILE.is_file() else credentials._OWN_ENV_FILE
            self.assertEqual(credentials.DEFAULT_ENV_FILE, expected)


if __name__ == "__main__":
    unittest.main()
