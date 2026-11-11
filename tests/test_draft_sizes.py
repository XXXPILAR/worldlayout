from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from workflow import drafting
from workflow.simple import run_simple

ROOT = Path(__file__).resolve().parents[1]
CATALOG = str(ROOT / "size_catalog.json")


class DraftSizesTests(unittest.TestCase):
    """The workflow without FastFill: the placement call gives every size with the poses, and the later steps keep
    the chosen draft's sizes; an object it gives no valid size keeps the catalog's."""

    def setUp(self):
        self.condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
        self.answer = {"objects": {
            "bed": {"width_m": 1.8, "depth_m": 2.2, "height_m": .55, "x": 1.5, "y": 2.0, "yaw_deg": 0},
            "nightstand": {"width_m": .5, "depth_m": .4, "height_m": .6, "x": .4, "y": .5, "yaw_deg": 90},
            "desk": {"width_m": 1.2, "depth_m": -1, "height_m": .75, "x": 3.5, "y": .5, "yaw_deg": 90},
            "chair": {"x": 3.5, "y": 1.2, "yaw_deg": 270},
            "lamp": {"width_m": .2, "depth_m": .2, "height_m": .4, "x": 3.5, "y": .5, "yaw_deg": 90}}}
        self.prompts = []

        def ask(instructions, prompt, **_kw):
            self.prompts.append((instructions, prompt))
            return json.dumps(self.answer), {"usage": {}, "cost_usd": 0.}
        self.ask = ask

    def test_the_request_has_no_sizes_and_asks_for_them(self):
        self.assertIn('"width_m": w', drafting.FREE_SIZE_INSTRUCTIONS)
        self.assertNotIn("do not change sizes", drafting.FREE_SIZE_INSTRUCTIONS)
        with mock.patch.object(drafting.llm_client, "ask", side_effect=self.ask):
            result = run_simple(self.condition, None, catalog_path=CATALOG, plan_llm=False, semantic=False,
                                repair=False, model="m", base_url="https://x", api_key="k", draft_sizes=True)
        instructions, prompt = self.prompts[0]
        self.assertTrue(instructions.startswith(drafting.FREE_SIZE_INSTRUCTIONS))
        objects = json.loads(prompt)["objects"]
        self.assertFalse(any("size_m" in o for o in objects))
        self.assertEqual({o["id"]: o["description"] for o in objects},
                         {o["id"]: o["description"] for o in self.condition["objects"]})
        d = result["diagnostics"]
        self.assertEqual(d["size_source"], {"draft": 3, "catalog": 2})  # desk (negative depth) and chair (none)
        self.assertEqual(d["sizes"]["bed"], [2.2, 1.8, .55])
        self.assertEqual(d["draft"]["sized_by_draft"], ["bed", "lamp", "nightstand"])
        boxes = {b["id"]: b["target_size_local_m"] for b in result["layout"]["objects"]}
        self.assertEqual(boxes["bed"], [2.2, 1.8, .55])
        self.assertEqual(boxes["chair"], d["sizes"]["chair"])

    def test_with_fixed_sizes_the_request_is_unchanged(self):
        with mock.patch.object(drafting.llm_client, "ask", side_effect=self.ask):
            run_simple(self.condition, None, catalog_path=CATALOG, plan_llm=False, semantic=False, repair=False,
                       model="m", base_url="https://x", api_key="k")
        instructions, prompt = self.prompts[0]
        self.assertTrue(instructions.startswith(drafting.DRAFT_INSTRUCTIONS))
        self.assertTrue(all("size_m" in o for o in json.loads(prompt)["objects"]))

    def test_given_sizes_and_draft_sizes_do_not_mix(self):
        with self.assertRaisesRegex(ValueError, "no sizes are given"):
            run_simple(self.condition, {"bed": [2, 1.6, .5]}, catalog_path=CATALOG, llm=False, draft_sizes=True)


if __name__ == "__main__":
    unittest.main()
