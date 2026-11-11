from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from workflow.catalog import SizeCatalog, build_catalog
from workflow.contract import check_request, validate_layout
from workflow.geometry import Geometry
from workflow.pipeline import run_draft_workflow
from workflow.planning import make_plan

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
        cls.result = run_draft_workflow(cls.condition, None, catalog_path=str(ROOT / "size_catalog.json"), llm=False)

    def test_no_furniture_bbox_in_input(self):
        self.assertIn(self.result["status"], ("clean", "best_effort"))
        self.assertTrue(all("target_size_local_m" not in obj and "bottom_center_m" not in obj
                            for obj in self.condition["objects"]))
        self.assertEqual({obj["id"] for obj in self.result["layout"]["objects"]},
                         {obj["id"] for obj in self.condition["objects"]})
        validate_layout(self.result["layout"], self.condition)

    def test_hard_geometry_and_support(self):
        plan = make_plan(self.condition, llm=False)
        geometry = Geometry(self.condition, plan.parents, plan.relations)
        placed = {obj["id"]: obj for obj in self.result["layout"]["objects"]}
        self.assertEqual(geometry.validate(placed), [])
        self.assertEqual(placed["lamp"]["support_parent"], "desk")
        self.assertAlmostEqual(placed["lamp"]["bottom_center_m"][2],
                               placed["desk"]["target_size_local_m"][2])


    def test_reject_size_as_condition(self):
        wrong = json.loads(json.dumps(self.condition))
        wrong["objects"][0]["fixed_size_local_m"] = [2, 1.5, .6]
        with self.assertRaisesRegex(ValueError, "size must not be an input"):
            check_request(wrong)

    def test_every_object_needs_a_description_beyond_its_category(self):
        for description in ("", "  ", "Desk", "desk"):
            wrong = json.loads(json.dumps(self.condition))
            wrong["objects"][2]["description"] = description
            with self.assertRaisesRegex(ValueError, r"1 of 5 objects have no description.*\['desk'\]"):
                check_request(wrong)
            check_request(wrong, require_descriptions=False)  # the older category-only test sets
            with mock.patch.dict(os.environ, {"WORKFLOW_ALLOW_BARE_DESCRIPTIONS": "1"}):
                check_request(wrong)
        check_request(self.condition)

    def test_reject_unverifiable_opening(self):
        wrong = json.loads(json.dumps(self.condition))
        wrong["room"]["openings"][0]["wall_index"] = 99
        with self.assertRaisesRegex(ValueError, "wall_index"):
            check_request(wrong)


    def test_train_only_catalog(self):
        source = {"condition": {"objects": [{"id": "x", "category": "test chair"}]},
                  "target": {"objects": [{"id": "x", "target_size_local_m": [.5, .6, .7]}]},
                  "validity": {"size": [[True, True, True]]}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            train = path / "train.jsonl"
            train.write_text((json.dumps(source) + "\n") * 20)
            build_catalog(train, path / "catalog.json")
            catalog = SizeCatalog(path / "catalog.json")
            self.assertEqual(catalog.categories["test chair"]["median"], [.5, .6, .7])
            fake = json.loads((path / "catalog.json").read_text())
            fake["split"] = "validation"
            (path / "catalog.json").write_text(json.dumps(fake))
            with self.assertRaisesRegex(ValueError, "train split"):
                SizeCatalog(path / "catalog.json")


    def test_thin_overlap_policy_is_explicit(self):
        condition = json.loads(json.dumps(self.condition))
        condition["room"]["openings"] = []
        condition["room"]["fixed_objects"] = []
        condition["objects"] = [
            {"id": "rug", "category": "rug", "description": "rug"},
            {"id": "chair", "category": "chair", "description": "chair"},
        ]
        condition["constraints"] = []
        parents = {"rug": "floor", "chair": "floor"}
        rug = {"id": "rug", "target_size_local_m": [2, 2, .02],
               "bottom_center_m": [2, 2, 0], "yaw_rad": 0, "support_parent": "floor"}
        chair = {"id": "chair", "target_size_local_m": [.5, .5, .8],
                 "bottom_center_m": [2, 2, 0], "yaw_rad": 0, "support_parent": "floor"}
        strict = Geometry(condition, parents, [])
        visual = Geometry(condition, parents, [], allow_thin_overlap=True)
        self.assertIn("collision", [issue.code for issue in strict.check_box(chair, {"rug": rug})])
        self.assertNotIn("collision", [issue.code for issue in visual.check_box(chair, {"rug": rug})])


if __name__ == "__main__":
    unittest.main()
