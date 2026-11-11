"""The whole path from the command line, ``python -m workflow``: rows -> FastFill's sizes (a tiny FastFill model built
here, the same loading and prediction code as the real checkpoint's) -> the workflow -> room_XXX.json. No LLM call."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflow import contract  # noqa: E402,F401  (puts the fastfill package on sys.path)
from fastfill.v2.model import build_model  # noqa: E402

N_OBJECTS = len(json.loads((ROOT / "examples" / "bedroom.json").read_text())["objects"])


def cli(*args):
    return subprocess.run([sys.executable, "-m", "workflow", *map(str, args)], cwd=ROOT, capture_output=True, text=True,
                          timeout=900)


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.checkpoint = tmp / "run" / "model-step-1"
        build_model({"backbone": "tiny", "max_objects": 64}).save_pretrained(cls.checkpoint)
        (cls.checkpoint / "checkpoint_manifest.json").write_text(json.dumps({"step": 1, "max_length": 4096}))
        condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
        task = {"required_ids": ["bed", "nightstand"], "articulation_ids": ["nightstand"]}
        cls.rows = tmp / "rows.jsonl"
        cls.rows.write_text(json.dumps({"condition": condition, "task": task}) + "\n"
                            + json.dumps({"condition": condition}) + "\n")  # the second: no task
        cls.fast = ("--no-llm", "--repair-seconds", "10")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _result(self, out, k):
        return json.loads((out / f"room_{k:03d}.json").read_text())

    def test_one_command_fastfill_then_the_workflow(self):
        out = Path(self.tmp.name) / "one"
        proc = cli("run", "--rows", self.rows, "--index", 0, "--checkpoint", self.checkpoint, "--device", "cpu",
                   *self.fast, "--out", out)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        result = self._result(out, 0)
        d = result["diagnostics"]
        self.assertEqual(result["sizes_from"], "fastfill")
        self.assertEqual(d["size_source"], {"given": N_OBJECTS, "catalog": 0})  # every size is FastFill's
        for box in result["layout"]["objects"]:  # and kept: no step resizes an object
            self.assertEqual(box["target_size_local_m"], d["sizes"][box["id"]])
        self.assertIn(result["status"], ("clean", "best_effort", "fallback_required_only"))
        self.assertEqual(d["task"]["required"], ["bed", "nightstand"])
        self.assertEqual(len((out / "sizes.jsonl").read_text().splitlines()), 1)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["sizes"]["device"], "cpu")
        self.assertIsNone(summary["llm"])
        again = cli("run", "--rows", self.rows, "--index", 0, "--checkpoint", self.checkpoint, "--device", "cpu",
                    *self.fast, "--out", out)  # resumed: the room is done, nothing runs again
        self.assertEqual(again.returncode, 0, again.stderr[-2000:])
        self.assertIn("skipping 1 rooms already done", again.stdout)
        self.assertEqual(len((out / "sizes.jsonl").read_text().splitlines()), 1)

    def test_sizes_then_placement_in_parallel(self):
        sizes = Path(self.tmp.name) / "sizes.jsonl"
        proc = cli("sizes", "--rows", self.rows, "--checkpoint", self.checkpoint, "--device", "cpu", "--out", sizes)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        predicted = [json.loads(line) for line in sizes.read_text().splitlines()]
        self.assertEqual(len(predicted), 2)  # aligned with the rows
        self.assertTrue(all(p["layout"] and not p["error"] for p in predicted))
        out = Path(self.tmp.name) / "two"
        proc = cli("run", "--rows", self.rows, "--sizes-file", sizes, "--workers", 2, *self.fast, "--out", out)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        task_room, plain_room = self._result(out, 0), self._result(out, 1)
        self.assertEqual(task_room["sizes_from"], "sizes_file")
        self.assertIsNotNone(task_room["diagnostics"]["task"])
        self.assertIsNone(plain_room["diagnostics"]["task"])  # no task: placed without the task mode
        file_sizes = {b["id"]: b["target_size_local_m"] for b in predicted[1]["layout"]["objects"]}
        self.assertEqual(plain_room["diagnostics"]["sizes"], file_sizes)


if __name__ == "__main__":
    unittest.main()
