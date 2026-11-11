from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

from workflow.catalog import SizeCatalog
from workflow.contract import check_request, fill_room_defaults
import numpy as np
from shapely.geometry import Polygon

from workflow.critic import (Assessment, Critic, Violation, facing_error, unreachable, wall_back_error, walls)
from workflow.geometry import Geometry
from workflow.loop import FixedSizes
from workflow.loop import Loop
from workflow.planning import _required_parents, make_plan
from workflow.solver import _pose

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "size_catalog.json"


def _room():
    condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
    condition, _ = fill_room_defaults(condition)
    return check_request(condition)


def _geometry(condition, plan):
    return Geometry(condition, plan.parents, plan.relations, hard_doors=True, hard_windows=False, door_rule="evaluator")


class CriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.condition = _room()
        cls.plan = make_plan(cls.condition, llm=False)
        cls.g = _geometry(cls.condition, cls.plan)
        cls.critic = Critic.for_plan(cls.g, cls.plan, templates=[])
        # a clean layout: bed against the top wall (2 mm off it: geometry.BOUNDARY_INSET_M), the rest spread out;
        # the lamp on the desk
        cls.placed = {
            "bed": _pose("bed", [2.1, 1.6, .5], 2.5, 2.948, 0., -math.pi / 2, "floor"),
            "nightstand": _pose("nightstand", [.45, .45, .55], 1.15, 3.7, 0., -math.pi / 2, "floor"),
            "desk": _pose("desk", [.6, 1.2, .75], 4.6, 1.5, 0., math.pi, "floor"),
            "chair": _pose("chair", [.5, .5, .9], 3.95, 1.5, 0., 0., "floor"),
        }
        cls.placed["lamp"] = _pose("lamp", [.25, .25, .45], 4.6, 1.5, .75, math.pi, "desk")

    def test_clean_layout_has_no_hard_violation(self):
        self.assertEqual(self.critic.assess(self.placed).hard, [], self.critic.assess(self.placed).summary())

    def test_sceneeval_contact_rules(self):
        """Contact as SceneEval's OOB and Support check it (geometry.BOUNDARY_INSET_M, WALL_CONTACT_M,
        SUPPORT_CENTRE_MARGIN)."""
        g = Geometry(self.condition, {**self.plan.parents, "art": "wall"}, [], hard_openings=False)

        def codes(box, placed=None):
            return {issue.code for issue in g.check_box(box, placed or {})}
        # flush with the top wall (y = 4) is out: a footprint keeps 1 mm inside the room
        self.assertIn("boundary", codes(_pose("bed", [2.1, 1.6, .5], 2.5, 2.95, 0., -math.pi / 2, "floor")))
        self.assertNotIn("boundary", codes(_pose("bed", [2.1, 1.6, .5], 2.5, 2.948, 0., -math.pi / 2, "floor")))
        # a painting 3 cm off the wall does not hang on it; snapped, it does: back 2 mm off, facing the room
        art = [.03, .8, .6]
        self.assertIn("wall_support", codes(_pose("art", art, 2., 4 - .015 - .03, 1.2, -math.pi / 2, "wall")))
        x, y, yaw = g.snap_to_wall(art, 2.1, 3.9, .3)
        hung = _pose("art", art, x, y, 1.2, yaw, "wall")
        self.assertEqual(codes(hung), set())
        self.assertAlmostEqual(g.wall_gap(hung), .002, places=4)
        self.assertAlmostEqual(x, 2.1)
        self.assertAlmostEqual(yaw, -math.pi / 2)
        # its back against the wall but facing it: not hung
        self.assertIn("wall_support", codes(_pose("art", art, x, y, 1.2, yaw + math.pi, "wall")))
        # a draft flush with the top wall, or 3 cm through it, is pushed just inside; one 20 cm through is left
        bed = [2.1, 1.6, .5]
        for y in (2.95, 2.98):
            x2, y2 = g.push_inside(bed, 2.5, y, -math.pi / 2)
            self.assertAlmostEqual(x2, 2.5)
            self.assertAlmostEqual(y2, 2.95 - .0012, places=6)
            self.assertNotIn("boundary", codes(_pose("bed", bed, x2, y2, 0., -math.pi / 2, "floor")))
        self.assertEqual(g.push_inside(bed, 2.5, 3.15, -math.pi / 2), (2.5, 3.15))
        # in a corner it moves off both walls
        x2, y2 = g.push_inside([.5, .5, .9], .25, 3.75, 0.)
        self.assertAlmostEqual(x2, .2512, places=6)
        self.assertAlmostEqual(y2, 3.7488, places=6)
        # on an object: the centre at least 1 cm inside the parent's top (the desk's spans y 0.9..2.1), overhang allowed
        desk = {"desk": self.placed["desk"]}
        self.assertNotIn("object_support", codes(_pose("lamp", [.25, .25, .45], 4.6, 2.08, .75, math.pi, "desk"), desk))
        self.assertIn("object_support", codes(_pose("lamp", [.25, .25, .45], 4.6, 2.095, .75, math.pi, "desk"), desk))

    def test_smaller_overlap_counts_as_progress(self):
        deep = {**self.placed, "chair": _pose("chair", [.5, .5, .9], 4.2, 1.5, 0., 0., "floor")}
        shallow = {**self.placed, "chair": _pose("chair", [.5, .5, .9], 4.1, 1.5, 0., 0., "floor")}
        a, b = self.critic.assess(deep), self.critic.assess(shallow)
        self.assertEqual(a.hard_count, b.hard_count)
        self.assertGreater(a.hard_magnitude, b.hard_magnitude)
        self.assertTrue(Critic.better(b, a)[0])
        self.assertFalse(Critic.better(a, b)[0])

    def test_hard_valid_layout_rejects_new_violation(self):
        clean = self.critic.assess(self.placed)
        worse = Assessment(hard=[Violation("collision", ("a", "b"), .02)], soft=[])
        self.assertFalse(Critic.better(worse, clean)[0])

    def test_door_rule_follows_the_evaluator(self):
        # the door strip is x 0.15-1.05, y 0-0.8: a chair there blocks it, a rug does not
        chair = {**self.placed, "chair": _pose("chair", [.5, .5, .9], .6, .4, 0., 0., "floor")}
        self.assertIn("door_clearance", {v.code for v in self.critic.assess(chair).hard})
        condition = copy.deepcopy(self.condition)
        condition["objects"][3]["category"] = "rug"
        g = _geometry(condition, self.plan)
        rug = {**self.placed, "chair": _pose("chair", [1., 1., .01], .6, .4, 0., 0., "floor")}
        self.assertNotIn("door_clearance", {v.code for v in Critic.for_plan(g, self.plan, templates=[]).assess(rug).hard})

    def test_templates_score_distance_continuously(self):
        templates = [{"a": "nightstand", "b": "bed", "kind": "near", "max_gap_m": .05, "rooms": 1}]
        critic = Critic.for_plan(self.g, self.plan, templates=templates)
        near = critic.assess(self.placed)
        far = critic.assess({**self.placed, "nightstand": _pose("nightstand", [.45, .45, .55], .5, 2.0, 0., 0., "floor")})
        cost = lambda a: sum(v.cost for v in a.soft if v.code == "template_near")  # noqa: E731
        self.assertGreater(cost(far), cost(near))

    def test_fixed_sizes_do_not_shrink_in_the_loop(self):
        sizes = FixedSizes({"bed": [2.1, 1.6, .5]})
        self.assertEqual(sizes.shrunk({"id": "bed", "category": "bed"}, 1), [])


class LoopTests(unittest.TestCase):
    """The loop's payload and answer parsing, which the review's fix request uses (semantic.py)."""

    def test_review_answer_respects_locked_supports_and_carries_children(self):
        condition = _room()
        plan = make_plan(condition, llm=False)
        g = _geometry(condition, plan)
        sizes = {"bed": [2.1, 1.6, .5], "nightstand": [.45, .45, .55], "desk": [.6, 1.2, .75], "chair": [.5, .5, .9],
                 "lamp": [.25, .25, .45]}
        if not hasattr(CriticTests, "placed"):
            CriticTests.setUpClass()
        placed = CriticTests.placed
        loop = Loop(g, plan, sizes, SizeCatalog(CATALOG), Critic.for_plan(g, plan, templates=[]),
                    locked_parents=_required_parents(condition))
        answer = {"objects": {"desk": {"x": 4.6, "y": 2.2, "yaw_deg": 180},
                              "lamp": {"x": 1.0, "y": 1.0, "yaw_deg": 0, "support": "floor"}}}
        new, parents = loop._apply(answer, {"desk", "lamp"}, placed)
        self.assertEqual(parents["lamp"], "desk")  # the input's support_parent is locked
        self.assertAlmostEqual(new["lamp"]["bottom_center_m"][2], .75)
        self.assertAlmostEqual(new["desk"]["bottom_center_m"][1], 2.2)
        answer = {"objects": {"desk": {"x": 4.6, "y": 2.2, "yaw_deg": 180}}}
        new, _ = loop._apply(answer, {"desk", "lamp"}, placed)  # no pose for the lamp: carried with the desk
        self.assertAlmostEqual(new["lamp"]["bottom_center_m"][1], 2.2)
        partial = {k: v for k, v in placed.items() if k != "lamp"}  # the lamp is not placed (left out)
        new, _ = loop._apply(answer, {"desk"}, partial)
        self.assertNotIn("lamp", new)
        self.assertAlmostEqual(new["desk"]["bottom_center_m"][1], 2.2)

    def test_review_request_carries_each_objects_description(self):
        condition = _room()
        plan = make_plan(condition, llm=False)
        g = _geometry(condition, plan)
        sizes = {"bed": [2.1, 1.6, .5], "nightstand": [.45, .45, .55], "desk": [.6, 1.2, .75], "chair": [.5, .5, .9],
                 "lamp": [.25, .25, .45]}
        if not hasattr(CriticTests, "placed"):
            CriticTests.setUpClass()
        loop = Loop(g, plan, sizes, SizeCatalog(CATALOG), Critic.for_plan(g, plan, templates=[]),
                    locked_parents=_required_parents(condition))
        payload = json.loads(loop._payload({"desk"}, CriticTests.placed))
        self.assertEqual({o["id"]: o["description"] for o in payload["objects"]},
                         {o["id"]: o["description"] for o in condition["objects"]})


def _box(ident, cat, x, y, sx, sy, sz, yaw):
    """A box in layout_quality.boxes' form (front: local +X)."""
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((sx / 2, sy / 2), (-sx / 2, sy / 2),
                                                                         (-sx / 2, -sy / 2), (sx / 2, -sy / 2))]
    return {"id": ident, "cat": cat, "head": cat.split()[-1], "poly": Polygon(corners), "front": np.array((c, s)),
            "side": np.array((-s, c)), "centre": np.array((x, y)), "size": (sx, sy, sz), "bottom": 0.}


class OrientationReachTests(unittest.TestCase):
    floor = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])

    def test_facing_allows_a_large_target_by_the_ray(self):
        table = _box("t", "table", 2., 2., 1., 3., .75, 0.)
        # 60 degrees off the table's centre, but the ray along the front still hits the long table
        chair = _box("c", "chair", .5, 2., .5, .5, .9, math.radians(55))
        self.assertGreater(abs(math.degrees(math.atan2(0., 1.5)) - 55), 45)
        self.assertEqual(facing_error(chair, table["poly"]), 0.)
        small = _box("s", "stool", 2., 2., .3, .3, .45, 0.)
        self.assertGreater(facing_error(chair, small["poly"]), 0.)
        # facing away: the ray must turn about 124 degrees to clip the table's far corner
        self.assertAlmostEqual(facing_error(_box("c", "chair", .5, 2., .5, .5, .9, math.pi), table["poly"]),
                               math.pi - math.atan2(1.5, 1.), places=6)

    def test_back_to_the_wall_needs_distance_and_direction(self):
        wl = walls(self.floor)
        self.assertEqual(len(wl), 4)
        # against the left wall (x = 0), facing +X into the room
        good = _box("w", "wardrobe", .3, 2., .6, 1.2, 2., 0.)
        self.assertEqual(wall_back_error(good, wl)[0], 0.)
        flipped = _box("w", "wardrobe", .3, 2., .6, 1.2, 2., math.pi)
        self.assertGreater(wall_back_error(flipped, wl)[0], .5)
        # facing away from a far wall in the middle of the room: SceneSmith's unbounded "away" ray would pass it
        middle = _box("w", "wardrobe", 2., 2., .6, 1.2, 2., 0.)
        self.assertGreater(wall_back_error(middle, wl)[0], 1.)
        self.assertGreater(wall_back_error(_box("w", "wardrobe", .3, 2., .6, 1.2, 2., math.radians(25)), wl)[0], 0.)

    def test_reach_finds_the_blocker(self):
        door = [Polygon([(1.5, 0), (2.5, 0), (2.5, .8), (1.5, .8)])]
        # a row of cabinets across the room cuts the sofa at the top off from the door at the bottom
        row = [_box(f"k{i}", "cabinet", .5 + i, 2., 1., .6, 1., math.pi / 2) for i in range(4)]
        sofa = _box("sofa", "sofa", 2., 3.5, .9, 2., .9, -math.pi / 2)
        missed = unreachable(self.floor, row + [sofa], [], door, .2, categories={"sofa"})
        self.assertEqual([m[0] for m in missed], ["sofa"])
        self.assertTrue(set(missed[0][2]) <= {f"k{i}" for i in range(4)} and missed[0][2])
        gap = [_box(f"k{i}", "cabinet", .5 + i, 2., 1., .6, 1., math.pi / 2) for i in (0, 3)]
        self.assertEqual(unreachable(self.floor, gap + [sofa], [], door, .2, categories={"sofa"}), [])

    def test_a_coffee_table_does_not_cut_off_its_sofa(self):
        door = [Polygon([(1.5, 0), (2.5, 0), (2.5, .8), (1.5, .8)])]
        sofa = _box("sofa", "sofa", 2., 3.5, .9, 2., .9, -math.pi / 2)
        table = _box("ct", "coffee table", 2., 2.7, .5, 1.6, .45, 0.)  # 0.3 m in front of the sofa
        self.assertEqual(unreachable(self.floor, [sofa, table], [], door, .2, categories={"sofa"}), [])

    def test_critic_flags_a_bed_turned_to_face_the_wall(self):
        condition = _room()
        plan = make_plan(condition, llm=False)
        critic = Critic.for_plan(_geometry(condition, plan), plan, templates=[],
                                 wall_rules={"against_wall": {"bed"}, "reach": set(), "reach_half_width_m": .2})
        if not hasattr(CriticTests, "placed"):
            CriticTests.setUpClass()
        placed = CriticTests.placed
        codes = [v.code for v in critic.assess(placed).soft]
        self.assertNotIn("wall_back", codes)
        bed = placed["bed"]
        flipped = {**placed, "bed": {**bed, "yaw_rad": bed["yaw_rad"] + math.pi}}
        self.assertIn("wall_back", [v.code for v in critic.assess(flipped).soft])


if __name__ == "__main__":
    unittest.main()
