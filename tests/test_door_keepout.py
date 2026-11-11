"""The critic's door rule "keepout" (infinite-task-world's entry keepout, 2026-11-05) and wall objects over windows."""

from __future__ import annotations

import math
import unittest

from workflow.contract import check_request, fill_room_defaults
from workflow.critic import Critic
from workflow.drafting import DRAFT_INSTRUCTIONS
from workflow.geometry import Geometry
from workflow.critic import critic_geometry
from workflow.planning import Plan
from workflow.solver import _pose

UP, DOWN = math.pi / 2, -math.pi / 2


def _condition(objects):
    """A 4 x 3 m room: a 0.9 m door on the bottom wall centred at x = 2 (2.1 m tall, keepout x 1.4-2.6, y 0-1),
    a 1.2 m window on the right wall centred at y = 1.5 (sill 0.9 m, top 2.1 m)."""
    return check_request(require_descriptions=False, condition=fill_room_defaults({
        "schema_version": "fastfill.v2",
        "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 3], [0, 3]],
                 "floor_z_m": 0, "height_m": 2.8,
                 "openings": [{"type": "door", "wall_index": 0, "position_on_wall": .5, "width_m": .9, "height_m": 2.1},
                              {"type": "window", "wall_index": 1, "position_on_wall": .5, "width_m": 1.2,
                               "height_m": 1.2, "sill_height_m": .9}],
                 "fixed_objects": []},
        "objects": [{"id": i, "category": c, "description": c} for i, c, _ in objects],
        "constraints": []})[0])


def _issues(boxes, parents, door_rule="keepout", wall_windows=True):
    condition = _condition([(b["id"], b["id"], None) for b in boxes])
    g = Geometry(condition, parents, [], hard_doors=True, hard_windows=False, door_rule=door_rule,
                 wall_windows=wall_windows)
    return {(i.code, i.ids[0]) for i in g.validate({b["id"]: b for b in boxes})}


class KeepoutTests(unittest.TestCase):
    def test_a_sconce_hung_in_the_doorway_blocks_it(self):
        sconce = _pose("sconce", [.23, .43, .37], 1.9, .115, 1.15, UP, "wall")
        self.assertIn(("door_clearance", "sconce"), _issues([sconce], {"sconce": "wall"}))

    def test_the_old_evaluator_rule_let_it_through(self):  # the bug the keepout rule fixes
        sconce = _pose("sconce", [.23, .43, .37], 1.9, .115, 1.15, UP, "wall")
        self.assertNotIn(("door_clearance", "sconce"), _issues([sconce], {"sconce": "wall"}, door_rule="evaluator"))

    def test_height_does_not_matter_a_clock_above_the_door_blocks_it(self):
        clock = _pose("clock", [.05, .3, .3], 2., .025, 2.3, UP, "wall")
        self.assertIn(("door_clearance", "clock"), _issues([clock], {"clock": "wall"}))

    def test_a_rug_in_front_of_the_door_is_allowed(self):
        rug = _pose("rug", [.6, 1., .02], 2., .4, 0., UP, "floor")
        self.assertNotIn(("door_clearance", "rug"), _issues([rug], {"rug": "floor"}))

    def test_a_thin_object_on_furniture_is_not_a_floor_covering(self):
        table = _pose("table", [.6, .6, .75], 2., .6, 0., 0., "floor")
        paper = _pose("paper", [.2, .3, .01], 2., .6, .75, 0., "table")
        found = _issues([table, paper], {"table": "floor", "paper": "table"})
        self.assertIn(("door_clearance", "table"), found)
        self.assertIn(("door_clearance", "paper"), found)

    def test_the_side_margin_counts_past_the_door_edge(self):
        # door edge at x = 2.45; the keepout runs to 2.6. A cabinet from x 2.55 to 2.95 is 5 cm in.
        cabinet = _pose("cabinet", [.5, .4, .8], 2.75, .26, 0., UP, "floor")
        self.assertIn(("door_clearance", "cabinet"), _issues([cabinet], {"cabinet": "floor"}))

    def test_touching_within_the_slack_is_allowed(self):
        cabinet = _pose("cabinet", [.5, .4, .8], 2.795, .26, 0., UP, "floor")  # x 2.595 to 2.995: 5 mm in
        self.assertNotIn(("door_clearance", "cabinet"), _issues([cabinet], {"cabinet": "floor"}))

    def test_one_metre_deep(self):
        near = _pose("chair", [.4, .4, .9], 2., 1.15, 0., UP, "floor")  # 0.95 to 1.35: 5 cm in
        far = _pose("stool", [.4, .4, .5], 1., 1.25, 0., UP, "floor")  # beside the keepout
        found = _issues([near, far], {"chair": "floor", "stool": "floor"})
        self.assertIn(("door_clearance", "chair"), found)
        self.assertNotIn(("door_clearance", "stool"), found)


class WallWindowTests(unittest.TestCase):
    def test_a_painting_over_the_window_is_hard(self):
        painting = _pose("painting", [.05, .6, .5], 3.975, 1.5, 1.2, math.pi, "wall")
        self.assertIn(("window_clearance", "painting"), _issues([painting], {"painting": "wall"}))

    def test_a_painting_above_the_window_is_fine(self):
        painting = _pose("painting", [.05, .6, .4], 3.975, 1.5, 2.2, math.pi, "wall")
        self.assertNotIn(("window_clearance", "painting"), _issues([painting], {"painting": "wall"}))

    def test_a_floor_object_in_front_of_the_window_stays_soft(self):
        shelf = _pose("shelf", [.4, 1., 1.5], 3.8, 1.5, 0., math.pi, "floor")  # its top above the sill
        self.assertNotIn(("window_clearance", "shelf"), _issues([shelf], {"shelf": "floor"}))
        condition = _condition([("shelf", "shelf", None)])
        g = critic_geometry(condition, Plan(parents={"shelf": "floor"}))
        critic = Critic(g, [], [], [])
        self.assertIn("window", {v.code for v in critic.soft({"shelf": shelf})})

    def test_the_hard_one_is_not_counted_again_as_soft(self):
        painting = _pose("painting", [.05, .6, .5], 3.975, 1.5, 1.2, math.pi, "wall")
        condition = _condition([("painting", "painting", None)])
        g = critic_geometry(condition, Plan(parents={"painting": "wall"}))
        critic = Critic(g, [], [], [])
        self.assertIn("window_clearance", {v.code for v in critic.hard({"painting": painting})})
        self.assertNotIn("window", {v.code for v in critic.soft({"painting": painting})})


class WiringTests(unittest.TestCase):
    def test_the_critic_uses_the_keepout(self):
        g = critic_geometry(_condition([("a", "a", None)]), Plan(parents={"a": "floor"}))
        self.assertEqual((g.door_rule, g.wall_windows), ("keepout", True))
        self.assertAlmostEqual(g.keepouts[0][0].area, 1.2 * 1.)
        self.assertAlmostEqual(g.doors[0][0].area, .9 * .8)  # the robot's door strips are unchanged

    def test_the_draft_prompt_states_the_rule(self):
        self.assertIn("1.0 m in front of a door, from 0.15 m left of it to 0.15 m right of it", DRAFT_INSTRUCTIONS)
        self.assertIn("never over a window", DRAFT_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
