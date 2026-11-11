"""④e (``simple.clear_wall_fronts``, 2026-11-07): a wall object over furniture slides along the walls to where the floor
in front of it is clear (SceneEval's accessibility strip), never into a violation; what stands on it moves with it."""

from __future__ import annotations

import math
import unittest

from shapely.ops import unary_union

from test_door_keepout import _condition  # a 4 x 3 m room, a door on the bottom wall, a window on the right one
from workflow.critic import Critic
from workflow.geometry import WALL_GAP_M, footprint
from workflow.critic import critic_geometry
from workflow.planning import Plan
from workflow.simple import (WALL_FRONT_NEED, WALL_FRONT_TARGET, clear_wall_fronts, front_free, front_strip)
from workflow.solver import _pose

UP, DOWN = math.pi / 2, -math.pi / 2
TOP = 3 - WALL_GAP_M  # the top wall's face (y = 3), what a wall object's back touches
SIDEBOARD = [.45, 1.2, .8]
PAINTING = [.05, .8, .6]


def _run(boxes):
    parents = {b["id"]: b["support_parent"] for b in boxes}
    plan = Plan(parents=parents)
    g = critic_geometry(_condition([(b["id"], b["id"], None) for b in boxes]), plan)
    critic = Critic.for_plan(g, plan, templates=[])
    placed = {b["id"]: b for b in boxes}
    after, report = clear_wall_fronts(critic, plan, placed)
    return g, critic, placed, after, report


def _free(g, layout, ident):
    others = [footprint(b) for i, b in layout.items() if i != ident and b["bottom_center_m"][2] < 2]
    return front_free(layout[ident], unary_union(others), g.polygon)


class FrontStripTests(unittest.TestCase):
    def test_the_strip_lies_in_front_as_sceneeval_draws_it(self):
        painting = _pose("painting", PAINTING, 1., TOP - .025, 1.3, DOWN, "wall")
        x0, y0, x1, y1 = front_strip(painting).bounds  # 0.05 m off its front, 0.2 m deep, as wide as it
        self.assertAlmostEqual(y1, TOP - .05 - .05, places=6)
        self.assertAlmostEqual(y1 - y0, .2, places=6)
        self.assertAlmostEqual(x1 - x0, .8, places=6)


class WallClearTests(unittest.TestCase):
    def test_a_painting_over_a_sideboard_slides_along_its_wall(self):
        sideboard = _pose("sideboard", SIDEBOARD, 1., 2.77, 0., DOWN, "floor")
        painting = _pose("painting", PAINTING, 1., TOP - .025, 1.3, DOWN, "wall")
        g, critic, placed, after, report = _run([sideboard, painting])
        self.assertLess(_free(g, placed, "painting"), WALL_FRONT_NEED)
        self.assertEqual([m["id"] for m in report["moved"]], ["painting"])
        self.assertFalse(report["moved"][0]["other_wall"])
        moved = after["painting"]
        # the nearest place with 0.65 of the strip free: 0.25 m of its 0.8 m still over the sideboard (to x = 1.6)
        self.assertAlmostEqual(moved["bottom_center_m"][0], 1.75, places=3)
        self.assertEqual(moved["bottom_center_m"][1:], painting["bottom_center_m"][1:])  # same wall, same height
        self.assertAlmostEqual(moved["yaw_rad"], painting["yaw_rad"])
        self.assertGreaterEqual(_free(g, after, "painting"), WALL_FRONT_TARGET)
        self.assertEqual(after["sideboard"], sideboard)  # nothing on the floor moves
        self.assertFalse(critic.hard(after))

    def test_a_painting_with_its_front_free_stays(self):
        painting = _pose("painting", PAINTING, 3., TOP - .025, 1.3, DOWN, "wall")
        _g, _critic, placed, after, report = _run([painting])
        self.assertEqual((report["wall_objects"], report["below_need"], report["moved"]), (1, 0, []))
        self.assertEqual(after, placed)

    def test_a_wall_covered_end_to_end_sends_it_to_another_wall(self):
        cabinet = _pose("cabinet", [.45, 2.98, .8], .23, 1.5, 0., 0., "floor")  # the whole left wall
        painting = _pose("painting", PAINTING, WALL_GAP_M + .025, 1.5, 1.3, 0., "wall")
        g, critic, _placed, after, report = _run([cabinet, painting])
        self.assertEqual([m["id"] for m in report["moved"]], ["painting"])
        self.assertTrue(report["moved"][0]["other_wall"])
        self.assertGreaterEqual(_free(g, after, "painting"), WALL_FRONT_TARGET)
        self.assertNotAlmostEqual(after["painting"]["yaw_rad"], 0.)  # it faces the room from its new wall
        self.assertFalse(critic.hard(after))

    def test_never_into_the_door_keepout(self):
        # the bottom wall: sideboard x 0.2-1.4, the door's keepout x 1.4-2.6; the first clear place right of the
        # sideboard that keeps out of the keepout is x = 2.9
        sideboard = _pose("sideboard", SIDEBOARD, .8, .23, 0., UP, "floor")
        painting = _pose("painting", [.05, .6, .6], .8, WALL_GAP_M + .025, 1.3, UP, "wall")
        g, critic, _placed, after, report = _run([sideboard, painting])
        self.assertEqual([m["id"] for m in report["moved"]], ["painting"])
        keepout = g.keepouts[0][0]
        self.assertLess(footprint(after["painting"]).intersection(keepout).area, 1e-6)
        self.assertAlmostEqual(after["painting"]["bottom_center_m"][0], 2.9, places=3)
        self.assertFalse(critic.hard(after))

    def test_what_stands_on_a_wall_shelf_moves_with_it(self):
        sideboard = _pose("sideboard", SIDEBOARD, 1., 2.77, 0., DOWN, "floor")
        shelf = _pose("shelf", [.25, .8, .03], 1., TOP - .125, 1.4, DOWN, "wall")
        book = _pose("book", [.2, .05, .25], 1.1, TOP - .125, 1.43, DOWN, "shelf")
        g, critic, _placed, after, report = _run([sideboard, shelf, book])
        self.assertEqual([(m["id"], m["with"]) for m in report["moved"]], [("shelf", ["book"])])
        dx = after["shelf"]["bottom_center_m"][0] - shelf["bottom_center_m"][0]
        self.assertGreater(dx, .5)
        self.assertAlmostEqual(after["book"]["bottom_center_m"][0] - book["bottom_center_m"][0], dx, places=3)
        self.assertEqual(after["book"]["bottom_center_m"][1:], book["bottom_center_m"][1:])
        self.assertGreaterEqual(_free(g, after, "shelf"), WALL_FRONT_TARGET)
        self.assertFalse(critic.hard(after))  # the book is still on the shelf


if __name__ == "__main__":
    unittest.main()
