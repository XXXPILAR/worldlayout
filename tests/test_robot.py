from __future__ import annotations

import math
import unittest

from workflow.contract import check_request, fill_room_defaults
from workflow.critic import Critic
from workflow.geometry import Geometry
from workflow.critic import Violation, identity
from workflow.drafting import robot_marks, robot_rule
from workflow.planning import Plan
from workflow.robot import Robot, RobotChecker, RobotZones, infer_parents, openable
from workflow.solver import _pose
from fastfill.v2.ops import layout_quality as lq

UP, DOWN = math.pi / 2, -math.pi / 2


def _condition(objects):
    """A 4 x 3 m room, one door on the bottom wall (left) and one on the top wall (left too)."""
    return check_request(require_descriptions=False, condition=fill_room_defaults({
        "schema_version": "fastfill.v2",
        "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 3], [0, 3]],
                 "floor_z_m": 0, "height_m": 2.8,
                 "openings": [{"type": "door", "wall_index": 0, "position_on_wall": .2, "width_m": .9, "height_m": 2.1},
                              {"type": "door", "wall_index": 2, "position_on_wall": .8, "width_m": .9, "height_m": 2.1}],
                 "fixed_objects": []},
        "objects": [{"id": i, "category": c, "description": c} for i, c in objects],
        "constraints": []})[0])


def _layout(*boxes):
    return {"objects": list(boxes)}


def _wall_to(x_right: float) -> dict:
    """A sofa across the room's middle from the left wall to ``x_right``."""
    return _pose("sofa", [.9, x_right, .8], x_right / 2, 1.5, 0., UP, "floor")


class DoorLinkTests(unittest.TestCase):
    def test_furniture_across_the_room_cuts_the_doors_apart(self):
        condition = _condition([("sofa", "sofa")])
        report = RobotChecker(Robot(width_m=.5), condition).report(_layout(_wall_to(4.)), {"sofa": "floor"})
        self.assertEqual(report["doors_unlinked"], 1)
        violations = RobotChecker(Robot(width_m=.5), condition).violations({"sofa": _wall_to(4.)}, {"sofa": "floor"})
        door = [v for v in violations if v.code == "robot_door"]
        self.assertEqual(door[0].ids, ("sofa",))  # what the repair moves

    def test_the_gap_left_decides_by_the_robot_width(self):
        condition = _condition([("sofa", "sofa")])
        layout = _layout(_wall_to(3.45))  # 0.55 m left at the right wall
        self.assertTrue(RobotChecker(Robot(width_m=.5), condition).report(layout, {"sofa": "floor"})["ok"])
        self.assertEqual(RobotChecker(Robot(width_m=.6), condition).report(layout, {"sofa": "floor"})["doors_unlinked"], 1)


class OpenableTests(unittest.TestCase):
    def test_categories(self):
        condition = _condition([("w", "wardrobe"), ("n", "bedside table"), ("c", "chair"), ("t", "dining table")])
        boxes = lq.boxes({"condition": condition}, _layout(
            _pose("w", [.6, 1.2, 2.], 1, 1, 0, 0, "floor"), _pose("n", [.4, .4, .5], 2, 1, 0, 0, "floor"),
            _pose("c", [.5, .5, .9], 3, 1, 0, 0, "floor"), _pose("t", [.9, 1.6, .75], 2, 2, 0, 0, "floor")))
        self.assertEqual([openable(b) for b in boxes], [True, True, False, False])

    def test_facing_the_wall_or_a_chair_in_front_blocks_it(self):
        condition = _condition([("w", "wardrobe"), ("c", "chair")])
        checker = RobotChecker(Robot(), condition)
        to_wall = _pose("w", [.6, 1.2, 2.], 2, 2.65, 0, UP, "floor")  # back to the room, doors to the top wall
        into_room = _pose("w", [.6, 1.2, 2.], 2, 2.65, 0, DOWN, "floor")
        chair = _pose("c", [.5, .5, .9], 2, 2.05, 0, UP, "floor")
        away = _pose("c", [.5, .5, .9], 3.5, .8, 0, UP, "floor")
        parents = {"w": "floor", "c": "floor"}
        blocked = {v.code: v for v in checker.violations({"w": to_wall, "c": away}, parents)}
        self.assertIn("the wall", blocked["robot_openable"].detail)
        blocked = {v.code: v for v in checker.violations({"w": into_room, "c": chair}, parents)}
        self.assertEqual(blocked["robot_openable"].ids, ("w", "c"))
        self.assertNotIn("robot_openable", {v.code for v in checker.violations({"w": into_room, "c": away}, parents)})


class ReachTests(unittest.TestCase):
    def test_a_cabinet_carrying_a_cup_must_be_reached(self):
        condition = _condition([("cab", "low cabinet"), ("cup", "cup"), ("sofa", "sofa")])
        cab = _pose("cab", [.45, .8, .8], 3.6, 2.4, 0, math.pi, "floor")
        cup = _pose("cup", [.1, .1, .12], 3.6, 2.4, .8, 0, "cab")
        # the sofa closes the top right corner off: from (2.4, 1.6) to the right wall, and up to the top wall
        sofa = _pose("sofa", [.9, 1.6, .8], 2.85, 2.2, 0, 0, "floor")
        bs = lq.boxes({"condition": condition}, _layout(cab, cup, sofa))
        self.assertEqual(infer_parents(bs), {"cab": "floor", "cup": "cab", "sofa": "floor"})
        checker = RobotChecker(Robot(reach_openable=False), condition)
        self.assertEqual(checker.required(bs, {"cab": "floor", "cup": "cab", "sofa": "floor"}), {"cab": "carries cup"})
        reach = [v for v in checker.violations({"cab": cab, "cup": cup, "sofa": sofa},
                                               {"cab": "floor", "cup": "cab", "sofa": "floor"}) if v.code == "robot_reach"]
        self.assertEqual(reach[0].ids, ("cab", "sofa"))
        far_sofa = _pose("sofa", [.9, 1.6, .8], 1.2, .9, 0, UP, "floor")
        self.assertFalse([v for v in checker.violations({"cab": cab, "cup": cup, "sofa": far_sofa},
                                                       {"cab": "floor", "cup": "cab", "sofa": "floor"})
                          if v.code == "robot_reach"])

    def test_task_objects_stand_for_the_furniture_under_them(self):
        condition = _condition([("desk", "desk"), ("book", "book")])
        desk = _pose("desk", [.6, 1.2, .75], 2, 1.5, 0, 0, "floor")
        book = _pose("book", [.2, .15, .03], 2, 1.5, .75, 0, "desk")
        bs = lq.boxes({"condition": condition}, _layout(desk, book))
        checker = RobotChecker(Robot(reach_ids=frozenset({"book"}), reach_supports=False), condition)
        self.assertEqual(checker.required(bs, {"desk": "floor", "book": "desk"}), {"desk": "carries task object book"})


class ArmTests(unittest.TestCase):
    def test_an_object_deep_on_a_counter_is_out_of_the_arms_reach(self):
        condition = _condition([("counter", "counter"), ("cup", "cup")])
        counter = _pose("counter", [1.2, 2., .9], 2.4, 2.4, 0, DOWN, "floor")  # 1.2 m deep, back to the top wall
        parents = {"counter": "floor", "cup": "counter"}
        checker = RobotChecker(Robot(), condition)
        back = _pose("cup", [.1, .1, .12], 2.4, 2.9, .9, 0, "counter")    # 1.1 m behind the front edge
        front = _pose("cup", [.1, .1, .12], 2.4, 2.0, .9, 0, "counter")   # 0.2 m behind it
        arm = [v for v in checker.violations({"counter": counter, "cup": back}, parents) if v.code == "robot_arm"]
        self.assertEqual(arm[0].ids, ("cup", "counter"))
        self.assertFalse([v for v in checker.violations({"counter": counter, "cup": front}, parents)
                          if v.code == "robot_arm"])
        report = checker.report(_layout(counter, back), parents)
        self.assertEqual((report["items"], report["items_unreached"]), (1, 1))


class CriticTests(unittest.TestCase):
    def test_robot_checks_are_hard_only_with_a_robot(self):
        condition = _condition([("sofa", "sofa")])
        critic = Critic(Geometry(condition, {"sofa": "floor"}, []), [], [], [], {"against_wall": set(), "reach": set(),
                                                                               "reach_half_width_m": .2})
        placed = {"sofa": _wall_to(4.)}
        self.assertFalse([v for v in critic.assess(placed).hard if v.code.startswith("robot_")])
        critic.robot = RobotChecker.for_critic(Robot(), critic)
        self.assertIn("robot_door", {v.code for v in critic.assess(placed).hard})


class ZoneTests(unittest.TestCase):
    """The search's form of the checks (Geometry.soft_score)."""

    def test_a_chair_in_front_of_a_wardrobe_costs_in_the_soft_score(self):
        condition = _condition([("w", "wardrobe"), ("c", "chair")])
        geometry = Geometry(condition, {"w": "floor", "c": "floor"}, [])
        w = _pose("w", [.6, 1.2, 2.], 2, 2.65, 0, DOWN, "floor")
        near, away = _pose("c", [.5, .5, .9], 2, 2.05, 0, UP, "floor"), _pose("c", [.5, .5, .9], 3.5, .8, 0, UP, "floor")
        self.assertEqual(geometry.soft_score({"w": w, "c": near})[0], geometry.soft_score({"w": w, "c": away})[0])
        geometry.robot_zones = RobotZones(Robot(), geometry)
        issues = geometry.soft_score({"w": w, "c": near})[1]
        self.assertEqual([(i.code, i.ids) for i in issues], [("robot_openable", ("w", "c"))])
        self.assertFalse(geometry.soft_score({"w": w, "c": away})[1])
        to_wall = _pose("w", [.6, 1.2, 2.], 2, 2.65, 0, UP, "floor")
        self.assertEqual([i.code for i in geometry.soft_score({"w": to_wall, "c": away})[1]], ["robot_openable"])

    def test_a_cup_deep_on_a_counter_and_a_boxed_in_shelf(self):
        condition = _condition([("counter", "counter"), ("cup", "cup")])
        parents = {"counter": "floor", "cup": "counter"}
        geometry = Geometry(condition, parents, [])
        geometry.robot_zones = RobotZones(Robot(), geometry)
        counter = _pose("counter", [1.2, 2., .9], 2.4, 2.4, 0, DOWN, "floor")
        back = _pose("cup", [.1, .1, .12], 2.4, 2.9, .9, 0, "counter")
        front = _pose("cup", [.1, .1, .12], 2.4, 2.0, .9, 0, "counter")
        self.assertEqual([(i.code, i.ids) for i in geometry.soft_score({"counter": counter, "cup": back})[1]],
                         [("robot_arm", ("cup", "counter"))])
        self.assertFalse(geometry.soft_score({"counter": counter, "cup": front})[1])
        # a bookshelf is used from its front only: a sofa there leaves no place to stand
        condition = _condition([("shelf", "bookshelf"), ("book", "book"), ("sofa", "sofa")])
        geometry = Geometry(condition, {"shelf": "floor", "book": "shelf", "sofa": "floor"}, [])
        geometry.robot_zones = RobotZones(Robot(), geometry)
        shelf = _pose("shelf", [.4, 2., 1.8], 2.4, 2.8, 0, DOWN, "floor")
        book = _pose("book", [.2, .15, .03], 2.4, 2.8, 1.8, 0, "shelf")
        sofa = _pose("sofa", [.9, 2., .8], 2.4, 2.1, 0, DOWN, "floor")
        codes = [(i.code, i.ids) for i in geometry.soft_score({"shelf": shelf, "book": book, "sofa": sofa})[1]]
        self.assertEqual(codes, [("robot_reach", ("shelf", "sofa"))])

    def test_legality_comes_before_the_robot_checks(self):
        from workflow.critic import Assessment
        collision, reach = Violation("collision", ("a", "b"), .2), Violation("robot_reach", ("c",), .5)
        # fixing a collision is kept even when it costs a robot check
        self.assertTrue(Critic.better(Assessment([reach]), Assessment([collision]))[0])
        # a robot repair is not bought with a collision
        self.assertFalse(Critic.better(Assessment([collision]), Assessment([reach]))[0])
        # at equal legality the robot tier decides, then the soft cost
        self.assertTrue(Critic.better(Assessment([collision]), Assessment([collision, reach]))[0])
        self.assertFalse(Critic.better(Assessment([collision, reach]), Assessment([collision]))[0])

    def test_robot_violations_keep_their_identity_when_the_blockers_change(self):
        a = Violation("robot_reach", ("cab", "sofa"), .3)
        b = Violation("robot_reach", ("cab", "table"), .2)
        self.assertEqual(identity(a), identity(b))
        self.assertNotEqual(identity(Violation("collision", ("a", "b"), .1)), identity(Violation("collision", ("a", "c"), .1)))
        self.assertEqual(identity(Violation("robot_door", ("x",), 1., detail="door 1 sealed")),
                         identity(Violation("robot_door", ("y",), .3, detail="door 1")))


class PlannedApproachTests(unittest.TestCase):
    """RobotZones.reserve fixes the approach side; with ``hard`` the search filters on the zones."""

    def setUp(self):
        condition = _condition([("t", "dining table"), ("cup", "cup"), ("sofa", "sofa")])
        self.geometry = Geometry(condition, {"t": "floor", "cup": "t", "sofa": "floor"}, [])
        self.table = _pose("t", [.8, 1.2, .75], 2, 1.5, 0, 0, "floor")    # front: +x
        self.cup = _pose("cup", [.1, .1, .1], 2, 1.5, .75, 0, "t")
        self.on_front = _pose("sofa", [.5, 1.2, .8], 2.7, 1.5, 0, 0, "floor")
        self.on_back = _pose("sofa", [.5, 1.2, .8], 1.3, 1.5, 0, 0, "floor")

    def test_the_draft_fixes_the_most_open_side(self):
        zones = RobotZones(Robot(), self.geometry)
        self.geometry.robot_zones = zones
        placed = {"t": self.table, "cup": self.cup, "sofa": self.on_front}
        self.assertEqual(zones.reserve(placed), {"t": 1})   # the front is covered: the back
        # once fixed, a sofa on that side blocks the robot although the front is free again
        codes = [i.code for i in zones.issues({"t": self.table, "cup": self.cup, "sofa": self.on_back})]
        self.assertIn("robot_reach", codes)
        self.assertNotIn("robot_reach", [i.code for i in zones.issues(placed)])
        zones.reserved.clear()  # without the plan any free side will do
        self.assertNotIn("robot_reach", [i.code for i in zones.issues({"t": self.table, "cup": self.cup,
                                                                        "sofa": self.on_back})])

    def test_hard_zones_filter_the_search_candidates(self):
        from collections import Counter
        from workflow.catalog import SizeCatalog
        from workflow.solver import Candidate, CandidateGenerator
        zones = RobotZones(Robot(), self.geometry, hard=True)
        self.geometry.robot_zones = zones
        zones.reserve({"t": self.table, "cup": self.cup, "sofa": self.on_front})
        gen = CandidateGenerator(self.geometry, SizeCatalog(None), Plan(dict(self.geometry.parents)), "local", 1)
        away = _pose("sofa", [.5, 1.2, .8], 3.6, 1.5, 0, 0, "floor")
        cands = [Candidate(self.on_back, 1., "a"), Candidate(away, 0., "b")]
        blocked = Counter()
        kept = gen._zone_filter("sofa", {"t": self.table, "cup": self.cup}, cands, blocked)
        self.assertEqual([c.origin for c in kept], ["b"])
        self.assertEqual(blocked["robot_zone"], 1)
        # when every candidate blocks, none is dropped: the object must still be placed
        self.assertEqual(len(gen._zone_filter("sofa", {"t": self.table, "cup": self.cup}, cands[:1], Counter())), 1)
        zones.hard = False
        self.assertEqual(len(gen._zone_filter("sofa", {"t": self.table, "cup": self.cup}, cands, Counter())), 2)


class DraftTests(unittest.TestCase):
    def test_the_request_marks_what_the_robot_needs(self):
        condition = _condition([("w", "wardrobe"), ("t", "dining table"), ("cup", "cup"), ("c", "chair")])
        plan = Plan({"w": "floor", "t": "floor", "cup": "t", "c": "floor"})
        sizes = {"w": [.6, 1.2, 2.], "t": [.9, 1.6, .75], "cup": [.1, .1, .1], "c": [.5, .5, .9]}
        marks = robot_marks(condition, plan, sizes, Robot(width_m=.5))
        self.assertEqual(marks, {"w": {"front_clearance_m": .6}, "t": {"robot_works_here": True}})
        self.assertIn("0.5 m wide", robot_rule(Robot(width_m=.5)))
        self.assertEqual(robot_rule(None), "")


class FallbackTests(unittest.TestCase):
    def test_no_object_is_ever_shrunk(self):
        from workflow.loop import FixedSizes
        request = {"id": "bed", "category": "king size bed"}
        fixed = FixedSizes({"bed": [2.2, 2.0, 1.0]})
        self.assertFalse(fixed.allow_shrink)
        self.assertEqual([fixed.shrunk(request, level) for level in (1, 2, 3, 4)], [[]] * 4)

    def test_with_a_robot_an_object_on_a_large_table_gets_spots_along_its_edges(self):
        from workflow.loop import FixedSizes
        from workflow.solver import CandidateGenerator
        condition = _condition([("t", "dining table"), ("cup", "cup")])
        plan = Plan({"t": "floor", "cup": "t"})
        sizes = {"t": [1.2, 2.4, .75], "cup": [.1, .1, .1]}
        geometry = Geometry(condition, plan.parents, [])
        table = _pose("t", sizes["t"], 2, 1.5, 0, 0, "floor")
        spots = lambda: {tuple(c.box["bottom_center_m"][:2]) for c in CandidateGenerator(
            geometry, FixedSizes(sizes), plan, "local", 1)._on_parent("cup", {"t": table})}
        without = spots()
        geometry.robot_zones = RobotZones(Robot(), geometry)
        edges = spots() - without
        self.assertIn((2.0, 2.6), edges)  # along y: 1.5 + (1.2 - half the cup - 5 cm)
        self.assertIn((2.5, 1.5), edges)  # along x: 2 + (0.6 - half the cup - 5 cm)


if __name__ == "__main__":
    unittest.main()


class RelocateTests(unittest.TestCase):
    def test_movers_are_the_blockers_then_the_furniture(self):
        from workflow.catalog import SizeCatalog
        from workflow.loop import Loop
        condition = _condition([("w", "wardrobe"), ("sofa", "sofa"), ("plant", "plant"), ("lamp", "lamp")])
        parents = {"w": "floor", "sofa": "floor", "plant": "floor", "lamp": "sofa"}
        geometry = Geometry(condition, parents, [])
        loop = Loop(geometry, Plan(parents), {}, SizeCatalog(None), None, locked_parents={}, relocate=True)
        reach = Violation("robot_reach", ("w", "sofa", "lamp", "plant"), .5)
        self.assertEqual(loop.movers(reach), ["sofa", "plant", "w"])   # the lamp stands on the sofa: it moves with it
        self.assertEqual(loop.movers(Violation("robot_door", ("plant", "sofa"), .3)), ["plant", "sofa"])
        self.assertEqual(loop.movers(Violation("robot_arm", ("lamp", "sofa"), .2)), [])
        self.assertEqual(loop.movers(Violation("collision", ("w", "sofa"), .2)), [])
