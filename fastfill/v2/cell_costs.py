"""Per-object costs of the position grid cells, for the expected-geometry loss.

For every object and every cell of the ``grid x grid`` position head (row-major, x index major, over the room's
normalized bounding box, like ``geometry.encode_grid_position``), the cost is the least violation the object's
upright footprint can reach with its bottom centre somewhere in that cell: the fraction of the footprint outside
the floor polygon, plus, for objects standing on the floor, the fraction inside a door's clearance zone. The
least value over ``sub x sub`` positions per cell (borders included) is taken so that a cell holding a wall-flush placement costs 0
although the cell centre itself would push the footprint through the wall.

Floor objects at most ``THIN_HEIGHT_M`` tall (carpets, mats) never block doors. Footprints use the label size and yaw (the cost depends on the position distribution only, so the size head is
never pushed to shrink objects); objects without a valid XY size or yaw get no cost row. The footprint is
sampled at ``foot x foot`` points after shrinking each side by ``shrink_m`` and looked up in a raster of the room
(``resolution_m`` pixels), so contact with a wall within the shrink margin is free.

A door's clearance zone is the rectangle on the room side of its wall, as wide as the door and
``door_depth_m`` deep, clipped to the floor polygon. Only wall-relative openings
(``batch.wall_relative_opening``) of type door define zones; windows never do.
"""
from __future__ import annotations

import math

import numpy as np

from fastfill.v2.schema import normalize_room

FLOOR_TOLERANCE_M = .05  # bottom within this of the floor counts as standing on the floor
THIN_HEIGHT_M = .05  # floor objects at most this tall (carpets, mats) never block a door
DEFAULTS = {"grid": 16, "door_depth_m": .8, "resolution_m": .04, "shrink_m": .03, "sub": 3, "foot": 4,
            "drop_label_conflicts": True}


def cell_cost_options(options: dict | None) -> dict:
    options = {**DEFAULTS, **(options or {})}
    if set(options) - set(DEFAULTS):
        raise ValueError(f"unknown cell cost options: {sorted(set(options) - set(DEFAULTS))}")
    for key in ("grid", "sub", "foot"):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"cell cost {key} must be a positive integer")
    if not isinstance(options["drop_label_conflicts"], bool):
        raise ValueError("cell cost drop_label_conflicts must be a boolean")
    for key in ("door_depth_m", "resolution_m", "shrink_m"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"cell cost {key} must be a finite nonnegative number")
    if options["resolution_m"] <= 0:
        raise ValueError("cell cost resolution_m must be positive")
    return options


def _signed_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2)


def door_zones(room: dict, depth_m: float) -> list[np.ndarray]:
    """Clearance rectangles (4 x 2 corner arrays, room side) of the room's wall-relative doors."""
    from fastfill.v2.batch import wall_relative_opening
    polygon = np.asarray(room["floor_polygon_xy_m"], dtype=np.float64)
    inward = 1. if _signed_area(polygon) > 0 else -1.  # counter-clockwise: the interior is left of each edge
    zones = []
    for opening in room.get("openings") or []:
        if not wall_relative_opening(opening) or opening["type"] != "door" or depth_m <= 0:
            continue
        k = opening["wall_index"]
        if k >= len(polygon):
            continue
        a, b = polygon[k], polygon[(k + 1) % len(polygon)]
        length = float(np.hypot(*(b - a)))
        if length <= 1e-9:
            continue
        u = (b - a) / length
        v = inward * np.array([-u[1], u[0]])
        centre = opening["position_on_wall"] * length
        s0, s1 = max(centre - opening["width_m"] / 2, 0.), min(centre + opening["width_m"] / 2, length)
        if s1 <= s0:
            continue
        p0, p1 = a + u * s0, a + u * s1
        zones.append(np.stack((p0, p1, p1 + v * depth_m, p0 + v * depth_m)))
    return zones


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for an (..., 2) array."""
    x, y = points[..., 0], points[..., 1]
    inside = np.zeros(points.shape[:-1], dtype=bool)
    for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if y0 == y1:
            continue
        crosses = (y0 > y) != (y1 > y)
        at = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < at)
    return inside


def box_corners(xy, size_xy, yaw: float) -> np.ndarray:
    """4 x 2 world corners of an upright footprint centred at ``xy``."""
    h0, h1 = size_xy[0] / 2, size_xy[1] / 2
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([(xy[0] + u * c - v * s, xy[1] + u * s + v * c) for u, v in ((h0, h1), (-h0, h1), (-h0, -h1), (h0, -h1))])


def vertical_extent(z_bottom: float, height: float, options: dict) -> tuple[float, float]:
    """Height interval of a box for occupancy, shrunk like its footprint so stacked boxes (a lamp on its table)
    do not overlap."""
    shrink = options["shrink_m"]
    return z_bottom + shrink, z_bottom + max(height - shrink, shrink)


class RoomRaster:
    """Outside-the-room and door-zone masks of a room on a square-pixel raster with a margin around it. The room's
    ``fixed_objects`` (columns, radiators, built-in units) start as blockers over their own height interval."""

    def __init__(self, room: dict, options: dict, margin_m: float):
        polygon = np.asarray(room["floor_polygon_xy_m"], dtype=np.float64)
        self.resolution = options["resolution_m"]
        self.low = polygon.min(0) - margin_m
        shape = np.ceil((polygon.max(0) + margin_m - self.low) / self.resolution).astype(int) + 1
        centres = self.low + (np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij"), -1)
                              + .5) * self.resolution
        inside = _inside(centres, polygon)
        door = np.zeros_like(inside)
        for zone in door_zones(room, options["door_depth_m"]):
            door |= _inside(centres, zone)
        self.outside, self.door = ~inside, door & inside
        self.centres, self.blockers = centres, []
        for fixed in room.get("fixed_objects", ()):
            (sx, sy, sz), (x, y, z) = fixed["size_local_m"], fixed["bottom_center_m"]
            self.block(box_corners((x, y), (sx, sy), fixed["yaw_rad"]), (z, z + sz))

    def block(self, corners: np.ndarray, z_range=(-math.inf, math.inf)) -> None:
        """Mark a placed footprint (4 x 2 corners) spanning heights ``z_range`` as occupied: it costs the boxes
        placed after it whose own height interval overlaps it."""
        self.blockers.append((_inside(self.centres, np.asarray(corners, dtype=np.float64)), *z_range))

    def occupied(self, z_range=(-math.inf, math.inf)) -> np.ndarray | None:
        """Union of the blockers whose height interval overlaps ``z_range`` (None: none does)."""
        masks = [m for m, lo, hi in self.blockers if lo < z_range[1] and z_range[0] < hi]
        return np.logical_or.reduce(masks) if masks else None

    def _index(self, points: np.ndarray):
        index = np.floor((points - self.low) / self.resolution).astype(np.int64)
        valid = (index >= 0).all(-1) & (index[..., 0] < self.outside.shape[0]) & (index[..., 1] < self.outside.shape[1])
        return valid, np.where(valid, index[..., 0], 0), np.where(valid, index[..., 1], 0)

    def lookup(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid, i, j = self._index(points)
        return np.where(valid, self.outside[i, j], True), np.where(valid, self.door[i, j], False)

    def lookup_occupied(self, points: np.ndarray, occupied: np.ndarray) -> np.ndarray:
        valid, i, j = self._index(points)
        return np.where(valid, occupied[i, j], False)


def footprint_offsets(size_xy, yaw: float, options: dict) -> np.ndarray:
    """(foot*foot, 2) sample points of the shrunk upright footprint about its bottom centre."""
    half = np.maximum(np.asarray(size_xy, dtype=np.float64) / 2 - options["shrink_m"], 0.)
    ticks = np.linspace(-1., 1., options["foot"]) if options["foot"] > 1 else np.zeros(1)
    u, v = [a.ravel() for a in np.meshgrid(ticks * half[0], ticks * half[1], indexing="ij")]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack((u * c - v * s, u * s + v * c), -1)


def cell_positions(room: dict, options: dict) -> np.ndarray:
    """(grid*grid, sub*sub, 2) world XY candidate bottom centres per cell, cells row-major (x index major)."""
    origin, scale = normalize_room(room)
    grid, sub = options["grid"], options["sub"]
    # Both cell borders are candidates (``sub`` > 1): whatever wall a label touches, some candidate lies at least as far
    # from it as the label itself, so the label's own cell stays free.
    frac = np.linspace(0., 1., sub) if sub > 1 else np.full(1, .5)
    cells = np.arange(grid)
    # normalized coordinate of cell i, sub-position k: (i + frac[k]) / grid
    x = (cells[:, None, None, None] + frac[None, None, :, None]) / grid  # (g, 1, sub, 1)
    y = (cells[None, :, None, None] + frac[None, None, None, :]) / grid  # (1, g, 1, sub)
    x, y = np.broadcast_arrays(x, y)
    xy = np.stack((x, y), -1).reshape(grid * grid, sub * sub, 2)
    return np.asarray(origin[:2]) + xy * np.asarray(scale[:2])


def cell_cost_maps(room: dict, boxes: list, options: dict | None = None, *, raster: RoomRaster | None = None,
                   parts: bool = False):
    """Cost rows for ``boxes`` = [(size_xy or None, yaw or None, on_floor), ...] (or with a fourth entry, below).

    Returns ``(costs, mask)``: float32 (n, grid*grid) and bool (n,); rows without a footprint are zero and
    unmasked. With ``parts=True`` also returns the outside / door fractions at each cell's least-cost position.
    Occupied footprints (the room's ``fixed_objects``, and those ``RoomRaster.block`` adds in remasking decoding)
    add the fraction of the footprint on those whose height interval overlaps the box's: an optional fourth entry
    ``(z_low, z_high)``, else, for floor objects (third entry set), every blocker.
    """
    options = cell_cost_options(options)
    n, cells = len(boxes), options["grid"] ** 2
    costs = np.zeros((n, cells), dtype=np.float32)
    split = {"outside": np.zeros((n, cells), dtype=np.float32), "door": np.zeros((n, cells), dtype=np.float32)}
    mask = np.zeros(n, dtype=bool)
    usable = [i for i, (size, yaw, *_) in enumerate(boxes) if size is not None and yaw is not None
              and all(math.isfinite(v) and v > 0 for v in size[:2]) and math.isfinite(yaw)]
    if usable:
        reach = max(math.hypot(*boxes[i][0][:2]) / 2 for i in usable) + options["resolution_m"]
        raster = raster or RoomRaster(room, options, reach)
        positions = cell_positions(room, options)
        for i in usable:
            size, yaw, on_floor = boxes[i][:3]
            z_range = boxes[i][3] if len(boxes[i]) > 3 else (-math.inf, math.inf) if on_floor else None
            points = positions[:, :, None, :] + footprint_offsets(size[:2], yaw, options)[None, None]
            outside, door = raster.lookup(points)
            outside_frac, door_frac = outside.mean(-1), door.mean(-1) * bool(on_floor)
            total = outside_frac + door_frac
            occupied = raster.occupied(z_range) if z_range is not None else None
            if occupied is not None:
                total = total + raster.lookup_occupied(points, occupied).mean(-1)
            best = total.argmin(-1)
            pick = lambda a: np.take_along_axis(a, best[:, None], -1)[:, 0]
            costs[i], split["outside"][i], split["door"][i] = pick(total), pick(outside_frac), pick(door_frac)
            mask[i] = True
    return (costs, mask, split) if parts else (costs, mask)


def blocks_doors(on_floor: bool, height) -> bool:
    """Whether an object can block a door: standing on the floor and taller than ``THIN_HEIGHT_M`` (unknown: taller)."""
    return bool(on_floor) and not (height is not None and math.isfinite(height) and height <= THIN_HEIGHT_M)


def sample_cell_costs(sample: dict, geometry: dict, origin, scale, options: dict | None = None):
    """Cost rows for a collated sample, from its ``batch._geometry_rows`` (label size and yaw, normalized position).

    An object stands on the floor when its label bottom is within ``FLOOR_TOLERANCE_M`` of a known floor, or,
    without a valid label height, when its request declares ``support_parent: floor``; door zones cost only
    objects that ``blocks_doors``. With ``drop_label_conflicts`` (default) an object whose own label cell already
    costs more than 0 gets no cost row, so the loss never pulls against the position cross-entropy.
    """
    options = cell_cost_options(options)
    room = sample["condition"]["room"]
    floor = room.get("floor_z_m")
    objects = sample["condition"].get("objects", [])
    boxes = []
    for i, obj in enumerate(objects):
        size_ok = all(geometry["size_valid"][i][:2])
        yaw = geometry["yaw"][i]
        size = geometry["size"][i] if size_ok else None
        yaw = yaw if geometry["yaw_valid"][i] and math.isfinite(yaw) else None
        if geometry["position_valid"][i][2] and isinstance(floor, (int, float)) and math.isfinite(floor):
            z = geometry["position"][i][2] * scale[2] + origin[2]
            on_floor = abs(z - floor) <= FLOOR_TOLERANCE_M
        else:
            on_floor = obj.get("support_parent") == "floor"
        height = geometry["size"][i][2] if geometry["size_valid"][i][2] else None
        box = (size, yaw, blocks_doors(on_floor, height))
        if height is not None and geometry["position_valid"][i][2]:  # fixed boxes cost only what their height meets
            box += (vertical_extent(geometry["position"][i][2] * scale[2] + origin[2], height, options),)
        boxes.append(box)
    costs, mask = cell_cost_maps(room, boxes, options)
    if options["drop_label_conflicts"]:
        from fastfill.v2.geometry import encode_grid_position
        import torch
        for i in np.flatnonzero(mask):
            if all(geometry["position_valid"][i][:2]):
                cell, _ = encode_grid_position(torch.tensor([geometry["position"][i][:2]], dtype=torch.float64), options["grid"])
                if costs[i, int(cell[0])] > 0:
                    mask[i], costs[i] = False, 0.
    return costs, mask
