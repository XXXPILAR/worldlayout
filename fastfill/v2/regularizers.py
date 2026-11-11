"""Optional scene regularizers, distinct from matched prediction-vs-GT GIoU.

Collision is normalized upright OBB intersection volume. Declared support pairs
are exempt only from volumetric overlap regularization; runtime verifies actual
contact. Boundary uses a convex input polygon's halfplanes; concave or unknown
boundaries fail explicitly when this training term is requested.
"""
from __future__ import annotations

import torch

from fastfill.v2.boxes import intersection_area
from fastfill.v2.geometry import bev_corners
from fastfill.v2.validation import effective_support_requests


def _convex_polygon(vertices):
    from shapely.geometry import Polygon
    polygon = Polygon(vertices)
    if not polygon.is_valid or polygon.area <= 0 or abs(polygon.convex_hull.area - polygon.area) > 1e-8:
        raise ValueError("boundary training regularizer requires a known convex polygon")
    area = sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(vertices, vertices[1:] + vertices[:1]))
    return vertices if area > 0 else list(reversed(vertices))


def _collision(p, s, y, q, t, z):
    original = p
    # Compute all volume factors in float64 before multiplying/dividing. A
    # float32 box side can be finite while its volume overflows or underflows.
    # Metal has no float64 kernels; CPU transfers remain part of autograd.
    values = (p, s, y, q, t, z)
    p, s, y, q, t, z = (value.cpu().double() if original.device.type == "mps" else value.double()
                        for value in values)
    offset = p.detach()
    p, q = p - offset, q - offset
    zero = sum((value * 0).sum() for value in (p, s, y, q, t, z))
    height = (torch.minimum(p[2] + s[2], q[2] + t[2]) - torch.maximum(p[2], q[2])).clamp_min(0)
    if not bool(height.detach() > 0):
        # Multiply before reducing: finite large coordinates can overflow a
        # sum even though the correct disjoint-box contribution is exactly zero.
        result = zero
    else:
        volume = intersection_area(p, s, y, q, t, z) * height
        denominator = (s.prod() + t.prod()) / 2
        if not bool(torch.isfinite(denominator) & (denominator > 0)):
            raise ValueError("collision normalization requires finite positive float64 box volume")
        result = volume / denominator + zero
    # Separate the CPU dtype conversion and transfer: a combined `.to` can
    # attempt a forbidden float64 conversion on Metal during backward.
    return result.to(dtype=original.dtype).to(original.device) if original.device.type == "mps" else result


def _boundary(p, s, y, room):
    if room.get("boundary_known", True) is not True:
        raise ValueError("boundary regularizer cannot use unknown/incomplete room boundaries")
    poly = p.new_tensor(_convex_polygon(room["floor_polygon_xy_m"]))
    edge = poly.roll(-1, 0) - poly
    offsets = bev_corners(p, s, y)[:, None] - poly[None]
    cross = edge[None, :, 0] * offsets[..., 1] - edge[None, :, 1] * offsets[..., 0]
    distance = cross / edge.norm(dim=-1).clamp_min(torch.finfo(p.dtype).tiny)
    return torch.relu(-distance).mean()


def regularizer_specs(condition, config):
    """One immutable enumeration for loss terms and eligibility counting."""
    objects, room = effective_support_requests(condition), condition["room"]
    if config.boundary and objects:
        if room.get("boundary_known", True) is not True:
            raise ValueError("boundary regularizer cannot use unknown/incomplete room boundaries")
        _convex_polygon(room["floor_polygon_xy_m"])
    pairs, fixed_pairs = [], []
    if config.collision:
        for i, obj in enumerate(objects):
            pairs.extend((i, j) for j in range(i)
                         if obj.get("support_parent") != objects[j]["id"]
                         and objects[j].get("support_parent") != obj["id"])
            fixed_pairs.extend((i, fixed) for fixed in room.get("fixed_objects", [])
                               if obj.get("support_parent") != fixed["id"])
    return objects, room, tuple(pairs), tuple(fixed_pairs)


def scene_regularizers(positions, sizes, yaws, batch, config, global_counts=True):
    from fastfill.v2.losses import _mean
    zero = (positions[batch["slot_mask"]] * 0).sum()
    collision, boundary = [], []
    if "conditions" not in batch:
        raise ValueError("scene regularizers require original conditions")
    for b, condition in enumerate(batch["conditions"]):
        objects, room, pairs, fixed_pairs = regularizer_specs(condition, config)
        if config.boundary:
            boundary.extend(_boundary(positions[b, i], sizes[b, i], yaws[b, i], room) for i in range(len(objects)))
        collision.extend(_collision(positions[b, i], sizes[b, i], yaws[b, i],
                                    positions[b, j], sizes[b, j], yaws[b, j]) for i, j in pairs)
        for i, fixed in fixed_pairs:
            q, t, z = [positions.new_tensor(fixed[k]) for k in ("bottom_center_m", "size_local_m", "yaw_rad")]
            collision.append(_collision(positions[b, i], sizes[b, i], yaws[b, i], q, t, z))
    terms = {"collision": collision, "boundary": boundary}
    return {**{key: _mean(sum(values, zero), len(values), global_counts)[0] for key, values in terms.items()},
            "sums": {key: sum((v.detach().float() for v in values), torch.zeros((), device=zero.device)) for key, values in terms.items()},
            "counts": {key: len(values) for key, values in terms.items()}}
