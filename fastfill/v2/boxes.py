"""Piecewise differentiable BEV oriented GIoU; convex hull enclosure of both boxes.

Sutherland-Hodgman intersection and a monotone-chain convex hull select topology
with detached comparisons. Original vertices retain autograd. Derivatives exist
within each topology; contacts, coincident edges and hull changes are nonsmooth.
No vertical supervision is supplied by this operator. Uses no compiled extension.
Metal inputs use CPU float64 polygon arithmetic and return float32 on Metal;
device transfers preserve autograd, at the cost of synchronization and transfer.
"""
from __future__ import annotations

import torch

from fastfill.v2.geometry import bev_corners


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _area(points, zero):
    if len(points) < 3:
        return zero
    p = torch.stack(points)
    p = p - p[0]  # translation invariant area, avoids cancellation of global offsets
    return (_cross_batch(p, p.roll(-1, 0)).sum() / 2).abs()


def _cross_batch(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _clip(subject, clip):
    output = list(subject.unbind(0))
    for a, b in zip(clip, clip.roll(-1, 0)):
        source, output = output, []
        if not source:
            break
        previous = source[-1]
        prev_side = _cross(b - a, previous - a)
        for current in source:
            side = _cross(b - a, current - a)
            inside, was_inside = bool(side.detach() >= 0), bool(prev_side.detach() >= 0)
            if inside != was_inside:
                # Signed distances have opposite signs; denominator is nonzero.
                fraction = prev_side / (prev_side - side)
                output.append(previous + fraction * (current - previous))
            if inside:
                output.append(current)
            previous, prev_side = current, side
    return output


def _hull(points):
    ordered = sorted(range(len(points)), key=lambda i: tuple(points[i].detach().cpu().tolist()))
    # Remove identical points using topology comparisons only.
    ordered = [i for k, i in enumerate(ordered) if k == 0 or not torch.equal(points[i].detach(), points[ordered[k - 1]].detach())]
    def chain(order):
        result = []
        for i in order:
            while len(result) >= 2 and bool(_cross(points[result[-1]] - points[result[-2]], points[i] - points[result[-1]]).detach() <= 0):
                result = result[:-1]
            result.append(i)
        return result
    indices = chain(ordered)[:-1] + chain(list(reversed(ordered)))[:-1]
    return [points[i] for i in indices]


def _corners(p, s, yaw):
    if p.shape != (3,) or s.shape != (3,) or yaw.ndim != 0:
        raise ValueError("BEV operator accepts individual 3D full-size boxes")
    if not all(torch.isfinite(t).all() for t in (p, s, yaw)) or (s <= 0).any():
        raise ValueError("box geometry must be finite and positive")
    return bev_corners(p, s, yaw)


def _working_pair(position, size, yaw, other_position, other_size, other_yaw):
    # Polygon topology is numerically fragile in float16/float32, especially for
    # tiny boxes far from the origin. Cast before constructing any vertices and
    # subtract a common origin. Autograd propagates through casts/subtractions.
    _corners(position, size, yaw)
    _corners(other_position, other_size, other_yaw)
    if position.device.type == "mps":
        position, size, yaw, other_position, other_size, other_yaw = (
            value.cpu().double()
            for value in (position, size, yaw, other_position, other_size, other_yaw))
    offset = position.detach().double()
    p, q = position.double() - offset, other_position.double() - offset
    s, t, y, z = size.double(), other_size.double(), yaw.double(), other_yaw.double()
    return _corners(p, s, y), _corners(q, t, z), s, t


def _result_on_input_device(result, position):
    return result.float().to(position.device) if position.device.type == "mps" else result


def intersection_area(position, size, yaw, other_position, other_size, other_yaw):
    a, b, _, _ = _working_pair(position, size, yaw, other_position, other_size, other_yaw)
    return _result_on_input_device(_area(_clip(a, b), (a.sum() + b.sum()) * 0), position)


def _overlap(position, size, yaw, other_position, other_size, other_yaw):
    a, b, size, other_size = _working_pair(position, size, yaw, other_position, other_size, other_yaw)
    zero = (a.sum() + b.sum()) * 0
    intersection = _area(_clip(a, b), zero)
    union = size[0] * size[1] + other_size[0] * other_size[1] - intersection
    if not torch.isfinite(union) or not bool(union.detach() > 0):
        raise ValueError("box union area must be finite and positive")
    return a, b, intersection, union, zero


def bev_iou(position, size, yaw, other_position, other_size, other_yaw):
    """Actual BEV IoU, kept distinct from generalized IoU."""
    _, _, intersection, union, _ = _overlap(position, size, yaw, other_position, other_size, other_yaw)
    return _result_on_input_device(intersection / union, position)


def bev_giou(position, size, yaw, other_position, other_size, other_yaw):
    a, b, intersection, union, zero = _overlap(position, size, yaw, other_position, other_size, other_yaw)
    enclosure = _area(_hull(torch.cat((a, b), 0)), zero)
    if not torch.isfinite(enclosure) or not bool(enclosure.detach() > 0):
        raise ValueError("box convex enclosure area must be finite and positive")
    return _result_on_input_device(intersection / union - (enclosure - union) / enclosure, position)
