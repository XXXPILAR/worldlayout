"""Given-pose conditioning for masked layout training (Stage 1).

A training room may reveal the reference pose of some requests ("given"); their
slots receive ``pose_features`` through the model's pose embedding, the other
slots are predicted conditioned on them, and given slots leave every
supervised term and every exchangeable matching group (fixed identity).
Training samples choose given requests in ``batch.augment_sample``
(``given_pose_p``); collate turns them into ``given_pose`` / ``given_pose_mask``.

Features are invariant under the label's own symmetries, so equivalent writings
of one box embed identically: a K1 box (``size_axis_swap_allowed``) is first
written with sx >= sy (a swap adds pi/2 to yaw), then yaw enters as
cos/sin(n * yaw) with n = 2 for K1 boxes (their remaining pi symmetry) and the
recorded ``yaw_symmetry_order`` otherwise.
"""
from __future__ import annotations

import math

import torch

POSE_FEATURES = 8  # normalized bottom centre (3), log size (3), cos/sin(n * yaw) (2)


def pose_features(position, size, yaw, symmetry, swap):
    """(..., 3) normalized bottom centre, (..., 3) positive size in m, (...) yaw in rad, (...) symmetry order,
    (...) K1 flag -> (..., POSE_FEATURES)."""
    swap = swap.bool()
    flip = swap & (size[..., 0] < size[..., 1])
    size = torch.where(flip[..., None], size[..., [1, 0, 2]], size)
    yaw = yaw + flip.to(yaw.dtype) * (math.pi / 2)
    order = torch.where(swap, torch.full_like(symmetry, 2), symmetry).to(yaw.dtype)
    angle = order * yaw
    return torch.cat((position, size.log(), angle.cos()[..., None], angle.sin()[..., None]), -1)
