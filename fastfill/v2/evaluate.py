"""All-request, stage-separated evaluation for structured and text v2 models.

Besides the reference errors the report carries label-only baselines (room centre,
per-category mean position, per-category median size, uniform yaw), mode-collapse
diagnostics (every predicted object, the label-complete ground truth, and the predicted
objects matched to it, each with the out-of-room fraction), symmetry-aware yaw error counts per order, and a per-source
breakdown of all of the above.

``--projection full minimal`` also evaluates the three-field projection
(``project_minimal``: ``batch.render_minimal_condition`` with the training
augmentation's regrouping) over boundary-known axis-aligned rectangular rooms; the report's top
level is the first projection and ``projections`` holds the others. Objects whose
``validity.size_axis_swap_allowed`` is true score size and yaw under the
box-equivalent minimum over (sx, sy, yaw + k pi/2 with sx/sy swapped for odd k),
with the plain-convention errors reported beside them. Target validation is
counted per check code as pass / violation / unknown. ``max_length`` comes from the
checkpoint (``io.load_checkpoint_config``) unless given explicitly; model settings
come from its ``model_config.json``. Reports record the checkpoint, its binding,
and the data and code sha256.

``baselines_paired`` scores the baselines on the model reference's own requests (those with a layout);
``validation`` counts validator collisions and clean rooms (no hard violation) for the layouts and, under the same
validator, for their labels. ``--save-head-outputs DIR`` keeps every request's head outputs;
``--from-head-outputs DIR`` re-decodes and scores them on the CPU with the online result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import math
from pathlib import Path
import time

import numpy as np
from shapely.geometry import Point, Polygon
import torch

from fastfill.v2.batch import (AUGMENT_DEFAULTS, AUGMENT_MAGNITUDES, OBJECT_BUDGET_ERROR, _geometry_rows, augment_sample, collate_samples, minimal_form_eligible,
                               load_tokenizer, room_normalization)
from fastfill.v2.boxes import bev_iou
from fastfill.v2.geometry import decode_yaw, wrap_yaw
from fastfill.v2.io import fingerprint, load_checkpoint_config, read_samples, run_metadata, safe_output, to_device
from fastfill.v2.matching import match_batch
from fastfill.v2.model import load_model, model_inputs
from fastfill.v2.schema import migrate_legacy_row, validate_condition, validate_layout
from fastfill.v2.validation import CHECK_STATUSES, effective_support_requests, footprint, validate_scene, validate_required_levels

BASELINE_METRICS = (("room_center_position", "bottom_center_error_m"), ("category_mean_position", "bottom_center_error_m"),
                    ("category_median_size", "log_size_error"), ("uniform_yaw", "yaw_error_rad"))
REFERENCE_METRICS = ("bottom_center_error_m", "log_size_error", "yaw_error_rad", "bev_iou",
                     "log_size_error_plain_convention", "yaw_error_rad_plain_convention")
COLLAPSE_SCOPE = ("predicted: every requested object (a legal exchange of IDs cannot change it); ground truth: request "
                  "slots whose labels form a trustworthy upright box (complete position and size, finite yaw); "
                  "predicted_matched: the predicted objects the reference matching assigns to those slots, the same "
                  "objects as ground truth, so compare predicted_matched with ground_truth")
COLLAPSE_KEYS = ("objects", "pairs", "bev_overlap_pairs", "same_category_pairs", "stacked_same_category_pairs",
                 "nearest_wall_distance_sum_m", "central_quarter_objects", "out_of_room_objects",
                 "footprint_outside_objects", "floor_objects", "door_blocking_objects")
OUT_OF_ROOM_M = .05  # bottom centre this far outside the floor polygon counts as out of the room
FOOTPRINT_VIOLATION_FRACTION = .1  # footprint share outside the room / inside a door zone that counts as a violation
DOOR_ZONE_DEPTH_M = .8  # ``cell_costs.door_zones`` depth used by door_blocking_objects
PROJECTIONS = ("full", "minimal")
# augment_sample with only the minimal form enabled is the K5 projection of a row: shared condition
# renderer, floor declarations dropped (z learned), exchangeable groups recomputed by the build rule.
MINIMAL_PROJECTION = {key: (1. if key == "minimal_form_p" else 0. if key.endswith("_p") or key in AUGMENT_MAGNITUDES else False)
                      for key in AUGMENT_DEFAULTS}
DEFAULT_MAX_LENGTH = 4096


def project_minimal(sample):
    """The three-field projection of an evaluation row, or None unless ``batch.minimal_form_eligible``
    (a boundary-known axis-aligned rectangle; such rows count as ``skipped_non_rectangular_rooms``)."""
    sample = migrate_legacy_row(sample)
    if not minimal_form_eligible(sample["condition"]["room"]):
        return None
    return augment_sample(sample, MINIMAL_PROJECTION, torch.Generator().manual_seed(0))


def collapse_score(counts):
    """Collapse score: BEV overlap rate (IoU > 0.3) + central-quarter fraction; lower is better.

    No objects -> None; objects without pairs contribute only the central fraction.
    It cannot see a layout pushed against or outside the walls (such a layout scores 0):
    read it only together with ``out_of_room_fraction`` and the reference errors.
    """
    if not counts["objects"]:
        return None
    return (counts["bev_overlap_pairs"] / counts["pairs"] if counts["pairs"] else 0.) + counts["central_quarter_objects"] / counts["objects"]


def batch_collapse_counts(predictions, batch, assignment=None):
    """Summed ``_collapse_counts`` of a collated batch, as the matching ``collapse_metrics`` column.

    Model outputs (yaw logits) are the predicted column: every request slot; with
    ``assignment`` (prediction slot -> label slot, e.g. the criterion's) only the slots
    assigned to label-complete ones, the predicted_matched column. Labels passed with
    their yaw in radians as ``predictions["yaw"]`` are the ground-truth column: only the
    label-complete slots.
    """
    origin, scale = batch["origin"].detach().float().cpu(), batch["scale"].detach().float().cpu()
    position = (predictions["position_normalized"].detach().float().cpu() * scale[:, None] + origin[:, None]).tolist()
    size = predictions["size"].detach().float().cpu().tolist()
    labels = "yaw" in predictions
    yaw = (predictions["yaw"].detach().float() if labels else decode_yaw(
        predictions["yaw_logits"].detach().float(), predictions["yaw_residuals"].detach().float())).cpu().tolist()
    complete = (batch["slot_mask"] & batch["validity"]["position"].all(-1) & batch["validity"]["size"].all(-1)
                & torch.isfinite(batch["targets"]["yaw"])).cpu()
    keep = complete if labels else batch["slot_mask"].cpu() if assignment is None else complete.gather(1, assignment.cpu())
    totals = dict.fromkeys(COLLAPSE_KEYS, 0)
    for b, objects in enumerate(batch["objects"]):
        slots = keep[b].nonzero().flatten().tolist()
        counts = _collapse_counts([(position[b][i], size[b][i], yaw[b][i]) for i in slots], [objects[i]["category"] for i in slots],
                                  batch["conditions"][b]["room"], origin[b].tolist(), scale[b].tolist())
        for key in totals:
            totals[key] += counts[key]
    return totals


GRID_DECODES = ("spread", "argmax")
PARENT_LATTICE = np.stack(np.meshgrid(*[np.linspace(-1, 1, 5)] * 2, indexing="ij"), -1).reshape(-1, 2)


def spread_grid_xy(logits, residuals, grid, size, yaw, position, room, *, requests=None, fixed_z=None, yaw_logits=None,
                   yaw_residuals=None, keep_yaw=None, top_k=64, max_overlap=.15, margin=.02, near_wall=.3, hang_centre=1.5,
                   inner=.05):
    """Collision-aware decoding of one scene's grid head (meters in, meters out); returns (xy, yaw, z).

    ``room`` may list ``fixed_objects`` (condition rows: bottom_center_m, size_local_m, yaw_rad): they never move,
    block cells like placed objects and can carry raised ones. ``requests`` (the slots' effective requests,
    ``validation.effective_support_requests``) supply ``id`` and ``support_parent``; ``fixed_z`` marks slots whose
    z the condition fixes (``fixed_position_mask[..., 2]``). Without them every object is undeclared.

    Objects are placed floor-standing before raised ones, parents before their declared children, largest
    footprint first; each takes its most probable cell whose footprint stays in the room (a footprint at most
    ``margin`` past a wall slides back inside first) and overlaps no placed
    or fixed object that shares its height interval (overlap / smaller footprint <= max_overlap). Identical
    requests, whose argmax cells coincide, therefore spread over their next most probable cells. If no
    candidate qualifies, the least overlapping one is used (for a declared child: on its parent).

    Declarations beat the predicted height: an object whose z is fixed stands at that z, one declared on
    "floor" stands on the floor, one declared on "wall" hangs at its rule height (below), and one declared on a request
    or fixed object stands on the parent's top: at its most probable top_k cell whose centre lies inside the
    parent's (placed) footprint, else at the free point of a 5 x 5 lattice over that footprint (inset by its
    own half extents) nearest its argmax decode, else at the least overlapping of these, so the declared
    support holds even at the cost of a collision. A declared child whose predicted bottom lies more than ``inner``
    below its parent's top (and not below its bottom) stands inside the parent instead, e.g. a book on a bookcase's
    inner shelf: it keeps that height (at least the parent's bottom) and the parent is no obstacle to it.

    A "wall" object is placed by rule: each top_k cell is projected
    onto the wall (side of the room's bounds) nearest that cell, back flush with it and facing into the room
    (local +X into the room, as every source wall object; a box predicted thinner along local Y with local +X along
    the wall nearest its argmax cell keeps that axis, turned a quarter; a ``keep_yaw`` slot keeps its yaw),
    sliding along the wall to stay inside; the first projection free of
    overlap wins (wall, floor and fixed objects sharing its height interval are obstacles), else the least
    overlapping one; one wider or deeper than the room is centred along that axis. It keeps its predicted z when
    that is over 0.15 m above the floor, else (a z near the floor is no evidence here) hangs centred ``hang_centre``
    above the floor; either is clamped to [floor, ceiling - its height] when ``room`` gives ``height_m`` (else only
    >= floor), and it is placed with the raised objects when it ends over 0.15 m above the floor.

    With yaw logits (they only switch the rule on), a floor-standing object whose candidate footprint ends within
    ``near_wall`` of a wall turns its back to that wall (as 97-99% of wall-adjacent training furniture) only by a
    half turn: its decoded ``yaw`` flips by pi when the flipped yaw faces away from that wall within 45 degrees, else
    it is kept. The model learns yaw only modulo pi (training takes the min over {y, y + pi}), so the flip chooses
    between the two facings of the predicted box and never turns the box: it is never rotated by a quarter turn and its
    footprint stays the same. By design it also turns an object that faced the wall on purpose (a chair facing a desk
    against it; symmetric yaw metrics do not see this), and only the nearest wall counts: in a corner, an object facing
    the second-nearest wall keeps facing it. Slots marked in ``keep_yaw`` (one bool per slot; ``serialize_predictions``
    marks objects whose orientation the request constrains) never turn: they keep ``yaw`` and their footprints are
    checked at it.

    An undeclared raised object (predicted bottom > 0.15 m above the floor) whose argmax cell lies over no placed or
    fixed floor-standing object holding its predicted bottom (from that object's bottom to 0.15 m over its top), and
    whose argmax footprint ends within ``near_wall`` of a wall, hangs there: it keeps its predicted z (clamped like a
    "wall" object's) at its most probable free cell (a painting over a sofa, a mirror on a bare wall; no declaration
    certifies the wall, so it is not projected onto it). Any other undeclared raised object rests on a
    floor-standing object: it takes its most probable cell (at least 10% as likely as its best) whose centre lies inside
    such an object's footprint and sits on the highest such top; with no such cell it goes to the floor at its
    most probable free cell. "Inside a footprint" (here and for declared parents) is tested in the supporting box's
    own frame, so a rotated table holds only points over its rotated top, and the parent lattice is laid out in it.
    """
    # ponytail: collisions between rotated footprints are compared by their axis-aligned bounds (exact for 90-degree
    # yaws, conservative otherwise); switch to shapely polygons if oblique furniture matters.
    n, cells = logits.shape
    index = torch.arange(cells)
    centres = torch.stack((torch.div(index, grid, rounding_mode="floor"), index % grid), -1).double()
    xy_norm = (centres + .5) / grid + residuals.double() / (2 * grid)  # n x cells x 2
    origin, scale = np.asarray(room["origin"], float), np.asarray(room["scale"], float)
    xy = xy_norm.numpy() * scale[:2] + origin[:2]
    size, yaw, position = np.asarray(size, float).reshape(-1, 3), np.asarray(yaw, float), np.asarray(position, float)
    requests = [{}] * n if requests is None else list(requests)
    fixed_z = np.zeros(n, bool) if fixed_z is None else np.asarray(fixed_z, bool)
    keep_yaw = np.zeros(n, bool) if keep_yaw is None else np.asarray(keep_yaw, bool)
    if len(requests) != n or fixed_z.shape != (n,) or keep_yaw.shape != (n,):
        raise ValueError("requests, fixed_z and keep_yaw need one entry per slot")

    def halves(angle, box):
        c, s = np.abs(np.cos(angle)), np.abs(np.sin(angle))
        return .5 * np.stack((c * box[..., 0] + s * box[..., 1], s * box[..., 0] + c * box[..., 1]), -1)

    def within(points, centres, box, angle):  # k points x m boxes: inside each box's footprint, in the box's own frame
        d = points[:, None] - centres[None]
        c, s = np.cos(angle), np.sin(angle)
        return ((np.abs(c * d[..., 0] + s * d[..., 1]) <= box[:, 0] / 2)
                & (np.abs(c * d[..., 1] - s * d[..., 0]) <= box[:, 1] / 2))

    fixed = room.get("fixed_objects", ())
    fixed_ids = {obj["id"]: j for j, obj in enumerate(fixed)}
    fixed_xy = np.array([obj["bottom_center_m"][:2] for obj in fixed], float).reshape(-1, 2)
    fixed_bottom = np.array([obj["bottom_center_m"][2] for obj in fixed], float)
    fixed_size = np.array([obj["size_local_m"] for obj in fixed], float).reshape(-1, 3)
    fixed_yaw = np.array([obj["yaw_rad"] for obj in fixed], float)
    fixed_half = halves(fixed_yaw, fixed_size)
    fixed_top, fixed_area = fixed_bottom + fixed_size[:, 2], 4 * fixed_half.prod(-1)
    fixed_standing = fixed_bottom - room["floor_z"] <= .15
    slots = {r["id"]: i for i, r in enumerate(requests) if "id" in r}
    parent = [r.get("support_parent") for r in requests]
    if any(p is not None and p not in slots and p not in fixed_ids and p not in ("floor", "wall") for p in parent):
        raise ValueError("unknown support parent")

    def depth(i):
        chain = 0
        while parent[i] in slots:
            i, chain = slots[parent[i]], chain + 1
            if chain > n:
                raise ValueError("support graph contains a cycle")
        return chain

    half = halves(yaw, size)
    area = 4 * half[:, 0] * half[:, 1]
    low = position[:, 2]
    lo, hi = np.asarray(room["bounds"][0], float), np.asarray(room["bounds"][1], float)
    on_object = np.array([p in slots or p in fixed_ids for p in parent], bool)
    on_floor = fixed_z | np.array([p == "floor" for p in parent], bool)
    on_wall = np.array([p == "wall" for p in parent], bool)
    ceiling = room.get("height_m")
    hung = np.where(low - room["floor_z"] > .15, low, room["floor_z"] + hang_centre - size[:, 2] / 2)
    hung = np.maximum(hung if ceiling is None else np.minimum(hung, room["floor_z"] + ceiling - size[:, 2]), room["floor_z"])
    raised = on_object | (~on_floor & (np.where(on_wall, hung, low) - room["floor_z"] > .15))
    stand = np.where(fixed_z, low, np.where(on_wall, hung, float(room["floor_z"])))  # floor-standing or wall-hung z
    order = sorted(range(n), key=lambda i: (bool(raised[i]), depth(i), -area[i]))
    # stable: equal logits rank by cell index on every platform (an unstable sort ordered them differently on aarch64)
    ranked = torch.sort(logits.double(), dim=-1, descending=True, stable=True).indices[:, :top_k].numpy()
    chosen, chosen_yaw, chosen_z = xy[np.arange(n), ranked[:, 0]].copy(), yaw.copy(), low.copy()
    away = np.array([0., np.pi, np.pi / 2, -np.pi / 2])  # from walls x=lo, x=hi, y=lo, y=hi into the room
    flipped = (yaw + 2 * np.pi) % (2 * np.pi) - np.pi  # the decoded yaw turned by pi, wrapped: the same box

    def boxes(js, f=slice(None)):  # placed slots js, then fixed objects f: centre, half extents, bottom, top, footprint
        js = np.asarray(js, int)  # area, local size, yaw
        return (np.concatenate((chosen[js], fixed_xy[f])), np.concatenate((half[js], fixed_half[f])),
                np.concatenate((chosen_z[js], fixed_bottom[f])), np.concatenate((chosen_z[js] + size[js, 2], fixed_top[f])),
                np.concatenate((area[js], fixed_area[f])), np.concatenate((size[js], fixed_size[f])),
                np.concatenate((chosen_yaw[js], fixed_yaw[f])))

    placed = []
    for i in order:
        cand = xy[i, ranked[i]]  # k x 2
        cand_yaw = np.full(len(cand), yaw[i])
        o_xy, o_half, o_bottom, o_top, o_area, o_size, o_yaw = boxes(placed)  # placed slots, then every fixed object
        inside, hang = None, False  # inside: the declared parent's box column this child stands in
        if raised[i] and not on_object[i] and not on_wall[i]:  # undeclared: on or in a standing object, else a near wall
            standing = np.concatenate((~raised[np.asarray(placed, int)], fixed_standing))
            holds = (standing & within(cand[:1], o_xy, o_size, o_yaw)[0]
                     & (o_bottom - margin <= low[i]) & (low[i] <= o_top + .15))
            hang = not holds.any() and np.concatenate((cand[0] - half[i] - lo, hi - cand[0] - half[i])).min() <= near_wall
        if on_object[i]:  # the declared parent's footprint and top; a lattice over that top follows the top_k cells
            if parent[i] in slots:
                j = slots[parent[i]]
                base, base_size, base_yaw, top, column = chosen[j], size[j], chosen_yaw[j], chosen_z[j] + size[j, 2], placed.index(j)
            else:
                j = fixed_ids[parent[i]]
                base, base_size, base_yaw, top, column = fixed_xy[j], fixed_size[j], fixed_yaw[j], fixed_top[j], len(placed) + j
            if o_bottom[column] - margin <= low[i] < top - inner:  # on an inner shelf: keeps its height in the parent
                inside = column
            # the lattice in the parent's frame, inset by the child's half extents in that frame; includes the centre
            lattice = PARENT_LATTICE * np.maximum(base_size[:2] / 2 - halves(yaw[i] - base_yaw, size[i]), 0)
            c, s = np.cos(base_yaw), np.sin(base_yaw)
            on_top = base + np.stack((c * lattice[:, 0] - s * lattice[:, 1], s * lattice[:, 0] + c * lattice[:, 1]), -1)
            on_top = on_top[np.argsort(((on_top - cand[0]) ** 2).sum(-1), kind="stable")]  # nearest the argmax decode first
            cand, cand_yaw = np.vstack((cand, on_top)), np.append(cand_yaw, np.full(len(on_top), yaw[i]))
        elif on_wall[i]:  # each cell's nearest wall: back flush with it, facing into the room, slid inside along it
            wall = np.stack((cand[:, 0] - lo[0], hi[0] - cand[:, 0], cand[:, 1] - lo[1], hi[1] - cand[:, 1]), -1).argmin(-1)
            turn = yaw[i] - away[wall[0]]  # a box predicted with its thin local Y across its argmax wall keeps that axis
            turn = np.sign(np.sin(turn)) * np.pi / 2 if size[i, 0] > size[i, 1] and abs(np.sin(turn)) > abs(np.cos(turn)) else 0.
            cand_yaw = np.arctan2(np.sin(away[wall] + turn), np.cos(away[wall] + turn)) if turn else away[wall]
            cand_yaw = np.where(keep_yaw[i], yaw[i], cand_yaw)
            wall_half = halves(cand_yaw, size[i])
            cand = np.clip(cand, lo + wall_half, hi - wall_half)
            rows, axis = np.arange(len(cand)), wall // 2
            cand[rows, axis] = np.where(wall % 2, hi[axis] - wall_half[rows, axis], lo[axis] + wall_half[rows, axis])
            cand = np.where(2 * wall_half > hi - lo, (lo + hi) / 2, cand)  # wider or deeper than the room: centred
        elif yaw_logits is not None and not raised[i] and not keep_yaw[i]:
            gaps = np.stack((cand[:, 0] - half[i, 0] - lo[0], hi[0] - cand[:, 0] - half[i, 0],
                             cand[:, 1] - half[i, 1] - lo[1], hi[1] - cand[:, 1] - half[i, 1]), -1)
            facing = (flipped[i] - away[gaps.argmin(-1)] + np.pi) % (2 * np.pi) - np.pi  # flipped yaw vs facing away
            cand_yaw = np.where((gaps.min(-1) < near_wall) & (np.abs(facing) <= np.pi / 4 + 1e-6), flipped[i], cand_yaw)
        cand_half = halves(cand_yaw, size[i])  # k x 2
        # a wall projection turns the box; elsewhere the footprint is the decoded one (a flip by pi keeps it: same area)
        cand_area = 4 * cand_half[:, 0] * cand_half[:, 1] if on_wall[i] else np.full(len(cand), area[i])
        over = (np.maximum(lo + cand_half - cand, 0) + np.maximum(cand + cand_half - hi, 0)).max(-1)
        snap = (over > 0) & (over <= margin)  # within the margin past a wall: slide back inside (the validator allows 1e-4)
        cand = np.where(snap[:, None], np.clip(cand, lo + cand_half, hi - cand_half), cand)
        outside = over > margin
        if on_object[i]:
            z = top if inside is None else max(low[i], o_bottom[inside])
            cand_z, supported = np.full(len(cand), z), within(cand, base[None], base_size[None], base_yaw)[:, 0]
        else:
            cand_z, supported = np.full(len(cand), hung[i] if hang else stand[i]), np.zeros(len(cand), bool)
        if raised[i] and not on_object[i] and not on_wall[i] and not hang:
            s_xy, _, _, s_top, _, s_size, s_yaw = boxes([j for j in placed if not raised[j]], fixed_standing)
            over_top = within(cand, s_xy, s_size, s_yaw)  # k x m centre in footprint
            highest = np.where(over_top, s_top, -np.inf).max(-1, initial=-np.inf)
            supported = np.isfinite(highest)
            cand_z = np.where(supported, highest, cand_z)
            supported &= (logits[i, ranked[i]] - logits[i, ranked[i, 0]]).numpy() >= -math.log(10)  # >= 10% of the best cell
        gap = (np.minimum(cand[:, None] + cand_half[:, None], o_xy + o_half)
               - np.maximum(cand[:, None] - cand_half[:, None], o_xy - o_half))
        inter = np.clip(gap, 0, None).prod(-1)
        vertical = (np.minimum(cand_z[:, None] + size[i, 2], o_top) - np.maximum(cand_z[:, None], o_bottom)) > margin
        if inside is not None:  # the parent it stands in is no obstacle
            vertical[:, inside] = False
        worst = np.where(vertical, inter / np.minimum(cand_area[:, None], o_area), 0).max(-1, initial=0.)
        free = ~outside & (worst <= max_overlap)
        ok = np.flatnonzero(free & supported if raised[i] else free)
        if not len(ok) and not on_object[i]:
            ok = np.flatnonzero(free)
        # least overlapping fallback; a declared child only among points on its parent (its lattice always is)
        pick = ok[0] if len(ok) else int(np.argmin(np.where(supported | ~on_object[i], worst + outside, np.inf)))
        chosen[i], chosen_yaw[i], half[i], area[i], chosen_z[i] = cand[pick], cand_yaw[pick], cand_half[pick], cand_area[pick], cand_z[pick]
        placed.append(i)
    return chosen, chosen_yaw, chosen_z


def _room_frame(batch, b):
    room = batch["conditions"][b]["room"]
    polygon = np.asarray(room["floor_polygon_xy_m"], float)
    floor = room.get("floor_z_m")
    return {"origin": batch["origin"][b].cpu().tolist(), "scale": batch["scale"][b].cpu().tolist(),
            "bounds": (polygon.min(0), polygon.max(0)), "floor_z": float(floor) if floor is not None else float(batch["origin"][b, 2]),
            "fixed_objects": room.get("fixed_objects", []), "height_m": room.get("height_m")}


def serialize_predictions(predictions, batch, *, grid_decode="spread"):
    """Validated layouts of a collated batch. ``spread`` (grid head only) re-decodes each scene with
    ``spread_grid_xy`` under its declared supports, fixed objects and fixed z, keeping the decoded yaw of every
    object named as ``object_id`` by a "faces_direction" or "faces" constraint (hard or soft); ``argmax`` keeps
    the raw head. A ``support_head`` model's ``support_logits`` give every object a ``support_parent``
    (``support.decode_support_parents``: declarations kept, the rest predicted), and spread places the undeclared
    ones on their predicted parents."""
    if grid_decode not in GRID_DECODES:
        raise ValueError(f"grid_decode must be one of {GRID_DECODES}")
    supports = None
    if "support_logits" in predictions:
        from fastfill.v2.support import decode_support_parents
        supports = [decode_support_parents(predictions["support_logits"][b, :len(objects)].detach().cpu(),
                                           effective_support_requests(batch["conditions"][b]))
                    for b, objects in enumerate(batch["objects"])]
    position = predictions["position_normalized"] * batch["scale"][:, None] + batch["origin"][:, None]
    yaw = decode_yaw(predictions["yaw_logits"], predictions["yaw_residuals"])
    if grid_decode == "spread" and "position_cell_logits" in predictions:
        position, yaw = position.detach().cpu().double().clone(), yaw.detach().cpu().double().clone()
        cells = predictions["position_cell_logits"].shape[-1]
        grid = math.isqrt(cells)
        if grid * grid != cells:
            raise ValueError(f"position_cell_logits hold {cells} cells, not a square grid")
        for b, objects in enumerate(batch["objects"]):
            n = len(objects)
            if n:
                faced = {c.get("object_id") for c in batch["conditions"][b].get("constraints", ())
                         if c.get("type") in ("faces_direction", "faces")}
                requests = effective_support_requests(batch["conditions"][b])
                if supports is not None:
                    requests = [{**r, "support_parent": parent} for r, parent in zip(requests, supports[b])]
                xy, spread_yaw, spread_z = spread_grid_xy(
                    predictions["position_cell_logits"][b, :n].detach().cpu().float(),
                    predictions["position_cell_residuals"][b, :n].detach().cpu().float(), grid,
                    predictions["size"][b, :n].detach().cpu().double().numpy(), yaw[b, :n].detach().cpu().double().numpy(),
                    position[b, :n].numpy(), _room_frame(batch, b),
                    requests=requests,
                    fixed_z=batch["fixed_position_mask"][b, :n, 2].cpu().numpy(),
                    yaw_logits=predictions["yaw_logits"][b, :n].detach().cpu().float().numpy(),
                    yaw_residuals=predictions["yaw_residuals"][b, :n].detach().cpu().float().numpy(),
                    keep_yaw=[obj["id"] in faced for obj in objects])
                position[b, :n, :2] = torch.from_numpy(xy)
                position[b, :n, 2] = torch.from_numpy(spread_z)
                yaw[b, :n] = torch.from_numpy(spread_yaw).to(yaw.dtype)
    result = []
    for b, objects in enumerate(batch["objects"]):
        layout = {"schema_version": "fastfill.v2", "objects": [
            {"id": obj["id"], "target_size_local_m": predictions["size"][b, i].detach().cpu().tolist(),
             "bottom_center_m": position[b, i].detach().cpu().tolist(),
             "yaw_rad": float(wrap_yaw(float(yaw[b, i].detach().cpu()))),
             **({"support_parent": supports[b][i]} if supports is not None else {})} for i, obj in enumerate(objects)]}
        validate_layout(layout, batch["conditions"][b])
        result.append(layout)
    return result


@torch.no_grad()
def predict_layout(model, tokenizer, condition, *, max_length=4096, device="cpu", grid_decode="spread", head=None):
    """``head`` (a dict) receives the forward pass's ``head_arrays`` before decoding."""
    validate_condition(condition)
    batch = to_device(collate_samples([{"condition": condition}], tokenizer,
        max_length=max_length, max_objects=model.config.max_objects), device)
    model.eval()
    predictions = model(**model_inputs(batch))
    if head is not None:
        head.update(head_arrays(predictions, batch))
    return serialize_predictions(predictions, batch, grid_decode=grid_decode)[0]


# What serialize_predictions reads: position_normalized carries the regressed z (and the argmax-decoded XY).
HEAD_KEYS = ("position_normalized", "size", "yaw_logits", "yaw_residuals", "position_cell_logits", "position_cell_residuals")
# What --save-head-outputs records in DIR/manifest.json, and --from-head-outputs reports as head_outputs_manifest.
HEAD_MANIFEST_KEYS = ("checkpoint", "checkpoint_binding", "max_length", "max_length_source", "grid_decode", "data_path",
                      "data_sha256", "implementation_sha256", "code_commit", "code_dirty")


def head_arrays(predictions, batch):
    """The head outputs of a one-request batch, exact copies of its object slots, plus its slot mask and ids."""
    n = len(batch["objects"][0])
    return {**{key: predictions[key][0, :n].detach().cpu().numpy() for key in HEAD_KEYS if key in predictions},
            "slot_mask": batch["slot_mask"][0, :n].cpu().numpy(), "ids": np.array([o["id"] for o in batch["objects"][0]], dtype=str)}


class StoredFailure(Exception):
    """A request whose saved head outputs record the online failure (no forward pass, e.g. over capacity)."""

    def __init__(self, error):
        super().__init__(error["message"])
        self.error = error


class HeadOutputMismatch(Exception):
    """Saved head outputs that do not belong to the evaluated rows: aborts the run, never a request failure."""


def decode_head_outputs(path, condition, *, grid_decode="spread"):
    """Re-decode one request's ``--save-head-outputs`` file on the CPU: the online ``serialize_predictions`` on the
    same arrays and the same condition, so the layout is identical. A file of another condition aborts the run."""
    with np.load(path, allow_pickle=False) as stored:
        stored = dict(stored)
    if json.dumps(json.loads(str(stored["condition"])), sort_keys=True) != json.dumps(condition, sort_keys=True):
        raise HeadOutputMismatch(f"{path} holds the head outputs of another condition")
    if "error_type" in stored:
        raise StoredFailure({"type": str(stored["error_type"]), "message": str(stored["error_message"])})
    from fastfill.v2.batch import TinyTokenizer  # the condition-derived fields only: origin, scale, fixed z, objects
    batch = collate_samples([{"condition": condition}], TinyTokenizer(), max_length=10**8, max_objects=10**6)
    if [o["id"] for o in batch["objects"][0]] != stored["ids"].tolist():
        raise HeadOutputMismatch(f"{path} holds the head outputs of other request ids")
    predictions = {key: torch.from_numpy(stored[key])[None] for key in HEAD_KEYS if key in stored}
    return serialize_predictions(predictions, batch, grid_decode=grid_decode)[0]


def _layout_tensors(layout, batch):
    """Float32 prediction tensors in batch slot order; only correspondence matching reads them."""
    objects = {o["id"]: o for o in layout["objects"]}
    ordered = [objects[o["id"]] for o in batch["objects"][0]]
    p = torch.tensor([[o["bottom_center_m"] for o in ordered]], dtype=torch.float32).reshape(1, -1, 3)
    s = torch.tensor([[o["target_size_local_m"] for o in ordered]], dtype=torch.float32).reshape(1, -1, 3)
    return {"position_normalized": (p - batch["origin"][:, None]) / batch["scale"][:, None],
            "size": s, "slot_mask": batch["slot_mask"]}


def _stat(values):
    return {"mean": float(np.mean(values)) if values else None, "valid_objects": len(values)}


def _pool(entries):
    """Label-weighted mean of per-request {"mean", "valid_objects"} measurements."""
    count = sum(e["valid_objects"] for e in entries)
    total = sum(e["mean"] * e["valid_objects"] for e in entries if e["mean"] is not None)
    return {"mean": total / count if count else None, "valid_objects": count}


def _yaw_error(predicted, label, order):
    period = 2 * math.pi / order
    return abs((predicted - label + period / 2) % period - period / 2)


def _log_size_error(predicted, label):
    return float(np.abs(np.log(predicted) - np.log(label)).mean())


def _box_equivalent_errors(p, t, size_valid, yaw_valid):
    """(size, yaw) errors of the label rewriting (sx, sy, sz, yaw + k pi/2), sy/sx swapped for odd k, k = 0..3,
    whose summed valid errors are smallest; the same candidate set as the training loss. None for invalid fields."""
    best = None
    for k in range(4):
        sx, sy, sz = t["target_size_local_m"] if size_valid else (1., 1., 1.)
        size = _log_size_error(p["target_size_local_m"], (sy, sx, sz) if k % 2 else (sx, sy, sz)) if size_valid else None
        yaw = _yaw_error(p["yaw_rad"], t["yaw_rad"] + k * math.pi / 2, 1) if yaw_valid else None
        if best is None or (size or 0.) + (yaw or 0.) < (best[0] or 0.) + (best[1] or 0.):
            best = (size, yaw)
    return best


def _uniform_yaw_error(order, swap, size=None):
    """Expected ``reference_metrics`` yaw error of a uniformly random yaw with a perfect size (the label's own;
    ``size`` None when it is not valid): pi / (2 order) modulo the symmetry order, and for a swap object the yaw of
    ``_box_equivalent_errors``' joint minimum, in closed form: with a = the yaw error modulo pi (uniform on
    [0, pi/2]) the quarter turn, error pi/2 - a plus its swap cost c (0 without a valid size), wins when a > m."""
    if not swap:
        return math.pi / (2 * order)
    c = _log_size_error(size, (size[1], size[0], size[2])) if size is not None else 0.
    m = min(math.pi / 2, math.pi / 4 + c / 2)
    return (m * m + (math.pi / 2 - m) ** 2) / math.pi  # pi / 8 for a square footprint, pi / 4 once c >= pi / 2


def _matched(layout, sample, hungarian):
    """The one-request batch and its legal assignment (prediction slot -> label slot)."""
    from fastfill.v2.batch import TinyTokenizer
    batch = collate_samples([sample], TinyTokenizer(), max_length=10**8, max_objects=10**6)
    return batch, match_batch(_layout_tensors(layout, batch), batch, enabled=hungarian)[0].tolist()


def reference_metrics(layout, sample, *, hungarian=True, include_iou=True):
    """Reference errors follow only legal correspondence; no GT used for inference.

    Errors are computed in float64 metres from the raw layout and target rows, so an
    exact imitation of the labels scores exactly zero. Groups come from collate's
    ``batch["exchangeable_group"]``; match_batch keeps position-incomplete groups at
    fixed identity and matches size-incomplete groups on position alone.
    ``log_size_error`` / ``yaw_error_rad`` use the box-equivalent minimum for objects
    with ``size_axis_swap_allowed``; the ``*_plain_convention`` keys keep the label as
    written (yaw modulo its symmetry order; modulo pi for swap objects, whose order 4
    only stands for the swap) for every object.
    """
    batch, assignment = _matched(layout, sample, hungarian)
    objects = batch["objects"][0]
    predicted = {o["id"]: o for o in layout["objects"]}
    labels = {o["id"]: o for o in sample["target"]["objects"]}
    valid = batch["validity"]
    values = {key: [] for key in REFERENCE_METRICS}
    by_order, box_equivalent = {}, 0
    for i, j in enumerate(assignment[:len(objects)]):
        p, t = predicted[objects[i]["id"]], labels.get(objects[j]["id"], {})
        pv, sv, yv = bool(valid["position"][0, j].all()), bool(valid["size"][0, j].all()), bool(valid["yaw"][0, j])
        if pv:
            values["bottom_center_error_m"].append(float(np.linalg.norm(np.subtract(p["bottom_center_m"], t["bottom_center_m"]))))
        order, swap = int(batch["yaw_symmetry_order"][0, j]), bool(batch["size_axis_swap_allowed"][0, j])
        size_error = _log_size_error(p["target_size_local_m"], t["target_size_local_m"]) if sv else None
        # A swap object's order 4 encodes the axis swap; as written, its box only repeats every pi.
        yaw_error = _yaw_error(p["yaw_rad"], t["yaw_rad"], 2 if swap else order) if yv else None
        values["log_size_error_plain_convention"] += [size_error] if sv else []
        values["yaw_error_rad_plain_convention"] += [yaw_error] if yv else []
        if (sv or yv) and swap:
            size_error, yaw_error = _box_equivalent_errors(p, t, sv, yv)
            box_equivalent += 1
        if sv:
            values["log_size_error"].append(size_error)
        if yv:
            values["yaw_error_rad"].append(yaw_error)
            by_order.setdefault(order, []).append(yaw_error)
        if include_iou and pv and sv and yv:
            values["bev_iou"].append(float(bev_iou(*(torch.tensor(v, dtype=torch.float64) for v in (
                p["bottom_center_m"], p["target_size_local_m"], p["yaw_rad"],
                t["bottom_center_m"], t["target_size_local_m"], t["yaw_rad"])))))
    groups = {}
    for i, name in enumerate(batch["exchangeable_group"][0] if hungarian else ()):
        if name:
            groups.setdefault(name, []).append(i)
    matched = [name for name, indices in groups.items() if len(indices) > 1]
    incomplete = [name for name in matched if not valid["position"][0, groups[name]].all()]
    position_only = [name for name in matched if name not in incomplete and not valid["size"][0, groups[name]].all()]
    matching_scope = ("exchangeable_complete_groups_fixed_incomplete_groups"
                      if hungarian and incomplete and len(incomplete) < len(matched) else
                      "fixed_incomplete_labels" if hungarian and incomplete else
                      "exchangeable_groups" if hungarian else "fixed")
    return {**{key: _stat(v) for key, v in values.items()},
            "yaw_error_rad_by_symmetry_order": {str(order): _stat(v) for order, v in sorted(by_order.items())},
            "box_equivalent_objects": box_equivalent, "matching_scope": matching_scope, "incomplete_groups": incomplete, "position_only_groups": position_only}


def _label_rows(sample):
    condition = sample["condition"]
    origin, scale = room_normalization(condition["room"])
    return condition["objects"], _geometry_rows(sample, origin, scale), origin, scale


def _collapse_counts(boxes, categories, room, origin, scale):
    """Pairwise/room statistics of upright boxes given as (bottom_center_m, size, yaw)."""
    from fastfill.v2.cell_costs import FLOOR_TOLERANCE_M, blocks_doors, door_zones
    polygons = [footprint({"_pos": p, "_size": s, "_yaw": y}) for p, s, y in boxes]
    floor = Polygon(room["floor_polygon_xy_m"])
    doors = [Polygon(zone).intersection(floor) for zone in door_zones(room, DOOR_ZONE_DEPTH_M)]
    floor_z = room.get("floor_z_m") or 0.
    counts = {"objects": len(boxes), "pairs": 0, "bev_overlap_pairs": 0, "same_category_pairs": 0,
              "stacked_same_category_pairs": 0, "nearest_wall_distance_sum_m": 0., "central_quarter_objects": 0,
              "out_of_room_objects": 0, "footprint_outside_objects": 0, "floor_objects": 0, "door_blocking_objects": 0}
    for i, (p, size, _) in enumerate(boxes):
        counts["nearest_wall_distance_sum_m"] += float(floor.exterior.distance(Point(p[:2])))
        counts["out_of_room_objects"] += int(floor.distance(Point(p[:2])) > OUT_OF_ROOM_M)  # 0 inside the polygon
        area = polygons[i].area
        if area > 0:
            outside = polygons[i].difference(floor).area
            counts["footprint_outside_objects"] += int(outside > FOOTPRINT_VIOLATION_FRACTION * area)
        if blocks_doors(abs(p[2] - floor_z) <= FLOOR_TOLERANCE_M, size[2]):
            counts["floor_objects"] += 1
            blocked = sum(polygons[i].intersection(door).area for door in doors) if area > 0 else 0.
            counts["door_blocking_objects"] += int(blocked > FOOTPRINT_VIOLATION_FRACTION * area)
        counts["central_quarter_objects"] += all(.25 <= (p[q] - origin[q]) / scale[q] <= .75 for q in (0, 1))
        for j in range(i + 1, len(boxes)):
            counts["pairs"] += 1
            intersection = polygons[i].intersection(polygons[j]).area
            union = polygons[i].area + polygons[j].area - intersection
            counts["bev_overlap_pairs"] += int(union > 0 and intersection / union > .3)
            if categories[i] == categories[j]:
                counts["same_category_pairs"] += 1
                counts["stacked_same_category_pairs"] += int(math.dist(p[:2], boxes[j][0][:2]) < .1)
    return counts


def collapse_metrics(layout, sample, *, hungarian=True):
    """Mode-collapse diagnostics: the prediction (every requested object), the ground truth
    (label-complete objects) and predicted_matched (the predicted objects that the reference
    matching assigns to the label-complete slots: the ground truth's objects, predicted).

    The predicted column never depends on which slots carry labels, so a legal exchange
    of IDs leaves it unchanged; predicted_matched follows the legal assignment, so it is
    unchanged too except inside groups the matching keeps at fixed identity. Compare
    predicted_matched, not predicted, with the ground truth: sources with masked labels
    (e.g. size-masked MansionWorld) are in predicted only. Stacking uses horizontal
    bottom-centre distance; the central quarter is the middle half of each normalized room
    axis; wall distance is to the floor polygon; an object is out of the room when its
    bottom centre lies more than ``OUT_OF_ROOM_M`` outside the floor polygon.
    """
    objects, rows, origin, scale = _label_rows(sample)
    labels = {o["id"]: o for o in sample["target"]["objects"]}
    predicted = {o["id"]: o for o in layout["objects"]}
    keep = [i for i in range(len(objects)) if all(rows["position_valid"][i]) and all(rows["size_valid"][i])
            and math.isfinite(rows["yaw"][i])]
    complete = set(keep)
    matched = [i for i, j in enumerate(_matched(layout, sample, hungarian)[1][:len(objects)]) if j in complete]
    room = sample["condition"]["room"]
    columns = {}
    for name, source, indices in (("predicted", predicted, range(len(objects))), ("predicted_matched", predicted, matched),
                                  ("ground_truth", labels, keep)):
        boxes = [(source[objects[i]["id"]]["bottom_center_m"], source[objects[i]["id"]]["target_size_local_m"],
                  source[objects[i]["id"]]["yaw_rad"]) for i in indices]
        columns[name] = _collapse_counts(boxes, [objects[i]["category"] for i in indices], room, origin, scale)
    return columns


def fit_baselines(rows):
    """Per-category label pools (normalized position, size) keyed by (row index, object id)."""
    pools, keys = {"position": {}, "size": {}}, {}
    for index, sample in enumerate(rows):
        try:
            objects, geometry, _, _ = _label_rows(sample)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"baseline fit row {index} has unusable labels: {exc}") from exc
        for i, obj in enumerate(objects):
            keys[(index, obj["id"])] = len(keys)
            for field in pools:
                if all(geometry[f"{field}_valid"][i]):
                    pools[field].setdefault(obj["category"], []).append((keys[(index, obj["id"])], geometry[field][i]))
    def arrays(entries):
        return np.array([g for g, _ in entries], dtype=np.int64), np.array([v for _, v in entries], dtype=np.float64).reshape(-1, 3)
    fit = {"keys": keys}
    for field, pool in pools.items():
        fit[field] = {category: arrays(entries) for category, entries in pool.items()}
        fit[field + "_all"] = arrays([entry for entries in pool.values() for entry in entries])
    return fit


def _estimate(fit, field, category, exclude, reduce):
    """Category statistic without the label keys in ``exclude``; falls back to the all-category pool."""
    for fallback, pool in ((False, fit[field].get(category)), (True, fit[field + "_all"])):
        if pool is not None:
            # ponytail: O(n_category) masked reduction per object; use a sorted-index trick if slow.
            vectors = pool[1][~np.isin(pool[0], list(exclude))]
            if len(vectors):
                return reduce(vectors, axis=0), fallback
    return None, True


def baseline_metrics(sample, fit, *, exclude_row=None):
    """Label-only predictors on one request; exclude_row removes every label of that row (row-level leave-one-out).

    Same-room duplicates often share one asset's exact size, so excluding only the
    object's own label would let its siblings stand in for it.
    Room centre predicts the floor-level centre of the room's XY bounds. Uniform yaw is
    the expected error of a uniformly random yaw scored like the model's (``_uniform_yaw_error``: pi / (2 * symmetry
    order), box-equivalent with the label's own size for swap-allowed objects).
    Median size scores swap-allowed objects box-equivalently, like the model.
    """
    objects, geometry, origin, scale = _label_rows(sample)
    labels = {o["id"]: o for o in sample["target"]["objects"]}
    values = {name: [] for name, _ in BASELINE_METRICS}
    fallbacks = 0
    center = np.array([origin[0] + scale[0] / 2, origin[1] + scale[1] / 2, origin[2]])
    own = {fit["keys"][(exclude_row, o["id"])] for o in objects if (exclude_row, o["id"]) in fit["keys"]}
    for i, obj in enumerate(objects):
        target = labels.get(obj["id"], {})
        if all(geometry["position_valid"][i]):
            gt = np.asarray(target["bottom_center_m"], dtype=np.float64)
            values["room_center_position"].append(float(np.linalg.norm(center - gt)))
            estimate, fallback = _estimate(fit, "position", obj["category"], own, np.mean)
            if estimate is not None:
                fallbacks += fallback
                values["category_mean_position"].append(float(np.linalg.norm(np.asarray(origin) + np.asarray(scale) * estimate - gt)))
        if all(geometry["size_valid"][i]):
            estimate, fallback = _estimate(fit, "size", obj["category"], own, np.median)
            if estimate is not None:
                fallbacks += fallback
                label = target["target_size_local_m"]
                error = _log_size_error(estimate, label)
                if geometry["swap"][i]:
                    error = min(error, _log_size_error(estimate, (label[1], label[0], label[2])))
                values["category_median_size"].append(error)
        if geometry["yaw_valid"][i]:
            values["uniform_yaw"].append(_uniform_yaw_error(geometry["symmetry"][i], geometry["swap"][i],
                target["target_size_local_m"] if all(geometry["size_valid"][i]) else None))
    return {**{name: {metric: _stat(values[name])} for name, metric in BASELINE_METRICS},
            "category_fallback_objects": fallbacks}


def evaluate_layout(layout, sample, *, resolver=None, commit_in_memory=False, hungarian=True,
                    asset_retries=2, repair_calls=0, repair_step_m=.25, max_seconds=10., required_levels=("bbox",)):
    sample = migrate_legacy_row(sample)
    condition = sample["condition"]
    validate_condition(condition)
    validate_layout(layout, condition)
    target = validate_scene(condition, layout["objects"], required_levels=required_levels)
    result = {"model": {"schema_success": True, "requested_ids_exactly_once": True, "positive_valid_size": True,
                         "target_geometry_valid": target["ok"], "reference": None},
              "raw_prediction": layout, "target_validation": target, "collapse": None, "asset": None, "system": None,
              "actual_resolved": None, "final_output": None}
    try:
        result["model"]["reference"] = reference_metrics(layout, sample, hungarian=hungarian)
        result["collapse"] = collapse_metrics(layout, sample, hungarian=hungarian)
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        # Reference-label eligibility is independent of model schema and runtime.
        result["reference_error"] = {"type": type(exc).__name__, "message": str(exc)}
    if resolver is not None:
        from fastfill.v2.runtime import AtomicMemoryHost, BoundedTranslationRepair, RuntimeBudget, run_pipeline
        host = AtomicMemoryHost() if commit_in_memory else None
        runtime = run_pipeline(condition, layout, resolver, host=host,
            budget=RuntimeBudget(max_asset_retries=asset_retries, max_repair_calls=repair_calls, max_seconds=max_seconds),
            repair=BoundedTranslationRepair(repair_step_m) if repair_calls else None,
            required_levels=required_levels,
            expected_world_version=0 if host else None,
            idempotency_key=str(sample["provenance"].get("scene_id", "evaluation")))
        result.update(asset=runtime["metrics"]["asset"], system=runtime["metrics"]["system"], runtime=runtime)
        result["actual_resolved"] = runtime.get("actual_first_pass_objects")
        result["final_output"] = runtime.get("final_objects")
    return result


def _model_summary(outcomes):
    n = len(outcomes)
    model = {key: sum(bool(o.get("model", {}).get(key)) for o in outcomes) / n for key in
             ("schema_success", "requested_ids_exactly_once", "positive_valid_size", "target_geometry_valid")}
    references = [o["model"]["reference"] for o in outcomes if o.get("model", {}).get("reference")]
    reference = {key: {**_pool([r[key] for r in references]),
                       "scope": "parsed predictions with valid labels; schema failures included in success denominator"}
                 for key in REFERENCE_METRICS}
    reference["box_equivalent_objects"] = sum(r["box_equivalent_objects"] for r in references)
    orders = sorted({order for r in references for order in r["yaw_error_rad_by_symmetry_order"]})
    reference["yaw_error_rad_by_symmetry_order"] = {
        order: _pool([r["yaw_error_rad_by_symmetry_order"][order] for r in references if order in r["yaw_error_rad_by_symmetry_order"]])
        for order in orders}
    model["reference"] = reference
    return model


def _baseline_summary(outcomes, *, paired=False):
    """Label-only baselines over every row, or (``paired``) over the rows the model's reference pool scores: the
    requests with a layout, so the same objects as ``model.reference`` (no-layout and over-capacity rows excluded)."""
    rows = [o["baselines"] for o in outcomes if o.get("baselines") and (not paired or o.get("model", {}).get("reference"))]
    if not rows:
        return None
    return {**{name: {metric: _pool([r[name][metric] for r in rows])} for name, metric in BASELINE_METRICS},
            "category_fallback_objects": sum(r["category_fallback_objects"] for r in rows), "requests": len(rows),
            "scope": ("requests with a layout scored against the labels: the model reference's own objects" if paired else
                      "all evaluation rows with valid labels, independent of model parsing")}


def _room_validation(validation):
    """One room's validator outcome: collision pairs (requested / with fixed objects), any hard violation, any hard unknown."""
    checks = validation["checks"]
    return {"collision_pairs": sum(c["code"] == "collision" for c in checks),
            "fixed_collision_pairs": sum(c["code"] == "fixed_collision" for c in checks),
            "hard_violation": any(c["hard"] and c["status"] == "violation" for c in checks),
            "hard_unknown": any(c["hard"] and c["status"] == "unknown" for c in checks)}


def _validation_summary(rooms):
    """Collisions and clean rooms over ``_room_validation`` rows. A clean room has no hard violation; unknown checks
    are not counted as violations, and the clean rooms with one are counted separately. ``clean_room_rate_known_only``
    leaves the rooms with a hard unknown out of the rate (None when every room has one, as in the three-field projection,
    whose undeclared supports are all unknown)."""
    if not rooms:
        return None
    clean = [r for r in rooms if not r["hard_violation"]]
    known = [r for r in rooms if not r["hard_unknown"]]
    return {"rooms": len(rooms), "rooms_with_collision": sum(r["collision_pairs"] + r["fixed_collision_pairs"] > 0 for r in rooms),
            "collision_pairs": sum(r["collision_pairs"] for r in rooms),
            "fixed_collision_pairs": sum(r["fixed_collision_pairs"] for r in rooms),
            "clean_rooms": len(clean), "clean_room_rate": len(clean) / len(rooms),
            "clean_rooms_with_hard_unknown": sum(r["hard_unknown"] for r in clean),
            "rooms_without_hard_unknown": len(known),
            "clean_room_rate_known_only": sum(not r["hard_violation"] for r in known) / len(known) if known else None}


def _collapse_summary(outcomes):
    rows = [o["collapse"] for o in outcomes if o.get("collapse")]
    if not rows:
        return None
    def rates(column):
        c = {key: sum(r[column][key] for r in rows) for key in rows[0][column]}
        ratio = lambda a, b: c[a] / c[b] if c[b] else None
        rooms = [r[column]["bev_overlap_pairs"] / r[column]["pairs"] for r in rows if r[column]["pairs"]]
        return {"bev_overlap_rate_iou_gt_0.3": ratio("bev_overlap_pairs", "pairs"),
                # every room with a pair weighs the same; the pooled rate above is dominated by the largest rooms
                "bev_overlap_rate_iou_gt_0.3_room_mean": float(np.mean(rooms)) if rooms else None,
                "duplicate_stacking_rate_lt_0.10m": ratio("stacked_same_category_pairs", "same_category_pairs"),
                "mean_nearest_wall_distance_m": ratio("nearest_wall_distance_sum_m", "objects"),
                "central_quarter_fraction": ratio("central_quarter_objects", "objects"), "collapse_score": collapse_score(c),
                "out_of_room_fraction": ratio("out_of_room_objects", "objects"),
                "footprint_outside_fraction": ratio("footprint_outside_objects", "objects"),
                "door_blocking_fraction_of_floor_objects": ratio("door_blocking_objects", "floor_objects"),  # thin ones excluded
                "objects": c["objects"], "pairs": c["pairs"], "same_category_pairs": c["same_category_pairs"]}
    return {**{column: rates(column) for column in rows[0]}, "requests": len(rows), "scope": COLLAPSE_SCOPE}


def _check_counts(outcomes):
    """Per check code: how many target-validation checks passed, were violated or stayed unknown.

    Rows whose own labels exceed the declared height (``provenance.height_conflict``)
    count their ceiling checks separately, so a known data conflict is not read as a model violation.
    """
    table = {}
    for outcome in outcomes:
        conflict = bool(outcome.get("provenance", {}).get("height_conflict"))
        for check in (outcome.get("target_validation") or {}).get("checks", ()):
            code = check["code"] + ("_on_height_conflict_rows" if conflict and check["code"] == "ceiling" else "")
            table.setdefault(code, dict.fromkeys(CHECK_STATUSES, 0))[check["status"]] += 1
    return dict(sorted(table.items()))


def summarize(outcomes, *, asset_evaluation_requested=False, commit_evaluation_requested=False, baseline_fit_source=None):
    n = len(outcomes)
    if not n:
        raise ValueError("no evaluation requests")
    model = _model_summary(outcomes)
    baselines, paired = _baseline_summary(outcomes), _baseline_summary(outcomes, paired=True)
    for summary in (baselines, paired):
        if summary is not None:
            summary["fit"] = baseline_fit_source
    def cohort(subset):
        return {"requests": len(subset), "model": _model_summary(subset), "baselines": _baseline_summary(subset),
                "baselines_paired": _baseline_summary(subset, paired=True), "collapse": _collapse_summary(subset),
                "validation": {"model": _validation_summary([_room_validation(o["target_validation"]) for o in subset
                                                             if o.get("target_validation")])},
                "target_validation_checks": _check_counts(subset)}
    sources = sorted({str(o.get("provenance", {}).get("source")) for o in outcomes})
    by_source = {source: cohort([o for o in outcomes if str(o.get("provenance", {}).get("source")) == source])
                 for source in sources}
    # explicit relations in the request or not: training rows carry extracted constraints in most rooms, so a model
    # leaning on them shows up as a gap between the two (outcomes without the count are left out)
    by_constraints = {name: cohort(subset) for name, keep in (("with_constraints", lambda c: c > 0),
                                                               ("without_constraints", lambda c: c == 0))
                      if (subset := [o for o in outcomes if o.get("constraint_count") is not None
                                     and keep(o["constraint_count"])])}
    validated = [o for o in outcomes if o.get("target_validation")]
    rooms = [_room_validation(o["target_validation"]) for o in validated]
    latency = [o["fastfill_latency_ms"] for o in outcomes if o.get("fastfill_latency_ms") is not None]
    latency_by_outcome = {}
    for status in ("success", "failure"):
        values = [o["fastfill_latency_ms"] for o in outcomes if o.get("fastfill_latency_ms") is not None
                  and o.get("fastfill_latency_status") == status]
        latency_by_outcome[status] = {"requests": len(values), "p50": float(np.percentile(values, 50)) if values else None,
                                     "p95": float(np.percentile(values, 95)) if values else None}
    wall = [o["evaluation_wall_time_ms"] for o in outcomes if o.get("evaluation_wall_time_ms") is not None]
    asset_evaluated = asset_evaluation_requested or any(o.get("asset") is not None for o in outcomes)
    inference_failures = sum("error" in o for o in outcomes)
    system_failures = sum(not o.get("runtime", {}).get("ok", False) for o in outcomes) if asset_evaluated else None
    strict = sum((not o.get("runtime", {}).get("ok", False))
                 if asset_evaluated else ("error" in o or not o.get("model", {}).get("target_geometry_valid", False)) for o in outcomes)
    report = {"requests": n, "failed_requests": strict, "failed_requests_strict": strict,
              "failed_requests_scope": ("strict: no layout, or a hard check not passed, unknown included (with assets: the "
                                        "runtime did not succeed); hard_violation_requests counts only violated hard checks"),
              # no layout, or a hard target check violated (unknown checks are not violations)
              "hard_violation_requests": sum("error" in o or not o.get("target_validation")
                                             or _room_validation(o["target_validation"])["hard_violation"] for o in outcomes),
              "validation": {"model": _validation_summary(rooms),
                             "ground_truth": _validation_summary([o["ground_truth_validation"] for o in validated
                                                                  if o.get("ground_truth_validation")]),
                             "scope": "requests with a validated layout; ground_truth: their labels as written, same validator"},
              "inference_failed_requests": inference_failures, "system_failed_requests": system_failures,
              # of the inference failures: requests with more objects than the checkpoint's max_objects
              "over_capacity_requests": sum(OBJECT_BUDGET_ERROR in o.get("error", {}).get("message", "") for o in outcomes),
              "target_validation_checks": _check_counts(outcomes),
              "target_validation_not_run": sum(not o.get("target_validation") for o in outcomes),
              "stage_denominator": n, "stage_failures": {
                  "model_schema": sum(not o.get("model", {}).get("schema_success", False) for o in outcomes),
                  "target_geometry": sum(not o.get("model", {}).get("target_geometry_valid", False) for o in outcomes),
                  "asset_resolution": sum(not o.get("asset") or o["asset"]["retrieval_coverage"] < 1 for o in outcomes) if asset_evaluated else None,
                  "actual_first_pass": sum(not o.get("asset", {}).get("first_pass_actual_geometry_validation", False)
                     if o.get("asset") else True for o in outcomes) if asset_evaluated else None}, "model": model,
              "baselines": baselines, "baselines_paired": paired, "collapse": _collapse_summary(outcomes), "by_source": by_source,
              "by_constraints": by_constraints,
              "latency_ms": {"fastfill_p50": float(np.percentile(latency, 50)) if latency else None,
                             "fastfill_p95": float(np.percentile(latency, 95)) if latency else None,
                             "fastfill_observed_requests": len(latency), "fastfill_by_outcome": latency_by_outcome,
                             "fastfill_scope": "All observed checkpoint generations, including failures; supplied prediction generation time is unknown",
                             "evaluation_wall_time_p50": float(np.percentile(wall, 50)) if wall else None,
                             "evaluation_wall_time_p95": float(np.percentile(wall, 95)) if wall else None}}
    if asset_evaluated:
        report["asset"] = {key: sum(float(o.get("asset", {}).get(key, 0) or 0) for o in outcomes if o.get("asset")) / n
                           for key in ("retrieval_coverage", "capability_satisfaction", "first_pass_actual_geometry_validation", "resolver_calls")}
        differences = [o["asset"]["target_actual_log_size_l1_mean"] for o in outcomes if o.get("asset") and o["asset"]["target_actual_log_size_l1_mean"] is not None]
        report["asset"]["target_actual_log_size_l1_mean"] = float(np.mean(differences)) if differences else None
        report["system"] = {key: sum(float(o.get("system", {}).get(key, 0) or 0) for o in outcomes if o.get("system")) / n
                            for key in ("repaired_success", "fallback_rate", "final_commit", "asset_retries", "repair_calls")}
        if not commit_evaluation_requested and not any(o.get("system", {}).get("commit_attempted") for o in outcomes if o.get("system")):
            report["system"]["final_commit"] = None
        runtime_latency = [o["system"]["end_to_end_latency_ms"] for o in outcomes if
                           o.get("system") and o["system"].get("end_to_end_latency_ms") is not None]
        end_to_end = [o["fastfill_latency_ms"]+o["system"]["end_to_end_latency_ms"] for o in outcomes if
                      o.get("fastfill_latency_ms") is not None and o.get("system") and
                      o["system"].get("end_to_end_latency_ms") is not None]
        report["latency_ms"].update(runtime_p50=float(np.percentile(runtime_latency, 50)) if runtime_latency else None,
                                    runtime_p95=float(np.percentile(runtime_latency, 95)) if runtime_latency else None,
                                    end_to_end_p50=float(np.percentile(end_to_end, 50)) if end_to_end else None,
                                    end_to_end_p95=float(np.percentile(end_to_end, 95)) if end_to_end else None)
        report["system"]["solver_success"] = None
    return report


def run_evaluation(data, output, *, checkpoint=None, baseline="structured", predictions=None,
                   catalog=None, device="cpu", max_length=None, max_samples=None,
                   commit_in_memory=False, hungarian=True, asset_retries=2, repair_calls=0, repair_step_m=.25,
                   max_seconds=10., max_new_tokens=2048, required_levels=("bbox",), baseline_fit=None,
                   projections=("full",), grid_decode="spread", save_head_outputs=None, head_outputs=None):
    """Evaluate every request once per projection; ``max_length=None`` binds to the checkpoint.

    ``save_head_outputs`` (structured checkpoint only): a new directory, apart from ``output``, receiving
    ``<projection>/row-<row>.npz`` per request (``head_arrays``, the condition, or the online failure) and, at the end,
    ``manifest.json`` (``HEAD_MANIFEST_KEYS`` of this run). ``head_outputs`` replaces the checkpoint with such a directory:
    every request is re-decoded on the CPU (``decode_head_outputs``) and scored exactly as online; its manifest is
    reported as ``head_outputs_manifest``."""
    validate_required_levels(required_levels)
    if sum(source is not None for source in (checkpoint, predictions, head_outputs)) != 1:
        raise ValueError("provide exactly one checkpoint, prediction JSONL or head-output directory")
    if save_head_outputs is not None and (checkpoint is None or baseline != "structured"):
        raise ValueError("head outputs are saved from a structured checkpoint")
    projections = tuple(projections)
    if not projections or len(set(projections)) != len(projections) or set(projections) - set(PROJECTIONS):
        raise ValueError(f"projections must be distinct names from {PROJECTIONS}")
    if predictions is not None and projections != ("full",):
        raise ValueError("supplied predictions answer the full condition; evaluate other projections from a checkpoint")
    if head_outputs is not None:
        missing = [p for p in projections if not (Path(head_outputs) / p).is_dir()]
        if missing:
            raise ValueError(f"{head_outputs} holds no head outputs of projection(s) {missing}")
    target = safe_output(output)
    if save_head_outputs is not None:
        save_head_outputs = safe_output(save_head_outputs)
        if save_head_outputs == target or target in save_head_outputs.parents or save_head_outputs in target.parents:
            raise ValueError("--save-head-outputs and --output must be separate directories, neither inside the other")
        for projection in projections:
            (save_head_outputs / projection).mkdir(parents=True, exist_ok=False)
    samples = read_samples(data, max_samples=max_samples)
    if baseline_fit:
        with Path(baseline_fit).open() as stream:
            fit = fit_baselines(json.loads(line) for line in stream if line.strip())
        fit_source = f"baseline_fit_file:{baseline_fit}"
    else:
        fit, fit_source = fit_baselines(samples), "evaluation_set_leave_one_out"
    model = tokenizer = binding = None
    max_length_source = "argument" if max_length is not None else None
    if checkpoint:
        binding = load_checkpoint_config(checkpoint)
        if max_length is None and binding["max_length"] is not None:
            max_length, max_length_source = binding["max_length"], binding["source"]
        if baseline == "structured":
            model = load_model(checkpoint, device=device)
            tokenizer_path = Path(checkpoint).parent / "tokenizer"
            tokenizer = load_tokenizer("tiny" if model.config.backbone == "tiny" else str(tokenizer_path), local_files_only=True)
        else:
            from fastfill.v2.text_sft import load_text_model
            model, tokenizer = load_text_model(checkpoint, device=device)
    if max_length is None:
        max_length, max_length_source = DEFAULT_MAX_LENGTH, "default"
    supplied = None
    if predictions:
        with Path(predictions).open() as stream:
            supplied = [line.rstrip("\n") for line in stream if line.strip()]
        if len(supplied) != len(samples):
            raise ValueError("predictions must have one row per selected request, including failures")
    resolver = None
    if catalog:
        from fastfill.v2.serve import load_catalog
        resolver = load_catalog(catalog)
    if commit_in_memory and resolver is None:
        raise ValueError("commit evaluation requires an asset catalog")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))  # CPU inference may use more threads

    def evaluate_row(i, sample, projection):
        start = time.perf_counter()
        raw, fastfill_latency, head = None, None, {} if save_head_outputs is not None else None
        try:
            if supplied is not None:
                raw = supplied[i]
                def reject_constant(value):
                    raise ValueError(f"nonfinite prediction JSON constant {value}")
                parsed = json.loads(raw, parse_constant=reject_constant)
                layout = parsed.get("layout", parsed) if isinstance(parsed, dict) else parsed
            elif head_outputs is not None:
                raw = layout = decode_head_outputs(Path(head_outputs) / projection / f"row-{i:06d}.npz", sample["condition"],
                                                   grid_decode=grid_decode)
            elif baseline == "structured":
                raw = layout = predict_layout(model, tokenizer, sample["condition"], max_length=max_length, device=device,
                                              grid_decode=grid_decode, head=head)
            else:
                from fastfill.v2.text_sft import generate_text
                raw = generate_text(model, tokenizer, sample["condition"], max_length=max_length,
                                    max_new_tokens=max_new_tokens, device=device)
                layout = json.loads(raw)
            fastfill_latency = (time.perf_counter() - start) * 1000 if checkpoint else None
            result = evaluate_layout(layout, sample, resolver=resolver, commit_in_memory=commit_in_memory,
                                    hungarian=hungarian, asset_retries=asset_retries, repair_calls=repair_calls,
                                    repair_step_m=repair_step_m, max_seconds=max_seconds, required_levels=required_levels)
        except (ValueError, KeyError, TypeError, RuntimeError, StoredFailure) as exc:
            if checkpoint and fastfill_latency is None:
                fastfill_latency = (time.perf_counter() - start) * 1000
            error = exc.error if isinstance(exc, StoredFailure) else {"type": type(exc).__name__, "message": str(exc)}
            result = {"error": error, "raw_prediction": raw,
                      "model": {"schema_success": False, "requested_ids_exactly_once": False, "positive_valid_size": False}}
        if head is not None:  # a failure before the forward pass is stored as that failure
            np.savez_compressed(save_head_outputs / projection / f"row-{i:06d}.npz", condition=np.array(json.dumps(sample["condition"])),
                                **(head or {"error_type": np.array(result["error"]["type"]),
                                            "error_message": np.array(result["error"]["message"])}))
        labelled = migrate_legacy_row(sample)
        result.update(row=i, provenance=sample["provenance"], constraint_count=len(sample["condition"].get("constraints", ())),
                      evaluation_wall_time_ms=(time.perf_counter() - start) * 1000,
                      fastfill_latency_ms=fastfill_latency,
                      fastfill_latency_status=("failure" if "error" in result else "success") if checkpoint else "unobserved",
                      baselines=baseline_metrics(sample, fit, exclude_row=None if baseline_fit else i),
                      ground_truth_validation=_room_validation(validate_scene(  # the labels as written, same validator
                          labelled["condition"], labelled["target"]["objects"], required_levels=required_levels)))
        return result

    reports, outcomes = {}, {}
    for projection in projections:
        rows = list(enumerate(samples)) if projection == "full" else [
            (i, projected) for i, projected in enumerate(map(project_minimal, samples)) if projected is not None]
        outcomes[projection] = [evaluate_row(i, sample, projection) for i, sample in rows]
        reports[projection] = summarize(outcomes[projection], asset_evaluation_requested=resolver is not None,
                                        commit_evaluation_requested=commit_in_memory,
                                        baseline_fit_source=fit_source) if rows else {"requests": 0}
        reports[projection].update(projection=projection, skipped_non_rectangular_rooms=len(samples) - len(rows))
    report = {**reports[projections[0]], "projections": {name: reports[name] for name in projections[1:]}}
    metadata = run_metadata(data)
    report.update(baseline=baseline if checkpoint else "supplied_head_outputs" if head_outputs else "supplied_predictions",
                  head_outputs=str(Path(head_outputs).resolve()) if head_outputs else None,
                  saved_head_outputs=str(save_head_outputs) if save_head_outputs else None, subset_limit=max_samples,
                  evaluation_matching="exchangeable_groups" if hungarian else "fixed", commit_scope="in_memory" if commit_in_memory else "not_attempted",
                  runtime_budget={"asset_retries": asset_retries, "repair_calls": repair_calls,
                                  "repair_step_m": repair_step_m, "max_seconds": max_seconds}, required_levels=list(required_levels),
                  geometry_level="upright_obb_bev_iou", source_root_modified=False,
                  checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None, checkpoint_binding=binding,
                  max_length=max_length if checkpoint else None, max_length_source=max_length_source if checkpoint else None,
                  grid_decode=grid_decode if checkpoint or head_outputs else None,
                  predictions_sha256=fingerprint(predictions) if predictions else None,
                  **{key: metadata[key] for key in ("data_path", "data_sha256", "implementation_sha256", "code_commit", "code_dirty")})
    manifest = Path(head_outputs) / "manifest.json" if head_outputs else None  # the saving run's checkpoint, data, code
    report["head_outputs_manifest"] = json.loads(manifest.read_text()) if manifest and manifest.is_file() else None
    if save_head_outputs is not None:
        (save_head_outputs / "manifest.json").write_text(json.dumps({key: report[key] for key in HEAD_MANIFEST_KEYS}, indent=2) + "\n")
    target.mkdir(parents=True, exist_ok=False)
    (target / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for index, projection in enumerate(projections):
        with (target / ("outcomes.jsonl" if not index else f"outcomes-{projection}.jsonl")).open("x") as stream:
            for outcome in outcomes[projection]:
                stream.write(json.dumps(outcome, allow_nan=False) + "\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--baseline", choices=["structured", "text"], default="structured")
    p.add_argument("--catalog", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, help="default: the checkpoint's training max_length")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--fixed-correspondence", action="store_true")
    p.add_argument("--commit-in-memory", action="store_true")
    p.add_argument("--asset-retries", type=int, default=2)
    p.add_argument("--repair-calls", type=int, default=0)
    p.add_argument("--repair-step-m", type=float, default=.25)
    p.add_argument("--required-levels", nargs="+", choices=("bbox", "mesh", "physics", "solver"), default=["bbox"])
    p.add_argument("--max-seconds", type=float, default=10.)
    p.add_argument("--baseline-fit", type=Path, help="JSONL rows for the category baselines; default: evaluation set, leave-one-out")
    p.add_argument("--projection", nargs="+", choices=PROJECTIONS, default=["full"], dest="projections",
                   help="full condition and/or its three-field projection (rectangular rooms); the first is the report's top level")
    p.add_argument("--grid-decode", choices=GRID_DECODES, default="spread",
                   help="grid position head: collision-aware spread (default) or plain per-object argmax")
    p.add_argument("--save-head-outputs", type=Path, metavar="DIR",
                   help="with --checkpoint: new directory for every request's head outputs (<projection>/row-<n>.npz)")
    p.add_argument("--from-head-outputs", type=Path, metavar="DIR", dest="head_outputs",
                   help="instead of --checkpoint: re-decode a --save-head-outputs directory on the CPU and score it")
    args = vars(p.parse_args(argv))
    args["hungarian"] = not args.pop("fixed_correspondence")
    print(json.dumps(run_evaluation(**args), indent=2))


if __name__ == "__main__":
    main()
