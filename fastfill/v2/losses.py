"""Continuous geometry supervision with field filtering and global counts.

Position head (chosen by the predictions, not the config): a ``grid_residual``
model also returns ``position_cell_logits`` / ``position_cell_residuals``; its
``position`` term is then ``position_cell * CE(GT cell) + position_residual *
L(GT-cell XY residual, half-cell units, mean of 2) + L(z)/3`` (z keeps exactly
its regression share). ``term_sums["position"]`` stays the decoded-position
error (identical definition for both heads, so runs compare); the grid parts
are logged as ``position_cell`` / ``position_residual`` / ``position_z``
(zero sums and counts under the regression head).

Box symmetry (``batch["size_axis_swap_allowed"]``): an object whose local axis
pairing is uncertain may be written as (sx, sy, yaw) or (sy, sx, yaw + pi/2).
Its candidates are k in {0,1,2,3}: yaw + k*pi/2 with (sx, sy, sz) for even k and
(sy, sx, sz) for odd k; the detached joint minimum of the weighted size, yaw CE
and yaw residual losses picks one. Without a valid yaw only the two size
orders compete. Requests that fix sx or sy pin the axis convention: size keeps
the written order and yaw only the box's own pi symmetry (order 2). A fixed sz
alone pins nothing (the swap never moves it).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from fastfill.v2.boxes import bev_giou
from fastfill.v2.geometry import decode_yaw, encode_grid_position, encode_yaw
from fastfill.v2.matching import match_batch, validate_matching_options
from fastfill.v2.objective import local_objective_count, supervision_masks

ELEMENTWISE_TYPES = {"l1", "smooth_l1"}
GRID_KEYS = ("position_cell_logits", "position_cell_residuals")
GRID_TERMS = ("position_cell", "position_residual", "position_z")


@dataclass(frozen=True)
class LossConfig:
    position: float = 1.
    size: float = 1.
    yaw_cls: float = 1.
    yaw_reg: float = 1.
    box: float = 0.
    box_operator: str = "bev_oriented_giou_convex_hull"
    hungarian: bool = True
    alpha_position: float = 1.
    alpha_size: float = 1.
    collision: float = 0.
    boundary: float = 0.
    position_type: str = "l1"
    size_type: str = "l1"
    smooth_l1_beta: float = 1.
    # Relative weights of the grid cell CE / residual inside ``position`` (grid_residual head only).
    position_cell: float = 1.
    position_residual: float = 1.
    # Expected cell cost (grid_residual head only): sum over cells of softmax(cell logits) x the cell's
    # boundary / door-clearance cost (``cell_costs``), so the gradient lowers the probability of bad cells
    # instead of only nudging the argmax cell's residual. ``door_clearance_m``: depth of a door's zone.
    expected_geometry: float = 0.
    door_clearance_m: float = .8
    # Support pointer CE (model.support_head): floor | wall | parent slot, on requests with a declared support
    # (``batch.support_targets``); logged as ``support`` and ``support_accuracy`` when on.
    support: float = 0.

    def __post_init__(self):
        weights = [getattr(self, key) for key in ("position", "size", "yaw_cls", "yaw_reg", "box", "collision", "boundary",
                                                  "position_cell", "position_residual", "expected_geometry",
                                                  "door_clearance_m", "support")]
        if any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("loss weights must be native finite nonnegative numbers")
        if self.position_type not in ELEMENTWISE_TYPES or self.size_type not in ELEMENTWISE_TYPES:
            raise ValueError(f"position_type/size_type must be one of {sorted(ELEMENTWISE_TYPES)}")
        beta = self.smooth_l1_beta
        if type(beta) not in {int, float} or not math.isfinite(beta) or beta <= 0:
            raise ValueError("smooth_l1_beta must be a native finite positive number")
        validate_matching_options(self.hungarian, self.alpha_position, self.alpha_size)


def _elementwise(error, kind, beta):
    if kind == "l1":
        return error.abs()
    return F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none", beta=beta)


def _mean(total, local_count, global_counts=True):
    # Every rank must reduce the same dtype, including ranks with no valid
    # float64 box terms. Counts also need no floating-point rounding.
    count = torch.tensor(local_count, dtype=torch.int64, device=total.device)
    world = 1
    if global_counts and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
        world = torch.distributed.get_world_size()
    # DDP averages gradients, so compensate once for global count.
    return total * world / count.clamp_min(1), int(count.item())


def _gather(tensor, assignment):
    index = assignment.reshape(*assignment.shape, *([1] * (tensor.ndim - 2)))
    return tensor.gather(1, index.expand_as(tensor))


def _safe_zero(predictions, valid):
    # Every head output enters the loss at least as 0: under DDP each rank must give every parameter a gradient,
    # also when its batch has no label for a head (a room without support declarations for the support head).
    return sum((value[valid] * 0).sum() for name, value in predictions.items()
               if name in {"position_normalized", "size", "yaw_logits", "yaw_residuals", "support_logits", *GRID_KEYS})


def _grid_position(predictions, gt, valid, coordinate_mask, cfg):
    """Per valid slot: cell CE and GT-cell residual loss (XY learned), z loss / 3 (z learned)."""
    logits, residuals = predictions["position_cell_logits"][valid], predictions["position_cell_residuals"][valid]
    grid = math.isqrt(logits.shape[-1])
    if grid * grid != logits.shape[-1] or residuals.shape != (*logits.shape, 2):
        raise ValueError("grid position logits must hold grid*grid cells with one XY residual pair per cell")
    xy = coordinate_mask[:, 0] & coordinate_mask[:, 1]
    cells, target = encode_grid_position(gt[xy, :2], grid)
    cell = F.cross_entropy(logits[xy], cells, reduction="none")
    selected = residuals[xy].gather(1, cells[:, None, None].expand(-1, 1, 2)).squeeze(1)
    residual = _elementwise(selected - target, cfg.position_type, cfg.smooth_l1_beta).mean(-1)
    z = coordinate_mask[:, 2]
    height = _elementwise(predictions["position_normalized"][valid][z, 2] - gt[z, 2], cfg.position_type, cfg.smooth_l1_beta) / 3
    return {"position_cell": cell, "position_residual": residual, "position_z": height}


class GeometryCriterion(nn.Module):
    def __init__(self, config=LossConfig()):
        super().__init__()
        if config.box and config.box_operator != "bev_oriented_giou_convex_hull":
            raise ValueError("only explicitly named BEV oriented convex-hull GIoU is implemented")
        self.config = config

    def forward(self, predictions, batch, *, global_counts=True):
        """``global_counts=False``: normalise by this rank's counts without the per-call all_reduce (diagnostics whose
        ranks run different numbers of batches)."""
        mask = batch["slot_mask"]
        grid = GRID_KEYS[0] in predictions
        for key in ("position_normalized", "size", "yaw_logits", "yaw_residuals", *(GRID_KEYS if grid else ())):
            if not torch.isfinite(predictions[key][mask]).all():
                raise ValueError(f"nonfinite valid prediction: {key}")
        if (predictions["size"][mask] <= 0).any():
            raise ValueError("predicted valid sizes must be positive")
        cfg = self.config
        given = batch.get("given_pose_mask", torch.zeros_like(mask))
        if (cfg.collision or cfg.boundary) and given.any():
            # Regularizers would score the given slots' own predictions, not the poses they were given.
            raise ValueError("scene regularizers are not implemented for given poses")
        assignment = match_batch(predictions, batch, cfg.hungarian, cfg.alpha_position, cfg.alpha_size, loss_config=cfg)
        targets = {k: _gather(v, assignment) for k, v in batch["targets"].items()}
        validity = {k: _gather(v, assignment) for k, v in batch["validity"].items() if k in {"position", "size", "yaw"}}
        masks = supervision_masks(batch, validity)
        active_local = local_objective_count(batch, cfg, validity)
        zero = _safe_zero(predictions, mask)
        # losses: per-term global means (gradient path). term_sums/term_counts: local
        # unweighted sums and valid counts, for window-level logging by the trainer.
        losses, counts, term_sums, term_counts = {}, {}, {}, {}
        per_slot = {}
        for key, pred_key, target_key in (("position", "position_normalized", "position_normalized"), ("size", "size", "size")):
            learn = ~batch.get(f"fixed_{key}_mask", torch.zeros_like(validity[key]))
            valid = masks[key]
            pred, gt = predictions[pred_key][valid], targets[target_key][valid]
            if not torch.isfinite(gt).all() or (key == "size" and (gt <= 0).any()):
                raise ValueError(f"invalid {key} target marked valid")
            # Fixed coords contribute zero /3 (masked before reduction; valid predictions are finite and positive).
            coordinate_mask = learn[valid]
            kind = getattr(cfg, f"{key}_type")
            error_of = lambda target: torch.where(coordinate_mask, _elementwise(
                pred.log() - target.log() if key == "size" else pred - target, kind, cfg.smooth_l1_beta), 0.).sum(-1) / 3
            per_slot[key] = error_of(gt)
            if key == "size":  # (sy, sx, sz) for box-symmetric slots; resolved jointly with yaw below
                per_slot["size_swapped"] = error_of(gt[:, [1, 0, 2]])
                continue
            total = per_slot[key].sum()
            term_sums[key], term_counts[key] = total.detach().float(), int(valid.sum())
            for name in GRID_TERMS:
                term_sums[name], term_counts[name] = torch.zeros((), device=zero.device), 0
            if grid:
                parts = _grid_position(predictions, gt, valid, coordinate_mask, cfg)
                for name, values in parts.items():
                    term_sums[name], term_counts[name] = values.detach().float().sum(), len(values)
                total = (cfg.position_cell * parts["position_cell"].sum() + cfg.position_residual *
                         parts["position_residual"].sum() + parts["position_z"].sum())
            losses[key], counts[key] = _mean(total + zero, int(valid.sum()), global_counts)
        size_valid = masks["size"]
        swap = (_gather(batch["size_axis_swap_allowed"], assignment) if "size_axis_swap_allowed" in batch
                else torch.zeros_like(mask)) & mask
        pinned = swap & batch.get("fixed_size_mask", torch.zeros_like(validity["size"]))[..., :2].any(-1)  # sx/sy only
        swap = swap & ~pinned
        size_row = torch.full(mask.shape, -1, dtype=torch.long, device=mask.device)
        size_row[size_valid] = torch.arange(int(size_valid.sum()), device=mask.device)
        plain, swapped = per_slot["size"], per_slot["size_swapped"]
        odd = swap[size_valid] & (swapped.detach() < plain.detach())  # size-only choice without a valid yaw
        angle_valid = masks["yaw"]
        logits, residuals = predictions["yaw_logits"][angle_valid], predictions["yaw_residuals"][angle_valid]
        yaw = targets["yaw"][angle_valid]
        if not torch.isfinite(yaw).all():
            raise ValueError("nonfinite yaw target marked valid")
        orders = batch.get("yaw_symmetry_order", torch.ones_like(mask, dtype=torch.long))
        orders = _gather(orders, assignment)[angle_valid]
        cls_terms, reg_terms = [], []
        for a, r, y, order, box_symmetric, pin, row in zip(logits, residuals, yaw, orders, swap[angle_valid],
                                                           pinned[angle_valid], size_row[angle_valid]):
            n = int(order)
            if n < 1 or n > 36:
                raise ValueError("yaw symmetry order must be 1..36 with source justification")
            n = 4 if box_symmetric else 2 if pin else n  # a pinned swap object's order 4 stood only for the swap
            candidates = y + torch.arange(n, device=y.device, dtype=y.dtype) * (2 * math.pi / n)
            bins, gt_residual = encode_yaw(candidates, a.shape[-1])
            cls = F.cross_entropy(a.expand(n, -1), bins, reduction="none")
            reg = F.smooth_l1_loss(r[bins], gt_residual, reduction="none", beta=1.)
            cost = cfg.yaw_cls * cls + cfg.yaw_reg * reg
            if box_symmetric and row >= 0:
                cost = cost + cfg.size * torch.stack((plain[row], swapped[row])).repeat(2)
            choice = cost.detach().argmin()
            if box_symmetric and row >= 0:
                odd[row] = bool(choice % 2)
            cls_terms.append(cls[choice])
            reg_terms.append(reg[choice])
        total = torch.where(odd, swapped, plain).sum()
        losses["size"], counts["size"] = _mean(total + zero, int(size_valid.sum()), global_counts)
        term_sums["size"], term_counts["size"] = total.detach().float(), int(size_valid.sum())
        losses["yaw_cls"], counts["yaw"] = _mean(sum(cls_terms, zero), int(angle_valid.sum()), global_counts)
        losses["yaw_reg"], _ = _mean(sum(reg_terms, zero), int(angle_valid.sum()), global_counts)
        for key, terms in (("yaw_cls", cls_terms), ("yaw_reg", reg_terms)):
            term_sums[key] = sum((t.detach().float() for t in terms), torch.zeros((), device=zero.device))
            term_counts[key] = int(angle_valid.sum())
        box_valid = masks["box"]
        box_terms = []
        yaw_pred = decode_yaw(predictions["yaw_logits"], predictions["yaw_residuals"])
        world_pred = predictions["position_normalized"] * batch["scale"][:, None] + batch["origin"][:, None]
        if cfg.box:
            for b, i in box_valid.nonzero().tolist():
                gt_pos = targets["position_normalized"][b, i] * batch["scale"][b] + batch["origin"][b]
                box_terms.append(1 - bev_giou(world_pred[b, i], predictions["size"][b, i], yaw_pred[b, i],
                                             gt_pos, targets["size"][b, i], targets["yaw"][b, i]))
        losses["box"], counts["box"] = _mean(sum(box_terms, zero), len(box_terms), global_counts)
        term_sums["box"] = sum((t.detach().float() for t in box_terms), torch.zeros((), device=zero.device))
        term_counts["box"] = len(box_terms)
        if cfg.collision or cfg.boundary:
            from fastfill.v2.regularizers import scene_regularizers
            regularizers = scene_regularizers(world_pred, predictions["size"], yaw_pred, batch, cfg, global_counts)
            term_sums.update(regularizers.pop("sums"))
            term_counts.update(regularizers.pop("counts"))
            losses.update(regularizers)
        else:
            losses.update(collision=zero, boundary=zero)
            term_sums.update(collision=zero.detach().float(), boundary=zero.detach().float())
            term_counts.update(collision=0, boundary=0)
        if cfg.expected_geometry and not grid:  # e.g. the trainer's room-centre baseline: no cell distribution to score
            losses["expected_geometry"], counts["expected_geometry"] = zero, 0
            term_sums["expected_geometry"], term_counts["expected_geometry"] = zero.detach().float(), 0
        elif cfg.expected_geometry:
            if "cell_costs" not in batch:
                raise ValueError("expected_geometry needs batches collated with cell_costs")
            costs = _gather(batch["cell_costs"], assignment)
            cost_valid = _gather(batch["cell_cost_mask"], assignment) & mask & ~given
            probabilities = predictions["position_cell_logits"][cost_valid].float().softmax(-1)
            terms = (probabilities * costs[cost_valid].to(probabilities.dtype)).sum(-1)
            losses["expected_geometry"], counts["expected_geometry"] = _mean(terms.sum() + zero, len(terms), global_counts)
            term_sums["expected_geometry"], term_counts["expected_geometry"] = terms.detach().sum(), len(terms)
        if cfg.support and "support_logits" not in predictions:  # e.g. the trainer's category baseline: no pointer
            losses["support"], counts["support"] = zero, 0
            for key in ("support", "support_accuracy"):
                term_sums[key], term_counts[key] = zero.detach().float(), 0
        elif cfg.support:
            if "support_target" not in batch:
                raise ValueError("loss.support needs collated support targets")
            target = batch["support_target"]
            # a parent label names a request slot; the slot regressing that request's target is the one matched to it
            inverse = torch.empty_like(assignment).scatter_(1, assignment, torch.arange(
                assignment.shape[1], device=assignment.device).expand_as(assignment))
            parents = (target - 2).clamp_min(0)
            target = torch.where(target >= 2, 2 + inverse.gather(1, parents), target)
            valid = (target >= 0) & mask
            logits = predictions["support_logits"][valid].float()
            terms = F.cross_entropy(logits, target[valid], reduction="none")
            losses["support"], counts["support"] = _mean(terms.sum() + zero, len(terms), global_counts)
            term_sums["support"], term_counts["support"] = terms.detach().sum(), len(terms)
            term_sums["support_accuracy"] = (logits.argmax(-1) == target[valid]).float().sum().detach()
            term_counts["support_accuracy"] = len(terms)
        loss = sum(getattr(cfg, k) * v for k, v in losses.items())
        return {"loss": loss, **losses, "counts": counts, "assignment": assignment,
                "active_objective_count_local": active_local,
                "term_sums": term_sums, "term_counts": term_counts}
