"""Detached assignment only within explicitly certified exchangeable groups."""
from __future__ import annotations

import json
import math

import torch
from scipy.optimize import linear_sum_assignment


def _rename_references(row, mapping, scalar_fields, list_fields=()):
    """Copy a schema row, renaming only its explicit object-reference fields.

    IDs share string representations with ordinary descriptions and attributes;
    those values, dictionary keys, group labels and asset-local surface IDs are
    independent namespaces. Nested metadata must never be interpreted as refs.
    ``object_ids`` is the existing top-level constraint reference-list extension.
    """
    renamed = {}
    for key, value in row.items():
        if key in scalar_fields and isinstance(value, str):
            renamed[key] = mapping.get(value, value)
        elif key in list_fields and isinstance(value, list):
            renamed[key] = [mapping.get(ref, ref) if isinstance(ref, str) else ref for ref in value]
        else:
            renamed[key] = value
    return renamed


def certify_group(objects, constraints, indices):
    """Check conditioning invariance under generators of the group's permutations."""
    fingerprints = [json.dumps({k: v for k, v in objects[i].items() if k not in {"id", "exchangeable_group"}},
                               sort_keys=True) for i in indices]
    if len(set(fingerprints)) != 1:
        raise ValueError("exchangeable group has distinct descriptions/requirements/roles")
    canonical = lambda rows: sorted(json.dumps(r, sort_keys=True) for r in rows)
    original = canonical(constraints)
    object_graph = canonical(objects)
    for i in indices[1:]:
        mapping = {objects[indices[0]]["id"]: objects[i]["id"], objects[i]["id"]: objects[indices[0]]["id"]}
        renamed_constraints = [_rename_references(c, mapping, ("object_id", "target_id", "parent_id"),
                                                   ("target_ids", "object_ids")) for c in constraints]
        renamed_objects = [_rename_references(o, mapping, ("id", "support_parent")) for o in objects]
        if canonical(renamed_constraints) != original or canonical(renamed_objects) != object_graph:
            raise ValueError("exchange would change a constraint or support role; use fixed identities")


def group_labels(objects, constraints, position_valid):
    """C5 exchangeable labels per request: identical non-id fields, complete position labels, certified swap.

    A signature set (identical request fields except ``id``) whose whole swap fails certification
    falls back to its members that no constraint reference field (the ones ``certify_group`` renames)
    and no ``support_parent`` names; that remainder always certifies and is grouped under the set's
    label when it has at least two members.
    """
    candidates = {}
    for i, obj in enumerate(objects):
        if all(position_valid[i]):
            signature = json.dumps({k: v for k, v in obj.items() if k != "id"}, sort_keys=True)
            candidates.setdefault(signature, []).append(i)
    referenced = {o.get("support_parent") for o in objects}
    for c in constraints:
        referenced.update(c.get(k) for k in ("object_id", "target_id", "parent_id"))
        referenced.update(ref for k in ("target_ids", "object_ids") for ref in c.get(k) or ())
    labels = [None] * len(objects)
    for number, indices in enumerate(candidates.values()):
        for members in (indices, [i for i in indices if objects[i]["id"] not in referenced]):
            if len(members) < 2:
                break
            try:
                certify_group(objects, constraints, members)
            except ValueError:
                continue
            for i in members:
                labels[i] = f"anonymous_{number}"
            break
    return labels


def validate_matching_options(enabled, alpha_position, alpha_size):
    if not isinstance(enabled, bool):
        raise ValueError("matching enabled/hungarian flag must be boolean")
    weights = (alpha_position, alpha_size)
    if (any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in weights)
            or alpha_position + alpha_size <= 0):
        raise ValueError("matching weights must be native finite nonnegative numbers and not both zero")


@torch.no_grad()
def match_batch(predictions, batch, enabled=True, alpha_position=1., alpha_size=1., *, loss_config=None):
    """Groups come from collate's ``batch["exchangeable_group"]`` (per sample, per
    request slot), never from the rendered condition objects. A group needs
    complete position labels on every member; otherwise it keeps fixed identity.
    Position cost (times ``alpha_position``): the decoded L1 for regression
    predictions; for grid_residual predictions (``position_cell_logits``) each
    (prediction slot, target) pair costs exactly the loss's ``position`` terms,
    ``losses._grid_position`` under ``loss_config`` (default ``LossConfig()``):
    ``position_cell * CE + position_residual * GT-cell residual + z / 3`` on the
    prediction slot's learned (not ``fixed_position_mask``) coordinates, summed in
    float64 (on the CPU for Metal, which has none).
    Size enters the cost only when every member also has a complete size label;
    a pair whose target object is box-symmetric (its own
    ``batch["size_axis_swap_allowed"]``) costs the minimum log-size error over the
    target's (sx, sy) and (sy, sx) orders, every other pair the written order.
    Members marked in ``batch["given_pose_mask"]`` (masked layout training) keep
    fixed identity: the rest of their group matches among itself.
    """
    validate_matching_options(enabled, alpha_position, alpha_size)
    if enabled and "exchangeable_group" not in batch:
        raise ValueError("exchangeable matching requires batch exchangeable_group from collate")
    grid = "position_cell_logits" in predictions
    if grid:
        from fastfill.v2.losses import GRID_KEYS, LossConfig, _grid_position  # losses imports this module
        cfg = LossConfig() if loss_config is None else loss_config
    valid = batch["slot_mask"]
    bsz, slots = valid.shape
    assignments = []
    for b in range(bsz):
        indices = list(range(slots))
        objects = batch["objects"][b]
        if len(objects) != int(valid[b].sum()) or not valid[b, :len(objects)].all():
            raise ValueError("slots must be contiguous and bound to every request")
        groups = {}
        if enabled:
            labels = batch["exchangeable_group"][b]
            if len(labels) != len(objects) or any(label is not None and not isinstance(label, str) for label in labels):
                raise ValueError("exchangeable_group must hold one string-or-null label per request slot")
            for i, label in enumerate(labels):
                if label:
                    groups.setdefault(label, []).append(i)
        if any(len(group) > 1 for group in groups.values()) and "conditions" not in batch:
            raise ValueError("exchangeable matching requires original conditions and constraint graph")
        conditions = batch.get("conditions", [{} for _ in range(bsz)])
        for group in groups.values():
            certify_group(objects, conditions[b].get("constraints", []), group)
            if "given_pose_mask" in batch:  # a given pose was revealed for its own target only
                group = [i for i in group if not bool(batch["given_pose_mask"][b, i])]
            if len(group) < 2 or not batch["validity"]["position"][b, group].all():
                continue
            pos, gt_pos = predictions["position_normalized"][b, group], batch["targets"]["position_normalized"][b, group]
            if not torch.isfinite(pos).all() or not torch.isfinite(gt_pos).all():
                raise ValueError("nonfinite position geometry in matching group")
            if grid:  # pair (i, j) at flat i * n + j: prediction slot group[i], target group[j]
                n = len(group)
                rows = torch.tensor(group, device=valid.device).repeat_interleave(n)
                cols = torch.tensor(group, device=valid.device).repeat(n)
                learn = ~batch.get("fixed_position_mask", torch.zeros_like(batch["validity"]["position"]))[b, rows]
                parts = _grid_position({k: predictions[k][b, rows] for k in ("position_normalized", *GRID_KEYS)},
                                       batch["targets"]["position_normalized"][b, cols], torch.ones_like(rows, dtype=torch.bool), learn, cfg)
                # Metal has no float64: its pair cost is built on the CPU; CPU/CUDA keep their device.
                host = torch.device("cpu") if valid.device.type == "mps" else valid.device
                part, learn = (lambda name: parts[name].to(host).double()), learn.to(host)
                pair = torch.zeros(n * n, dtype=torch.float64, device=host)
                pair[learn[:, 0] & learn[:, 1]] = (cfg.position_cell * part("position_cell") +
                                                   cfg.position_residual * part("position_residual"))
                pair[learn[:, 2]] += part("position_z")
                cost = alpha_position * pair.view(n, n)
            else:
                cost = alpha_position * (pos[:, None] - gt_pos[None]).abs().sum(-1)
            if batch["validity"]["size"][b, group].all():
                size, gt_size = predictions["size"][b, group], batch["targets"]["size"][b, group]
                if not torch.isfinite(size).all() or not torch.isfinite(gt_size).all() or (size <= 0).any() or (gt_size <= 0).any():
                    raise ValueError("nonfinite or nonpositive size geometry in matching group")
                size_cost = (size.log()[:, None] - gt_size.log()[None]).abs().sum(-1)
                if "size_axis_swap_allowed" in batch:  # columns are targets: each pair follows its target's flag
                    swapped = (size.log()[:, None] - gt_size.log()[None, :, [1, 0, 2]]).abs().sum(-1)
                    size_cost = torch.where(batch["size_axis_swap_allowed"][b, group][None], torch.minimum(size_cost, swapped), size_cost)
                cost = cost + (alpha_size * size_cost).to(cost.device)  # a no-op except for an mps grid cost
            if not torch.isfinite(cost).all():
                raise ValueError("matching cost is not finite")
            # SciPy deterministic ordered input; exact ties use its row/column order.
            rows, cols = linear_sum_assignment(cost.cpu().double().numpy())
            chosen = {group[int(row)]: group[int(col)] for row, col in zip(rows, cols)}
            indices = [chosen.get(k, old) for k, old in enumerate(indices)]
        assignments.append(indices)
    return torch.tensor(assignments, dtype=torch.long, device=valid.device)


def permute_relations(assignment, parents=None, relation_matrix=None):
    """Negative parent indices represent external supports and remain unchanged."""
    if (assignment.ndim != 1 or assignment.dtype not in {torch.int32, torch.int64} or
            not torch.equal(assignment.sort().values, torch.arange(len(assignment), device=assignment.device, dtype=assignment.dtype))):
        raise ValueError("relation assignment must be a one-to-one permutation")
    if parents is not None and parents.shape != assignment.shape:
        raise ValueError("parent targets must have one value per assigned object")
    if relation_matrix is not None and relation_matrix.shape != (len(assignment), len(assignment)):
        raise ValueError("relation matrix shape must match assigned objects")
    inverse = assignment.argsort()
    new_parents = None
    if parents is not None:
        selected = parents[assignment]
        external = selected < 0
        if ((selected >= len(assignment)) & ~external).any():
            raise ValueError("parent index out of range")
        new_parents = torch.where(external, selected, inverse[selected.clamp_min(0)])
    matrix = None if relation_matrix is None else relation_matrix[assignment][:, assignment]
    return new_parents, matrix
