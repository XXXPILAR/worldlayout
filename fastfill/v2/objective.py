"""Label/term eligibility shared by training preflight and the criterion.

Eligibility is independent of numeric loss: a correctly predicted supervised
example remains eligible. Scene regularizers need reliable input geometry,
not fabricated target labels. Counts describe eligible terms, not coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def supervision_masks(batch, validity=None):
    validity = batch["validity"] if validity is None else validity
    mask = batch["slot_mask"]
    if "given_pose_mask" in batch:  # masked layout training: a given request is a condition, never a target
        mask = mask & ~batch["given_pose_mask"]
    fields = {}
    for key in ("position", "size"):
        learn = ~batch.get(f"fixed_{key}_mask", torch.zeros_like(validity[key]))
        fields[key] = mask & validity[key].all(-1) & learn.any(-1)
    fields["yaw"] = mask & validity["yaw"]
    # The implemented BEV operator deliberately retains the frozen complete
    # geometry-label policy. Yaw is always a learned head; it is not fixed by
    # a request field, even when position or size coordinates are prescribed.
    fields["box"] = mask & validity["position"].all(-1) & validity["size"].all(-1) & validity["yaw"]
    return fields


def local_objective_count(batch, config, validity=None):
    masks = supervision_masks(batch, validity)
    count = sum(int(masks[key].sum()) for key in ("position", "size", "box") if getattr(config, key) > 0)
    if config.yaw_cls > 0 or config.yaw_reg > 0:
        count += int(masks["yaw"].sum())
    if config.collision or config.boundary:
        from fastfill.v2.regularizers import regularizer_specs
        if "conditions" not in batch:
            raise ValueError("scene regularizers require original conditions")
        for condition in batch["conditions"]:
            objects, _, pairs, fixed_pairs = regularizer_specs(condition, config)
            if config.boundary:
                count += len(objects)
            if config.collision:
                count += len(pairs) + len(fixed_pairs)
    return count


@dataclass(frozen=True)
class ObjectiveWindow:
    """Accumulate local eligibility until Accelerate synchronizes a window."""

    local_count: int = 0

    def add(self, local_count):
        if isinstance(local_count, bool) or not isinstance(local_count, int) or local_count < 0:
            raise ValueError("active objective count must be a nonnegative integer")
        return ObjectiveWindow(self.local_count + local_count)

    def global_count(self, accelerator):
        # All ranks enter this boundary, including ranks with zero local terms.
        count = torch.tensor(self.local_count, dtype=torch.int64, device=accelerator.device)
        return int(accelerator.reduce(count, reduction="sum").item())
