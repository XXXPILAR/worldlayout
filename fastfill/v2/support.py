"""Support pointer decoding: the ``support_head``'s logits (floor | wall | request slots) -> ``support_parent`` ids.

A request that declares its support (``support_parent``, or a hard ``on`` constraint merged in by
``validation.effective_support_requests``) keeps it. The others take their most probable parent that keeps the
support graph acyclic, the most confident slots first; floor and wall never close a cycle, so every slot gets one.
"""
from __future__ import annotations

import torch

from fastfill.v2.batch import SUPPORT_CLASSES


def _closes_cycle(parents: dict, child: str, parent: str) -> bool:
    seen = set()
    while parent is not None and parent not in seen:
        if parent == child:
            return True
        seen.add(parent)
        parent = parents.get(parent)
    return False


def decode_support_parents(logits: torch.Tensor, requests: list[dict]) -> list[str]:
    """``logits`` (n, 2 + n') of one scene's ``n`` requests (``requests``: effective support requests, in slot
    order) -> one parent per request: "floor", "wall" or another request's id."""
    ids = [r["id"] for r in requests]
    names = list(SUPPORT_CLASSES) + ids
    probability = logits[:, :len(names)].float().softmax(-1)
    parents = {r["id"]: r["support_parent"] for r in requests if r.get("support_parent") is not None}
    order = sorted((i for i, r in enumerate(requests) if r.get("support_parent") is None),
                   key=lambda i: -float(probability[i].max()))
    for i in order:
        for k in probability[i].argsort(descending=True).tolist():
            if names[k] != ids[i] and not _closes_cycle(parents, ids[i], names[k]):
                parents[ids[i]] = names[k]
                break
    return [parents[i] for i in ids]
