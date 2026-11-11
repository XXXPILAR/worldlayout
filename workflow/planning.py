"""One semantic planning pass; all hard constraints remain owned by the input.

The plan holds the support graph (``parents``), soft relation preferences, functional groups (an anchor and its
members) and a placement priority; sizes are step ②'s and poses step ③'s. By default one LLM call proposes them
(``make_plan(..., llm=True)``); the input's support_parent values and hard ``on`` constraints stay locked, every LLM
field is checked, and any LLM failure falls back to the rule plan with the reason recorded in ``Plan.llm``.
``llm=False`` gives the rule plan alone (an ablation, and the offline tests).

Groups act through the solver's soft relations: a member gets a soft ``near`` to its anchor unless a relation
between the two already exists. Priority orders placement among objects whose supports are placed (1 = first).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from . import llm as llm_client

GROUP_NEAR_M = 1.5


@dataclass
class Plan:
    parents: dict[str, str]
    relations: list[dict] = field(default_factory=list)
    groups: list[dict] = field(default_factory=list)          # {"anchor": id, "members": [ids]}
    priority: dict[str, int] = field(default_factory=dict)     # id -> rank, 1 = place first
    draft_poses: dict[str, dict] = field(default_factory=dict)  # the poses the solver's local search starts from
    source: str = "rules"
    llm: dict = field(default_factory=dict)                    # calls, dropped fields, error


def _required_parents(condition: dict) -> dict[str, str]:
    parents = {obj["id"]: obj["support_parent"] for obj in condition["objects"]
               if obj.get("support_parent") is not None}
    for relation in condition["constraints"]:
        if relation["type"] == "on" and relation.get("hard", True):
            ident, parent = relation["object_id"], relation.get("target_id", relation.get("parent_id"))
            if ident in parents and parents[ident] != parent:
                raise ValueError(f"{ident}: support_parent {parents[ident]!r} conflicts with hard on -> {parent!r}")
            parents[ident] = parent
    return parents


def _would_cycle(parents: dict[str, str]) -> bool:
    for ident in parents:
        seen, current = set(), ident
        while current in parents:
            if current in seen:
                return True
            seen.add(current)
            current = parents[current]
    return False


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", (text or "").lower())


def _head(obj: dict) -> str:
    """The category's head noun ("bedside table" -> table, "table lamp" -> lamp, "tv stand" -> stand)."""
    words = _words(obj["category"])
    return words[-1] if words else ""


def _is(obj: dict, *heads: str) -> bool:
    return _head(obj) in heads


def _is_nightstand(obj: dict) -> bool:
    words = _words(obj["category"])
    return _head(obj) == "nightstand" or (_head(obj) in ("table", "stand", "cabinet") and
                                          ("bedside" in words or "night" in words))


def _is_desk_lamp(obj: dict) -> bool:
    label = " ".join(_words(f"{obj['category']} {obj.get('description', '')}"))
    return _is(obj, "lamp", "light") and re.search(r"\b(table|desk|bedside|reading) (lamp|light)\b", label) is not None


def _is_wall_item(obj: dict) -> bool:
    label = " ".join(_words(f"{obj['category']} {obj.get('description', '')}"))
    return re.search(r"\b(painting|wall art|wall clock|wall shelf|wall mounted|poster|picture frame)\b", label) is not None


def _plain_table(obj: dict) -> bool:
    """A desk or a dining / work table (not a bedside, side, coffee or end table)."""
    words = set(_words(obj["category"]))
    return _is(obj, "desk") or (_is(obj, "table") and not words & {"bedside", "night", "side", "coffee", "end", "console"})


def _heuristic_plan(condition: dict) -> Plan:
    objects = condition["objects"]
    parents = _required_parents(condition)

    def assign(ident: str, parent: str) -> None:
        if ident not in parents and parent != ident and not _would_cycle({**parents, ident: parent}):
            parents[ident] = parent

    def round_robin(items: list[dict], targets: list[str]) -> None:
        for k, obj in enumerate(items):
            if targets:
                assign(obj["id"], targets[k % len(targets)])

    for obj in objects:
        if _is_wall_item(obj):
            assign(obj["id"], "wall")
    lamps = [o for o in objects if _is_desk_lamp(o) and o["id"] not in parents]
    desk_lamps = [o for o in lamps if "desk" in _words(o.get("description", "")) + _words(o["category"])]
    nightstands = [o["id"] for o in objects if _is_nightstand(o)]
    desks = [o["id"] for o in objects if _is(o, "desk")]
    tables = [o["id"] for o in objects if _is(o, "table") and o["id"] not in nightstands]
    round_robin(desk_lamps, desks or nightstands or tables)
    round_robin([o for o in lamps if o not in desk_lamps], nightstands or desks or tables)
    beds = [o["id"] for o in objects if _is(o, "bed")]
    sofas = [o["id"] for o in objects if _is(o, "sofa", "couch")]
    round_robin([o for o in objects if _is(o, "pillow") and o["id"] not in parents], beds or sofas)
    for obj in objects:
        parents.setdefault(obj["id"], "floor")

    relations: list[dict] = []
    groups: dict[str, list[str]] = {}

    def only(candidates: list[str], ident: str) -> str | None:
        candidates = [c for c in candidates if c != ident]
        return candidates[0] if len(candidates) == 1 else None

    tables_for_chairs = [o["id"] for o in objects if _plain_table(o)]
    tvs = [o["id"] for o in objects if _is(o, "tv", "television")]
    for obj in objects:
        ident = obj["id"]
        if _is(obj, "chair") and parents[ident] == "floor":
            target = only(tables_for_chairs, ident)
            if target:
                relations += [{"type": "near", "object_id": ident, "target_id": target, "max_distance_m": .9,
                               "hard": False, "origin": "heuristic"},
                              {"type": "faces", "object_id": ident, "target_id": target, "tolerance_rad": math.pi / 3,
                               "hard": False, "origin": "heuristic"}]
                groups.setdefault(target, []).append(ident)
        elif _is(obj, "sofa", "couch"):
            target = only(tvs, ident)
            if target:
                relations.append({"type": "faces", "object_id": ident, "target_id": target, "hard": False,
                                  "origin": "heuristic"})
                groups.setdefault(target, []).append(ident)
        elif _is_nightstand(obj):
            target = only(beds, ident)
            if target:
                relations.append({"type": "near", "object_id": ident, "target_id": target, "max_distance_m": .6,
                                  "hard": False, "origin": "heuristic"})
                groups.setdefault(target, []).append(ident)
    return Plan(parents, relations, [{"anchor": a, "members": m} for a, m in groups.items()])


SEMANTIC_INSTRUCTIONS = """You plan the structure of an indoor furniture layout; positions and sizes are decided later. \
The user message is the room condition as JSON: floor polygon (metres, world x/y, z up), openings, fixed objects \
(obstacles that cannot move), the objects to place (id, category, description, sometimes support_parent) and \
constraints.

Answer with one JSON object only, no prose:
{"objects": {"<id>": {"support": "floor" | "wall" | "<id of another object or fixed object it rests on>",
                      "priority": n}},
 "groups": [{"anchor": "<id>", "members": ["<id>", ...]}],
 "relations": [{"type": "near", "object_id": "<id>", "target_id": "<id>", "max_distance_m": m},
               {"type": "faces", "object_id": "<id>", "target_id": "<id>"},
               {"type": "against_wall", "object_id": "<id>"}]}

Rules:
- Include every object id exactly once under "objects"; do not add or remove objects.
- support: what the object stands on or hangs from ("wall" for wall-mounted items such as paintings).
- priority: placement order, 1 = place first (large anchor furniture first, items that depend on others later).
- groups: functional groups (e.g. a table and its chairs, a bed and its nightstands); the anchor is the object the \
others are arranged around.
- relations are soft preferences you add; the input constraints already hold and must not be restated or changed.
- Keep a given support_parent and the input's "on" constraints as they are."""


def _number(value, low: float, high: float) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and low <= value <= high else None


def _apply_llm(plan: Plan, condition: dict, proposal: dict) -> dict:
    """Merges the checked parts of ``proposal`` into ``plan``; returns counts of what was dropped and why."""
    if not isinstance(proposal, dict):
        raise ValueError("LLM plan is not a JSON object")
    ids = [obj["id"] for obj in condition["objects"]]
    id_set = set(ids)
    fixed_ids = {obj["id"] for obj in condition["room"].get("fixed_objects", [])}
    locked = _required_parents(condition)
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    entries = proposal.get("objects") if isinstance(proposal.get("objects"), dict) else {}
    dropped["objects_missing"] = len(id_set - set(entries))
    ranked = []
    for ident, entry in entries.items():
        if ident not in id_set or not isinstance(entry, dict):
            drop("unknown_object"); continue
        parent = entry.get("support")
        if isinstance(parent, str) and ident not in locked and parent != plan.parents.get(ident):
            if parent in id_set | fixed_ids | {"floor", "wall"} and parent != ident and \
                    not _would_cycle({**plan.parents, ident: parent}):
                plan.parents[ident] = parent
            else:
                drop("support_invalid")
        rank = _number(entry.get("priority"), 1, 10 ** 6)
        if rank is not None:
            ranked.append((rank, ids.index(ident), ident))
        elif entry.get("priority") is not None:
            drop("priority_invalid")
    plan.priority = {ident: k + 1 for k, (_, _, ident) in enumerate(sorted(ranked))}

    known = id_set | fixed_ids
    for relation in proposal.get("relations") or []:
        if not isinstance(relation, dict) or relation.get("object_id") not in id_set:
            drop("relation_invalid"); continue
        kind, ident, target = relation.get("type"), relation["object_id"], relation.get("target_id")
        if kind == "against_wall":
            plan.relations.append({"type": kind, "object_id": ident, "hard": False, "origin": "llm"})
        elif kind in ("near", "faces") and target in known and target != ident:
            clean = {"type": kind, "object_id": ident, "target_id": target, "hard": False, "origin": "llm"}
            if kind == "near":
                distance = _number(relation.get("max_distance_m", 1.0), 0., 100.)
                clean["max_distance_m"] = max(.2, min(2.5, distance if distance is not None else 1.0))
            plan.relations.append(clean)
        else:
            drop("relation_invalid")
    groups = []
    for group in proposal.get("groups") or []:
        anchor = group.get("anchor") if isinstance(group, dict) else None
        members = [m for m in (group.get("members") or []) if m in id_set and m != anchor] if isinstance(group, dict) else []
        if anchor in known and members:
            groups.append({"anchor": anchor, "members": list(dict.fromkeys(members))})
        else:
            drop("group_invalid")
    if groups:
        plan.groups = groups
    return {k: v for k, v in dropped.items() if v}


def _group_relations(plan: Plan) -> None:
    """Groups enter the solver as soft near relations member -> anchor (when no relation links the two yet)."""
    linked = {frozenset((r["object_id"], r.get("target_id"))) for r in plan.relations if r.get("target_id")}
    for group in plan.groups:
        for member in group["members"]:
            if frozenset((member, group["anchor"])) not in linked and plan.parents.get(member) in ("floor", "wall"):
                plan.relations.append({"type": "near", "object_id": member, "target_id": group["anchor"],
                                       "max_distance_m": GROUP_NEAR_M, "hard": False, "origin": "group"})


def make_plan(condition: dict, *, llm: bool = True, model: str | None = None,
              base_url: str | None = None, api_key: str | None = None) -> Plan:
    """Step ①: support, priority, groups and relations (no sizes or poses)."""
    plan = _heuristic_plan(condition)
    if llm:
        try:
            text, call = llm_client.ask(SEMANTIC_INSTRUCTIONS,
                                        json.dumps(condition, ensure_ascii=False),
                                        model=model, base_url=base_url, api_key=api_key)
            plan.llm["calls"] = [call]
            first, last = text.find("{"), text.rfind("}")
            if first < 0 or last <= first:
                raise ValueError("LLM response does not contain a JSON object")
            plan.llm["dropped"] = _apply_llm(plan, condition, json.loads(text[first:last + 1]))
            plan.source = "llm+rules"
        except (RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValueError) and "requires --model" in str(exc):
                raise  # a configuration error, not an LLM failure
            plan = _fallback(condition, plan, exc)
    _group_relations(plan)
    return plan


def _fallback(condition: dict, failed: Plan, exc: Exception) -> Plan:
    plan = _heuristic_plan(condition)
    plan.source = "rules (llm failed)"
    plan.llm = {"calls": failed.llm.get("calls", []), "error": f"{type(exc).__name__}: {exc}"[:300]}
    return plan
