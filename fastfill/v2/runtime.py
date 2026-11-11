"""Offline v2 asset loop and the external WorldEdge transaction contract.

Asset.canonical_transform maps raw asset coordinates (including its pivot,
source scale/axis conventions) to a metre, Z-up, bbox-bottom-centred frame.
The world transform is T(bottom_center) @ Rz(yaw) @ canonical_transform.
No mesh rescaling or deletion is performed to satisfy validation.

Callbacks must implement their own I/O timeout: max_seconds is checked at
callback boundaries, rather than claiming to interrupt arbitrary host code.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from threading import RLock
from time import monotonic
from typing import Protocol

from .validation import _capability_names, _polygon, _validate_support_bbox, _vector, effective_support_requests, validate_required_levels, validate_scene


IDENTITY = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))


@dataclass(frozen=True)
class SupportSurface:
    """Verified horizontal surface in the asset canonical frame, not bbox top."""
    surface_id: str
    local_polygon_xy_m: tuple
    local_z_m: float

    def __post_init__(self):
        if not isinstance(self.surface_id, str) or not self.surface_id:
            raise ValueError("support surface needs a stable ID")
        _polygon(self.local_polygon_xy_m)
        _vector((self.local_z_m,), 1)
        object.__setattr__(self, "local_polygon_xy_m", tuple(tuple(p) for p in self.local_polygon_xy_m))

    def as_dict(self):
        return {"surface_id": self.surface_id, "local_polygon_xy_m": [list(p) for p in self.local_polygon_xy_m],
                "local_z_m": self.local_z_m}


@dataclass(frozen=True)
class Asset:
    ref: str
    category: str
    actual_size_local_m: tuple
    description: str = ""
    canonical_transform: tuple = IDENTITY
    semantic_front_local: tuple | None = None
    capabilities: tuple | None = None
    support_surfaces: tuple = ()
    provenance: tuple = ()

    def __post_init__(self):
        if not isinstance(self.ref, str) or not self.ref.strip() or not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("asset needs a reference and category")
        if not isinstance(self.description, str):
            raise ValueError("asset description must be a string")
        object.__setattr__(self, "actual_size_local_m", _vector(self.actual_size_local_m, 3, positive=True))
        rows = tuple(_vector(row, 4) for row in self.canonical_transform)
        if len(rows) != 4 or rows[3] != (0., 0., 0., 1.):
            raise ValueError("canonical_transform must be a finite affine 4x4 transform")
        determinant = (rows[0][0]*(rows[1][1]*rows[2][2]-rows[1][2]*rows[2][1]) -
                       rows[0][1]*(rows[1][0]*rows[2][2]-rows[1][2]*rows[2][0]) +
                       rows[0][2]*(rows[1][0]*rows[2][1]-rows[1][1]*rows[2][0]))
        if abs(determinant) < 1e-12:
            raise ValueError("canonical_transform is singular")
        object.__setattr__(self, "canonical_transform", rows)
        if self.semantic_front_local is not None:
            front = _vector(self.semantic_front_local, 3)
            if math.hypot(front[0], front[1]) < 1e-12 or abs(front[2]) > 1e-6:
                raise ValueError("semantic front must be a nonzero horizontal direction")
            if front[0] <= 0 or abs(front[1] / math.hypot(front[0], front[1])) > 1e-6:
                raise ValueError("known semantic front must already be canonical +X; normalize asset size and transform upstream")
            object.__setattr__(self, "semantic_front_local", (1., 0., 0.))
        if self.capabilities is not None:
            object.__setattr__(self, "capabilities", _capability_names(self.capabilities))
        if any(not isinstance(surface, SupportSurface) for surface in self.support_surfaces):
            raise ValueError("invalid verified support surface metadata")
        _validate_support_bbox([surface.as_dict() for surface in self.support_surfaces], self.actual_size_local_m)
        object.__setattr__(self, "support_surfaces", tuple(self.support_surfaces))
        if not isinstance(self.provenance, (tuple, list)) or any(not isinstance(x, str) for x in self.provenance):
            raise ValueError("asset provenance must be a list/tuple of source references")
        object.__setattr__(self, "provenance", tuple(self.provenance))


def _eligible(asset, request, prediction):
    # Asset has no verified typed-attribute evidence contract. Description text
    # or an arbitrary metadata claim cannot satisfy explicit color/material/etc.
    if request.get("attributes"):
        return False
    if asset.category.casefold().strip() != request["category"].casefold().strip():
        return False
    required = set(request.get("required_capabilities", ()))
    if required and (asset.capabilities is None or not required.issubset(asset.capabilities)):
        return False
    if request.get("semantic_front_required") and asset.semantic_front_local is None:
        return False
    size = asset.actual_size_local_m
    fixed = request.get("fixed_size_local_m")
    if fixed is not None and any(b is not None and abs(a-b) > 1e-6 for a, b in zip(size, fixed)):
        return False
    bounds = request.get("size_bounds_local_m")
    if bounds is not None and any(x < lo-1e-6 or x > hi+1e-6 for x, lo, hi in zip(size, bounds["min"], bounds["max"])):
        return False
    tolerance = request.get("retrieval_tolerance_log", request.get("retrieval_tolerance"))
    if tolerance is not None:
        limits = (tolerance,) * 3 if isinstance(tolerance, (int, float)) else tuple(tolerance)
        if len(limits) != 3 or any(not math.isfinite(x) or x < 0 for x in limits):
            raise ValueError("retrieval_tolerance is a scalar or three max absolute log ratios")
        if any(abs(math.log(a/b)) > limit+1e-12 for a, b, limit in
               zip(size, prediction["target_size_local_m"], limits)):
            return False
    return True


class Resolver(Protocol):
    def resolve(self, request_object, prediction, *, excluded_refs=()) -> Asset | None: ...


@dataclass(frozen=True)
class CatalogResolver:
    """Category/capability reference catalog; unverified typed attributes reject.

    Natural-language descriptions are not a verified attribute evidence source.
    Attribute-bearing requests need an explicit external evidence contract before
    this reference loop can admit them.
    """
    assets: tuple

    def __post_init__(self):
        if any(not isinstance(asset, Asset) for asset in self.assets):
            raise ValueError("catalog entries must be Asset instances")
        if len({a.ref for a in self.assets}) != len(self.assets):
            raise ValueError("duplicate asset references")
        object.__setattr__(self, "assets", tuple(self.assets))

    def resolve(self, request_object, prediction, *, excluded_refs=()):
        options = tuple(a for a in self.assets if a.ref not in excluded_refs and _eligible(a, request_object, prediction))
        return min(options, key=lambda a: (sum(abs(math.log(x/y)) for x, y in
                   zip(a.actual_size_local_m, prediction["target_size_local_m"])), a.ref), default=None)


def _transform(asset, position, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    world = ((c, -s, 0, position[0]), (s, c, 0, position[1]), (0, 0, 1, position[2]), (0, 0, 0, 1))
    return [[sum(world[i][k]*asset.canonical_transform[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def reconcile(assets, condition, prediction):
    """Resolve actual geometry and propagate verified support heights without changing targets."""
    requests = {obj["id"]: obj for obj in effective_support_requests(condition)}
    records = {obj["id"]: {**deepcopy(obj), "target_bottom_center_m": list(obj["bottom_center_m"]),
               "target_yaw_rad": obj["yaw_rad"], "actual_size_local_m": list(assets[obj["id"]].actual_size_local_m),
               "asset_ref": assets[obj["id"]].ref,
               "capabilities": list(assets[obj["id"]].capabilities) if assets[obj["id"]].capabilities is not None else None,
               "semantic_front_local": list(assets[obj["id"]].semantic_front_local) if
                    assets[obj["id"]].semantic_front_local is not None else None,
               "support_parent": requests[obj["id"]].get("support_parent"),
               "support_surfaces": [surface.as_dict() for surface in assets[obj["id"]].support_surfaces],
               "provenance": list(assets[obj["id"]].provenance)} for obj in prediction["objects"]}
    done = frozenset()
    fixed = {obj["id"]: obj for obj in condition["room"].get("fixed_objects", ())}
    while len(done) < len(records):
        ready = tuple(ident for ident in records if ident not in done and
                      (requests[ident].get("support_parent") not in records or requests[ident].get("support_parent") in done))
        if not ready:
            raise ValueError("support graph contains a cycle")
        for ident in ready:
            parent = requests[ident].get("support_parent")
            parent_obj = records.get(parent, fixed.get(parent))
            surfaces = parent_obj.get("support_surfaces", ()) if parent_obj else ()
            surface_id = requests[ident].get("support_surface_id")
            surfaces = tuple(sf for sf in surfaces if surface_id is None or sf["surface_id"] == surface_id)
            # Multiple unselected surfaces are ambiguous and remain unknown until a selector is supplied.
            if len(surfaces) == 1:
                p = records[ident]["bottom_center_m"]
                records = {**records, ident: {**records[ident], "bottom_center_m":
                           [p[0], p[1], parent_obj["bottom_center_m"][2]+surfaces[0]["local_z_m"]],
                           "support_surface_id": surfaces[0]["surface_id"]}}
        done = done.union(ready)
    return [{**records[obj["id"]], "asset_transform": _transform(assets[obj["id"]],
             records[obj["id"]]["bottom_center_m"], records[obj["id"]]["yaw_rad"])} for obj in prediction["objects"]]


@dataclass(frozen=True)
class RuntimeBudget:
    max_asset_retries: int = 2
    max_repair_calls: int = 0
    max_seconds: float = 10.0

    def __post_init__(self):
        for count in (self.max_asset_retries, self.max_repair_calls):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("retry and repair budgets must be nonnegative integers")
        if isinstance(self.max_seconds, bool) or not math.isfinite(self.max_seconds) or self.max_seconds < 0:
            raise ValueError("time budget must be finite and nonnegative")


@dataclass(frozen=True)
class BoundedTranslationRepair:
    """Small deterministic XY move for boundary/box penetration; no geometry changes.

    Moves one offending object and its support descendants per call. Other
    constraints may still fail, so every returned scene requires revalidation.
    This is a reference local repair, not a proof of feasible re-layout.
    """
    max_step_m: float = .25

    def __post_init__(self):
        if isinstance(self.max_step_m, bool) or not math.isfinite(self.max_step_m) or self.max_step_m <= 0:
            raise ValueError("translation step must be finite and positive")

    def __call__(self, condition, objects, report):
        index = {obj["id"]: obj for obj in objects}
        failure = next((check for check in report["checks"] if check["status"] == "violation" and
                        check["code"] in ("boundary", "collision", "fixed_collision")), None)
        if failure is None:
            return deepcopy(objects)
        ident = next((ident for ident in failure["object_ids"] if ident in index), None)
        if ident is None:
            return deepcopy(objects)
        obj = index[ident]
        if failure["code"] == "boundary":
            if condition["room"].get("boundary_known") is False:
                return deepcopy(objects)
            point = _polygon(condition["room"]["floor_polygon_xy_m"]).representative_point()
            delta = (point.x - obj["bottom_center_m"][0], point.y - obj["bottom_center_m"][1])
        else:
            fixed = {fixed["id"]: fixed for fixed in condition["room"].get("fixed_objects", ())}
            other = next((index.get(i, fixed.get(i)) for i in failure["object_ids"] if i != ident), None)
            if other is None:
                return deepcopy(objects)
            delta = tuple(obj["bottom_center_m"][q]-other["bottom_center_m"][q] for q in (0, 1))
        norm = math.hypot(*delta)
        delta = tuple(x * min(self.max_step_m, norm)/norm for x in delta) if norm > 1e-12 else (self.max_step_m, 0.)
        parents = {req["id"]: req.get("support_parent") for req in effective_support_requests(condition)}
        affected = frozenset((ident,))
        while True:
            expanded = affected.union(child for child, parent in parents.items() if parent in affected)
            if expanded == affected:
                break
            affected = expanded
        return [{**obj, "bottom_center_m": [obj["bottom_center_m"][0]+delta[0],
                    obj["bottom_center_m"][1]+delta[1], obj["bottom_center_m"][2]]}
                if obj["id"] in affected else deepcopy(obj) for obj in objects]


class Host(Protocol):
    """A real adapter must atomically check version, deduplicate, and commit all objects."""
    def commit(self, objects, *, idempotency_key, expected_world_version) -> dict: ...


@dataclass(frozen=True)
class _HostState:
    version: int = 0
    objects: tuple = ()
    requests: tuple = ()


class AtomicMemoryHost:
    """Offline reference transaction sink; never changes an external world."""
    def __init__(self):
        self._state = _HostState()
        self._lock = RLock()

    def snapshot(self):
        with self._lock:
            return {"version": self._state.version, "objects": deepcopy(list(self._state.objects)),
                    "commits": len(self._state.requests)}

    def commit(self, objects, *, idempotency_key, expected_world_version):
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("commit requires a stable idempotency key")
        if isinstance(expected_world_version, bool) or not isinstance(expected_world_version, int):
            raise ValueError("commit requires an integer world version")
        payload = deepcopy(list(objects))
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        with self._lock:
            previous = next((x for x in self._state.requests if x[0] == idempotency_key), None)
            if previous:
                if previous[1] != fingerprint:
                    raise ValueError("idempotency key already used for a different scene")
                return {"committed": True, "world_version": previous[2], "idempotent": True}
            if self._state.version != expected_world_version:
                raise ValueError("world version conflict")
            ids = [obj.get("id") for obj in payload]
            existing = {obj["id"] for obj in self._state.objects}
            if any(not isinstance(ident, str) or not ident for ident in ids) or len(ids) != len(set(ids)) or existing.intersection(ids):
                raise ValueError("commit object IDs must be new, nonempty and unique")
            version = self._state.version + 1
            self._state = _HostState(version, self._state.objects + tuple(payload),
                                    self._state.requests + ((idempotency_key, fingerprint, version),))
            return {"committed": True, "world_version": version, "idempotent": False}


def _repair_is_pose_only(before, after):
    if not isinstance(after, (list, tuple)) or len(after) != len(before):
        return False
    index = {obj.get("id"): obj for obj in after if isinstance(obj, dict)}
    if len(index) != len(before) or set(index) != {obj["id"] for obj in before}:
        return False
    allowed = {"bottom_center_m", "yaw_rad", "asset_transform"}
    return all({k: v for k, v in obj.items() if k not in allowed} ==
               {k: v for k, v in index[obj["id"]].items() if k not in allowed} for obj in before)


def _result(condition, raw, target, first, final, first_objects, objects, diagnostics, counts, start, commit=None):
    n = len(condition.get("objects", ())) if isinstance(condition, dict) else 0
    mismatch = [sum(abs(math.log(a/b)) for a, b in zip(obj["actual_size_local_m"], obj["target_size_local_m"]))
                for obj in objects]
    return {"ok": bool(final and final["ok"] and not counts.get("fatal")), "committed": bool(commit and commit.get("committed")),
            "raw_model_output": deepcopy(raw), "actual_first_pass_objects": deepcopy(first_objects),
            "final_objects": deepcopy(objects), "validation": {"target": target, "actual_first_pass": first, "final": final},
            "diagnostics": list(diagnostics), "commit": commit,
            "attempts": list(counts.get("attempts", ())),
            "metrics": {"model": {"schema_success": counts.get("schema_success", False),
                         "requested_ids_exactly_once": counts.get("schema_success", False),
                         "positive_valid_size": counts.get("schema_success", False),
                         "target_geometry_valid": bool(target and target["ok"])},
              "asset": {"retrieval_coverage": counts.get("resolved", 0)/n if n else 1.,
                        "resolver_calls": counts.get("resolver_calls", 0),
                        "target_actual_log_size_l1_mean": sum(mismatch)/len(mismatch) if mismatch else None,
                        "capability_satisfaction": bool(objects) and not counts.get("asset_filter_failure", False),
                        "first_pass_actual_geometry_validation": bool(first and first["ok"])},
              "system": {"repaired_success": bool(final and final["ok"] and
                         (counts.get("repair_calls", 0) or counts.get("asset_retries", 0))),
                         "fallback_rate": 0., "final_commit": bool(commit and commit.get("committed")),
                         "commit_attempted": counts.get("commit_attempted", False),
                         "solver_success": None, "fastfill_latency_ms": None,
                         "end_to_end_latency_ms": (monotonic()-start)*1000,
                         "asset_retries": counts.get("asset_retries", 0), "repair_calls": counts.get("repair_calls", 0)}}}


def run_pipeline(condition, prediction, resolver, *, host=None, budget=RuntimeBudget(), repair=None,
                 expected_world_version=None, idempotency_key=None, required_levels=("bbox",)):
    """Resolve, reconcile, validate, bounded retry/pose repair, then one atomic commit.

    repair(condition, actual_objects, report) receives copies and may change
    positions/yaw only. It must explicitly account for support dependencies.
    A real host adapter carries responsibility for an atomic external transaction;
    this function never calls it for a failed or unknown required validation.
    """
    start, diagnostics, counts = monotonic(), (), {}
    raw, objects, first_objects, target, first, final = deepcopy(prediction), [], [], None, None, None
    def finish(commit=None):
        return _result(condition, raw, target, first, final, first_objects, objects, diagnostics, counts, start, commit)
    def expired():
        return monotonic()-start >= budget.max_seconds
    if expired():
        diagnostics = ({"code": "budget_exhausted", "phase": "start"},)
        return finish()
    try:
        from .schema import validate_condition, validate_layout
        required_levels = validate_required_levels(required_levels)
        validate_condition(condition)
        validate_layout(prediction, condition)
        requests = {obj["id"]: obj for obj in effective_support_requests(condition)}
        counts = {**counts, "schema_success": True}
        target = validate_scene(condition, prediction["objects"], required_levels=required_levels)
        predictions = {o["id"]: o for o in prediction["objects"]}
        assets, excluded = {}, {ident: () for ident in requests}
        for ident in requests:
            if expired():
                raise TimeoutError("asset_resolution")
            counts = {**counts, "resolver_calls": counts.get("resolver_calls", 0)+1}
            asset = resolver.resolve(deepcopy(requests[ident]), deepcopy(predictions[ident]), excluded_refs=())
            counts = {**counts, "attempts": counts.get("attempts", ())+
                      ({"type": "asset_resolution", "object_id": ident, "asset_ref": asset.ref if isinstance(asset, Asset) else None},)}
            if asset is None or not isinstance(asset, Asset) or not _eligible(asset, requests[ident], predictions[ident]):
                diagnostics += ({"code": "asset_unavailable", "object_id": ident,
                                 "requirements": deepcopy(requests[ident]),
                                 "target_size_local_m": list(predictions[ident]["target_size_local_m"])},)
                counts = {**counts, "asset_filter_failure": True}
                return finish()
            assets = {**assets, ident: asset}
            counts = {**counts, "resolved": len(assets)}
        objects = reconcile(assets, condition, prediction)
        first_objects = deepcopy(objects)
        first = final = validate_scene(condition, objects, stage="actual", required_levels=required_levels)
        while not final["ok"] and counts.get("asset_retries", 0) < budget.max_asset_retries:
            if expired():
                raise TimeoutError("asset_retry")
            failed = tuple(ident for c in final["checks"] if c["hard"] and c["status"] != "pass" for ident in c["object_ids"])
            candidates = tuple(ident for ident in requests if ident in failed) or tuple(requests)
            changed = False
            for ident in candidates:
                if counts.get("asset_retries", 0) >= budget.max_asset_retries:
                    break
                if expired():
                    raise TimeoutError("asset_retry")
                excluded = {**excluded, ident: excluded[ident]+(assets[ident].ref,)}
                counts = {**counts, "asset_retries": counts.get("asset_retries", 0)+1,
                          "resolver_calls": counts.get("resolver_calls", 0)+1}
                asset = resolver.resolve(deepcopy(requests[ident]), deepcopy(predictions[ident]), excluded_refs=excluded[ident])
                counts = {**counts, "attempts": counts.get("attempts", ())+
                          ({"type": "asset_retry", "object_id": ident, "asset_ref": asset.ref if isinstance(asset, Asset) else None,
                            "excluded_refs": list(excluded[ident])},)}
                if asset is not None and isinstance(asset, Asset) and asset.ref not in excluded[ident] and _eligible(asset, requests[ident], predictions[ident]):
                    assets, changed = {**assets, ident: asset}, True
                    diagnostics += ({"code": "asset_reselected", "object_id": ident, "asset_ref": asset.ref},)
                    break
            if not changed:
                break
            objects = reconcile(assets, condition, prediction)
            final = validate_scene(condition, objects, stage="actual", required_levels=required_levels)
        while not final["ok"] and repair is not None and counts.get("repair_calls", 0) < budget.max_repair_calls:
            if expired():
                raise TimeoutError("repair")
            counts = {**counts, "repair_calls": counts.get("repair_calls", 0)+1}
            repaired = repair(deepcopy(condition), deepcopy(objects), deepcopy(final))
            counts = {**counts, "attempts": counts.get("attempts", ())+
                      ({"type": "pose_repair", "call": counts["repair_calls"]},)}
            if not _repair_is_pose_only(objects, repaired):
                diagnostics += ({"code": "invalid_repair", "reason": "repair must preserve IDs, targets, assets and actual sizes"},)
                break
            objects = [{**obj, "asset_transform": _transform(assets[obj["id"]], obj["bottom_center_m"], obj["yaw_rad"])} for obj in repaired]
            final = validate_scene(condition, objects, stage="actual", required_levels=required_levels)
        if expired():
            raise TimeoutError("before_commit")
        if not final["ok"]:
            diagnostics += ({"code": "validation_failed", "required_failures":
                            [c for c in final["checks"] if c["hard"] and c["status"] != "pass"]},)
            return finish()
        if host is not None:
            try:
                counts = {**counts, "commit_attempted": True}
                commit = host.commit(deepcopy(objects), idempotency_key=idempotency_key,
                                     expected_world_version=expected_world_version)
                if not isinstance(commit, dict) or commit.get("committed") is not True:
                    raise ValueError("host did not acknowledge an atomic commit")
                return finish(commit)
            except (ValueError, RuntimeError, OSError) as exc:
                diagnostics += ({"code": "commit_failed", "message": str(exc)},)
                counts = {**counts, "fatal": True}
                return finish()
        return finish()
    except TimeoutError as exc:
        diagnostics += ({"code": "budget_exhausted", "phase": str(exc)},)
    except (ValueError, KeyError, TypeError) as exc:
        diagnostics += ({"code": "invalid_input_or_geometry", "message": str(exc)},)
    except (RuntimeError, OSError) as exc:
        diagnostics += ({"code": "runtime_failure", "message": str(exc)},)
    counts = {**counts, "fatal": True}
    return finish()
