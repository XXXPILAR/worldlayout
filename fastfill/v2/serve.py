"""Offline JSON runner for the v2 runtime; no network service or real Host writes.

The demo catalog ranks by category, typed capabilities and target-size distance.
Nuanced natural-language retrieval needs an external semantic Resolver adapter.
"""
import argparse
import json
from pathlib import Path
import sys

from .runtime import Asset, AtomicMemoryHost, BoundedTranslationRepair, CatalogResolver, RuntimeBudget, SupportSurface, run_pipeline


def asset_from_dict(value):
    allowed = {"ref", "category", "description", "actual_size_local_m", "canonical_transform",
               "semantic_front_local", "capabilities", "support_surfaces", "provenance"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("invalid catalog asset fields")
    required = {"ref", "category", "actual_size_local_m"}
    if not required.issubset(value):
        raise ValueError("asset needs ref, category and actual_size_local_m")
    surfaces = tuple(SupportSurface(**surface) for surface in value.get("support_surfaces", ()))
    return Asset(**{**value, "support_surfaces": surfaces})


def load_catalog(path):
    value = _read(path)
    if isinstance(value, dict):
        if set(value) != {"assets"}:
            raise ValueError("catalog object must contain only assets")
        value = value["assets"]
    if not isinstance(value, list):
        raise ValueError("catalog must be an asset array or {assets: array}")
    return CatalogResolver(tuple(asset_from_dict(asset) for asset in value))


def _read(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    with Path(path).open() as handle:
        return json.load(handle, parse_constant=reject_constant)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New repository/output JSON path; defaults to stdout")
    parser.add_argument("--max-asset-retries", type=int, default=2)
    parser.add_argument("--max-repair-calls", type=int, default=0)
    parser.add_argument("--repair-step-m", type=float, default=.25)
    parser.add_argument("--max-seconds", type=float, default=10.)
    parser.add_argument("--required-levels", nargs="+", choices=("bbox", "mesh", "physics", "solver"), default=["bbox"])
    parser.add_argument("--commit-in-memory", action="store_true", help="Exercise only the offline atomic Host contract")
    parser.add_argument("--idempotency-key", default="offline-request")
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            forbidden = Path("/Volumes/harddisk/3D_Room_Collections").resolve()
            if args.output.resolve().is_relative_to(forbidden):
                raise ValueError("source datasets are read-only; choose a new output path")
            if args.output.exists():
                raise ValueError("output already exists; choose a new path")
        host = AtomicMemoryHost() if args.commit_in_memory else None
        report = run_pipeline(_read(args.condition), _read(args.prediction), load_catalog(args.catalog),
                              host=host, budget=RuntimeBudget(args.max_asset_retries, args.max_repair_calls, args.max_seconds),
                              repair=BoundedTranslationRepair(args.repair_step_m) if args.max_repair_calls else None,
                              required_levels=tuple(args.required_levels), idempotency_key=args.idempotency_key,
                              expected_world_version=0 if host else None)
        if host:
            report = {**report, "offline_host_snapshot": host.snapshot()}
        output = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as handle:
                handle.write(output)
        else:
            sys.stdout.write(output)
        return 0 if report["ok"] else 2
    except (ValueError, TypeError, KeyError, OSError) as exc:
        sys.stderr.write(json.dumps({"error": "invalid_runtime_request", "message": str(exc)}) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
