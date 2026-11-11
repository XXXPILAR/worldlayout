"""The command line, ``python -m workflow``: rooms (and their tasks) -> FastFill's sizes -> the workflow -> layouts.

    # both steps on one machine (a GPU, or Apple's MPS, and the LLM API)
    python -m workflow run --rows rows.jsonl --index 0 1 --checkpoint <run>/model-step-4500 --out my_out
    # or in two: FastFill's sizes where the GPU is, then the placement where the API is
    python -m workflow sizes --rows rows.jsonl --checkpoint <run>/model-step-4500 --out my_sizes.jsonl
    python -m workflow run --rows rows.jsonl --sizes-file my_sizes.jsonl --workers 4 --out my_out
    # without FastFill: the placement call gives every size (the size ablation)
    python -m workflow run --rows rows.jsonl --draft-sizes --out my_out

Each room is a row: ``{"condition": <fastfill.v2 condition>, "task": {"required_ids": [...],
"articulation_ids": [...]}}`` (``task`` optional: without it the room is placed without the task mode). FastFill
(``fastfill_sizes.FastFillSizer``) is one forward pass per room with the checkpoint loaded once, spread decode; its
sizes are kept and its positions discarded. A room it cannot take (over its object or context budget) gets the size
catalog's sizes, recorded as ``size_error``. The defaults are the paper's settings: gpt-6.1-sol through the relay at
reasoning medium, at most 30 requests a minute (shared by the workers), a 0.5 m robot planned with a 0.1 m margin
and a 0.4 m arm, hard zones, relocation, two drafts, three repair passes of at most 180 s. ``--no-llm`` makes no LLM
call (rules plan, program placement, no review). A room whose ``room_XXX.json`` exists is skipped (rerun to resume).

-> sizes: one line per row of --rows, ``{"layout", "error", "latency_s"}`` (a row not run: no layout).
-> run: <out>/room_XXX.json (the workflow's result, the sizes' source, the LLM calls and their cost),
   <out>/llm_calls/room_XXX.jsonl (every LLM call; never the key), <out>/sizes.jsonl (with --checkpoint),
   <out>/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

RELAY_URL = "https://api.kbsfdxc.net/codex/v1"


def device_of(name: str) -> str:
    if name != "auto":
        return name
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_rows(a) -> dict[int, dict]:
    from fastfill.v2.schema import migrate_legacy_row

    if a.input:
        rows = {0: json.loads(Path(a.input).read_text())}
    else:
        lines = Path(a.rows).read_text().splitlines()
        rows = {k: json.loads(lines[k]) for k in (a.index or [k for k, line in enumerate(lines) if line.strip()])}
    return {k: migrate_legacy_row(r) for k, r in rows.items()}  # as FastFill reads rows


def predict_sizes(rows: dict[int, dict], checkpoint, device: str) -> tuple[dict[int, dict], dict]:
    """{row: {"layout", "error", "latency_s"}} from FastFill, the model loaded once."""
    from .fastfill_sizes import FastFillSizer

    start = time.perf_counter()
    sizer = FastFillSizer(checkpoint, device=device_of(device))
    info = {"checkpoint": str(checkpoint), "device": sizer.device, "max_length": sizer.max_length,
            "load_s": round(time.perf_counter() - start, 1)}
    print(f"FastFill {checkpoint} on {sizer.device}: loaded in {info['load_s']} s", flush=True)
    out = {}
    for k, row in rows.items():
        start = time.perf_counter()
        try:
            sizes, yaws = sizer.predict(row["condition"])
            layout, error = {"objects": [{"id": i, "target_size_local_m": s, "yaw_rad": yaws[i]}
                                         for i, s in sizes.items()]}, None
        except ValueError as exc:  # over the model's object or context budget: the size catalog's sizes
            layout, error = None, f"{type(exc).__name__}: {exc}"[:300]
        out[k] = {"layout": layout, "error": error, "latency_s": round(time.perf_counter() - start, 4)}
    return out, info


def sizes_main(a) -> int:
    rows = read_rows(a)
    predicted, info = predict_sizes(rows, a.checkpoint, a.device)
    n = len(Path(a.rows).read_text().splitlines()) if a.rows else 1
    lines = [json.dumps(predicted.get(k, {"layout": None, "error": "not run", "latency_s": None})) for k in range(n)]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(json.dumps({**info, "rooms": len(predicted), "errors": sum(bool(p["error"]) for p in predicted.values()),
                      "out": a.out}))
    return 0


def llm_cost(diagnostics: dict) -> tuple[int, float]:
    calls = (diagnostics.get("plan_llm") or {}).get("calls", []) + (diagnostics.get("draft") or {}).get("calls", []) + \
        [c for c in (diagnostics.get("semantic") or {}).get("llm_calls", []) if "usage" in c]
    return len(calls), round(sum(c.get("cost_usd") or 0 for c in calls), 4)


def room_job(k: int, row: dict, predicted: dict | None, sizes_from: str, opts: dict, out: str) -> dict:
    """One room through the workflow (in a worker process); writes room_XXX.json and returns its summary line."""
    os.environ["WORKFLOW_LLM_TAG"] = f"room_{k:03d}"
    from .credentials import api_settings
    from .pipeline import run_draft_workflow
    from .robot import Robot

    settings = {} if opts["no_llm"] else api_settings(opts["api"], base_url=opts["base_url"], model=opts["model"])
    task = row.get("task") or {}
    layout = (predicted or {}).get("layout")
    sizes = {b["id"]: b["target_size_local_m"] for b in layout["objects"]} if layout else None
    start = time.perf_counter()
    result = run_draft_workflow(
        row["condition"], sizes, llm=not opts["no_llm"], **settings, draft_sizes=opts["draft_sizes"],
        robot=Robot(width_m=opts["robot_width"], arm_m=opts["robot_arm"]), robot_margin=opts["robot_margin"],
        hard_zones=True, relocate=True, drafts=opts["drafts"], repair_rounds=opts["repair_rounds"],
        repair_seconds=opts["repair_seconds"],
        required=set(task["required_ids"]) if "required_ids" in task else None,
        articulation=set(task.get("articulation_ids") or ()) if "required_ids" in task else None)
    d = result["diagnostics"]
    if predicted is not None and predicted.get("error"):
        d["size_error"] = predicted["error"]
    calls, cost = llm_cost(d)
    result.update(wall_s=round(time.perf_counter() - start, 1), llm_calls=calls, cost_usd=cost,
                  sizes_from=sizes_from if sizes is not None else ("draft" if opts["draft_sizes"] else "catalog"),
                  fastfill_latency_s=(predicted or {}).get("latency_s"))
    (Path(out) / f"room_{k:03d}.json").write_text(json.dumps(result))
    t = d.get("task") or {}
    return {"status": result["status"], "seconds": result["wall_s"], "llm_calls": calls, "cost_usd": cost,
            "objects": len(row["condition"]["objects"]), "kept": len(result["layout"]["objects"]),
            "hard_issues": len(d["hard_issues"]), "required_ok": t.get("required_ok"),
            "size_error": d.get("size_error")}


def run_main(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WORKFLOW_RPM", str(a.rpm))
    os.environ.setdefault("WORKFLOW_RPM_FILE", str(out.resolve() / ".rpm_state.json"))
    os.environ.setdefault("WORKFLOW_LLM_ATTEMPTS", "0")  # a busy relay is retried until it answers
    os.environ.setdefault("WORKFLOW_REASONING_EFFORT", a.reasoning)
    os.environ["WORKFLOW_LLM_LOG_DIR"] = str(out.resolve() / "llm_calls")
    rows = read_rows(a)
    done = {k for k in rows if (out / f"room_{k:03d}.json").exists()}
    todo = {k: r for k, r in rows.items() if k not in done}
    if done:
        print(f"skipping {len(done)} rooms already done: {sorted(done)}", flush=True)
    if not a.no_llm:  # the settings must be there before any room starts (the key is never printed)
        from .credentials import api_settings
        api_settings(a.api, base_url=a.base_url, model=a.model)
    info = {}
    if a.draft_sizes:
        predicted, sizes_from = {}, "draft"
    elif a.sizes_file:
        lines = Path(a.sizes_file).read_text().splitlines()
        predicted, sizes_from = {k: json.loads(lines[k]) for k in todo}, "sizes_file"
    elif todo:
        predicted, info = predict_sizes(todo, a.checkpoint, a.device)
        sizes_from = "fastfill"
        with open(out / "sizes.jsonl", "a") as f:
            f.writelines(json.dumps({"row": k, **p}) + "\n" for k, p in predicted.items())
    else:
        predicted, sizes_from = {}, "fastfill"
    opts = {k: getattr(a, k) for k in ("no_llm", "draft_sizes", "api", "base_url", "model", "robot_width", "robot_arm",
                                       "robot_margin", "drafts", "repair_rounds", "repair_seconds")}
    lines = {}
    if a.workers <= 1:
        for k, row in todo.items():
            lines[k] = room_job(k, row, predicted.get(k), sizes_from, opts, str(out))
            print(f"room {k}: {json.dumps(lines[k])}", flush=True)
    else:
        with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as pool:
            jobs = {pool.submit(room_job, k, row, predicted.get(k), sizes_from, opts, str(out)): k
                    for k, row in todo.items()}
            for job in jobs:
                lines[jobs[job]] = job.result()
                print(f"room {jobs[job]}: {json.dumps(lines[jobs[job]])}", flush=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"rooms": {}}
    summary.update(rows=a.input or a.rows, sizes=a.sizes_file or info or summary.get("sizes"),
                   llm=None if a.no_llm else {"api": a.api, "base_url": a.base_url, "model": a.model,
                                              "reasoning_effort": os.environ["WORKFLOW_REASONING_EFFORT"]})
    summary["rooms"].update({str(k): v for k, v in lines.items()})
    summary_path.write_text(json.dumps(summary, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m workflow", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name, help_text in (("run", "FastFill's sizes (or a sizes file), then the workflow, for each room"),
                            ("sizes", "FastFill's sizes only (where the GPU is)")):
        p = sub.add_parser(name, help=help_text)
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--rows", help="a JSONL of rows (condition, task)")
        src.add_argument("--input", help="one row as a JSON file")
        p.add_argument("--index", type=int, nargs="+", help="with --rows: the rows to run, from 0 (default: all)")
        p.add_argument("--device", default="auto", help="auto (cuda, else mps, else cpu), cuda, mps or cpu")
        p.add_argument("--out", required=True)
        if name == "sizes":
            p.add_argument("--checkpoint", type=Path, required=True,
                           help="the FastFill model-step directory (tokenizer/ beside it)")
            continue
        sizes = p.add_mutually_exclusive_group(required=True)
        sizes.add_argument("--checkpoint", type=Path, help="the FastFill model-step directory (tokenizer/ beside it)")
        sizes.add_argument("--sizes-file", help="sizes predicted earlier (`sizes`), aligned with --rows")
        sizes.add_argument("--draft-sizes", action="store_true",
                           help="no FastFill: the placement call gives every object's size (needs the LLM)")
        p.add_argument("--workers", type=int, default=1, help="rooms placed in parallel")
        p.add_argument("--api", default="relay", choices=("relay", "official"))
        p.add_argument("--base-url", help=f"the relay's URL (default {RELAY_URL})")
        p.add_argument("--model", default="gpt-6.1-sol")
        p.add_argument("--reasoning", default="medium", help="the LLM's reasoning effort (medium, xhigh)")
        p.add_argument("--rpm", type=int, default=30, help="LLM requests per minute, at most, all workers together")
        p.add_argument("--no-llm", action="store_true", help="no LLM call: rules plan, program placement, no review")
        p.add_argument("--robot-width", type=float, default=.5)
        p.add_argument("--robot-arm", type=float, default=.4)
        p.add_argument("--robot-margin", type=float, default=.1)
        p.add_argument("--drafts", type=int, default=2)
        p.add_argument("--repair-rounds", type=int, default=3)
        p.add_argument("--repair-seconds", type=float, default=180.)
    a = ap.parse_args(argv)
    if a.index and not a.rows:
        ap.error("--index needs --rows")
    if a.command == "sizes":
        return sizes_main(a)
    if a.sizes_file and a.input:
        ap.error("--sizes-file needs --rows (it is aligned with them)")
    if a.draft_sizes and a.no_llm:
        ap.error("--draft-sizes needs the LLM: the placement call gives the sizes")
    if a.base_url is None and a.api == "relay":
        a.base_url = RELAY_URL
    return run_main(a)
