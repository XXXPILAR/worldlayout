"""V3.1's metrics for v2 layouts: the rooms and layouts of fastfill.v2 rows are converted to V3.1's IR and scored by
V3.1's own code (``fastfill.evaluate.judge``: ``validate.check`` raw, then the interface's <= 10 cm ``snap_inside``
repair, acceptance as serve.py), every metric next to the reference layout of the same rooms scored the same way, as
in the V3.1 report (Parse / Complete / Valid / 1 mm Snap / Accepted rate, per-object OOB / support / ceiling rates,
rooms with each collision kind, mean OOR).

Conversion (the same for every method and for the reference):
  boundary   floor_polygon_xy_m (boundary_type polygon when boundary_known, else unknown: no OOB check, as V3.1)
  height     height_m; z measured from floor_z_m
  objects    category, target_size_local_m -> size, bottom_center_m -> pos, yaw_rad -> yaw
  fixed      room fixed_objects (size_local_m, bottom_center_m, yaw_rad)
  parent     the condition's declared support_parent (an object id; floor -> none); an object declared on a wall is left
             out of the support check (V3.1's IR has no wall support); an undeclared object's parent is found from the
             layout itself: none when its bottom is within 5 cm of the floor, else the object whose top is within 5 cm of
             its bottom and whose footprint holds its centre (the largest such), else none (a support failure)
  constraints  none: V3.1's reported test metrics had no constraints (constraint_satisfaction {}); v2's constraint
             pass rate is reported by fastfill.v2.evaluate
A request without a layout (or missing ids) fails, as an unparsable V3.1 answer.

    python -m fastfill.v2.ops.v3_metrics --rows fastfill/v2/ops/test300/rows.jsonl --output DIR \\
        --method "GPT-5.6=fastfill/v2/ops/test300/gpt56_predictions.jsonl" --method "E remask4L=.../predictions_remask4L.jsonl"
writes DIR/v3_metrics.json and DIR/v3_core_metrics.png (V3.1's eval_core_metrics chart, one bar per method + GT).
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path

from shapely.geometry import Polygon

from fastfill.interface import snap_inside
from fastfill.scene import apply, footprint
from fastfill.validate import check, support_contains
from fastfill.v2.schema import migrate_legacy_row

TOL = .05
BUCKETS = ("main", "severe", "tuck", "contained", "fixed")
CORE = (("parse_rate", "Parse\nRate"), ("complete_rate", "Complete\nRate"), ("valid_rate", "Valid\nRate"),
        ("inside_1mm_rate_repaired", "1mm Snap\nRate"), ("accepted_rate", "Accepted\nRate"))


def ir_room(row):
    c = row["condition"]
    room = c["room"]
    z0 = room.get("floor_z_m") or 0.
    declared = {o["id"]: o.get("support_parent") for o in c["objects"]}
    gt = {o["id"]: o for o in row["target"]["objects"]}
    objects = [{"id": o["id"], "category": o.get("category") or "object", "size": list(gt[o["id"]]["target_size_local_m"]),
                "pos": [*gt[o["id"]]["bottom_center_m"][:2], gt[o["id"]]["bottom_center_m"][2] - z0],
                "yaw": gt[o["id"]]["yaw_rad"]} for o in c["objects"]]
    fixed = [{"id": f["id"], "category": f.get("category") or "fixed", "size": list(f["size_local_m"]),
              "pos": [*f["bottom_center_m"][:2], f["bottom_center_m"][2] - z0], "yaw": f["yaw_rad"]}
             for f in room.get("fixed_objects") or []]
    return {"uid": row["provenance"].get("scene_id"), "source": row["provenance"]["source"],
            "group": row["provenance"].get("group"), "objects": objects, "fixed": fixed, "constraints": [],
            "boundary": [list(p) for p in room["floor_polygon_xy_m"]],
            "boundary_type": "polygon" if room.get("boundary_known", True) else "unknown",
            "height": room.get("height_m"), "meta": {}}, declared, z0


def placements(room, declared, layout, z0):
    """{id: {pos, yaw, on}} in V3.1's form; None when the layout is missing or incomplete."""
    if layout is None:
        return None
    by_id = {o["id"]: o for o in layout["objects"]}
    if set(by_id) != {o["id"] for o in room["objects"]}:
        return None
    size = {o["id"]: o["size"] for o in room["objects"]}
    pl = {i: {"pos": [*o["bottom_center_m"][:2], o["bottom_center_m"][2] - z0], "yaw": o["yaw_rad"], "on": None}
          for i, o in by_id.items()}
    for i in by_id:
        size[i] = list(by_id[i]["target_size_local_m"])
    boxes = {i: {"pos": p["pos"], "yaw": p["yaw"], "size": size[i]} for i, p in pl.items()}
    for i, p in pl.items():
        d = declared.get(i)
        if d in boxes and d != i:
            p["on"] = d
        elif d in ("floor", "wall"):
            continue
        elif abs(p["pos"][2]) > TOL:
            under = [j for j, b in boxes.items() if j != i and abs(b["pos"][2] + b["size"][2] - p["pos"][2]) <= TOL
                     and support_contains(b, p["pos"], TOL)]
            if under:
                p["on"] = max(under, key=lambda j: Polygon(footprint(boxes[j])).area)
    return pl, size


def judge(room, pl, size, wall):
    """fastfill.evaluate.judge on the predicted sizes, wall-declared objects left out of the support check."""
    room = {**room, "objects": [{**o, "size": size[o["id"]]} for o in room["objects"]]}
    pred = apply(room, pl)
    c = check(pred)
    fails = [i for i in c["support_fail"] if i not in wall]
    r = {"valid": not (c["oob"] or fails or c["ceiling"]), "oob_objects": len(c["oob"]), "support_fail": len(fails),
         "ceiling_objects": len(c["ceiling"]), "oor": c["oor"], "coll_fixed": len(c["fixed_collisions"]),
         **{f"coll_{k}": len(c["collisions"][k]) for k in BUCKETS[:4]}}
    floor = Polygon(room["boundary"])
    rep = copy.deepcopy(pl)
    snap_inside(rep, room, floor, (), room["boundary"])
    fixed = apply(room, rep)
    c2 = check(fixed)
    grown = floor.buffer(1e-3)
    inside = lambda scene: all(grown.covers(Polygon(footprint(o))) for o in scene["objects"])
    r["inside_1mm"], r["inside_1mm_repaired"] = inside(pred), inside(fixed)
    r["valid_repaired"] = not (c2["oob"] or [i for i in c2["support_fail"] if i not in wall] or c2["ceiling"])
    r["repaired_objects"] = sum(rep[i]["pos"] != pl[i]["pos"] for i in pl)
    r["accepted"] = r["valid_repaired"] and r["inside_1mm_repaired"] and not c2["fixed_blocking"]
    return r


def score(rows, layouts):
    out = []
    for row, layout in zip(rows, layouts):
        room, declared, z0 = ir_room(row)
        wall = {i for i, d in declared.items() if d == "wall"}
        gt_pl, gt_size = placements(room, declared, row["target"], z0)
        r = {"source": room["source"], "group": room["group"], "n": len(room["objects"]),
             "gt": judge(room, gt_pl, gt_size, wall)}
        got = placements(room, declared, layout, z0)
        r["parsed"] = layout is not None
        r["error"] = None if got else ("no_layout" if layout is None else "missing_ids")
        out.append({**r, **(judge(room, *got, wall) if got else {"valid": False})})
    return out


def summarize(rs):
    """fastfill.evaluate.summarize's pooled figures."""
    n, ok = len(rs), [r for r in rs if r["error"] is None]
    objs = sum(r["n"] for r in ok)
    rate = lambda num, den: num / den if den else None
    s = {"rooms": n, "parse_rate": sum(r["parsed"] for r in rs) / n, "complete_rate": len(ok) / n,
         "errors": dict(collections.Counter(r["error"] for r in rs if r["error"]))}
    for k, key in (("valid_rate", "valid"), ("valid_rate_repaired", "valid_repaired"), ("inside_1mm_rate", "inside_1mm"),
                   ("inside_1mm_rate_repaired", "inside_1mm_repaired"), ("accepted_rate", "accepted")):
        s[k] = sum(r.get(key, False) for r in rs) / n
        s[f"{k}_gt"] = sum(r["gt"][key] for r in rs) / n
    for k, key in (("repaired_object_rate", "repaired_objects"), ("oob_object_rate", "oob_objects"),
                   ("support_fail_rate", "support_fail"), ("ceiling_object_rate", "ceiling_objects")):
        s[k], s[f"{k}_gt"] = rate(sum(r[key] for r in ok), objs), rate(sum(r["gt"][key] for r in ok), objs)
    for k in BUCKETS:
        name = "fixed_collision" if k == "fixed" else k
        s[f"rooms_with_{name}"] = rate(sum(r[f"coll_{k}"] > 0 for r in ok), len(ok))
        s[f"rooms_with_{name}_gt"] = rate(sum(r["gt"][f"coll_{k}"] > 0 for r in ok), len(ok))
    s["mean_oor"], s["mean_oor_gt"] = rate(sum(r["oor"] for r in ok), len(ok)), rate(sum(r["gt"]["oor"] for r in ok), len(ok))
    return s


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)
    fig, ax = plt.subplots(figsize=(max(12, 2.2 * len(CORE) * (len(names) + 1) / 2), 6))
    width = .8 / (len(names) + 1)
    colors = plt.cm.tab10.colors
    for k, name in enumerate(names + ["Ground Truth"]):
        s = results[names[0]] if name == "Ground Truth" else results[name]
        vals = [100 * (s.get(f"{key}_gt", 1.) if name == "Ground Truth" else s[key]) for key, _ in CORE]  # GT always parses
        bars = ax.bar([i + k * width for i in range(len(CORE))], vals, width, label=name,
                      color="#a9b1bf" if name == "Ground Truth" else colors[k % 10])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .8, f"{v:.1f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks([i + .4 - width / 2 for i in range(len(CORE))], [label for _, label in CORE])
    ax.set(ylabel="Percentage (%)", ylim=(0, 118))
    ax.set_title(f"V3.1 metrics on the {results[names[0]]['rooms']} test300 rooms (V3.1 evaluator, same rooms for every method)",
                 fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=.5)
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--method", action="append", required=True, help="NAME=predictions.jsonl (one {layout} per row)")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    rows = [migrate_legacy_row(json.loads(l)) for l in a.rows.read_text().splitlines() if l.strip()]
    results = {}
    for spec in a.method:
        name, path = spec.split("=", 1)
        layouts = [json.loads(l).get("layout") for l in Path(path).read_text().splitlines() if l.strip()]
        if len(layouts) != len(rows):
            raise SystemExit(f"{path}: {len(layouts)} predictions for {len(rows)} rows")
        results[name] = summarize(score(rows, layouts))
        s = results[name]
        print(f"{name:28s} parse {s['parse_rate']:.3f} complete {s['complete_rate']:.3f} valid {s['valid_rate']:.3f} "
              f"(GT {s['valid_rate_gt']:.3f}) 1mm {s['inside_1mm_rate_repaired']:.3f} (GT {s['inside_1mm_rate_repaired_gt']:.3f}) "
              f"accepted {s['accepted_rate']:.3f} (GT {s['accepted_rate_gt']:.3f}) collision rooms {s['rooms_with_main']:.3f} "
              f"(GT {s['rooms_with_main_gt']:.3f})", flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "v3_metrics.json").write_text(json.dumps(results, indent=1) + "\n")
    figure(results, a.output / "v3_core_metrics.png")


if __name__ == "__main__":
    main()
