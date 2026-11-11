"""Train-only empirical size prior; no test or validation target is read."""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

DEFAULT_SIZES = {
    "bed": [2.0, 1.5, 0.6], "nightstand": [0.5, 0.5, 0.55],
    "sofa": [0.9, 2.0, 0.85], "armchair": [0.85, 0.85, 0.85],
    "chair": [0.55, 0.55, 0.85], "table": [1.2, 0.75, 0.75],
    "coffee table": [1.0, 0.55, 0.45], "desk": [1.2, 0.65, 0.75],
    "wardrobe": [0.65, 1.3, 2.0], "bookshelf": [0.4, 0.9, 1.8],
    "cabinet": [0.5, 0.9, 0.9], "tv stand": [0.45, 1.4, 0.6],
    "lamp": [0.3, 0.3, 1.4], "table lamp": [0.25, 0.25, 0.45],
    "rug": [1.8, 2.2, 0.02], "floor mat": [0.8, 1.1, 0.02],
    "painting": [0.06, 0.8, 0.6], "pillow": [0.4, 0.4, 0.15],
}


def _key(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace("_", " ").strip())


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def build_catalog(train_jsonl: str | Path, output: str | Path, *, cap: int = 2048) -> dict:
    rng = random.Random(17)
    buckets: dict[str, list[list[float]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    with Path(train_jsonl).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            requests = {obj["id"]: obj for obj in row["condition"]["objects"]}
            targets = row["target"]["objects"]
            valid = row.get("validity", {}).get("size", [])
            for index, obj in enumerate(targets):
                if index >= len(valid) or not all(valid[index]):
                    continue
                size = obj["target_size_local_m"]
                if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in size):
                    continue
                request = requests.get(obj["id"])
                if request is None:
                    continue
                category = _key(request["category"])
                counts[category] += 1
                bucket = buckets[category]
                if len(bucket) < cap:
                    bucket.append(size)
                else:
                    pick = rng.randrange(counts[category])
                    if pick < cap:
                        bucket[pick] = size
    result = {
        "source": str(Path(train_jsonl).resolve()), "split": "train", "cap_per_category": cap,
        "categories": {
            name: {
                "count": counts[name],
                "median": [round(median(row[axis] for row in values), 4) for axis in range(3)],
                "p15": [round(_quantile([row[axis] for row in values], .15), 4) for axis in range(3)],
                "p85": [round(_quantile([row[axis] for row in values], .85), 4) for axis in range(3)],
            }
            for name, values in sorted(buckets.items()) if counts[name] >= 20
        },
    }
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


class SizeCatalog:
    def __init__(self, path: str | Path | None = None):
        self.categories = {}
        if path is not None:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("split") != "train":
                raise ValueError("size catalog must be built from the train split")
            self.categories = data["categories"]

    def candidates(self, request: dict, llm_hint: list[float] | None = None) -> list[tuple[list[float], float, str]]:
        category, description = _key(request["category"]), _key(request["description"])
        record = self.categories.get(category)
        if record and record["count"] >= 20:
            base, low, high = record["median"], record["p15"], record["p85"]
            source = "train_catalog"
        else:
            matched = next((key for key in sorted(DEFAULT_SIZES, key=len, reverse=True)
                            if key in category or key in description), None)
            base = DEFAULT_SIZES.get(matched, [0.7, 0.7, 0.8])
            low, high, source = [v * .75 for v in base], [v * 1.25 for v in base], "fallback_prior"
        scale = 1.0
        if re.search(r"\b(large|big|king|oversized)\b|大号|大型", description):
            scale = 1.18
        elif re.search(r"\b(small|single|compact|tiny)\b|小号|小型", description):
            scale = .82
        if category == "bed" or " bed" in category:
            if re.search(r"\bking\b", description):
                base = [2.05, 1.95, base[2]]
            elif re.search(r"\bsingle\b", description):
                base = [2.0, 0.95, base[2]]
        if "table lamp" in description:
            base = DEFAULT_SIZES["table lamp"]
        variants = [([round(v * scale, 3) for v in base], 0.0, source)]
        variants += [([round(v * scale, 3) for v in low], -0.3, source),
                     ([round(v * scale, 3) for v in high], -0.3, source)]
        if llm_hint is not None and len(llm_hint) == 3 and all(
                isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in llm_hint):
            # within a third of the category's 15th to three times its 85th percentile (train), per axis: wide
            # enough for a category whose median misfits the object, narrow enough to catch swapped units
            if all(lo / 3 <= v <= 3 * hi for lo, hi, v in zip(low, high, llm_hint)):
                variants.insert(0, ([round(float(v), 3) for v in llm_hint], .1, "llm_hint"))
        seen, result = set(), []
        for size, score, origin in variants:
            signature = tuple(size)
            if signature not in seen:
                seen.add(signature)
                result.append((size, score, origin))
        return result
