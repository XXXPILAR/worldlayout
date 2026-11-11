"""Dataset safety, immutable outputs and reproducible run metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import torch

from fastfill.v2.schema import migrate_legacy_row

SOURCE_ROOT = Path("/Volumes/harddisk/3D_Room_Collections")
# RoomGenBench benchmark rooms (SAGE layouts) never train; qualification writes them to test
# and every training-data reader refuses them. multisource_verify keeps an independent copy.
ROOMGENBENCH_HOLDOUT_GROUPS = ("sage:layout_61ebde9f", "sage:layout_6b049b06", "sage:layout_fef15043",
                               "sage:layout_60ee2ae3", "sage:layout_46cdcce3")
HOLDOUT_REASON = "roomgenbench_benchmark_room"


def safe_output(path, *, create=False):
    target = Path(path).resolve()
    source = SOURCE_ROOT.resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("generated outputs must stay outside the read-only source root and ancestors")
    if target.exists():
        raise FileExistsError(f"output already exists; choose a new run path: {target}")
    if create:
        target.mkdir(parents=True, exist_ok=False)
    return target


def read_samples(path, *, training=False, max_samples=None):
    """Read audited v2 rows; pre-C1 rows get their exchangeable groups migrated into validity once, here."""
    if max_samples is not None and (isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1):
        raise ValueError("max_samples must be a positive integer")
    path = Path(path)
    with path.open() as stream:
        rows = []
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples is not None and len(rows) >= max_samples:
                    break
    if not rows:
        raise ValueError("dataset is empty")
    for row in rows:
        if row.get("schema_version") != "fastfill.v2" or not isinstance(row.get("provenance"), dict):
            raise ValueError("audited v2 sample with provenance is required")
        if training and row["provenance"].get("split") != "train":
            raise ValueError("training accepts only explicitly marked training-split rows")
        if training and (row["provenance"].get("group") in ROOMGENBENCH_HOLDOUT_GROUPS or row["provenance"].get("holdout_reason")):
            raise ValueError("RoomGenBench benchmark room in training data; build training data through qualification")
    return [migrate_legacy_row(row) for row in rows]


def ensure_disjoint(train, holdout):
    def identity(row):
        p = row["provenance"]
        if not p.get("source") or not p.get("house_id"):
            raise ValueError("split integrity needs source and underlying house identity")
        return p["source"], p["house_id"]
    if {identity(r) for r in train} & {identity(r) for r in holdout}:
        raise ValueError("training and held-out scenes share an underlying source house")


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_tree(directory):
    """One digest over every file of a directory (sorted relative path + content hash); e.g. a saved tokenizer."""
    directory = Path(directory)
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    listing = [(str(path.relative_to(directory)), fingerprint(path)) for path in files]
    return hashlib.sha256(json.dumps(listing).encode()).hexdigest()


def backbone_provenance(backbone, *, local_files_only=True):
    """Backbone identity for run manifests: resolved directory, HF snapshot revision (hub id) and config.json sha256.

    A hub id resolves inside the HF cache to ``.../snapshots/<revision>/config.json``;
    a local directory has no revision, so its config.json digest identifies it
    (that digest does not pin the weights).
    """
    if backbone == "tiny":
        return {"path": "tiny", "revision": None, "config_sha256": None}
    from transformers.utils import cached_file
    config_path = Path(cached_file(backbone, "config.json", local_files_only=local_files_only))
    # Resolve the directory, not config.json: a relative path or a symlink to a snapshot becomes the absolute
    # snapshots/<revision> directory, while its files (symlinks into blobs/) would lose the revision.
    snapshot = config_path.parent.resolve()
    return {"path": str(snapshot), "revision": snapshot.name if snapshot.parent.name == "snapshots" else None,
            "config_sha256": fingerprint(config_path)}


CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
CHECKPOINT_KEYS = ("step", "max_length", "data_sha256", "validation_data_sha256", "config_sha256", "implementation_sha256")


def load_checkpoint_config(model_dir):
    """Settings bound to an exported checkpoint, so evaluation and prediction never guess them.

    Structured exports hold ``model_config.json`` (position head, grid, size
    reference, yaw bins, ...; ``model.load_model`` applies it) and the
    ``checkpoint_manifest.json`` train.py writes beside it (``CHECKPOINT_KEYS``
    plus backbone and tokenizer identity). Exports older than that manifest read
    max_length and hashes from the run's ``run_manifest.json`` one directory up.
    Text SFT exports (``text_config.json``) carry their own max_length and run
    metadata. Values that no file records are None; ``source`` names the file used.
    """
    model_dir = Path(model_dir)
    text = model_dir / "text_config.json"
    if text.exists():
        config = json.loads(text.read_text())
        values = {**config.get("run_metadata", {}), "step": config.get("steps"), "max_length": config.get("max_length")}
        return {"model": None, **{key: values.get(key) for key in CHECKPOINT_KEYS}, "source": str(text)}
    model = json.loads((model_dir / "model_config.json").read_text()) if (model_dir / "model_config.json").exists() else None
    values, source = {}, None
    if (model_dir / CHECKPOINT_MANIFEST).exists():
        source = model_dir / CHECKPOINT_MANIFEST
        values = json.loads(source.read_text())
    elif (model_dir.parent / "run_manifest.json").exists():
        source = model_dir.parent / "run_manifest.json"
        manifest = json.loads(source.read_text())
        # A run directory holds several exports; only the run-wide settings bind here, not the step.
        values = {**manifest, "step": None, "max_length": manifest.get("training", {}).get("max_length")}
    return {"model": model, **{key: values.get(key) for key in CHECKPOINT_KEYS},
            "source": str(source) if source else None}


def run_metadata(data_path):
    import importlib.metadata
    packages = {}
    for name in ("torch", "transformers", "peft", "scipy", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=False)
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    hashes = {str(path.relative_to(repo_root)): fingerprint(path) for path in source_files}
    tree_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"data_path": str(Path(data_path).resolve()), "data_sha256": fingerprint(data_path),
            "code_commit": result.stdout.strip() if result.returncode == 0 else None,
            "code_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "implementation_files_sha256": hashes, "implementation_sha256": tree_hash,
            "packages": packages}


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    # Condition metadata lists remain Python values; tensor fields live in dicts.
    return value
