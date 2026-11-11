"""Independent JSON text SFT baseline; assistant token CE only.

Example offline smoke (not a Qwen training result)::

    python -m fastfill.v2.text_sft --data samples.jsonl --output out/text-smoke \
        --backbone tiny --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .batch import TinyTokenizer, _geometry_rows, load_tokenizer, room_normalization, tokenize_condition
from .schema import validate_condition
from .io import fingerprint, read_samples, run_metadata, safe_output


def _answer(sample: dict) -> dict:
    condition = sample["condition"]
    validate_condition(condition)
    origin, scale = room_normalization(condition.get("room", {}))
    rows = _geometry_rows(sample, origin, scale)
    if not all(all(row) for row in rows["position_valid"] + rows["size_valid"]) or not all(rows["yaw_valid"]):
        raise ValueError("text SFT requires complete audited geometry labels")
    targets = {o["id"]: o for o in sample.get("target", {}).get("objects", [])}
    return {"schema_version": "fastfill.v2", "objects": [{
        "id": obj["id"], "target_size_local_m": targets[obj["id"]]["target_size_local_m"],
        "bottom_center_m": targets[obj["id"]]["bottom_center_m"], "yaw_rad": targets[obj["id"]]["yaw_rad"],
    } for obj in condition["objects"]]}


def collate_text_samples(samples: list[dict], tokenizer: Any, *, max_length: int = 4096) -> dict:
    if not samples:
        raise ValueError("cannot batch zero samples")
    sequences, labels, prompt_lengths = [], [], []
    for sample in samples:
        prompt, _ = tokenize_condition(sample["condition"], tokenizer)
        answer = json.dumps(_answer(sample), ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False)
        response = tokenizer.encode(answer, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            response = response + [tokenizer.eos_token_id]
        if len(prompt) + len(response) > max_length:
            raise ValueError("text sample exceeds context budget; no partial target truncation")
        sequences.append(prompt + response)
        labels.append([-100] * len(prompt) + response)
        prompt_lengths.append(len(prompt))
    length = max(map(len, sequences))
    ids = torch.full((len(samples), length), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    label_tensor = torch.full_like(ids, -100)
    for row, (sequence, target) in enumerate(zip(sequences, labels)):
        ids[row, :len(sequence)] = torch.tensor(sequence)
        attention[row, :len(sequence)] = True
        label_tensor[row, :len(target)] = torch.tensor(target)
    return {"input_ids": ids, "attention_mask": attention, "labels": label_tensor,
            "prompt_lengths": prompt_lengths}


class TinyTextModel(nn.Module):
    """Causal byte GRU, reserved for baseline plumbing smoke tests."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(258, hidden_size, padding_idx=0)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, 258)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_: Any) -> Any:
        hidden, _ = self.encoder(self.embedding(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


def build_text_model(backbone: str, *, hidden_size: int = 64, lora_rank: int = 8,
                     lora_alpha: int = 16, local_files_only: bool = False,
                     backbone_dtype: str = "float32") -> nn.Module:
    if backbone == "tiny":
        return TinyTextModel(hidden_size)
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(backbone, local_files_only=local_files_only)
    if not str(config.model_type).startswith("qwen"):
        raise ValueError("text baseline backbone must be a Qwen-family checkpoint")
    model = AutoModelForCausalLM.from_pretrained(backbone, config=config,
        torch_dtype=getattr(torch, backbone_dtype), local_files_only=local_files_only)
    model.config.use_cache = False
    if lora_rank:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_rank,
            lora_alpha=lora_alpha, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    return model


def text_loss(model: nn.Module, batch: dict) -> torch.Tensor:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    labels = batch["labels"][:, 1:].reshape(-1)
    if not (labels != -100).any():
        raise ValueError("no assistant target tokens available for CE")
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels, ignore_index=-100)


def _read_samples(path: Path) -> list[dict]:
    samples = read_samples(path, training=True)
    for sample in samples:
        _answer(sample)
    return samples


def _verify_source_fingerprint(metadata):
    try:
        matches = fingerprint(metadata["data_path"]) == metadata["data_sha256"]
    except OSError:
        matches = False
    if not matches:
        raise RuntimeError("text training input fingerprint changed during the run")


@torch.no_grad()
def generate_text(model: nn.Module, tokenizer: Any, condition: dict, *, max_new_tokens: int = 1024,
                  max_length: int = 4096, device: str | torch.device = "cpu") -> str:
    validate_condition(condition)
    tokens, _ = tokenize_condition(condition, tokenizer)
    if len(tokens) + max_new_tokens > max_length:
        raise ValueError("text generation exceeds configured context budget")
    model.eval()
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    if not isinstance(model, TinyTextModel):
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0, len(tokens):], skip_special_tokens=True)
    generated = []
    hidden, state = model.encoder(model.embedding(ids))
    for _ in range(max_new_tokens):
        next_token = model.lm_head(hidden[:, -1]).argmax(-1)
        if int(next_token[0]) == tokenizer.eos_token_id:
            break
        generated.append(int(next_token[0]))
        hidden, state = model.encoder(model.embedding(next_token[:, None]), state)
    return tokenizer.decode(generated)


def load_text_model(directory: str | Path, *, device: str | torch.device = "cpu",
                    local_files_only: bool = False) -> tuple[nn.Module, Any]:
    directory = Path(directory)
    config = json.loads((directory / "text_config.json").read_text())
    if config["backbone"] == "tiny":
        model = build_text_model("tiny", hidden_size=config["hidden_size"])
        model.load_state_dict(torch.load(directory / "text_model.pt", map_location="cpu", weights_only=True))
        tokenizer = TinyTokenizer()
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        if config["lora_rank"]:
            from peft import PeftModel
            base = build_text_model(config["backbone"], lora_rank=0, local_files_only=local_files_only,
                                    backbone_dtype=config["backbone_dtype"])
            model = PeftModel.from_pretrained(base, directory / "model")
        else:
            model = AutoModelForCausalLM.from_pretrained(directory / "model", local_files_only=True)
    return model.to(device), tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="audited training JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=64, help="tiny backend only")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--backbone-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="one offline optimizer step; requires explicit tiny backend")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("steps, batch size and learning rate must be positive and finite")
    if args.dry_run and args.backbone != "tiny":
        parser.error("offline dry-run requires --backbone tiny")
    output = safe_output(args.output)
    torch.manual_seed(args.seed)
    metadata = run_metadata(args.data)
    samples = _read_samples(args.data)
    _verify_source_fingerprint(metadata)
    tokenizer = load_tokenizer(args.backbone, local_files_only=args.local_files_only)
    for row, sample in enumerate(samples):
        try:
            collate_text_samples([sample], tokenizer, max_length=args.max_length)
        except ValueError as exc:
            raise ValueError(f"text training row {row} failed preflight: {exc}") from exc
    model = build_text_model(args.backbone, hidden_size=args.hidden_size, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, local_files_only=args.local_files_only,
        backbone_dtype=args.backbone_dtype).to(args.device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    steps = 1 if args.dry_run else args.steps
    log = []
    for step in range(steps):
        indices = torch.randint(len(samples), (args.batch_size,), generator=generator).tolist()
        batch = collate_text_samples([samples[i] for i in indices], tokenizer, max_length=args.max_length)
        batch = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = text_loss(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite assistant CE at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        record = {"step": step + 1, "assistant_ce": float(loss.detach())}
        log.append(record)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            print(json.dumps(record), flush=True)
    _verify_source_fingerprint(metadata)
    safe_output(output, create=True)
    tokenizer.save_pretrained(output)
    if args.backbone == "tiny":
        torch.save(model.state_dict(), output / "text_model.pt")
    else:
        model.save_pretrained(output / "model")
    config = {key: value for key, value in vars(args).items() if key not in {"data", "output"}}
    config.update(data=str(args.data), implementation="text_token_ce", offline_smoke=args.backbone == "tiny")
    config["run_metadata"] = metadata
    (output / "text_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "text_training_log.json").write_text(json.dumps(log, indent=2) + "\n")


if __name__ == "__main__":
    main()
