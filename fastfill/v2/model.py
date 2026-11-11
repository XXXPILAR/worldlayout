"""Qwen condition memory, bidirectional request slots, continuous geometry heads.

``position_head`` selects the XY parameterization: ``regression`` emits
normalized (x, y, z) directly; ``grid_residual`` emits logits over
``position_grid`` x ``position_grid`` cells plus a tanh XY residual per cell in
half-cell units (the yaw bin+residual scheme) and regresses z. Both decode to
``position_normalized``, so matching, regularizers, evaluation and the hand-off
never see the head type; the grid head additionally returns
``position_cell_logits`` / ``position_cell_residuals`` for the criterion.

``pose_conditioning`` adds ``given_pose_embedding``: given requests
(``given_pose`` / ``given_pose_mask``, see ``masked_layout``) add the embedding of
their pose features to their slot before the bidirectional decoder, so every
other slot attends to them. Its last layer starts at zero (a new model behaves
like one without it), and outputs of given slots are not replaced by their poses.

The explicit ``tiny`` backend is a causal GRU for offline correctness checks.
It is not a Qwen model and must not be used as evidence of model quality.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from fastfill.v2.geometry import decode_grid_position
from fastfill.v2.masked_layout import POSE_FEATURES

SUPPORT_MASKED = -1e4  # support logit of an impossible parent (the slot itself, padding)

MODEL_INPUT_KEYS = (
    "input_ids", "attention_mask", "object_spans", "slot_mask",
    "fixed_position_normalized", "fixed_position_mask", "fixed_size", "fixed_size_mask",
    "given_pose", "given_pose_mask",
)


def model_inputs(batch: dict) -> dict:
    return {key: batch[key] for key in MODEL_INPUT_KEYS if key in batch}


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "Qwen/Qwen2.5-0.5B-Instruct"
    decoder_dim: int = 128
    decoder_heads: int = 4
    decoder_layers: int = 2
    decoder_ffn_multiplier: int = 4
    dropout: float = 0.0
    yaw_bins: int = 12
    max_objects: int = 128
    size_reference: tuple[float, float, float] = (1., 1., 1.)
    size_log_limit: float = 10.0
    residual_mode: str = "tanh"
    position_head: str = "regression"
    position_grid: int = 16
    train_backbone: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    backbone_dtype: str = "float32"
    local_files_only: bool = False
    tiny_hidden_size: int = 64
    tiny_layers: int = 1
    pose_conditioning: bool = False
    # Support pointer head: per slot, logits over floor | wall | every slot (``support_logits``, b x n x 2 + n).
    support_head: bool = False
    # Recompute backbone activations in the backward pass: needed to fit 27B-class backbones at 8192 tokens.
    gradient_checkpointing: bool = False
    # Run the backbone under bf16 autocast at inference (eval mode), as it trained under mixed precision bf16: a full
    # fine-tuning checkpoint keeps its fp32 master weights, and fp32 attention has no memory-efficient kernel, so it
    # holds heads x tokens^2 scores (45 GiB on a 19,399-token room for Qwen3-4B).
    inference_autocast_bf16: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.backbone, str) or not self.backbone.strip():
            raise ValueError("backbone must be a nonempty checkpoint identifier or path")
        positive_integers = ("decoder_dim", "decoder_heads", "decoder_layers", "decoder_ffn_multiplier",
                             "yaw_bins", "max_objects", "tiny_hidden_size", "tiny_layers", "lora_alpha", "position_grid")
        for key in positive_integers:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if isinstance(self.lora_rank, bool) or not isinstance(self.lora_rank, int) or self.lora_rank < 0:
            raise ValueError("LoRA rank must be a nonnegative integer")
        for key in ("dropout", "lora_dropout"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and in [0,1]")
        for key in ("train_backbone", "local_files_only", "pose_conditioning", "support_head", "gradient_checkpointing",
                    "inference_autocast_bf16"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be a boolean")
        if (not isinstance(self.lora_target_modules, (list, tuple)) or not self.lora_target_modules or
                any(not isinstance(value, str) or not value.strip() for value in self.lora_target_modules)):
            raise ValueError("LoRA target modules must contain nonempty module names")
        if self.decoder_dim <= 0 or self.decoder_heads <= 0 or self.decoder_dim % self.decoder_heads:
            raise ValueError("decoder width must be positive and divisible by attention heads")
        if self.decoder_layers < 1 or self.yaw_bins < 1 or self.max_objects < 1:
            raise ValueError("decoder layers, yaw bins and object budget must be positive")
        if (not isinstance(self.size_reference, (list, tuple)) or len(self.size_reference) != 3 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not (0 < v < float("inf")) for v in self.size_reference)):
            raise ValueError("size reference must contain three positive finite values")
        if (isinstance(self.size_log_limit, bool) or not isinstance(self.size_log_limit, (int, float)) or
                not 0 < self.size_log_limit <= 30):
            raise ValueError("log size limit must lie in (0,30]")
        limits = torch.finfo(torch.float32)
        if any(v * math.exp(-self.size_log_limit) < limits.tiny or
               v * math.exp(self.size_log_limit) > limits.max for v in self.size_reference):
            raise ValueError("size reference and exponent bounds must preserve positive finite float32 sizes")
        if self.residual_mode not in {"tanh", "unbounded"}:
            raise ValueError("residual mode must be tanh or unbounded")
        if self.position_head not in {"regression", "grid_residual"}:
            raise ValueError("position head must be regression or grid_residual")
        if self.backbone_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported backbone dtype")
        if self.train_backbone and self.lora_rank:
            raise ValueError("select full backbone training or LoRA, not both")


class TinyConditionBackbone(nn.Module):
    def __init__(self, hidden_size: int, layers: int = 1):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, model_type="offline_tiny_gru")
        self.embedding = nn.Embedding(258, hidden_size, padding_idx=0)
        self.encoder = nn.GRU(hidden_size, hidden_size, layers, batch_first=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: Any) -> Any:
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(input_ids),
            attention_mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.encoder(packed)
        memory, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=input_ids.shape[1])
        return SimpleNamespace(last_hidden_state=memory)


def _hidden_size(config) -> int:
    """Width of the text model's last hidden state; multimodal checkpoints (e.g. Qwen3.5) keep it in ``text_config``."""
    text = getattr(config, "text_config", None)
    return (text if text is not None else config).hidden_size


def _qwen_backbone(config: ModelConfig) -> nn.Module:
    from transformers import AutoConfig, AutoModel
    pretrained_config = AutoConfig.from_pretrained(config.backbone, local_files_only=config.local_files_only)
    if not str(pretrained_config.model_type).startswith("qwen"):
        raise ValueError("structured condition backbone must be a Qwen-family checkpoint")
    backbone = AutoModel.from_pretrained(config.backbone, config=pretrained_config,
        torch_dtype=getattr(torch, config.backbone_dtype), local_files_only=config.local_files_only)
    backbone.config.use_cache = False
    if config.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if config.lora_rank:
        from peft import LoraConfig, TaskType, get_peft_model
        backbone = get_peft_model(backbone, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules)))
    elif not config.train_backbone:
        backbone.requires_grad_(False)
    return backbone


class StructuredFastFillModel(nn.Module):
    def __init__(self, config: ModelConfig, backbone: nn.Module):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.memory_projection = nn.Linear(_hidden_size(backbone.config), config.decoder_dim)
        self.slot_seed = nn.Embedding(config.max_objects, config.decoder_dim)
        layer = nn.TransformerDecoderLayer(config.decoder_dim, config.decoder_heads,
            dim_feedforward=config.decoder_dim * config.decoder_ffn_multiplier,
            dropout=config.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, config.decoder_layers, norm=nn.LayerNorm(config.decoder_dim))
        # grid_residual: one Linear (so trainer gradient logging by name is unchanged) holding
        # G*G cell logits | G*G tanh XY residual pairs in half-cell units | one regressed z.
        cells = config.position_grid ** 2 if config.position_head == "grid_residual" else 0
        self.position_head = nn.Linear(config.decoder_dim, 3 * cells + 1 if cells else 3)
        self.size_head = nn.Linear(config.decoder_dim, 3)
        self.yaw_logits_head = nn.Linear(config.decoder_dim, config.yaw_bins)
        self.yaw_residual_head = nn.Linear(config.decoder_dim, config.yaw_bins)
        self.register_buffer("size_reference", torch.tensor(config.size_reference, dtype=torch.float32))
        if config.pose_conditioning:
            self.given_pose_embedding = nn.Sequential(nn.Linear(POSE_FEATURES, config.decoder_dim), nn.GELU(),
                                                      nn.Linear(config.decoder_dim, config.decoder_dim))
            nn.init.zeros_(self.given_pose_embedding[-1].weight)
            nn.init.zeros_(self.given_pose_embedding[-1].bias)
        if config.support_head:
            self.support_query = nn.Linear(config.decoder_dim, config.decoder_dim)
            self.support_key = nn.Linear(config.decoder_dim, config.decoder_dim)
            self.support_anchor = nn.Parameter(torch.randn(2, config.decoder_dim) * .02)  # floor, wall

    def _slots(self, memory: torch.Tensor, object_spans: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = object_spans.shape
        if n > self.config.max_objects:
            raise ValueError("number of slots exceeds model object budget")
        starts, ends = object_spans.unbind(-1)
        if (((starts < 0) | (ends > memory.shape[1]) | (ends <= starts)) & slot_mask).any():
            raise ValueError("valid slot span must be inside condition token memory")
        # Prefix differences lose short object spans at late token positions in
        # bf16, and fp16 prefixes can overflow despite finite token features.
        # Accumulate at least in float32, retain float64 when supplied, and keep
        # the cast differentiable before returning the pooled memory precision.
        accumulator = memory.to(torch.float64 if memory.dtype == torch.float64 else torch.float32)
        cumulative = torch.cat((accumulator.new_zeros((b, 1, memory.shape[-1])), accumulator.cumsum(1)), dim=1)
        start_indices = starts.clamp(0, memory.shape[1]).unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        end_indices = ends.clamp(0, memory.shape[1]).unsqueeze(-1).expand_as(start_indices)
        pooled = ((cumulative.gather(1, end_indices) - cumulative.gather(1, start_indices)) /
                  (ends - starts).clamp_min(1).unsqueeze(-1)).to(memory.dtype)
        seed = self.slot_seed(torch.arange(n, device=memory.device)).unsqueeze(0)
        return (pooled + seed) * slot_mask.unsqueeze(-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                object_spans: torch.Tensor, slot_mask: torch.Tensor,
                fixed_position_normalized: torch.Tensor | None = None,
                fixed_position_mask: torch.Tensor | None = None,
                fixed_size: torch.Tensor | None = None,
                fixed_size_mask: torch.Tensor | None = None,
                given_pose: torch.Tensor | None = None,
                given_pose_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        encoded = self.encode(input_ids, attention_mask, object_spans, slot_mask)
        return self.decode(encoded, fixed_position_normalized=fixed_position_normalized,
                           fixed_position_mask=fixed_position_mask, fixed_size=fixed_size, fixed_size_mask=fixed_size_mask,
                           given_pose=given_pose, given_pose_mask=given_pose_mask)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, object_spans: torch.Tensor,
               slot_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Backbone pass and slot pooling: everything that does not depend on given poses, so remasking decoding
        (``remask``) runs it once per request and ``decode`` once per round."""
        if not attention_mask.bool().any(dim=1).all():
            raise ValueError("every condition requires at least one unmasked token")
        cast = self.config.inference_autocast_bf16 and not self.training
        with torch.autocast(input_ids.device.type, dtype=torch.bfloat16) if cast else contextlib.nullcontext():
            memory = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                                   return_dict=True).last_hidden_state
        memory = self.memory_projection(memory.to(self.memory_projection.weight.dtype))
        return {"memory": memory, "slots": self._slots(memory, object_spans, slot_mask),
                "attention_mask": attention_mask, "slot_mask": slot_mask}

    def decode(self, encoded: dict[str, torch.Tensor], *,
               fixed_position_normalized: torch.Tensor | None = None,
               fixed_position_mask: torch.Tensor | None = None,
               fixed_size: torch.Tensor | None = None,
               fixed_size_mask: torch.Tensor | None = None,
               given_pose: torch.Tensor | None = None,
               given_pose_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        memory, slots = encoded["memory"], encoded["slots"]
        attention_mask, slot_mask = encoded["attention_mask"], encoded["slot_mask"]
        if self.config.pose_conditioning:
            if given_pose_mask is None:
                given_pose_mask = torch.zeros_like(slot_mask, dtype=torch.bool)
                given_pose = slots.new_zeros((*slot_mask.shape, POSE_FEATURES))
            given = given_pose_mask.bool() & slot_mask.bool()
            if given_pose is None or given_pose.shape != (*given.shape, POSE_FEATURES):
                raise ValueError(f"given_pose must hold {POSE_FEATURES} features per slot")
            if not torch.isfinite(given_pose[given]).all():
                raise ValueError("given pose features must be finite")
            # Applied even with nothing given: under DDP every rank must use every parameter in each step.
            slots = slots + self.given_pose_embedding(given_pose.to(slots.dtype)) * given.unsqueeze(-1)
        elif given_pose_mask is not None and given_pose_mask.any():
            raise ValueError("given poses need a model built with pose_conditioning")
        if slots.shape[1]:
            safe_mask = slot_mask.bool().clone()
            safe_mask[~safe_mask.any(1), 0] = True  # avoid all-masked attention for empty scenes
            slots = self.decoder(slots, memory, tgt_key_padding_mask=~safe_mask,
                                 memory_key_padding_mask=~attention_mask.bool())
        active = slot_mask.unsqueeze(-1)
        raw_position = self.position_head(slots)
        grid = {}
        if self.config.position_head == "grid_residual":
            g = self.config.position_grid
            cell_logits = raw_position[..., :g * g]
            cell_residuals = raw_position[..., g * g:3 * g * g].reshape(*raw_position.shape[:-1], g * g, 2).tanh()
            xy = decode_grid_position(cell_logits, cell_residuals, g)
            raw_position = torch.cat((xy, raw_position[..., -1:].to(xy.dtype)), -1)
            grid = {"position_cell_logits": cell_logits * active, "position_cell_residuals": cell_residuals * active.unsqueeze(-1)}
        # Exponentiation always uses float32, even under fp16 autocast; positive
        # numerical bounds validated above remain valid in mixed precision.
        raw_size = self.size_reference.float() * self.size_head(slots).float().clamp(
            -self.config.size_log_limit, self.config.size_log_limit).exp()
        position, size = raw_position, raw_size
        if fixed_position_mask is not None:
            position = torch.where(fixed_position_mask, fixed_position_normalized, position)
        if fixed_size_mask is not None:
            if ((~torch.isfinite(fixed_size) | (fixed_size <= 0)) & fixed_size_mask).any():
                raise ValueError("fixed sizes must be positive and finite")
            size = torch.where(fixed_size_mask, fixed_size, size)
        residual = self.yaw_residual_head(slots)
        if self.config.residual_mode == "tanh":
            residual = residual.tanh()
        if self.config.support_head:
            grid["support_logits"] = self._support_logits(slots, slot_mask.bool())
        return {"position_normalized": position * active, "size": size * active,
                "yaw_logits": self.yaw_logits_head(slots) * active, "yaw_residuals": residual * active,
                "slot_mask": slot_mask, "unconstrained_position_normalized": raw_position * active,
                "unconstrained_size": raw_size * active, **grid}

    def _support_logits(self, slots: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        """Scaled dot products of each slot's query with the floor / wall anchors and every slot's key; a slot can
        not stand on itself or on padding (``SUPPORT_MASKED`` logit), padded rows are zero."""
        query = self.support_query(slots)
        keys = torch.cat((self.support_anchor.to(slots.dtype).expand(slots.shape[0], -1, -1), self.support_key(slots)), 1)
        logits = (query @ keys.transpose(1, 2)).float() / math.sqrt(slots.shape[-1])
        n = slots.shape[1]
        blocked = torch.eye(n, dtype=torch.bool, device=slots.device)[None] | ~slot_mask[:, None, :]
        logits[..., 2:] = logits[..., 2:].masked_fill(blocked, SUPPORT_MASKED)
        return logits * slot_mask[..., None]

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model_config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")
        state = self.state_dict()
        if self.config.backbone != "tiny":
            if self.config.lora_rank or self.config.train_backbone:
                self.backbone.save_pretrained(directory / "backbone")
            state = {key: value for key, value in state.items() if not key.startswith("backbone.")}
        torch.save(state, directory / "geometry_model.pt")


def build_model(config: ModelConfig | dict | None = None) -> StructuredFastFillModel:
    config = ModelConfig(**config) if isinstance(config, dict) else config or ModelConfig()
    backbone = (TinyConditionBackbone(config.tiny_hidden_size, config.tiny_layers)
                if config.backbone == "tiny" else _qwen_backbone(config))
    return StructuredFastFillModel(config, backbone)


def load_model(directory: str | Path, *, device: str | torch.device = "cpu",
               local_files_only: bool | None = None) -> StructuredFastFillModel:
    directory = Path(directory)
    config = ModelConfig(**json.loads((directory / "model_config.json").read_text()))
    if local_files_only is not None:
        config = replace(config, local_files_only=local_files_only)
    if config.backbone != "tiny" and config.train_backbone:
        model = build_model(replace(config, backbone=str(directory / "backbone")))
        model.config = config
    elif config.backbone != "tiny" and config.lora_rank:
        from peft import PeftModel
        backbone = _qwen_backbone(replace(config, lora_rank=0))
        backbone = PeftModel.from_pretrained(backbone, directory / "backbone", is_trainable=True)
        model = StructuredFastFillModel(config, backbone)
    else:
        model = build_model(config)
    state = torch.load(directory / "geometry_model.pt", map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=config.backbone == "tiny")
    if config.backbone != "tiny" and (incompatible.unexpected_keys or
            any(not key.startswith("backbone.") for key in incompatible.missing_keys)):
        raise ValueError("checkpoint geometry decoder/head keys do not match model configuration")
    return model.to(device)
