from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import diffusers
import torch
import yaml

from .config import TrainerConfig
from .localized_outputs import sync_output_root_localized_outputs


def build_environment_payload(config: TrainerConfig) -> dict[str, Any]:
    gpu_name = "cpu"
    gpu_memory_gb = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

    controlnet_version = config.controlnet_model or "auto-resolved local cache"

    payload = {
        "environment": {
            "gpu": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "platform": platform.platform(),
            "controlnet_version": controlnet_version,
        },
        "seeds": list(config.seed_list) if config.seed_list else [int(config.seed)],
        "hyperparameters": {
            "optimizer": config.optimizer_name,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "lr_scheduler": config.lr_scheduler_name,
            "warmup_steps": config.warmup_steps,
            "batch_size": config.batch_size,
            "total_epochs": config.epochs,
            "gradient_clip": config.max_grad_norm,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "mixed_precision": config.mixed_precision,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "controlnet_conditioning_scale": config.controlnet_conditioning_scale,
        },
        "data": {
            "dataset": Path(config.dataset_root).name,
            "dataset_root": config.dataset_root,
            "train_split": config.train_ratio,
            "val_split": config.val_ratio,
            "test_split": config.test_ratio,
            "split_seed": config.split_seed,
            "augmentation": {
                "horizontal_flip": config.horizontal_flip,
                "random_crop": config.random_crop,
                "color_jitter": config.color_jitter,
            },
        },
    }
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


def _normalized_seed_list(seeds: Any) -> list[int]:
    if not isinstance(seeds, list):
        return []
    normalized: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        try:
            seed_int = int(seed)
        except Exception:
            continue
        if seed_int in seen:
            continue
        seen.add(seed_int)
        normalized.append(seed_int)
    return normalized


def _payload_without_seeds(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    normalized = dict(payload)
    normalized.pop("seeds", None)
    return normalized


def _write_environment_lock(yaml_path: Path, json_path: Path, payload: dict[str, Any]) -> None:
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def ensure_environment_lock(config: TrainerConfig) -> Path:
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    yaml_path = output_root / "experiment_config.lock.yaml"
    json_path = output_root / "experiment_config.lock.json"
    payload = build_environment_payload(config)

    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle)
        if _payload_without_seeds(existing) != _payload_without_seeds(payload):
            raise RuntimeError(
                f"Environment lock mismatch: {yaml_path} already exists and differs from current config."
            )
        existing_payload = existing if isinstance(existing, dict) else {}
        existing_seeds = _normalized_seed_list(existing_payload.get("seeds"))
        requested_seeds = _normalized_seed_list(payload.get("seeds"))
        merged_seeds = list(dict.fromkeys(existing_seeds + requested_seeds))
        if merged_seeds != existing_seeds:
            existing_payload["seeds"] = merged_seeds
            _write_environment_lock(yaml_path, json_path, existing_payload)
        sync_output_root_localized_outputs(output_root)
        return yaml_path

    _write_environment_lock(yaml_path, json_path, payload)
    sync_output_root_localized_outputs(output_root)
    return yaml_path
