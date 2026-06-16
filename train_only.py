#!/usr/bin/env python3
"""
Module 1: pure training only.

This entry keeps training completely separated from inference and metrics:
- train the model
- save per-epoch LoRA checkpoints
- save logs and training curves
- do not generate validation images
- do not compute FID / LPIPS / SSIM during or after training
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from sci_color_lab.ablation import get_group
from sci_color_lab.config import TrainerConfig
from sci_color_lab.environment import ensure_environment_lock
from sci_color_lab.trainer import ColorizationTrainer


DEFAULT_SEEDS = [42, 123, 456]


def load_trainer_config(path: str | Path) -> TrainerConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "trainer" in payload and isinstance(payload["trainer"], dict):
        payload = payload["trainer"]
    return TrainerConfig(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 1: pure training without inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--group-id", type=str, help="Single ablation group id, for example E_FULL")
    group.add_argument("--groups", nargs="+", help="Multiple ablation group ids")
    parser.add_argument("--config-json", type=str, required=True, help="TrainerConfig json path")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seed list, default [42, 123, 456]")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs from config")
    parser.add_argument("--output-root", type=str, default=None, help="Override output root from config")
    return parser.parse_args()


def build_group_list(args: argparse.Namespace) -> list[str]:
    if args.group_id:
        return [args.group_id]
    return list(args.groups or [])


def build_seed_list(args: argparse.Namespace, config: TrainerConfig) -> list[int]:
    seeds = args.seeds or config.seed_list or DEFAULT_SEEDS
    normalized = [int(seed) for seed in seeds]
    if not normalized:
        return list(DEFAULT_SEEDS)
    return normalized


def prepare_config(base_config: TrainerConfig, *, epochs: int | None, output_root: str | None, seeds: list[int]) -> TrainerConfig:
    payload = asdict(base_config)
    config = TrainerConfig(**payload)
    if epochs is not None:
        config.epochs = int(epochs)
    if output_root:
        config.output_root = output_root
    config.seed_list = list(seeds)
    config.seed = int(seeds[0])
    config.defer_generation_metrics_until_seed_end = False
    config.eval_every_epoch = False
    config.preview_every_epoch = False
    config.final_eval_on_test = False
    config.save_every_epoch = True
    return config


def main() -> int:
    args = parse_args()
    base_config = load_trainer_config(args.config_json)
    seeds = build_seed_list(args, base_config)
    group_ids = build_group_list(args)
    shared_config = prepare_config(
        base_config,
        epochs=args.epochs,
        output_root=args.output_root,
        seeds=seeds,
    )

    ensure_environment_lock(shared_config)

    completed: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for group_id in group_ids:
        group = get_group(group_id)
        for seed in seeds:
            run_config = prepare_config(
                shared_config,
                epochs=shared_config.epochs,
                output_root=shared_config.output_root,
                seeds=seeds,
            )
            run_config.seed = int(seed)
            try:
                trainer = ColorizationTrainer(config=run_config, group=group, seed=seed)
                run_dir = trainer.train_without_preview()
                completed.append(
                    {
                        "group_id": group_id,
                        "seed": int(seed),
                        "run_dir": str(run_dir.resolve()),
                    }
                )
                print(f"[train_only] completed group={group_id} seed={seed} -> {run_dir}")
            except Exception as exc:
                failure = {
                    "group_id": group_id,
                    "seed": int(seed),
                    "error": str(exc),
                }
                failures.append(failure)
                print(json.dumps({"event": "seed_failed", **failure}, ensure_ascii=False))

    summary = {
        "mode": "train_only",
        "config_json": str(Path(args.config_json).resolve()),
        "output_root": str(Path(shared_config.output_root).resolve()),
        "groups": group_ids,
        "seeds": seeds,
        "completed": completed,
        "failures": failures,
    }

    if failures:
        failure_path = Path(shared_config.output_root) / "train_failures.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
