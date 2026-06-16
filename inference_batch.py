#!/usr/bin/env python3
"""
Module 2: centralized batch inference from saved LoRA checkpoints.

This script loads the checkpoints saved by `train_only.py`, resolves the
corresponding LoRA for each epoch, and writes generated validation images
without doing any metric computation.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from sci_color_lab.config import TrainerConfig
from sci_color_lab.data import EvaluationRecord, PairRecord, select_validation_records
from sci_color_lab.memory import PeakMemoryMonitor
from sci_color_lab.pipeline import InferenceEngine
from sci_color_lab.trainer import load_json, save_json

DEFAULT_SHARED_SELECTION_MANIFEST = (Path(__file__).resolve().parent / "shared_validation_selection_top20.json").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 2: centralized LoRA batch inference")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories, for example outputs/E_FULL/seed_42")
    parser.add_argument(
        "--epochs",
        type=str,
        default="all",
        help="Epoch list like 2,4,6 or 'all' to use every saved LoRA checkpoint",
    )
    parser.add_argument("--max-samples", type=int, default=20, help="Maximum validation samples, <=0 means all")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Diffusion inference steps")
    parser.add_argument("--guidance-scale", type=float, default=6.5, help="CFG guidance scale")
    parser.add_argument("--controlnet-scale", type=float, default=None, help="Optional ControlNet conditioning scale override")
    parser.add_argument("--scheduler", type=str, default="unipc", help="Scheduler name passed to InferenceEngine")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--dtype", type=str, default="fp16", help="Inference dtype")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing epoch inference outputs")
    parser.add_argument(
        "--shared-selection-manifest",
        type=str,
        default=str(DEFAULT_SHARED_SELECTION_MANIFEST),
        help="Canonical shared validation selection manifest path; empty disables manifest creation/loading.",
    )
    parser.add_argument(
        "--no-shared-selection",
        action="store_true",
        help="Disable canonical shared validation selection enforcement for this run.",
    )
    return parser.parse_args()


def parse_epochs(text: str) -> list[int] | None:
    normalized = str(text).strip().lower()
    if not normalized or normalized in {"all", "*", "auto"}:
        return None
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    return sorted(set(values))


def load_run_context(run_dir: Path) -> tuple[dict, dict, TrainerConfig, str, int]:
    metadata = load_json(run_dir / "run_metadata.json")
    if not metadata:
        raise FileNotFoundError(f"run_metadata.json not found in {run_dir}")
    summary = load_json(run_dir / "run_summary.json")
    trainer_cfg = TrainerConfig(**metadata["trainer_config"])
    group_id = str(metadata.get("group", {}).get("group_id") or summary.get("group_id") or run_dir.parent.name)
    seed_text = summary.get("seed")
    if seed_text is None:
        try:
            seed_text = int(str(run_dir.name).split("_")[-1])
        except Exception:
            seed_text = 0
    return metadata, summary, trainer_cfg, group_id, int(seed_text)


def build_records_from_payload(items: list[dict]) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        records.append(
            EvaluationRecord(
                image_id=str(item.get("image_id", "")),
                lineart_path=str(item.get("lineart_path", "")),
                color_path=str(item.get("color_path", "")),
                has_reference=bool(item.get("has_reference", False)),
                source=str(item.get("source", "")),
            )
        )
    return [record for record in records if record.image_id and record.lineart_path]


def record_to_payload(record: EvaluationRecord) -> dict[str, object]:
    return {
        "image_id": record.image_id,
        "lineart_path": record.lineart_path,
        "color_path": record.color_path,
        "has_reference": bool(record.has_reference),
        "source": record.source,
    }


def apply_shared_selection(
    records: list[EvaluationRecord],
    *,
    max_samples: int,
    validation_source: str,
    validation_note: str,
    manifest_path: str,
) -> tuple[list[EvaluationRecord], str]:
    if not records:
        return [], ""

    requested = int(max_samples)
    default_records = list(records)
    if requested > 0:
        default_records = default_records[:requested]

    manifest_text = str(manifest_path).strip()
    if not manifest_text:
        return default_records, ""

    lookup: dict[str, EvaluationRecord] = {}
    duplicate_ids: set[str] = set()
    for record in records:
        if record.image_id in lookup:
            duplicate_ids.add(record.image_id)
            continue
        lookup[record.image_id] = record
    if duplicate_ids:
        duplicate_list = ", ".join(sorted(duplicate_ids))
        raise RuntimeError(f"Duplicate image_id detected in validation records: {duplicate_list}")

    manifest_file = Path(manifest_text)
    payload = load_json(manifest_file)
    if isinstance(payload.get("数据"), dict):
        payload = payload.get("数据", {})
    manifest_ids = [str(item).strip() for item in payload.get("image_ids", []) if str(item).strip()]
    if not manifest_ids:
        manifest_ids = [
            str(item.get("image_id", "")).strip()
            for item in payload.get("records", [])
            if isinstance(item, dict) and str(item.get("image_id", "")).strip()
        ]

    if manifest_ids:
        if requested > 0 and len(manifest_ids) != requested:
            raise RuntimeError(
                f"Shared selection manifest count {len(manifest_ids)} does not match requested max_samples={requested}: {manifest_file}"
            )
        missing_ids = [image_id for image_id in manifest_ids if image_id not in lookup]
        if missing_ids:
            raise RuntimeError(
                f"Shared selection manifest contains image_ids not found in current validation records: {missing_ids}"
            )
        selected_records = [lookup[image_id] for image_id in manifest_ids]
        return selected_records, str(manifest_file.resolve())

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "mode": "shared_validation_selection",
        "record_count": len(default_records),
        "requested_max_samples": requested,
        "source": validation_source,
        "note": validation_note,
        "image_ids": [record.image_id for record in default_records],
        "records": [record_to_payload(record) for record in default_records],
    }
    with manifest_file.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, ensure_ascii=False, indent=2)
    return default_records, str(manifest_file.resolve())


def load_validation_records(
    run_dir: Path,
    trainer_cfg: TrainerConfig,
    max_samples: int,
    *,
    shared_selection_manifest: str = "",
) -> tuple[list[EvaluationRecord], str, str, str]:
    selection_payload = load_json(run_dir / "validation_selection.json")
    records = build_records_from_payload(selection_payload.get("metric_records", []))
    source = str(selection_payload.get("source", ""))
    note = str(selection_payload.get("note", ""))

    if not records:
        split_payload = load_json(run_dir / "dataset_split.json")
        split_val = [PairRecord(**item) for item in split_payload.get("val", []) if isinstance(item, dict)]
        if split_val:
            selection = select_validation_records(
                split_val_records=split_val,
                validation_dataset_root=trainer_cfg.validation_dataset_root,
                validation_color_dir_name=trainer_cfg.validation_color_dir_name,
                validation_lineart_dir_name=trainer_cfg.validation_lineart_dir_name,
                prefer_external_validation_dataset=bool(trainer_cfg.prefer_external_validation_dataset),
            )
            records = list(selection.metric_records)
            source = selection.source
            note = selection.note

    if not records:
        raise RuntimeError(f"No validation records available for inference under {run_dir}")

    records, resolved_manifest_path = apply_shared_selection(
        records,
        max_samples=max_samples,
        validation_source=source,
        validation_note=note,
        manifest_path=shared_selection_manifest,
    )

    return records, source, note, resolved_manifest_path


def discover_checkpoint_epochs(run_dir: Path) -> list[int]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    epoch_to_step: dict[int, int] = {}
    for checkpoint_dir in checkpoints_dir.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        metadata = load_json(checkpoint_dir / "checkpoint.json")
        try:
            epoch = int(metadata.get("epoch", 0) or 0)
        except Exception:
            epoch = 0
        try:
            step = int(metadata.get("step", 0) or 0)
        except Exception:
            step = 0
        if epoch <= 0:
            continue
        epoch_to_step[epoch] = max(epoch_to_step.get(epoch, -1), step)
    return sorted(epoch_to_step)


def resolve_checkpoint_dir(run_dir: Path, epoch: int) -> Path | None:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    best_step = -1
    best_path: Path | None = None
    for checkpoint_dir in checkpoints_dir.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        metadata = load_json(checkpoint_dir / "checkpoint.json")
        try:
            checkpoint_epoch = int(metadata.get("epoch", 0) or 0)
        except Exception:
            checkpoint_epoch = 0
        if checkpoint_epoch != int(epoch):
            continue
        try:
            checkpoint_step = int(metadata.get("step", 0) or 0)
        except Exception:
            checkpoint_step = 0
        if checkpoint_step >= best_step:
            best_step = checkpoint_step
            best_path = checkpoint_dir

    if best_path is not None:
        return best_path

    legacy_epoch_dir = checkpoints_dir / f"epoch_{int(epoch):03d}"
    if legacy_epoch_dir.exists():
        return legacy_epoch_dir

    for name in ("latest", "best_fid"):
        candidate = checkpoints_dir / name
        metadata = load_json(candidate / "checkpoint.json")
        try:
            checkpoint_epoch = int(metadata.get("epoch", 0) or 0)
        except Exception:
            checkpoint_epoch = 0
        if candidate.exists() and checkpoint_epoch == int(epoch):
            return candidate

    return None


def inference_complete(eval_dir: Path, expected_samples: int) -> bool:
    generated_dir = eval_dir / "generated"
    target_dir = eval_dir / "target"
    lineart_dir = eval_dir / "lineart"
    if not generated_dir.exists() or not lineart_dir.exists():
        return False
    generated_count = len(list(generated_dir.glob("*.png")))
    lineart_count = len(list(lineart_dir.glob("*.png")))
    target_count = len(list(target_dir.glob("*.png"))) if target_dir.exists() else 0
    return generated_count >= expected_samples and lineart_count >= expected_samples and target_count >= expected_samples


def build_inference_engine(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    trainer_cfg: TrainerConfig,
    scheduler_name: str,
    device: str,
    dtype: str,
    cpu_offload: bool,
) -> InferenceEngine:
    return InferenceEngine(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        scheduler_name=scheduler_name,
        device=device,
        dtype=dtype,
        cpu_offload=cpu_offload,
        enable_xformers=bool(trainer_cfg.enable_xformers),
    )


@torch.no_grad()
def run_inference_for_epoch(
    *,
    run_dir: Path,
    group_id: str,
    seed: int,
    trainer_cfg: TrainerConfig,
    checkpoint_dir: Path,
    epoch: int,
    records: list[EvaluationRecord],
    validation_source: str,
    validation_note: str,
    selection_manifest_path: str,
    engine: InferenceEngine,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_scale: float | None,
    device: str,
) -> dict[str, object]:
    eval_dir = run_dir / "evaluations" / "validation_epochs" / f"epoch_{int(epoch):03d}"
    generated_dir = eval_dir / "generated"
    target_dir = eval_dir / "target"
    lineart_dir = eval_dir / "lineart"
    generated_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    lineart_dir.mkdir(parents=True, exist_ok=True)

    memory_monitor = PeakMemoryMonitor(device=device).start()
    inference_times: list[float] = []
    rows: list[dict[str, object]] = []

    for index, record in enumerate(tqdm(records, desc=f"{run_dir.name} epoch {epoch:03d}", leave=False), start=1):
        lineart_image = Image.open(record.lineart_path).convert("RGB").resize(
            (trainer_cfg.image_width, trainer_cfg.image_height),
            Image.NEAREST,
        )
        target_image = None
        if record.color_path and Path(record.color_path).exists():
            target_image = Image.open(record.color_path).convert("RGB").resize(
                (trainer_cfg.image_width, trainer_cfg.image_height),
                Image.LANCZOS,
            )

        file_name = f"{index:03d}_{record.image_id}.png"
        lineart_path = lineart_dir / file_name
        generated_path = generated_dir / file_name
        target_path = target_dir / file_name

        lineart_image.save(lineart_path)
        if target_image is not None:
            target_image.save(target_path)

        started = time.perf_counter()
        result_np = engine.colorize(
            lineart_image=np.array(lineart_image),
            prompt=trainer_cfg.prompt_template,
            negative_prompt=trainer_cfg.negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            controlnet_scale=float(
                trainer_cfg.controlnet_conditioning_scale if controlnet_scale is None else controlnet_scale
            ),
            seed=int(seed + index),
            width=int(trainer_cfg.image_width),
            height=int(trainer_cfg.image_height),
        )
        inference_time_ms = (time.perf_counter() - started) * 1000.0
        Image.fromarray(result_np).save(generated_path)

        inference_times.append(inference_time_ms)
        rows.append(
            {
                "image_id": record.image_id,
                "file_name": file_name,
                "generated_path": str(generated_path.resolve()),
                "target_path": str(target_path.resolve()) if target_image is not None else "",
                "lineart_path": str(lineart_path.resolve()),
                "inference_time_ms": inference_time_ms,
                "has_reference": bool(target_image is not None),
                "source": record.source,
            }
        )

        del lineart_image, target_image, result_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    peak_memory = memory_monitor.stop()

    summary_payload = {
        "group_id": group_id,
        "seed": int(seed),
        "epoch": int(epoch),
        "split": "validation_epoch",
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "validation_source": validation_source,
        "validation_note": validation_note,
        "shared_selection_manifest_path": selection_manifest_path,
        "selection_image_ids": [record.image_id for record in records],
        "params_m": load_json(run_dir / "run_summary.json").get("params_m"),
        "flops_g": load_json(run_dir / "run_summary.json").get("flops_g"),
        "num_samples": len(rows),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "controlnet_scale": float(trainer_cfg.controlnet_conditioning_scale if controlnet_scale is None else controlnet_scale),
        "rows": rows,
    }
    save_json(eval_dir / "generation_records.json", summary_payload)

    metrics_stub = {
        "group_id": group_id,
        "seed": int(seed),
        "epoch": int(epoch),
        "split": "validation_epoch",
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "generated_dir": str(generated_dir.resolve()),
        "target_dir": str(target_dir.resolve()),
        "lineart_dir": str(lineart_dir.resolve()),
        "validation_source": validation_source,
        "validation_note": validation_note,
        "shared_selection_manifest_path": selection_manifest_path,
        "selection_image_ids": [record.image_id for record in records],
        "params_m": summary_payload.get("params_m"),
        "flops_g": summary_payload.get("flops_g"),
        "generated_samples_count": len(rows),
        "generation_records_path": str((eval_dir / "generation_records.json").resolve()),
        "avg_inference_time_ms": sum(inference_times) / len(inference_times) if inference_times else 0.0,
        "gpu_memory_peak_gb": peak_memory.get("gpu_memory_peak_gb"),
        "gpu_memory_reserved_peak_gb": peak_memory.get("gpu_memory_reserved_peak_gb"),
        "cpu_memory_peak_gb": peak_memory.get("cpu_memory_peak_gb"),
        "metrics_computed": False,
    }
    save_json(eval_dir / "metrics.json", metrics_stub)

    return {
        "group_id": group_id,
        "seed": int(seed),
        "epoch": int(epoch),
        "run_dir": str(run_dir.resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "num_samples": len(rows),
        "shared_selection_manifest_path": selection_manifest_path,
        "avg_inference_time_ms": metrics_stub["avg_inference_time_ms"],
        "peak_gpu_memory_gb": metrics_stub["gpu_memory_peak_gb"],
    }


def main() -> int:
    args = parse_args()
    requested_epochs = parse_epochs(args.epochs)

    all_results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    print("[inference_batch] configuration")
    print(f"  run_dirs={args.run_dirs}")
    print(f"  epochs={args.epochs}")
    print(f"  max_samples={args.max_samples}")
    print(f"  num_inference_steps={args.num_inference_steps}")
    print(f"  guidance_scale={args.guidance_scale}")
    print(f"  controlnet_scale={args.controlnet_scale}")
    print(f"  device={args.device}")
    print(f"  dtype={args.dtype}")
    print(f"  cpu_offload={args.cpu_offload}")
    print(f"  overwrite={args.overwrite}")
    print(f"  shared_selection_manifest={args.shared_selection_manifest}")
    print(f"  no_shared_selection={args.no_shared_selection}")

    for run_dir_text in args.run_dirs:
        run_dir = Path(run_dir_text)
        if not run_dir.exists():
            failures.append({"run_dir": str(run_dir), "error": "run directory not found"})
            continue

        try:
            _, _, trainer_cfg, group_id, seed = load_run_context(run_dir)
            records, validation_source, validation_note, selection_manifest_path = load_validation_records(
                run_dir,
                trainer_cfg,
                args.max_samples,
                shared_selection_manifest="" if args.no_shared_selection else args.shared_selection_manifest,
            )
        except Exception as exc:
            failures.append({"run_dir": str(run_dir), "error": str(exc)})
            continue

        epochs = requested_epochs or discover_checkpoint_epochs(run_dir)
        if not epochs:
            failures.append({"run_dir": str(run_dir), "error": "no saved checkpoint epochs found"})
            continue

        print(
            f"[inference_batch] run={run_dir} epochs={epochs} samples={len(records)} "
            f"shared_selection_manifest={selection_manifest_path or 'disabled'}"
        )

        for epoch in epochs:
            checkpoint_dir = resolve_checkpoint_dir(run_dir, epoch)
            if checkpoint_dir is None:
                failures.append(
                    {
                        "run_dir": str(run_dir),
                        "epoch": int(epoch),
                        "error": "checkpoint for epoch not found",
                    }
                )
                continue

            eval_dir = run_dir / "evaluations" / "validation_epochs" / f"epoch_{int(epoch):03d}"
            if not args.overwrite and inference_complete(eval_dir, len(records)):
                all_results.append(
                    {
                        "group_id": group_id,
                        "seed": int(seed),
                        "epoch": int(epoch),
                        "run_dir": str(run_dir.resolve()),
                        "eval_dir": str(eval_dir.resolve()),
                        "checkpoint_dir": str(checkpoint_dir.resolve()),
                        "num_samples": len(records),
                        "shared_selection_manifest_path": selection_manifest_path,
                        "skipped": True,
                    }
                )
                continue

            if eval_dir.exists():
                shutil.rmtree(eval_dir)

            engine: InferenceEngine | None = None
            try:
                engine = build_inference_engine(
                    run_dir=run_dir,
                    checkpoint_dir=checkpoint_dir,
                    trainer_cfg=trainer_cfg,
                    scheduler_name=args.scheduler,
                    device=args.device,
                    dtype=args.dtype,
                    cpu_offload=args.cpu_offload,
                )
                result = run_inference_for_epoch(
                    run_dir=run_dir,
                    group_id=group_id,
                    seed=seed,
                    trainer_cfg=trainer_cfg,
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    records=records,
                    validation_source=validation_source,
                    validation_note=validation_note,
                    selection_manifest_path=selection_manifest_path,
                    engine=engine,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    controlnet_scale=args.controlnet_scale,
                    device=args.device,
                )
                all_results.append(result)
            except Exception as exc:
                failures.append(
                    {
                        "run_dir": str(run_dir),
                        "epoch": int(epoch),
                        "checkpoint_dir": str(checkpoint_dir),
                        "error": str(exc),
                    }
                )
            finally:
                if engine is not None:
                    del engine
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = {
        "mode": "inference_batch",
        "run_dirs": [str(Path(item).resolve()) for item in args.run_dirs],
        "requested_epochs": args.epochs,
        "max_samples": int(args.max_samples),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "controlnet_scale": args.controlnet_scale,
        "scheduler": args.scheduler,
        "device": args.device,
        "dtype": args.dtype,
        "cpu_offload": bool(args.cpu_offload),
        "overwrite": bool(args.overwrite),
        "shared_selection_manifest": (
            ""
            if args.no_shared_selection or not str(args.shared_selection_manifest).strip()
            else str(Path(args.shared_selection_manifest).resolve())
        ),
        "results": all_results,
        "failures": failures,
    }

    summary_path: Path
    if args.run_dirs:
        summary_path = Path(args.run_dirs[0]).parent / "inference_batch_summary.json"
    else:
        summary_path = Path("inference_batch_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
