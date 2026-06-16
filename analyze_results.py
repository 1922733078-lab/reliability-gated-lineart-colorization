#!/usr/bin/env python3
"""
Module 3: compute metrics from centralized inference outputs and generate plots.

Workflow:
1. scan `evaluations/validation_epochs/epoch_*`
2. compute or refresh metrics for every generated epoch
3. update per-run summaries and best checkpoint pointers
4. export aggregate reports and plots
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from sci_color_lab.metrics import compute_saved_epoch_metrics, count_trainable_params_m, should_compute_fid
from sci_color_lab.plotting import export_training_curves
from sci_color_lab.reporting import (
    build_group_summary,
    collect_epoch_metric_logs,
    collect_train_step_logs,
    discover_run_dirs,
    export_experiment_reports,
    load_run_table,
)
from sci_color_lab.trainer import load_json, load_jsonl, now_iso, save_json, save_jsonl


EPOCH_METRIC_FIELDS = [
    "fid",
    "precision",
    "recall",
    "f_score",
    "pr_curve_auc",
    "lpips",
    "lpips_mean",
    "lpips_std",
    "lpips_source",
    "lpips_internal",
    "ssim",
    "ssim_mean",
    "ssim_std",
    "ssim_source",
    "ssim_internal",
    "edge_consistency",
    "color_bleeding_rate",
    "histogram_correlation",
    "inference_time_ms",
    "subgroup_metrics",
    "eval_dir",
    "generated_dir",
    "target_dir",
    "lineart_dir",
    "archive_dir",
    "helper_metrics_available",
    "helper_metric_reports",
    "helper_metric_errors",
    "pr_curve_csv",
    "pr_curve_plot",
    "pr_metrics_error",
    "fid_computed",
    "fid_source",
    "fid_internal",
    "params_m",
    "flops_g",
    "validation_source",
    "validation_note",
    "generation_records_path",
    "generated_samples_count",
    "checkpoint_dir",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 3: post-hoc metric computation and plotting")
    parser.add_argument("--output-root", type=str, required=True, help="Output root, for example outputs/")
    parser.add_argument("--groups", nargs="+", default=None, help="Optional group filter")
    parser.add_argument(
        "--epochs",
        type=str,
        default="all",
        help="Epoch filter like 2,4,6 or 'all'",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Metric device")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if metrics.json already exists")
    parser.add_argument("--no-excel", action="store_true", help="Skip Excel workbook export")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Plot directory under output_root/analysis")
    return parser.parse_args()


def parse_epochs(text: str) -> list[int] | None:
    normalized = str(text).strip().lower()
    if not normalized or normalized in {"all", "*", "auto"}:
        return None
    return sorted({int(item.strip()) for item in str(text).split(",") if item.strip()})


def discover_target_run_dirs(output_root: Path, groups_filter: list[str] | None) -> list[Path]:
    run_dirs = discover_run_dirs(output_root)
    if not groups_filter:
        return run_dirs
    allowed = {item.strip().upper() for item in groups_filter}
    filtered: list[Path] = []
    for run_dir in run_dirs:
        metadata = load_json(run_dir / "run_metadata.json")
        group_id = str(metadata.get("group", {}).get("group_id") or run_dir.parent.name).upper()
        if group_id in allowed:
            filtered.append(run_dir)
    return filtered


def discover_evaluation_dirs(run_dir: Path, epochs_filter: list[int] | None) -> list[tuple[int, Path]]:
    validation_root = run_dir / "evaluations" / "validation_epochs"
    if not validation_root.exists():
        return []
    allowed = set(epochs_filter or [])
    items: list[tuple[int, Path]] = []
    for eval_dir in sorted(validation_root.glob("epoch_*")):
        if not eval_dir.is_dir():
            continue
        try:
            epoch = int(eval_dir.name.split("_")[-1])
        except Exception:
            continue
        if allowed and epoch not in allowed:
            continue
        if not list((eval_dir / "generated").glob("*.png")):
            continue
        items.append((epoch, eval_dir))
    return items


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

    for name in ("best_fid", "latest"):
        candidate = checkpoints_dir / name
        metadata = load_json(candidate / "checkpoint.json")
        try:
            checkpoint_epoch = int(metadata.get("epoch", 0) or 0)
        except Exception:
            checkpoint_epoch = 0
        if candidate.exists() and checkpoint_epoch == int(epoch):
            return candidate

    return None


def metric_payload_ready(metrics_payload: dict[str, Any]) -> bool:
    return bool(metrics_payload) and (
        metrics_payload.get("ssim_mean") is not None
        or metrics_payload.get("ssim") is not None
        or metrics_payload.get("lpips_mean") is not None
        or metrics_payload.get("fid") is not None
    )


def build_epoch_metric_update(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    update = {key: metrics_payload.get(key) for key in EPOCH_METRIC_FIELDS}
    update["eval_gpu_memory_peak_gb"] = metrics_payload.get("gpu_memory_peak_gb")
    update["eval_gpu_memory_reserved_peak_gb"] = metrics_payload.get("gpu_memory_reserved_peak_gb")
    update["eval_cpu_memory_peak_gb"] = metrics_payload.get("cpu_memory_peak_gb")
    update["generation_metrics_deferred"] = False
    update["posthoc_metrics_completed"] = True
    update["metrics_computed_at"] = now_iso()
    return update


def copy_named_checkpoint(run_dir: Path, source_checkpoint_dir: str | Path, name: str, epoch: int) -> str:
    source_path = Path(source_checkpoint_dir)
    target_path = run_dir / "checkpoints" / name
    if not source_path.exists():
        resolved = resolve_checkpoint_dir(run_dir, epoch)
        if resolved is None or not resolved.exists():
            return ""
        source_path = resolved
    else:
        try:
            source_resolved = source_path.resolve()
            target_resolved = target_path.resolve()
        except Exception:
            source_resolved = source_path
            target_resolved = target_path
        if source_resolved == target_resolved:
            resolved = resolve_checkpoint_dir(run_dir, epoch)
            if resolved is None or not resolved.exists():
                return str(target_path.resolve()) if target_path.exists() else ""
            try:
                if resolved.resolve() == target_resolved:
                    return str(target_path.resolve()) if target_path.exists() else ""
            except Exception:
                pass
            source_path = resolved
    if target_path.exists():
        shutil.rmtree(target_path)
    shutil.copytree(source_path, target_path)
    source_metadata = load_json(source_path / "checkpoint.json")
    save_json(
        target_path / "checkpoint.json",
        {
            "name": name,
            "epoch": int(epoch),
            "step": source_metadata.get("step"),
            "saved_at": now_iso(),
            "source_checkpoint_dir": str(source_path.resolve()),
        },
    )
    return str(target_path.resolve())


def refresh_epoch_logs(
    run_dir: Path,
    *,
    group_id: str,
    seed: int,
    epoch_payloads: dict[int, dict[str, Any]],
    best_fid: float | None,
    best_fid_epoch: int,
    best_fid_checkpoint_path: str,
    best_fid_preview_path: str,
    best_fid_eval_dir: str,
    best_fid_archive_dir: str,
    best_fid_learning_rate: float | None,
) -> list[dict[str, Any]]:
    history_path = run_dir / "logs" / "epoch_history.json"
    metric_log_path = run_dir / "logs" / "metrics.jsonl"

    history_rows = load_json(history_path).get("epochs", [])
    history_by_epoch: dict[int, dict[str, Any]] = {}
    for row in history_rows:
        try:
            epoch = int(row.get("epoch", 0) or 0)
        except Exception:
            epoch = 0
        if epoch <= 0:
            continue
        history_by_epoch[epoch] = dict(row)

    for epoch, metrics_payload in epoch_payloads.items():
        base_row = history_by_epoch.get(
            epoch,
            {
                "timestamp": now_iso(),
                "event": "epoch_end",
                "group_id": group_id,
                "seed": int(seed),
                "epoch": int(epoch),
            },
        )
        base_row.update(build_epoch_metric_update(metrics_payload))
        base_row["event"] = "epoch_end"
        base_row["group_id"] = group_id
        base_row["seed"] = int(seed)
        base_row["epoch"] = int(epoch)
        base_row["best_fid"] = best_fid
        base_row["best_fid_epoch"] = int(best_fid_epoch)
        base_row["best_fid_checkpoint_path"] = best_fid_checkpoint_path
        base_row["best_fid_preview_path"] = best_fid_preview_path
        base_row["best_fid_eval_dir"] = best_fid_eval_dir
        base_row["best_fid_archive_dir"] = best_fid_archive_dir
        base_row["best_fid_learning_rate"] = best_fid_learning_rate
        history_by_epoch[epoch] = base_row

    ordered_rows = [history_by_epoch[epoch] for epoch in sorted(history_by_epoch)]
    save_json(history_path, {"epochs": ordered_rows})

    preserved_rows = [row for row in load_jsonl(metric_log_path) if str(row.get("event", "")) != "epoch_end"]
    save_jsonl(metric_log_path, preserved_rows + ordered_rows)
    return ordered_rows


def update_run_artifacts(
    run_dir: Path,
    *,
    group_id: str,
    seed: int,
    epoch_payloads: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    status_path = run_dir / "train_status.json"
    summary = load_json(summary_path)
    status = load_json(status_path)

    best_epoch = 0
    best_payload: dict[str, Any] = {}
    best_fid: float | None = None
    for epoch, payload in sorted(epoch_payloads.items()):
        fid_value = payload.get("fid")
        if fid_value is None:
            continue
        try:
            fid = float(fid_value)
        except Exception:
            continue
        if best_fid is None or fid < best_fid:
            best_fid = fid
            best_epoch = int(epoch)
            best_payload = payload

    latest_epoch = max(epoch_payloads) if epoch_payloads else 0
    latest_payload = epoch_payloads.get(latest_epoch, {})

    best_checkpoint_path = str(summary.get("best_fid_checkpoint_path", ""))
    best_preview_path = str(summary.get("best_fid_preview_path", ""))
    best_eval_dir = str(summary.get("best_fid_eval_dir", ""))
    best_archive_dir = str(summary.get("best_fid_archive_dir", ""))
    best_learning_rate = summary.get("best_fid_learning_rate")

    history_rows = load_json(run_dir / "logs" / "epoch_history.json").get("epochs", [])
    learning_rate_by_epoch: dict[int, float] = {}
    for row in history_rows:
        try:
            epoch = int(row.get("epoch", 0) or 0)
        except Exception:
            epoch = 0
        if epoch <= 0:
            continue
        learning_rate = row.get("learning_rate")
        if learning_rate is not None:
            try:
                learning_rate_by_epoch[epoch] = float(learning_rate)
            except Exception:
                pass

    if best_epoch > 0 and best_payload:
        source_checkpoint_dir = str(best_payload.get("checkpoint_dir") or "")
        if not source_checkpoint_dir:
            resolved = resolve_checkpoint_dir(run_dir, best_epoch)
            source_checkpoint_dir = str(resolved.resolve()) if resolved is not None else ""
        if source_checkpoint_dir:
            best_checkpoint_path = copy_named_checkpoint(run_dir, source_checkpoint_dir, "best_fid", best_epoch)

        generated_dir = Path(best_payload.get("generated_dir", ""))
        preview_candidates = sorted(generated_dir.glob("*.png")) if generated_dir.exists() else []
        best_preview_path = str(preview_candidates[0].resolve()) if preview_candidates else best_preview_path
        best_eval_dir = str(Path(best_payload.get("eval_dir", "")).resolve()) if best_payload.get("eval_dir") else best_eval_dir
        best_archive_dir = str(best_payload.get("archive_dir", "")) or best_archive_dir
        best_learning_rate = learning_rate_by_epoch.get(best_epoch, best_learning_rate)

    ordered_history = refresh_epoch_logs(
        run_dir,
        group_id=group_id,
        seed=seed,
        epoch_payloads=epoch_payloads,
        best_fid=best_fid,
        best_fid_epoch=best_epoch,
        best_fid_checkpoint_path=best_checkpoint_path,
        best_fid_preview_path=best_preview_path,
        best_fid_eval_dir=best_eval_dir,
        best_fid_archive_dir=best_archive_dir,
        best_fid_learning_rate=best_learning_rate,
    )

    latest_preview_path = best_preview_path
    if not latest_preview_path and latest_payload.get("generated_dir"):
        latest_generated_dir = Path(str(latest_payload["generated_dir"]))
        preview_candidates = sorted(latest_generated_dir.glob("*.png")) if latest_generated_dir.exists() else []
        if preview_candidates:
            latest_preview_path = str(preview_candidates[0].resolve())

    summary.update(
        {
            "group_id": group_id,
            "seed": int(seed),
            "best_fid": best_fid,
            "best_fid_epoch": int(best_epoch),
            "best_fid_checkpoint_path": best_checkpoint_path,
            "best_fid_preview_path": best_preview_path,
            "best_fid_eval_dir": best_eval_dir,
            "best_fid_archive_dir": best_archive_dir,
            "best_fid_learning_rate": best_learning_rate,
            "latest_preview_path": latest_preview_path,
            "latest_eval_gpu_memory_peak_gb": latest_payload.get("gpu_memory_peak_gb"),
            "latest_eval_gpu_memory_reserved_peak_gb": latest_payload.get("gpu_memory_reserved_peak_gb"),
            "latest_eval_cpu_memory_peak_gb": latest_payload.get("cpu_memory_peak_gb"),
            "latest_eval_archive_dir": latest_payload.get("archive_dir", summary.get("latest_eval_archive_dir", "")),
            "validation_source": latest_payload.get("validation_source", summary.get("validation_source", "")),
            "validation_note": latest_payload.get("validation_note", summary.get("validation_note", "")),
            "updated_at": now_iso(),
        }
    )
    save_json(summary_path, summary)

    status.update(
        {
            "group_id": group_id,
            "seed": int(seed),
            "updated_at": now_iso(),
            "best_fid": best_fid,
            "best_fid_epoch": int(best_epoch),
            "best_fid_checkpoint_path": best_checkpoint_path,
            "best_fid_preview_path": best_preview_path,
            "best_fid_eval_dir": best_eval_dir,
            "best_fid_archive_dir": best_archive_dir,
            "best_fid_learning_rate": best_learning_rate,
            "latest_preview_path": latest_preview_path,
            "latest_eval_archive_dir": latest_payload.get("archive_dir", status.get("latest_eval_archive_dir", "")),
        }
    )
    save_json(status_path, status)

    return {
        "best_fid": best_fid,
        "best_fid_epoch": int(best_epoch),
        "best_fid_checkpoint_path": best_checkpoint_path,
        "history_rows": ordered_history,
    }


def compute_metrics_for_run(
    run_dir: Path,
    *,
    device: str,
    epochs_filter: list[int] | None,
    force: bool,
) -> dict[str, Any]:
    metadata = load_json(run_dir / "run_metadata.json")
    if not metadata:
        return {"run_dir": str(run_dir), "computed": 0, "skipped": 0, "failures": [{"error": "missing run_metadata.json"}]}

    trainer_cfg = metadata.get("trainer_config", {})
    total_epochs = int(trainer_cfg.get("epochs", 0) or 0)
    helper_tools_root = str(trainer_cfg.get("helper_tools_root", ""))
    archive_root = str(trainer_cfg.get("inference_archive_root", ""))
    group_id = str(metadata.get("group", {}).get("group_id") or run_dir.parent.name)

    summary = load_json(run_dir / "run_summary.json")
    seed = int(summary.get("seed") or int(str(run_dir.name).split("_")[-1]))
    params_m = summary.get("params_m")
    if params_m is None:
        params_m = count_trainable_params_m(run_dir)
    flops_g = summary.get("flops_g")
    validation_selection = load_json(run_dir / "validation_selection.json")
    validation_source = str(validation_selection.get("source", summary.get("validation_source", "")))
    validation_note = str(validation_selection.get("note", summary.get("validation_note", "")))

    epoch_payloads: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    computed = 0
    skipped = 0

    for epoch, eval_dir in discover_evaluation_dirs(run_dir, epochs_filter):
        existing_metrics = load_json(eval_dir / "metrics.json")
        try:
            if force or not metric_payload_ready(existing_metrics):
                payload = compute_saved_epoch_metrics(
                    eval_dir=eval_dir,
                    device=device,
                    compute_fid=should_compute_fid(epoch, total_epochs),
                    compute_pr_curve=epoch == total_epochs,
                    helper_tools_root=helper_tools_root,
                    archive_root=archive_root,
                    group_id=group_id,
                    seed=seed,
                    epoch=epoch,
                    params_m=params_m,
                    flops_g=flops_g,
                    validation_source=validation_source,
                    validation_note=validation_note,
                )
                computed += 1
            else:
                payload = existing_metrics
                skipped += 1
        except Exception as exc:
            failures.append({"epoch": int(epoch), "eval_dir": str(eval_dir), "error": str(exc)})
            continue

        payload["group_id"] = payload.get("group_id") or group_id
        payload["seed"] = int(payload.get("seed") or seed)
        payload["epoch"] = int(payload.get("epoch") or epoch)
        payload["checkpoint_dir"] = payload.get("checkpoint_dir") or (
            str(resolve_checkpoint_dir(run_dir, epoch).resolve()) if resolve_checkpoint_dir(run_dir, epoch) is not None else ""
        )
        payload["split"] = payload.get("split") or "validation_epoch"
        epoch_payloads[int(epoch)] = payload

    update_result = update_run_artifacts(run_dir, group_id=group_id, seed=seed, epoch_payloads=epoch_payloads) if epoch_payloads else {}
    curve_paths = export_training_curves(run_dir)

    return {
        "run_dir": str(run_dir.resolve()),
        "group_id": group_id,
        "seed": int(seed),
        "computed": computed,
        "skipped": skipped,
        "failures": failures,
        "epochs": sorted(epoch_payloads),
        "best_fid": update_result.get("best_fid"),
        "best_fid_epoch": update_result.get("best_fid_epoch"),
        "loss_curve_path": curve_paths.get("loss_curve_path"),
        "lr_curve_path": curve_paths.get("lr_curve_path"),
    }


DISABLED_METRIC_COMPARISON_PLOTS = [
    "fid_mean_comparison.png",
    "lpips_mean_comparison.png",
    "precision_mean_comparison.png",
    "recall_mean_comparison.png",
    "ssim_mean_comparison.png",
]

DISABLED_DIAGNOSTIC_PLOT_GLOBS = [
    "*_train_gpu_memory_by_epoch.png",
]


def remove_disabled_metric_comparison_plots(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name in DISABLED_METRIC_COMPARISON_PLOTS:
        path = output_dir / file_name
        if path.exists():
            path.unlink()


def remove_disabled_diagnostic_plots(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in DISABLED_DIAGNOSTIC_PLOT_GLOBS:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def plot_training_curves_for_runs(output_root: Path, groups_filter: list[str] | None = None) -> dict[str, str]:
    paths: dict[str, str] = {}
    run_table = load_run_table(output_root)
    if groups_filter:
        allowed = {item.strip().upper() for item in groups_filter}
        run_table = run_table[run_table["group_id"].astype(str).str.upper().isin(allowed)]

    for _, row in run_table.iterrows():
        run_dir = Path(str(row["run_dir"]))
        plot_paths = export_training_curves(run_dir)
        for key, value in plot_paths.items():
            if value:
                paths[f"{row['group_id']}_{row['seed']}_{key}"] = value
    return paths


def plot_epoch_metric_trends(epoch_metrics: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if epoch_metrics.empty:
        return paths

    frame = epoch_metrics.copy()
    if "event" in frame.columns:
        frame = frame[frame["event"].astype(str) == "epoch_end"].copy()
    if frame.empty:
        return paths

    metric_specs = [
        ("fid", ("fid",), "FID", True),
        ("lpips", ("lpips_mean", "lpips"), "LPIPS", True),
        ("ssim", ("ssim_mean", "ssim"), "SSIM", False),
    ]
    palette = ["#355070", "#6d597a", "#b56576", "#2a9d8f", "#e76f51", "#1d3557"]

    for group_id, group_frame in frame.groupby("group_id", sort=True):
        group_slug = str(group_id).strip().lower().replace(" ", "_")
        for metric_key, column_candidates, label, lower_is_better in metric_specs:
            value_column = next((item for item in column_candidates if item in group_frame.columns), "")
            if not value_column:
                continue

            data = group_frame[["seed", "epoch", value_column]].copy()
            data["seed"] = pd.to_numeric(data["seed"], errors="coerce")
            data["epoch"] = pd.to_numeric(data["epoch"], errors="coerce")
            data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
            data = data.dropna(subset=["seed", "epoch", value_column]).sort_values(["seed", "epoch"])
            if data.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
            for index, (seed, seed_frame) in enumerate(data.groupby("seed", sort=True)):
                seed_frame = seed_frame.sort_values("epoch")
                ax.plot(
                    seed_frame["epoch"],
                    seed_frame[value_column],
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    color=palette[index % len(palette)],
                    label=f"seed {int(seed)}",
                )

            mean_frame = data.groupby("epoch", as_index=False)[value_column].mean().sort_values("epoch")
            ax.plot(
                mean_frame["epoch"],
                mean_frame[value_column],
                linestyle="--",
                linewidth=2.4,
                color="#1b4332",
                label="seed mean",
            )

            best_index = mean_frame[value_column].idxmin() if lower_is_better else mean_frame[value_column].idxmax()
            best_row = mean_frame.loc[best_index]
            ax.scatter(best_row["epoch"], best_row[value_column], s=40, color="#111111", zorder=5)
            ax.annotate(
                f"{best_row[value_column]:.3f} @ epoch {int(best_row['epoch'])}",
                xy=(best_row["epoch"], best_row[value_column]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                color="#111111",
            )

            epoch_ticks = sorted({int(epoch) for epoch in data["epoch"].tolist()})
            ax.set_xticks(epoch_ticks)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.set_title(f"{group_id} {label} by Epoch")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()

            path = output_dir / f"{group_slug}_{metric_key}_by_epoch.png"
            fig.savefig(path)
            plt.close(fig)
            paths[f"{group_id}_{metric_key}_by_epoch"] = str(path.resolve())

    return paths


def _slugify_group(group_id: Any) -> str:
    return str(group_id).strip().lower().replace(" ", "_")


def plot_loss_curves(train_steps: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if train_steps.empty:
        return paths

    frame = train_steps.copy()
    if "event" in frame.columns:
        frame = frame[frame["event"].astype(str) == "train"].copy()
    if frame.empty or "loss" not in frame.columns:
        return paths

    palette = ["#355070", "#6d597a", "#b56576", "#2a9d8f", "#e76f51", "#1d3557"]
    for group_id, group_frame in frame.groupby("group_id", sort=True):
        data = group_frame[["seed", "global_step", "loss"]].copy()
        data["seed"] = pd.to_numeric(data["seed"], errors="coerce")
        data["global_step"] = pd.to_numeric(data["global_step"], errors="coerce")
        data["loss"] = pd.to_numeric(data["loss"], errors="coerce")
        data = data.dropna(subset=["seed", "global_step", "loss"]).sort_values(["seed", "global_step"])
        if data.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
        for index, (seed, seed_frame) in enumerate(data.groupby("seed", sort=True)):
            seed_frame = seed_frame.sort_values("global_step").copy()
            color = palette[index % len(palette)]
            pre_window = max(11, min(31, len(seed_frame) // 18 if len(seed_frame) >= 90 else 11))
            span = max(35, min(90, len(seed_frame) // 6 if len(seed_frame) >= 120 else 35))
            seed_frame["loss_smooth"] = (
                seed_frame["loss"]
                .rolling(window=pre_window, min_periods=1, center=True)
                .mean()
                .ewm(span=span, adjust=False)
                .mean()
            )
            ax.plot(
                seed_frame["global_step"],
                seed_frame["loss_smooth"],
                color=color,
                linewidth=0.9,
                solid_capstyle="round",
                label=f"seed {int(seed)}",
            )

        mean_frame = data.groupby("global_step", as_index=False)["loss"].mean().sort_values("global_step")
        mean_pre_window = max(15, min(41, len(mean_frame) // 16 if len(mean_frame) >= 90 else 15))
        mean_span = max(45, min(110, len(mean_frame) // 5 if len(mean_frame) >= 120 else 45))
        mean_frame["loss_smooth"] = (
            mean_frame["loss"]
            .rolling(window=mean_pre_window, min_periods=1, center=True)
            .mean()
            .ewm(span=mean_span, adjust=False)
            .mean()
        )
        ax.plot(
            mean_frame["global_step"],
            mean_frame["loss_smooth"],
            color="#1b4332",
            linestyle="--",
            linewidth=1.1,
            solid_capstyle="round",
            label="seed mean",
        )
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{group_id} Smoothed Training Loss Curve")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        path = output_dir / f"{_slugify_group(group_id)}_training_loss_by_step.png"
        fig.savefig(path)
        plt.close(fig)
        paths[f"{group_id}_training_loss_by_step"] = str(path.resolve())

    return paths


def plot_epoch_time_trends(epoch_metrics: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if epoch_metrics.empty or "epoch_time_seconds" not in epoch_metrics.columns:
        return paths

    frame = epoch_metrics.copy()
    if "event" in frame.columns:
        frame = frame[frame["event"].astype(str) == "epoch_end"].copy()
    if frame.empty:
        return paths

    palette = ["#355070", "#6d597a", "#b56576", "#2a9d8f", "#e76f51", "#1d3557"]
    for group_id, group_frame in frame.groupby("group_id", sort=True):
        data = group_frame[["seed", "epoch", "epoch_time_seconds"]].copy()
        data["seed"] = pd.to_numeric(data["seed"], errors="coerce")
        data["epoch"] = pd.to_numeric(data["epoch"], errors="coerce")
        data["epoch_time_seconds"] = pd.to_numeric(data["epoch_time_seconds"], errors="coerce")
        data = data.dropna(subset=["seed", "epoch", "epoch_time_seconds"]).sort_values(["seed", "epoch"])
        if data.empty:
            continue
        data["epoch_time_minutes"] = data["epoch_time_seconds"] / 60.0

        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        epochs = sorted({int(epoch) for epoch in data["epoch"].tolist()})
        seeds = sorted({int(seed) for seed in data["seed"].tolist()})
        x_base = list(range(len(epochs)))
        width = 0.8 / max(len(seeds) + 1, 1)

        for index, seed in enumerate(seeds):
            seed_frame = data[data["seed"] == seed].sort_values("epoch")
            seed_by_epoch = (
                seed_frame.set_index(seed_frame["epoch"].astype(int))["epoch_time_minutes"].reindex(epochs)
            )
            positions = [x + (index - (len(seeds) - 1) / 2.0) * width for x in x_base]
            ax.bar(
                positions,
                seed_by_epoch.tolist(),
                width=width * 0.92,
                color=palette[index % len(palette)],
                alpha=0.88,
                label=f"seed {seed}",
            )

        mean_by_epoch = data.groupby("epoch", as_index=False)["epoch_time_minutes"].mean().sort_values("epoch")
        mean_series = mean_by_epoch.set_index(mean_by_epoch["epoch"].astype(int))["epoch_time_minutes"].reindex(epochs)
        mean_positions = [x + (len(seeds) - (len(seeds) - 1) / 2.0) * width for x in x_base]
        ax.bar(
            mean_positions,
            mean_series.tolist(),
            width=width * 0.92,
            color="#1b4332",
            alpha=0.88,
            label="seed mean",
        )

        ax.set_xticks(x_base)
        ax.set_xticklabels([str(epoch) for epoch in epochs])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Epoch Time (minutes)")
        ax.set_title(f"{group_id} Epoch Time Comparison")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(loc="best")
        fig.tight_layout()

        path = output_dir / f"{_slugify_group(group_id)}_epoch_time_by_epoch.png"
        fig.savefig(path)
        plt.close(fig)
        paths[f"{group_id}_epoch_time_by_epoch"] = str(path.resolve())

    return paths


def plot_gpu_memory_trends(epoch_metrics: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    required_columns = {"gpu_memory_peak_gb", "gpu_memory_reserved_peak_gb"}
    if epoch_metrics.empty or not required_columns.intersection(set(epoch_metrics.columns)):
        return paths

    frame = epoch_metrics.copy()
    if "event" in frame.columns:
        frame = frame[frame["event"].astype(str) == "epoch_end"].copy()
    if frame.empty:
        return paths

    palette = ["#355070", "#6d597a", "#b56576", "#2a9d8f", "#e76f51", "#1d3557"]
    for group_id, group_frame in frame.groupby("group_id", sort=True):
        columns = [column for column in ["seed", "epoch", "gpu_memory_peak_gb", "gpu_memory_reserved_peak_gb"] if column in group_frame.columns]
        data = group_frame[columns].copy()
        data["seed"] = pd.to_numeric(data["seed"], errors="coerce")
        data["epoch"] = pd.to_numeric(data["epoch"], errors="coerce")
        for column in ["gpu_memory_peak_gb", "gpu_memory_reserved_peak_gb"]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        memory_columns = [column for column in ["gpu_memory_peak_gb", "gpu_memory_reserved_peak_gb"] if column in data.columns]
        data = data.dropna(subset=["seed", "epoch"] + memory_columns).sort_values(["seed", "epoch"])
        if data.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160, sharex=True)
        plot_specs = [
            ("gpu_memory_peak_gb", "Allocated Peak (GB)"),
            ("gpu_memory_reserved_peak_gb", "Reserved Peak (GB)"),
        ]
        for axis, (column, ylabel) in zip(axes, plot_specs):
            if column not in data.columns:
                axis.set_visible(False)
                continue
            for index, (seed, seed_frame) in enumerate(data.groupby("seed", sort=True)):
                seed_frame = seed_frame.sort_values("epoch")
                axis.plot(
                    seed_frame["epoch"],
                    seed_frame[column],
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    color=palette[index % len(palette)],
                    label=f"seed {int(seed)}",
                )
            mean_frame = data.groupby("epoch", as_index=False)[column].mean().sort_values("epoch")
            axis.plot(
                mean_frame["epoch"],
                mean_frame[column],
                linestyle="--",
                linewidth=2.4,
                color="#1b4332",
                label="seed mean",
            )
            axis.set_xticks(sorted({int(epoch) for epoch in data["epoch"].tolist()}))
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best")

        fig.suptitle(f"{group_id} Training GPU Memory by Epoch")
        fig.tight_layout()
        path = output_dir / f"{_slugify_group(group_id)}_train_gpu_memory_by_epoch.png"
        fig.savefig(path)
        plt.close(fig)
        paths[f"{group_id}_train_gpu_memory_by_epoch"] = str(path.resolve())

    return paths


def plot_ablation_summary(
    group_summary: pd.DataFrame,
    single_contrib: dict[str, float],
    pair_interactions: dict[str, float],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if single_contrib:
        items = sorted(single_contrib.items(), key=lambda item: abs(item[1]), reverse=True)
        labels = [item[0] for item in items]
        values = [item[1] for item in items]
        colors = ["#2d6a4f" if value > 0 else "#c44536" for value in values]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        ax.barh(labels, values, color=colors)
        ax.set_xlabel("Contribution on FID")
        ax.set_title("Single Module Contributions")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.25, axis="x")
        fig.tight_layout()
        path = output_dir / "single_module_contributions.png"
        fig.savefig(path)
        plt.close(fig)
        paths["single_contrib"] = str(path.resolve())

    if pair_interactions:
        items = sorted(pair_interactions.items(), key=lambda item: abs(item[1]), reverse=True)[:20]
        labels = [item[0] for item in items]
        values = [item[1] for item in items]
        colors = ["#2d6a4f" if value > 0 else "#c44536" for value in values]
        fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
        ax.barh(labels, values, color=colors)
        ax.set_xlabel("Interaction Effect on FID")
        ax.set_title("Top 20 Interaction Effects")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.25, axis="x")
        fig.tight_layout()
        path = output_dir / "interaction_effects.png"
        fig.savefig(path)
        plt.close(fig)
        paths["interactions"] = str(path.resolve())

    return paths


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.exists():
        raise SystemExit(f"Output root not found: {output_root}")

    epochs_filter = parse_epochs(args.epochs)
    run_dirs = discover_target_run_dirs(output_root, args.groups)
    print(f"[analyze_results] output_root={output_root}")
    print(f"[analyze_results] run_count={len(run_dirs)}")

    run_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        result = compute_metrics_for_run(
            run_dir,
            device=args.device,
            epochs_filter=epochs_filter,
            force=args.force,
        )
        run_results.append(result)
        for failure in result.get("failures", []):
            failure_payload = dict(failure)
            failure_payload["run_dir"] = result["run_dir"]
            failures.append(failure_payload)

    report_paths = export_experiment_reports(output_root, write_excel=not args.no_excel)
    training_plot_paths = plot_training_curves_for_runs(output_root, args.groups)

    run_table = load_run_table(output_root)
    if args.groups:
        allowed = {item.strip().upper() for item in args.groups}
        run_table = run_table[run_table["group_id"].astype(str).str.upper().isin(allowed)]

    metric_plot_paths: dict[str, str] = {}
    epoch_metric_plot_paths: dict[str, str] = {}
    diagnostic_plot_paths: dict[str, str] = {}
    ablation_plot_paths: dict[str, str] = {}
    plots_dir = output_root / "analysis" / args.plot_dir
    remove_disabled_metric_comparison_plots(plots_dir)
    remove_disabled_diagnostic_plots(plots_dir)
    if not run_table.empty:
        group_summary, single_contrib, pair_interactions = build_group_summary(run_table)
        epoch_metric_frame = collect_epoch_metric_logs(output_root)
        train_step_frame = collect_train_step_logs(output_root)
        if args.groups:
            allowed = {item.strip().upper() for item in args.groups}
            epoch_metric_frame = epoch_metric_frame[
                epoch_metric_frame["group_id"].astype(str).str.upper().isin(allowed)
            ]
            train_step_frame = train_step_frame[
                train_step_frame["group_id"].astype(str).str.upper().isin(allowed)
            ]
        epoch_metric_plot_paths = plot_epoch_metric_trends(epoch_metric_frame, plots_dir)
        diagnostic_plot_paths.update(plot_loss_curves(train_step_frame, plots_dir))
        diagnostic_plot_paths.update(plot_epoch_time_trends(epoch_metric_frame, plots_dir))
        ablation_plot_paths = plot_ablation_summary(group_summary, single_contrib, pair_interactions, plots_dir)

    execution_summary = {
        "mode": "analyze_results",
        "output_root": str(output_root.resolve()),
        "groups": args.groups or [],
        "epochs": args.epochs,
        "device": args.device,
        "force": bool(args.force),
        "run_results": run_results,
        "failures": failures,
        "report_files": report_paths,
        "training_plots": training_plot_paths,
        "metric_plots": metric_plot_paths,
        "epoch_metric_plots": epoch_metric_plot_paths,
        "diagnostic_plots": diagnostic_plot_paths,
        "ablation_plots": ablation_plot_paths,
    }

    execution_summary_path = output_root / "analysis" / "analysis_execution_summary.json"
    execution_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with execution_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(execution_summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(execution_summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
