from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.utils import get_column_letter
from scipy.stats import ttest_rel

from .ablation import build_ablation_summary
from .localized_outputs import sync_analysis_localized_outputs


PRIMARY_METRIC_COLUMNS = [
    "fid",
    "precision",
    "recall",
    "f_score",
    "pr_curve_auc",
    "lpips_mean",
    "ssim_mean",
    "edge_consistency_mean",
    "color_bleeding_rate_mean",
    "histogram_correlation_mean",
    "inference_time_ms_mean",
    "params_m",
    "flops_g",
    "train_gpu_memory_peak_gb",
    "train_gpu_memory_reserved_peak_gb",
    "train_cpu_memory_peak_gb",
    "latest_epoch_time_seconds",
    "seed_elapsed_seconds",
    "eval_gpu_memory_peak_gb",
    "eval_gpu_memory_reserved_peak_gb",
    "eval_cpu_memory_peak_gb",
]


def _infer_seed_from_run_dir(run_dir: Path) -> int | None:
    name = run_dir.name
    if not name.startswith("seed_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return None


def discover_run_dirs(output_root: str | Path) -> list[Path]:
    root = Path(output_root)
    if not root.exists():
        return []
    run_dirs = [path.parent for path in root.rglob("run_metadata.json")]
    return sorted(set(run_dirs), key=lambda item: item.stat().st_mtime, reverse=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (JSONDecodeError, OSError):
        return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        return []
    return rows


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _normalize_excel_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _prepare_frame_for_excel(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    prepared = frame.copy()
    for column in prepared.columns:
        prepared[column] = prepared[column].map(_normalize_excel_value)
    return prepared


def _write_excel_sheet(writer: pd.ExcelWriter, sheet_name: str, frame: pd.DataFrame) -> None:
    export_frame = _prepare_frame_for_excel(frame)
    export_frame.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    sampled = export_frame.head(200)
    for column_index, column_name in enumerate(export_frame.columns, start=1):
        width = len(str(column_name)) + 2
        if not sampled.empty:
            sample_lengths = sampled[column_name].map(lambda value: len(str(value)) if value is not None else 0)
            width = max(width, int(sample_lengths.max()) + 2 if not sample_lengths.empty else width)
        worksheet.column_dimensions[get_column_letter(column_index)].width = min(max(width, 10), 60)


def _run_base_info(run_dir: Path) -> dict[str, Any]:
    summary = _load_json(run_dir / "run_summary.json")
    metadata = _load_json(run_dir / "run_metadata.json")
    return {
        "run_dir": str(run_dir.resolve()),
        "group_id": summary.get("group_id", metadata.get("group", {}).get("group_id", "")),
        "group_name": summary.get("group_name", metadata.get("group", {}).get("display_name", "")),
        "seed": summary.get("seed", _infer_seed_from_run_dir(run_dir)),
        "status": summary.get("status", "unknown"),
    }


def collect_train_step_logs(output_root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs(output_root):
        base = _run_base_info(run_dir)
        log_path = run_dir / "logs" / "train.jsonl"
        for payload in _load_jsonl(log_path):
            row = dict(base)
            row["source_file"] = str(log_path.resolve())
            row.update(payload)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sort_columns = [column for column in ["group_id", "seed", "epoch", "global_step", "timestamp"] if column in frame.columns]
    return frame.sort_values(sort_columns).reset_index(drop=True) if sort_columns else frame


def collect_epoch_metric_logs(output_root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs(output_root):
        base = _run_base_info(run_dir)
        log_path = run_dir / "logs" / "metrics.jsonl"
        for payload in _load_jsonl(log_path):
            row = dict(base)
            row["source_file"] = str(log_path.resolve())
            row.update(payload)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sort_columns = [column for column in ["group_id", "seed", "epoch", "timestamp"] if column in frame.columns]
    return frame.sort_values(sort_columns).reset_index(drop=True) if sort_columns else frame


def collect_per_sample_metrics(output_root: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in discover_run_dirs(output_root):
        base = _run_base_info(run_dir)
        for csv_path in sorted((run_dir / "evaluations").rglob("per_sample_metrics.csv")):
            frame = _load_csv(csv_path)
            if frame.empty:
                continue
            metrics = _load_json(csv_path.parent / "metrics.json")
            frame = frame.copy()
            for key, value in base.items():
                frame[key] = value
            frame["eval_dir"] = str(csv_path.parent.resolve())
            frame["per_sample_metrics_path"] = str(csv_path.resolve())
            frame["eval_split"] = metrics.get("split", "")
            frame["validation_source"] = metrics.get("validation_source", "")
            frame["validation_note"] = metrics.get("validation_note", "")
            frame["checkpoint_dir"] = metrics.get("checkpoint_dir", "")
            frame["archive_dir"] = metrics.get("archive_dir", "")
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    sort_columns = [column for column in ["group_id", "seed", "eval_split", "image_id"] if column in frame.columns]
    return frame.sort_values(sort_columns).reset_index(drop=True) if sort_columns else frame


def _build_workbook_index(
    *,
    analysis_dir: Path,
    workbook_path: Path,
    run_frame: pd.DataFrame,
    train_steps_frame: pd.DataFrame,
    epoch_metrics_frame: pd.DataFrame,
    per_sample_frame: pd.DataFrame,
    report_files: dict[str, str],
) -> pd.DataFrame:
    rows = [
        {"section": "概览", "item": "generated_at", "value": pd.Timestamp.utcnow().isoformat()},
        {"section": "概览", "item": "analysis_dir", "value": str(analysis_dir.resolve())},
        {"section": "概览", "item": "workbook_path", "value": str(workbook_path.resolve())},
        {"section": "概览", "item": "run_count", "value": len(run_frame)},
        {"section": "概览", "item": "completed_run_count", "value": int((run_frame.get("status") == "completed").sum()) if not run_frame.empty and "status" in run_frame.columns else 0},
        {"section": "概览", "item": "train_step_log_rows", "value": len(train_steps_frame)},
        {"section": "概览", "item": "epoch_metric_rows", "value": len(epoch_metrics_frame)},
        {"section": "概览", "item": "per_sample_rows", "value": len(per_sample_frame)},
        {"section": "Sheet说明", "item": "单次运行汇总", "value": "每个 group + seed 一行，汇总 run 级指标、路径、最佳结果与资源占用。"},
        {"section": "Sheet说明", "item": "组级汇总", "value": "按 group 聚合的均值/标准差结果。"},
        {"section": "Sheet说明", "item": "Seed平均汇总", "value": "与组级汇总一致，保留独立 sheet 便于直接引用。"},
        {"section": "Sheet说明", "item": "最优检查点", "value": "每个 run 的 best_fid / best_val_loss checkpoint 路径。"},
        {"section": "Sheet说明", "item": "训练Step日志", "value": "来自 train.jsonl 的 step 级 loss / lr 记录。"},
        {"section": "Sheet说明", "item": "Epoch指标", "value": "来自 metrics.jsonl 的 epoch 级 loss / FID / Precision / Recall / LPIPS / SSIM / 时间 / 显存等记录。"},
        {"section": "Sheet说明", "item": "逐图评估", "value": "聚合所有 per_sample_metrics.csv 的逐图评估结果。"},
        {"section": "Sheet说明", "item": "单模块贡献", "value": "自动分析得到的单模块贡献。"},
        {"section": "Sheet说明", "item": "交互效应", "value": "自动分析得到的模块交互效应。"},
        {"section": "Sheet说明", "item": "配对T检验", "value": "使用相同 seed 的组间配对 t 检验结果。"},
    ]
    for name, path in report_files.items():
        rows.append({"section": "文件索引", "item": name, "value": path})
    return pd.DataFrame(rows)


def _preferred_metrics_file(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "evaluations").rglob("metrics.json"))
    if not candidates:
        return None
    test_candidates = [path for path in candidates if "test_" in str(path)]
    if test_candidates:
        return sorted(test_candidates)[-1]
    val_candidates = [path for path in candidates if "val_" in str(path)]
    if val_candidates:
        return sorted(val_candidates)[-1]
    return candidates[-1]


def load_run_table(output_root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs(output_root):
        summary = _load_json(run_dir / "run_summary.json")
        metadata = _load_json(run_dir / "run_metadata.json")
        metrics_path = _preferred_metrics_file(run_dir)
        metrics = _load_json(metrics_path) if metrics_path else {}
        epoch_history = _load_json(run_dir / "logs" / "epoch_history.json").get("epochs", [])
        row = {
            "run_dir": str(run_dir.resolve()),
            "group_id": summary.get("group_id", metadata.get("group", {}).get("group_id", "")),
            "group_name": summary.get("group_name", metadata.get("group", {}).get("display_name", "")),
            "seed": summary.get("seed", _infer_seed_from_run_dir(run_dir)),
            "status": summary.get("status", "unknown"),
            "best_train_loss": summary.get("best_train_loss"),
            "best_val_loss": summary.get("best_val_loss"),
            "best_fid": summary.get("best_fid"),
            "best_fid_epoch": summary.get("best_fid_epoch"),
            "best_fid_checkpoint_path": summary.get("best_fid_checkpoint_path", ""),
            "best_fid_preview_path": summary.get("best_fid_preview_path", ""),
            "best_fid_eval_dir": summary.get("best_fid_eval_dir", ""),
            "best_fid_archive_dir": summary.get("best_fid_archive_dir", ""),
            "best_fid_learning_rate": summary.get("best_fid_learning_rate"),
            "quality_monitor_metric": summary.get("quality_monitor_metric", ""),
            "consecutive_quality_decline_epochs": summary.get("consecutive_quality_decline_epochs"),
            "best_fid_lr_recovery_count": summary.get("best_fid_lr_recovery_count"),
            "best_fid_lr_recovery_active": summary.get("best_fid_lr_recovery_active"),
            "best_val_loss_checkpoint_path": summary.get("best_val_loss_checkpoint_path", ""),
            "latest_preview_path": summary.get("latest_preview_path", ""),
            "updated_at": summary.get("updated_at", metadata.get("created_at", "")),
            "environment_lock_path": summary.get("environment_lock_path", ""),
            "validation_source": summary.get("validation_source", ""),
            "preview_source": summary.get("preview_source", ""),
            "validation_note": summary.get("validation_note", ""),
            "validation_selection_path": summary.get("validation_selection_path", ""),
            "loss_curve_path": summary.get("loss_curve_path", ""),
            "lr_curve_path": summary.get("lr_curve_path", ""),
            "params_m": summary.get("params_m"),
            "flops_g": summary.get("flops_g"),
            "train_gpu_memory_peak_gb": summary.get("train_gpu_memory_peak_gb"),
            "train_gpu_memory_reserved_peak_gb": summary.get("train_gpu_memory_reserved_peak_gb"),
            "train_cpu_memory_peak_gb": summary.get("train_cpu_memory_peak_gb"),
            "latest_epoch_time_seconds": summary.get("latest_epoch_time_seconds"),
            "latest_epoch_time_hms": summary.get("latest_epoch_time_hms", ""),
            "seed_elapsed_seconds": summary.get("seed_elapsed_seconds"),
            "seed_elapsed_hms": summary.get("seed_elapsed_hms", ""),
            "latest_eval_gpu_memory_peak_gb": summary.get("latest_eval_gpu_memory_peak_gb"),
            "latest_eval_gpu_memory_reserved_peak_gb": summary.get("latest_eval_gpu_memory_reserved_peak_gb"),
            "latest_eval_cpu_memory_peak_gb": summary.get("latest_eval_cpu_memory_peak_gb"),
            "latest_eval_archive_dir": summary.get("latest_eval_archive_dir", ""),
            "selected_metrics_path": str(metrics_path.resolve()) if metrics_path else "",
            "eval_split": metrics.get("split", ""),
            "eval_gpu_memory_peak_gb": metrics.get("gpu_memory_peak_gb"),
            "eval_gpu_memory_reserved_peak_gb": metrics.get("gpu_memory_reserved_peak_gb"),
            "eval_cpu_memory_peak_gb": metrics.get("cpu_memory_peak_gb"),
            "archive_dir": metrics.get("archive_dir", ""),
        }
        row.update(metrics)
        row.update(_derive_module_specific_metrics(epoch_history, metrics))
        rows.append(row)
    return pd.DataFrame(rows)


def _derive_module_specific_metrics(epoch_history: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    derived: dict[str, Any] = {}
    if epoch_history:
        losses = [item.get("train_loss") for item in epoch_history if item.get("train_loss") is not None]
        if len(losses) > 1:
            diffs = pd.Series(losses).diff().dropna()
            derived["loss_curve_smoothness"] = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
        best_fid_epoch = metrics.get("best_fid_epoch")
        if best_fid_epoch is not None:
            derived["epochs_to_best_fid"] = best_fid_epoch

    subgroup_base = metrics.get("eval_dir") or Path(metrics.get("generated_dir", "")).parent
    subgroup_path = Path(subgroup_base) / "subgroup_metrics.json"
    subgroup = _load_json(subgroup_path) if subgroup_path.exists() else {}
    if subgroup.get("color_complexity"):
        groups = subgroup["color_complexity"]
        if len(groups) >= 2:
            values = [item.get("ssim", 0.0) for item in groups]
            derived["complexity_performance_gap"] = float(max(values) - min(values))
    if subgroup.get("region_scale"):
        groups = subgroup["region_scale"]
        if len(groups) >= 2:
            values = [item.get("ssim", 0.0) for item in groups]
            derived["scale_performance_gap"] = float(max(values) - min(values))
    return derived


def build_group_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    if frame.empty:
        return pd.DataFrame(), {}, {}
    available = [column for column in PRIMARY_METRIC_COLUMNS if column in frame.columns]
    if not available:
        grouped = frame[["group_id", "group_name", "seed", "status"]].copy()
        return grouped, {}, {}
    grouped = frame.groupby("group_id", dropna=False)[available].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join([part for part in column if part]).strip("_") for column in grouped.columns.to_flat_index()]
    seed_stats = (
        frame.groupby("group_id", dropna=False)
        .agg(
            group_name=("group_name", "first"),
            seed_count=("seed", "nunique"),
            seeds=("seed", lambda values: ",".join(str(int(value)) for value in sorted(pd.Series(values).dropna().unique()))),
        )
        .reset_index()
    )
    grouped = seed_stats.merge(grouped, on="group_id", how="left")

    metric_by_group: dict[str, float] = {}
    if "fid_mean" in grouped.columns:
        metric_by_group = {
            row["group_id"]: -float(row["fid_mean"])
            for _, row in grouped[["group_id", "fid_mean"]].dropna().iterrows()
        }
    summary = build_ablation_summary(metric_by_group)
    return grouped, summary.single_contributions, summary.pair_interactions


def build_paired_t_tests(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "fid" not in frame.columns:
        return pd.DataFrame()
    metrics = [
        column
        for column in [
            "fid",
            "precision",
            "recall",
            "f_score",
            "pr_curve_auc",
            "lpips_mean",
            "ssim_mean",
            "edge_consistency_mean",
            "color_bleeding_rate_mean",
        ]
        if column in frame.columns
    ]
    full = frame[frame["group_id"] == "E_FULL"]
    if full.empty:
        return pd.DataFrame()
    rows = []
    for group_id in sorted(item for item in frame["group_id"].dropna().unique() if item != "E_FULL"):
        group_frame = frame[frame["group_id"] == group_id]
        merged = full.merge(group_frame, on="seed", suffixes=("_full", "_other"))
        if merged.empty:
            continue
        for metric in metrics:
            left = merged[f"{metric}_full"].dropna()
            right = merged[f"{metric}_other"].dropna()
            shared = min(len(left), len(right))
            if shared < 2:
                continue
            stat = ttest_rel(left.iloc[:shared], right.iloc[:shared])
            rows.append(
                {
                    "group_id": group_id,
                    "metric": metric,
                    "t_statistic": float(stat.statistic),
                    "p_value": float(stat.pvalue),
                    "n": shared,
                }
            )
    return pd.DataFrame(rows)


def export_experiment_reports(output_root: str | Path, *, write_excel: bool = True) -> dict[str, str]:
    output_root = Path(output_root)
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_frame = load_run_table(output_root)
    group_summary, single_contributions, pair_interactions = build_group_summary(run_frame)
    t_tests = build_paired_t_tests(run_frame)
    train_steps_frame = collect_train_step_logs(output_root)
    epoch_metrics_frame = collect_epoch_metric_logs(output_root)
    per_sample_frame = collect_per_sample_metrics(output_root)

    run_csv = analysis_dir / "per_run_summary.csv"
    group_csv = analysis_dir / "group_summary.csv"
    seed_average_csv = analysis_dir / "seed_average_summary.csv"
    best_csv = analysis_dir / "best_checkpoints.csv"
    single_csv = analysis_dir / "single_module_contributions.csv"
    interaction_csv = analysis_dir / "interaction_effects.csv"
    ttest_csv = analysis_dir / "paired_t_tests.csv"
    step_log_csv = analysis_dir / "train_step_logs.csv"
    epoch_log_csv = analysis_dir / "epoch_metric_logs.csv"
    per_sample_csv = analysis_dir / "per_sample_metrics_all.csv"
    workbook_path = analysis_dir / "experiment_summary.xlsx"

    run_frame.to_csv(run_csv, index=False)
    group_summary.to_csv(group_csv, index=False)
    group_summary.to_csv(seed_average_csv, index=False)
    train_steps_frame.to_csv(step_log_csv, index=False)
    epoch_metrics_frame.to_csv(epoch_log_csv, index=False)
    per_sample_frame.to_csv(per_sample_csv, index=False)

    best_columns = [column for column in ["group_id", "seed", "best_fid", "best_fid_epoch", "best_fid_checkpoint_path", "best_val_loss", "best_val_loss_checkpoint_path"] if column in run_frame.columns]
    best_frame = run_frame[best_columns].copy() if best_columns else pd.DataFrame()
    if best_columns:
        best_frame.to_csv(best_csv, index=False)
    else:
        pd.DataFrame().to_csv(best_csv, index=False)

    single_frame = pd.DataFrame(
        [{"module": key, "contribution": value} for key, value in single_contributions.items()]
    )
    single_frame.to_csv(single_csv, index=False)
    interaction_frame = pd.DataFrame(
        [{"module_pair": key, "interaction": value} for key, value in pair_interactions.items()]
    )
    interaction_frame.to_csv(interaction_csv, index=False)
    t_tests.to_csv(ttest_csv, index=False)

    report_files = {
        "per_run_summary": str(run_csv.resolve()),
        "group_summary": str(group_csv.resolve()),
        "seed_average_summary": str(seed_average_csv.resolve()),
        "best_checkpoints": str(best_csv.resolve()),
        "train_step_logs": str(step_log_csv.resolve()),
        "epoch_metric_logs": str(epoch_log_csv.resolve()),
        "per_sample_metrics_all": str(per_sample_csv.resolve()),
        "single_module_contributions": str(single_csv.resolve()),
        "interaction_effects": str(interaction_csv.resolve()),
        "paired_t_tests": str(ttest_csv.resolve()),
    }
    if write_excel:
        report_files["excel_workbook"] = str(workbook_path.resolve())
    workbook_index = _build_workbook_index(
        analysis_dir=analysis_dir,
        workbook_path=workbook_path,
        run_frame=run_frame,
        train_steps_frame=train_steps_frame,
        epoch_metrics_frame=epoch_metrics_frame,
        per_sample_frame=per_sample_frame,
        report_files=report_files,
    )

    if write_excel:
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            _write_excel_sheet(writer, "说明", workbook_index)
            _write_excel_sheet(writer, "单次运行汇总", run_frame)
            _write_excel_sheet(writer, "组级汇总", group_summary)
            _write_excel_sheet(writer, "Seed平均汇总", group_summary)
            _write_excel_sheet(writer, "最优检查点", best_frame)
            _write_excel_sheet(writer, "训练Step日志", train_steps_frame)
            _write_excel_sheet(writer, "Epoch指标", epoch_metrics_frame)
            _write_excel_sheet(writer, "逐图评估", per_sample_frame)
            _write_excel_sheet(writer, "单模块贡献", single_frame)
            _write_excel_sheet(writer, "交互效应", interaction_frame)
            _write_excel_sheet(writer, "配对T检验", t_tests)

    summary_json = analysis_dir / "analysis_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "single_module_contributions": single_contributions,
                "interaction_effects": pair_interactions,
                "report_files": report_files,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    localized_workbook_path = sync_analysis_localized_outputs(analysis_dir) if write_excel else None

    result = {
        "analysis_dir": str(analysis_dir.resolve()),
        "analysis_summary_json": str(summary_json.resolve()),
    }
    result.update(report_files)
    if localized_workbook_path is not None:
        result["experiment_summary_chinese"] = str(localized_workbook_path.resolve())
    return result
