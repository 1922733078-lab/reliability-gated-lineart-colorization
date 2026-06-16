from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _trend(values: list[float | None]) -> str:
    filtered = [item for item in values if item is not None]
    if len(filtered) < 2:
        return "insufficient_data"
    return "improved" if filtered[-1] < filtered[0] else "worsened_or_flat"


def build_summary(run_dir: Path) -> dict[str, Any]:
    run_summary = _load_json(run_dir / "run_summary.json")
    epoch_history_payload = _load_json(run_dir / "logs" / "epoch_history.json")
    epochs = epoch_history_payload.get("epochs", []) if isinstance(epoch_history_payload, dict) else []

    epoch_rows: list[dict[str, Any]] = []
    for row in epochs:
        if not isinstance(row, dict):
            continue
        epoch_rows.append(
            {
                "epoch": row.get("epoch"),
                "train_loss": row.get("train_loss"),
                "val_loss": row.get("val_loss"),
                "fid": row.get("fid"),
                "lpips": row.get("lpips"),
                "ssim": row.get("ssim"),
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "f_score": row.get("f_score"),
                "train_gpu_memory_peak_gb": row.get("gpu_memory_peak_gb"),
                "eval_gpu_memory_peak_gb": row.get("eval_gpu_memory_peak_gb"),
                "epoch_time_seconds": row.get("epoch_time_seconds"),
                "generation_metrics_deferred": row.get("generation_metrics_deferred"),
                "posthoc_metrics_completed": row.get("posthoc_metrics_completed"),
                "wgan_disabled_reason": row.get("wgan_disabled_reason", ""),
                "collapse_guard_triggered": bool(row.get("collapse_guard_triggered", False)),
            }
        )

    fid_values = [_safe_float(row.get("fid")) for row in epoch_rows]
    val_loss_values = [_safe_float(row.get("val_loss")) for row in epoch_rows]
    train_gpu_values = [_safe_float(row.get("train_gpu_memory_peak_gb")) for row in epoch_rows]
    eval_gpu_values = [_safe_float(row.get("eval_gpu_memory_peak_gb")) for row in epoch_rows]

    warnings: list[str] = []
    latest_precision = _safe_float(run_summary.get("latest_precision"))
    latest_f_score = _safe_float(run_summary.get("latest_f_score"))
    if latest_precision is not None and latest_precision <= 0.0:
        warnings.append("latest_precision <= 0, precision signal is weak")
    if latest_f_score is not None and latest_f_score <= 0.0:
        warnings.append("latest_f_score <= 0, quality-threshold hits are weak")
    if any(bool(row.get("collapse_guard_triggered")) for row in epoch_rows):
        warnings.append("collapse guard was triggered in at least one epoch")
    if any(str(row.get("wgan_disabled_reason", "")).strip() for row in epoch_rows):
        warnings.append("WGAN was disabled by guard in at least one epoch")

    deferred_ok = all(
        bool(row.get("generation_metrics_deferred")) is False or bool(row.get("posthoc_metrics_completed"))
        for row in epoch_rows
    ) if epoch_rows else True
    if not deferred_ok:
        warnings.append("some deferred generation metrics were not completed")

    overview = {
        "status": run_summary.get("status"),
        "group_id": run_summary.get("group_id"),
        "seed": run_summary.get("seed"),
        "epoch": run_summary.get("epoch"),
        "best_fid": run_summary.get("best_fid"),
        "best_fid_epoch": run_summary.get("best_fid_epoch"),
        "best_val_loss": run_summary.get("best_val_loss"),
        "latest_val_loss": run_summary.get("latest_val_loss"),
        "latest_precision": run_summary.get("latest_precision"),
        "latest_recall": run_summary.get("latest_recall"),
        "latest_f_score": run_summary.get("latest_f_score"),
        "seed_elapsed_hms": run_summary.get("seed_elapsed_hms"),
        "train_gpu_memory_peak_gb": run_summary.get("train_gpu_memory_peak_gb"),
        "latest_eval_gpu_memory_peak_gb": run_summary.get("latest_eval_gpu_memory_peak_gb"),
    }

    trends = {
        "fid_trend": _trend(fid_values),
        "val_loss_trend": _trend(val_loss_values),
        "avg_train_gpu_memory_peak_gb": mean([v for v in train_gpu_values if v is not None]) if any(v is not None for v in train_gpu_values) else None,
        "avg_eval_gpu_memory_peak_gb": mean([v for v in eval_gpu_values if v is not None]) if any(v is not None for v in eval_gpu_values) else None,
    }

    return {
        "run_dir": str(run_dir.resolve()),
        "overview": overview,
        "trends": trends,
        "warnings": warnings,
        "epochs": epoch_rows,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    overview = summary.get("overview", {})
    trends = summary.get("trends", {})
    warnings = summary.get("warnings", [])
    epochs = summary.get("epochs", [])

    lines = [
        "# 训练自动汇总",
        "",
        f"- Run: `{summary.get('run_dir', '')}`",
        f"- Status: `{overview.get('status', '')}`",
        f"- Group/Seed: `{overview.get('group_id', '')}` / `{overview.get('seed', '')}`",
        f"- Epoch: `{overview.get('epoch', '')}`",
        f"- Best FID: `{overview.get('best_fid', '')}` (epoch `{overview.get('best_fid_epoch', '')}`)",
        f"- Best Val Loss: `{overview.get('best_val_loss', '')}`",
        f"- Latest Precision/Recall/F-score: `{overview.get('latest_precision', '')}` / `{overview.get('latest_recall', '')}` / `{overview.get('latest_f_score', '')}`",
        f"- Train/Eval Peak GPU (GB): `{overview.get('train_gpu_memory_peak_gb', '')}` / `{overview.get('latest_eval_gpu_memory_peak_gb', '')}`",
        f"- Total Time: `{overview.get('seed_elapsed_hms', '')}`",
        "",
        "## 趋势",
        "",
        f"- FID trend: `{trends.get('fid_trend', '')}`",
        f"- Val loss trend: `{trends.get('val_loss_trend', '')}`",
        f"- Avg train GPU peak (GB): `{trends.get('avg_train_gpu_memory_peak_gb', '')}`",
        f"- Avg eval GPU peak (GB): `{trends.get('avg_eval_gpu_memory_peak_gb', '')}`",
        "",
        "## 风险提示",
    ]
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- 无明显风险提示")

    lines.extend([
        "",
        "## 分轮次简表",
        "",
        "| epoch | train_loss | val_loss | fid | lpips | ssim | precision | recall | f_score | train_gpu_gb | eval_gpu_gb | posthoc_done |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])

    for row in epochs:
        lines.append(
            "| {epoch} | {train_loss} | {val_loss} | {fid} | {lpips} | {ssim} | {precision} | {recall} | {f_score} | {train_gpu} | {eval_gpu} | {posthoc} |".format(
                epoch=row.get("epoch", ""),
                train_loss=row.get("train_loss", ""),
                val_loss=row.get("val_loss", ""),
                fid=row.get("fid", ""),
                lpips=row.get("lpips", ""),
                ssim=row.get("ssim", ""),
                precision=row.get("precision", ""),
                recall=row.get("recall", ""),
                f_score=row.get("f_score", ""),
                train_gpu=row.get("train_gpu_memory_peak_gb", ""),
                eval_gpu=row.get("eval_gpu_memory_peak_gb", ""),
                posthoc=row.get("posthoc_metrics_completed", ""),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto summarize one training run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory, e.g. outputs/E_FULL/seed_42")
    parser.add_argument(
        "--output-name",
        type=str,
        default="auto_summary",
        help="Output basename. Will write <name>.json and <name>.md under run_dir/analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    summary = build_summary(run_dir)
    analysis_dir = run_dir / "analysis"
    json_path = analysis_dir / f"{args.output_name}.json"
    md_path = analysis_dir / f"{args.output_name}.md"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_markdown(summary, md_path)

    print(json.dumps({"summary_json": str(json_path.resolve()), "summary_md": str(md_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
