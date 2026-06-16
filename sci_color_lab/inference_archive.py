from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .localized_outputs import sync_archive_localized_outputs


DEFAULT_INFERENCE_ARCHIVE_ROOT = Path("artifacts/inference_archive")

IMAGE_SUBDIRS = ("generated", "target", "lineart")
REPORT_FILES = (
    "metrics.json",
    "per_sample_metrics.csv",
    "subgroup_metrics.json",
    "helper_tool_metrics.json",
)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _archive_root(archive_root: str | Path | None) -> Path:
    if archive_root is None:
        return DEFAULT_INFERENCE_ARCHIVE_ROOT
    text = str(archive_root).strip()
    if not text:
        return DEFAULT_INFERENCE_ARCHIVE_ROOT
    return Path(text)


def archive_evaluation_outputs(
    *,
    source_eval_dir: str | Path,
    archive_root: str | Path | None,
    archive_kind: str,
    group_id: str,
    seed: int,
    epoch: int | None = None,
    split_name: str | None = None,
    checkpoint_label: str | None = None,
) -> str:
    source_eval_dir = Path(source_eval_dir)
    archive_base = _archive_root(archive_root)

    if archive_kind == "training_validation":
        if epoch is None:
            raise ValueError("epoch is required when archive_kind=training_validation")
        destination = archive_base / "训练过程验证" / f"group_{group_id}" / f"seed_{seed}" / f"epoch_{epoch:03d}"
    elif archive_kind == "standalone_evaluation":
        split_label = (split_name or "unknown").strip() or "unknown"
        checkpoint_name = (checkpoint_label or "final").strip() or "final"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = (
            archive_base
            / "独立评估推理"
            / f"group_{group_id}"
            / f"seed_{seed}"
            / f"split_{split_label}"
            / f"checkpoint_{checkpoint_name}"
            / timestamp
        )
    else:
        raise ValueError(f"Unsupported archive_kind: {archive_kind}")

    destination.mkdir(parents=True, exist_ok=True)
    for directory_name in IMAGE_SUBDIRS:
        _copy_tree(source_eval_dir / directory_name, destination / directory_name)

    reports_dir = destination / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for file_name in REPORT_FILES:
        _copy_file(source_eval_dir / file_name, reports_dir / file_name)

    helper_runtime_dir = source_eval_dir / "helper_tools_runtime"
    if helper_runtime_dir.exists():
        _copy_tree(helper_runtime_dir, destination / "helper_tools_runtime")

    manifest_path = destination / "archive_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_eval_dir": str(source_eval_dir.resolve()),
                "archive_dir": str(destination.resolve()),
                "archive_kind": archive_kind,
                "group_id": group_id,
                "seed": int(seed),
                "epoch": epoch,
                "split_name": split_name,
                "checkpoint_label": checkpoint_label,
                "archived_at": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    sync_archive_localized_outputs(destination)
    return str(destination.resolve())
