from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED_COLORS = {
    42: "#1f77b4",
    123: "#ff7f0e",
    456: "#2ca02c",
}

PREFERRED_FONTS = [
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-seed training metric comparison plots.")
    parser.add_argument("--group", default="E0", help="Experiment group id, for example E0.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Seed ids to plot together.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/examples",
        help="Directory to save the generated comparison plots.",
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Project root directory.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Primary experiment output root. Legacy per-group output roots are auto-detected as fallback.",
    )
    return parser.parse_args()


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = PREFERRED_FONTS
    plt.rcParams["axes.unicode_minus"] = False


def _load_jsonl_rows(log_path: Path, *, event_name: str) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") == event_name:
                rows.append(payload)
    return rows


def _resolve_run_dir(root_dir: Path, output_root: str, group: str, seed: int) -> Path:
    candidates = [
        root_dir / output_root / group / f"seed_{seed}",
        root_dir / f"outputs_{group.lower()}_adamw8bit_12epoch" / group / f"seed_{seed}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _plot_metric(
    seed_series: dict[int, tuple[list[float], list[float]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
) -> str:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)

    for seed, (xs, ys) in seed_series.items():
        if not xs or not ys:
            continue
        color = SEED_COLORS.get(seed, None)
        ax.plot(
            xs,
            ys,
            label=f"Seed {seed}",
            color=color,
            linewidth=2.0,
            marker="o" if len(xs) <= 20 else None,
            markersize=4,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path.resolve())


def _collect_epoch_metric_series(
    root_dir: Path,
    output_root: str,
    group: str,
    seeds: list[int],
    metric_key: str,
) -> dict[int, tuple[list[float], list[float]]]:
    series: dict[int, tuple[list[float], list[float]]] = {}
    for seed in seeds:
        run_dir = _resolve_run_dir(root_dir, output_root, group, seed)
        log_path = run_dir / "logs" / "metrics.jsonl"
        rows = _load_jsonl_rows(log_path, event_name="epoch_end")
        xs = [float(row.get("epoch", 0)) for row in rows if metric_key in row]
        ys = [float(row.get(metric_key, 0.0)) for row in rows if metric_key in row]
        series[seed] = (xs, ys)
    return series


def _collect_loss_series(
    root_dir: Path,
    output_root: str,
    group: str,
    seeds: list[int],
) -> dict[int, tuple[list[float], list[float]]]:
    series: dict[int, tuple[list[float], list[float]]] = {}
    for seed in seeds:
        run_dir = _resolve_run_dir(root_dir, output_root, group, seed)
        log_path = run_dir / "logs" / "train.jsonl"
        rows = _load_jsonl_rows(log_path, event_name="train")
        xs = [float(row.get("global_step", 0)) for row in rows]
        ys = [float(row.get("loss", 0.0)) for row in rows]
        series[seed] = (xs, ys)
    return series


def generate_group_seed_comparison_plots(
    *,
    root_dir: Path,
    output_root: str,
    group: str,
    seeds: list[int],
    output_dir: Path,
) -> dict[str, str]:
    _configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    fid_series = _collect_epoch_metric_series(root_dir, output_root, group, seeds, "fid")
    lpips_series = _collect_epoch_metric_series(root_dir, output_root, group, seeds, "lpips")
    ssim_series = _collect_epoch_metric_series(root_dir, output_root, group, seeds, "ssim")
    loss_series = _collect_loss_series(root_dir, output_root, group, seeds)

    paths["fid"] = _plot_metric(
        fid_series,
        title=f"{group} FID Comparison Across Seeds",
        xlabel="Epoch",
        ylabel="FID",
        save_path=output_dir / f"{group}_FID三随机种子对比图.png",
    )
    paths["lpips"] = _plot_metric(
        lpips_series,
        title=f"{group} LPIPS Comparison Across Seeds",
        xlabel="Epoch",
        ylabel="LPIPS",
        save_path=output_dir / f"{group}_LPIPS三随机种子对比图.png",
    )
    paths["ssim"] = _plot_metric(
        ssim_series,
        title=f"{group} SSIM Comparison Across Seeds",
        xlabel="Epoch",
        ylabel="SSIM",
        save_path=output_dir / f"{group}_SSIM三随机种子对比图.png",
    )
    paths["loss"] = _plot_metric(
        loss_series,
        title=f"{group} Training Loss Comparison Across Seeds",
        xlabel="Global Step",
        ylabel="Loss",
        save_path=output_dir / f"{group}_Loss三随机种子对比图.png",
    )

    manifest_path = output_dir / f"{group}_三随机种子绘图输出.json"
    manifest = {
        "实验组": group,
        "随机种子": seeds,
        "输出目录": str(output_dir.resolve()),
        "图像路径": paths,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["manifest"] = str(manifest_path.resolve())
    return paths


def main() -> None:
    args = parse_args()
    paths = generate_group_seed_comparison_plots(
        root_dir=Path(args.root_dir),
        output_root=args.output_root,
        group=args.group,
        seeds=args.seeds,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
