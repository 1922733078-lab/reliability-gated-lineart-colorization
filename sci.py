#!/usr/bin/env python3
"""Unified CLI entry for train / inference / analyze."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    print(f"\n{'=' * 72}")
    print(f"[run] {description}")
    print(f"command: {' '.join(cmd)}")
    print(f"{'=' * 72}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[error] command failed with exit code {result.returncode}")
    return int(result.returncode)


def load_config_output_root(config_json: str | None, cli_output_root: str | None) -> str:
    if cli_output_root:
        return cli_output_root
    if not config_json:
        return "outputs"

    config_path = Path(config_json)
    if not config_path.exists():
        return "outputs"

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return "outputs"

    if isinstance(payload, dict) and "trainer" in payload and isinstance(payload["trainer"], dict):
        payload = payload["trainer"]
    if isinstance(payload, dict):
        output_root = payload.get("output_root")
        if output_root:
            return str(output_root)
    return "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SCI workflow entry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sci.py train --group-id E_FULL --config-json config.json --epochs 12 --seeds 42 123 456
  python sci.py inference --run-dirs outputs/E_FULL/seed_42 --epochs all
  python sci.py analyze --output-root outputs --groups E_FULL
  python sci.py full --group-id E_FULL --config-json config.json --epochs 12 --seeds 42 123 456
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="subcommands")

    train_parser = subparsers.add_parser("train", help="pure training only")
    train_parser.add_argument("--group-id", type=str, required=True, help="Ablation group id")
    train_parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seed list")
    train_parser.add_argument("--config-json", type=str, required=True, help="Trainer config json")
    train_parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    train_parser.add_argument("--output-root", type=str, default=None, help="Override output root")

    inference_parser = subparsers.add_parser("inference", help="centralized batch inference from saved LoRA checkpoints")
    inference_parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories")
    inference_parser.add_argument("--epochs", type=str, default="all", help="Epoch list or 'all'")
    inference_parser.add_argument("--max-samples", type=int, default=20, help="Max validation samples")
    inference_parser.add_argument("--num-inference-steps", type=int, default=20, help="Diffusion inference steps")
    inference_parser.add_argument("--guidance-scale", type=float, default=6.5, help="CFG guidance scale")
    inference_parser.add_argument("--controlnet-scale", type=float, default=None, help="Optional ControlNet conditioning scale override")
    inference_parser.add_argument("--scheduler", type=str, default="unipc", help="Scheduler name")
    inference_parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    inference_parser.add_argument("--dtype", type=str, default="fp16", help="Inference dtype")
    inference_parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload")
    inference_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing inference outputs")

    analyze_parser = subparsers.add_parser("analyze", help="compute metrics and generate plots")
    analyze_parser.add_argument("--output-root", type=str, required=True, help="Output root")
    analyze_parser.add_argument("--groups", nargs="+", default=None, help="Optional group filter")
    analyze_parser.add_argument("--epochs", type=str, default="all", help="Epoch list or 'all'")
    analyze_parser.add_argument("--device", type=str, default="cuda", help="Metric device")
    analyze_parser.add_argument("--force", action="store_true", help="Force metric recomputation")
    analyze_parser.add_argument("--no-excel", action="store_true", help="Skip Excel workbook export")
    analyze_parser.add_argument("--plot-dir", type=str, default="plots", help="Plot directory name")

    full_parser = subparsers.add_parser("full", help="train -> inference -> analyze")
    full_parser.add_argument("--group-id", type=str, required=True, help="Ablation group id")
    full_parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seed list")
    full_parser.add_argument("--config-json", type=str, required=True, help="Trainer config json")
    full_parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    full_parser.add_argument("--inference-epochs", type=str, default="all", help="Epoch list or 'all'")
    full_parser.add_argument("--output-root", type=str, default=None, help="Override output root")
    full_parser.add_argument("--device", type=str, default="cuda", help="Device for inference / metrics")
    full_parser.add_argument("--dtype", type=str, default="fp16", help="Inference dtype")
    full_parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload during inference")
    full_parser.add_argument("--max-samples", type=int, default=20, help="Max validation samples for inference")
    full_parser.add_argument("--num-inference-steps", type=int, default=20, help="Diffusion inference steps")
    full_parser.add_argument("--guidance-scale", type=float, default=6.5, help="CFG guidance scale")
    full_parser.add_argument("--controlnet-scale", type=float, default=None, help="Optional ControlNet conditioning scale override")
    full_parser.add_argument("--scheduler", type=str, default="unipc", help="Scheduler name")
    full_parser.add_argument("--force-analyze", action="store_true", help="Force metric recomputation in analyze stage")
    full_parser.add_argument("--no-excel", action="store_true", help="Skip Excel workbook export")
    full_parser.add_argument("--skip-train", action="store_true", help="Skip training stage")
    full_parser.add_argument("--skip-inference", action="store_true", help="Skip inference stage")
    full_parser.add_argument("--skip-analyze", action="store_true", help="Skip analyze stage")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    script_dir = Path(__file__).resolve().parent

    if args.command == "train":
        cmd = [
            sys.executable,
            str(script_dir / "train_only.py"),
            "--group-id",
            args.group_id,
            "--config-json",
            args.config_json,
        ]
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.output_root:
            cmd.extend(["--output-root", args.output_root])
        if args.seeds:
            cmd.extend(["--seeds"] + [str(seed) for seed in args.seeds])
        return run_command(cmd, f"train group={args.group_id}")

    if args.command == "inference":
        cmd = [
            sys.executable,
            str(script_dir / "inference_batch.py"),
            "--run-dirs",
            *args.run_dirs,
            "--epochs",
            args.epochs,
            "--max-samples",
            str(args.max_samples),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--scheduler",
            args.scheduler,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
        ]
        if args.controlnet_scale is not None:
            cmd.extend(["--controlnet-scale", str(args.controlnet_scale)])
        if args.cpu_offload:
            cmd.append("--cpu-offload")
        if args.overwrite:
            cmd.append("--overwrite")
        return run_command(cmd, f"inference over {len(args.run_dirs)} run dirs")

    if args.command == "analyze":
        cmd = [
            sys.executable,
            str(script_dir / "analyze_results.py"),
            "--output-root",
            args.output_root,
            "--epochs",
            args.epochs,
            "--device",
            args.device,
            "--plot-dir",
            args.plot_dir,
        ]
        if args.groups:
            cmd.extend(["--groups"] + args.groups)
        if args.force:
            cmd.append("--force")
        if args.no_excel:
            cmd.append("--no-excel")
        return run_command(cmd, "analyze metrics and plots")

    if args.command == "full":
        output_root = load_config_output_root(args.config_json, args.output_root)
        seeds = args.seeds or [42, 123, 456]
        run_dirs = [str((Path(output_root) / args.group_id / f"seed_{int(seed)}").resolve()) for seed in seeds]

        if not args.skip_train:
            train_cmd = [
                sys.executable,
                str(script_dir / "train_only.py"),
                "--group-id",
                args.group_id,
                "--config-json",
                args.config_json,
            ]
            if args.epochs is not None:
                train_cmd.extend(["--epochs", str(args.epochs)])
            if output_root:
                train_cmd.extend(["--output-root", output_root])
            if seeds:
                train_cmd.extend(["--seeds"] + [str(seed) for seed in seeds])
            result = run_command(train_cmd, f"[1/3] train group={args.group_id}")
            if result != 0:
                return result

        if not args.skip_inference:
            inference_cmd = [
                sys.executable,
                str(script_dir / "inference_batch.py"),
                "--run-dirs",
                *run_dirs,
                "--epochs",
                args.inference_epochs,
                "--max-samples",
                str(args.max_samples),
                "--num-inference-steps",
                str(args.num_inference_steps),
                "--guidance-scale",
                str(args.guidance_scale),
                "--scheduler",
                args.scheduler,
                "--device",
                args.device,
                "--dtype",
                args.dtype,
            ]
            if args.controlnet_scale is not None:
                inference_cmd.extend(["--controlnet-scale", str(args.controlnet_scale)])
            if args.cpu_offload:
                inference_cmd.append("--cpu-offload")
            result = run_command(inference_cmd, "[2/3] centralized inference")
            if result != 0:
                return result

        if not args.skip_analyze:
            analyze_cmd = [
                sys.executable,
                str(script_dir / "analyze_results.py"),
                "--output-root",
                output_root,
                "--groups",
                args.group_id,
                "--epochs",
                args.inference_epochs,
                "--device",
                args.device,
            ]
            if args.force_analyze:
                analyze_cmd.append("--force")
            if args.no_excel:
                analyze_cmd.append("--no-excel")
            result = run_command(analyze_cmd, "[3/3] analyze metrics and plots")
            if result != 0:
                return result

        print(f"\n{'=' * 72}")
        print("workflow completed")
        print(f"output_root: {Path(output_root).resolve()}")
        print(f"{'=' * 72}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
