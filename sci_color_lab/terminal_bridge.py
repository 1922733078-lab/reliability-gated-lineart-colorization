from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror training progress summaries into a target terminal.")
    parser.add_argument("--output-root", required=True, help="Experiment output root.")
    parser.add_argument("--group", required=True, help="Ablation group id, for example E0.")
    parser.add_argument("--tty-path", required=True, help="Target tty path, for example /dev/pts/3.")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Polling interval in seconds.")
    return parser.parse_args()


def _write_line(tty_path: Path, message: str) -> None:
    tty_path.parent.mkdir(parents=True, exist_ok=True)
    with tty_path.open("a", encoding="utf-8", errors="ignore") as handle:
        handle.write(message.rstrip() + "\n")
        handle.flush()


def _format_train(payload: dict) -> str:
    return (
        f"[{payload.get('group_id', '-')}/seed_{payload.get('seed', '-')}] "
        f"epoch {payload.get('epoch', '-')}, "
        f"step {payload.get('global_step', '-')}/{payload.get('total_optimizer_steps', '-')}, "
        f"loss={float(payload.get('loss', 0.0)):.6f}, "
        f"lr={float(payload.get('lr', 0.0)):.2e}"
    )


def _format_epoch_end(payload: dict) -> str:
    parts = [
        f"[{payload.get('group_id', '-')}/seed_{payload.get('seed', '-')}]",
        f"epoch {payload.get('epoch', '-')}",
        f"train_loss={payload.get('train_loss')}",
        f"val_loss={payload.get('val_loss')}",
        f"fid={payload.get('fid')}",
        f"lpips={payload.get('lpips')}",
        f"ssim={payload.get('ssim')}",
        f"params_m={payload.get('params_m')}",
        f"flops_g={payload.get('flops_g')}",
    ]
    return " ".join(parts)


def _read_new_lines(path: Path, offsets: dict[str, int]) -> list[str]:
    if not path.exists():
        return []
    key = str(path.resolve())
    previous = offsets.get(key)
    current_size = path.stat().st_size
    if previous is None or previous > current_size:
        offsets[key] = current_size
        return []
    if previous == current_size:
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(previous)
        chunk = handle.read()
    offsets[key] = current_size
    return [line for line in chunk.splitlines() if line.strip()]


def _print_initial_snapshot(output_root: Path, group: str, tty_path: Path) -> None:
    run_root = output_root / group
    if not run_root.exists():
        _write_line(tty_path, f"[bridge] waiting for {run_root}")
        return
    for summary_path in sorted(run_root.glob("seed_*/train_status.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        _write_line(
            tty_path,
            "[bridge:init] "
            + " ".join(
                [
                    f"group={payload.get('group_id', '-')}",
                    f"seed={payload.get('seed', '-')}",
                    f"state={payload.get('state', '-')}",
                    f"epoch={payload.get('epoch', '-')}",
                    f"step={payload.get('global_step', '-')}/{payload.get('total_optimizer_steps', '-')}",
                    f"train_loss={payload.get('train_loss', payload.get('best_train_loss'))}",
                    f"val_loss={payload.get('val_loss', payload.get('best_val_loss'))}",
                    f"fid={payload.get('fid', payload.get('best_fid'))}",
                ]
            ),
        )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    tty_path = Path(args.tty_path)
    offsets: dict[str, int] = {}

    _write_line(tty_path, f"[bridge] mirroring {args.group} progress from {output_root} to {tty_path}")
    _print_initial_snapshot(output_root, args.group, tty_path)

    while True:
        run_root = output_root / args.group
        for log_path in sorted(run_root.glob("seed_*/logs/train.jsonl")):
            for line in _read_new_lines(log_path, offsets):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") == "train":
                    _write_line(tty_path, _format_train(payload))
        for log_path in sorted(run_root.glob("seed_*/logs/metrics.jsonl")):
            for line in _read_new_lines(log_path, offsets):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") == "epoch_end":
                    _write_line(tty_path, _format_epoch_end(payload))
        time.sleep(max(args.poll_seconds, 0.2))


if __name__ == "__main__":
    main()
