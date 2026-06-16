#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sci_color_lab.lineart_features import discover_lineart_records, extract_lineart_features_from_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract deterministic lineart-only features into CSV.")
    parser.add_argument("--lineart-dir", type=str, required=True, help="Directory containing lineart images.")
    parser.add_argument("--file-list", type=str, default="", help="Optional TXT/CSV/JSON file describing explicit images to extract.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan the lineart directory.")
    parser.add_argument("--output-csv", type=str, required=True, help="Destination CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_lineart_records(
        lineart_dir=args.lineart_dir,
        file_list_path=args.file_list or None,
        recursive=bool(args.recursive),
    )
    feature_rows = [record.to_dict() for record in extract_lineart_features_from_records(records)]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feature_rows).sort_values("image_id").to_csv(output_path, index=False, encoding="utf-8")
    print(f"[extract_lineart_features] records={len(feature_rows)} output={output_path.resolve()}")


if __name__ == "__main__":
    main()
