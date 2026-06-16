from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class PairRecord:
    image_id: str
    color_path: str
    lineart_path: str


@dataclass(frozen=True)
class EvaluationRecord:
    image_id: str
    lineart_path: str
    color_path: str = ""
    has_reference: bool = False
    source: str = ""


@dataclass(frozen=True)
class SplitBundle:
    train: tuple[PairRecord, ...]
    val: tuple[PairRecord, ...]
    test: tuple[PairRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": [record.__dict__ for record in self.train],
            "val": [record.__dict__ for record in self.val],
            "test": [record.__dict__ for record in self.test],
        }


@dataclass(frozen=True)
class ValidationSelection:
    metric_records: tuple[EvaluationRecord, ...]
    preview_records: tuple[EvaluationRecord, ...]
    source: str
    preview_source: str
    has_reference: bool
    external_root: str
    external_paired_count: int
    external_lineart_only_count: int
    fallback_count: int
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_records": [record.__dict__ for record in self.metric_records],
            "preview_records": [record.__dict__ for record in self.preview_records],
            "source": self.source,
            "preview_source": self.preview_source,
            "has_reference": self.has_reference,
            "external_root": self.external_root,
            "external_paired_count": self.external_paired_count,
            "external_lineart_only_count": self.external_lineart_only_count,
            "fallback_count": self.fallback_count,
            "note": self.note,
        }


def normalize_lineart_image(lineart_gray: np.ndarray) -> np.ndarray:
    lineart_gray = lineart_gray.astype(np.uint8)
    white_ratio = float(np.mean(lineart_gray > 200))
    if white_ratio <= 0.5:
        lineart_gray = 255 - lineart_gray
    if lineart_gray.min() != lineart_gray.max():
        lineart_gray = cv2.normalize(lineart_gray, None, 0, 255, cv2.NORM_MINMAX)
    return lineart_gray


def build_control_image(lineart_gray: np.ndarray, controlnet_name: str) -> np.ndarray:
    normalized = normalize_lineart_image(lineart_gray)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    name = controlnet_name.lower()
    if "canny" in name:
        return (255 - binary).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    line_mask = binary < 128
    thick_lines = cv2.dilate(line_mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
    return np.where(thick_lines, 0, 255).astype(np.uint8)


def discover_pairs(dataset_root: str | Path, color_dir_name: str, lineart_dir_name: str) -> list[PairRecord]:
    root = Path(dataset_root)
    color_dir = root / color_dir_name
    lineart_dir = root / lineart_dir_name
    if not color_dir.exists():
        raise FileNotFoundError(f"Color directory not found: {color_dir}")
    if not lineart_dir.exists():
        raise FileNotFoundError(f"Lineart directory not found: {lineart_dir}")

    lineart_lookup: dict[str, Path] = {}
    for path in sorted(lineart_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            lineart_lookup[path.stem] = path.resolve()

    records: list[PairRecord] = []
    for path in sorted(color_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = lineart_lookup.get(path.stem)
        if match is None:
            continue
        records.append(
            PairRecord(
                image_id=path.stem,
                color_path=str(path.resolve()),
                lineart_path=str(match),
            )
        )
    if not records:
        raise RuntimeError(f"No matched image pairs found in {color_dir} and {lineart_dir}.")
    return records


def pair_records_to_evaluation(records: list[PairRecord] | tuple[PairRecord, ...], source: str) -> list[EvaluationRecord]:
    return [
        EvaluationRecord(
            image_id=record.image_id,
            lineart_path=record.lineart_path,
            color_path=record.color_path,
            has_reference=True,
            source=source,
        )
        for record in records
    ]


def _list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS])


def discover_external_validation_records(
    dataset_root: str | Path,
    color_dir_name: str,
    lineart_dir_name: str,
) -> tuple[list[EvaluationRecord], list[EvaluationRecord], dict[str, Any]]:
    root = Path(dataset_root)
    info = {
        "root": str(root.resolve()) if root.exists() else str(root),
        "exists": root.exists(),
        "paired_count": 0,
        "lineart_only_count": 0,
        "color_count": 0,
        "lineart_count": 0,
    }
    if not root.exists():
        return [], [], info

    paired_records: list[EvaluationRecord] = []
    try:
        matched_pairs = discover_pairs(root, color_dir_name, lineart_dir_name)
        paired_records = pair_records_to_evaluation(matched_pairs, source="external_validation")
    except Exception:
        paired_records = []

    color_dir = root / color_dir_name
    lineart_dir = root / lineart_dir_name
    color_images = _list_images(color_dir) if color_dir.exists() else []
    lineart_images = _list_images(lineart_dir) if lineart_dir.exists() else []
    if not lineart_images and root.exists():
        lineart_images = sorted(
            [
                item.resolve()
                for item in root.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

    paired_lineart_paths = {str(Path(record.lineart_path).resolve()) for record in paired_records}
    lineart_only_records: list[EvaluationRecord] = []
    seen_image_ids: dict[str, int] = {}
    for path in lineart_images:
        path_str = str(path)
        if path_str in paired_lineart_paths:
            continue
        image_id = path.stem
        duplicate_index = seen_image_ids.get(image_id, 0)
        seen_image_ids[image_id] = duplicate_index + 1
        if duplicate_index:
            image_id = f"{image_id}_{duplicate_index + 1}"
        lineart_only_records.append(
            EvaluationRecord(
                image_id=image_id,
                lineart_path=path_str,
                has_reference=False,
                source="external_validation_lineart_only",
            )
        )

    info.update(
        {
            "paired_count": len(paired_records),
            "lineart_only_count": len(lineart_only_records),
            "color_count": len(color_images),
            "lineart_count": len(lineart_images),
        }
    )
    return paired_records, lineart_only_records, info


def select_validation_records(
    *,
    split_val_records: list[PairRecord] | tuple[PairRecord, ...],
    validation_dataset_root: str,
    validation_color_dir_name: str,
    validation_lineart_dir_name: str,
    prefer_external_validation_dataset: bool,
) -> ValidationSelection:
    fallback_records = pair_records_to_evaluation(split_val_records, source="split_val")
    external_paired, external_lineart_only, info = discover_external_validation_records(
        validation_dataset_root,
        validation_color_dir_name,
        validation_lineart_dir_name,
    )

    if prefer_external_validation_dataset and external_paired:
        metric_records = tuple(external_paired)
        source = "external_paired_validation"
        note = "Using paired external validation dataset for LPIPS/SSIM/FID and other generation metrics."
    else:
        metric_records = tuple(fallback_records)
        if external_lineart_only:
            source = "split_val_with_external_preview"
            note = (
                "External validation dataset currently contains lineart-only samples, "
                "so quantitative metrics fall back to the fixed validation split."
            )
        elif info["exists"]:
            source = "split_val_fallback"
            note = "External validation dataset exists but no usable paired references were found."
        else:
            source = "split_val"
            note = "External validation dataset not found. Using the fixed validation split."

    if external_lineart_only:
        preview_records = tuple(external_lineart_only)
        preview_source = "external_lineart_only"
    elif metric_records:
        preview_records = metric_records
        preview_source = source
    else:
        preview_records = tuple(fallback_records)
        preview_source = "split_val"

    return ValidationSelection(
        metric_records=metric_records,
        preview_records=preview_records,
        source=source,
        preview_source=preview_source,
        has_reference=bool(metric_records and all(record.has_reference for record in metric_records)),
        external_root=info["root"],
        external_paired_count=int(info["paired_count"]),
        external_lineart_only_count=int(info["lineart_only_count"]),
        fallback_count=len(fallback_records),
        note=note,
    )


def summarize_dataset(dataset_root: str | Path, color_dir_name: str, lineart_dir_name: str) -> dict[str, Any]:
    records = discover_pairs(dataset_root, color_dir_name, lineart_dir_name)
    color_dir = Path(dataset_root) / color_dir_name
    lineart_dir = Path(dataset_root) / lineart_dir_name
    color_count = len([path for path in color_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS])
    lineart_count = len([path for path in lineart_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS])
    sample_sizes: list[tuple[int, int]] = []
    for record in records[:5]:
        with Image.open(record.color_path) as image:
            sample_sizes.append(image.size)
    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "color_dir": str(color_dir.resolve()),
        "lineart_dir": str(lineart_dir.resolve()),
        "color_count": color_count,
        "lineart_count": lineart_count,
        "matched_count": len(records),
        "sample_sizes": sample_sizes,
    }


def create_or_load_split(
    dataset_root: str | Path,
    color_dir_name: str,
    lineart_dir_name: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    output_path: str | Path,
    use_all_training_pairs_for_training: bool = False,
) -> SplitBundle:
    output_path = Path(output_path)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload.get("_metadata", {})
        split_mode_matches = metadata.get("use_all_training_pairs_for_training") == bool(use_all_training_pairs_for_training)
        ratio_matches = (
            metadata.get("train_ratio") == float(train_ratio)
            and metadata.get("val_ratio") == float(val_ratio)
            and metadata.get("test_ratio") == float(test_ratio)
        )
        seed_matches = metadata.get("split_seed") == int(split_seed)
        if metadata and split_mode_matches and ratio_matches and seed_matches:
            return SplitBundle(
                train=tuple(PairRecord(**item) for item in payload.get("train", [])),
                val=tuple(PairRecord(**item) for item in payload.get("val", [])),
                test=tuple(PairRecord(**item) for item in payload.get("test", [])),
            )
        if not metadata and not use_all_training_pairs_for_training:
            return SplitBundle(
                train=tuple(PairRecord(**item) for item in payload.get("train", [])),
                val=tuple(PairRecord(**item) for item in payload.get("val", [])),
                test=tuple(PairRecord(**item) for item in payload.get("test", [])),
            )

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    records = discover_pairs(dataset_root, color_dir_name, lineart_dir_name)
    shuffled = list(records)
    random.Random(split_seed).shuffle(shuffled)

    if use_all_training_pairs_for_training:
        train_slice = shuffled
        val_slice = []
        test_slice = []
    else:
        total_count = len(shuffled)
        train_count = max(1, int(total_count * train_ratio))
        val_count = max(1, int(total_count * val_ratio))
        if train_count + val_count >= total_count:
            val_count = max(1, total_count - train_count - 1)
        test_count = max(1, total_count - train_count - val_count)
        train_slice = shuffled[:train_count]
        val_slice = shuffled[train_count : train_count + val_count]
        test_slice = shuffled[train_count + val_count : train_count + val_count + test_count]

    bundle = SplitBundle(train=tuple(train_slice), val=tuple(val_slice), test=tuple(test_slice))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **bundle.to_dict(),
                "_metadata": {
                    "dataset_root": str(Path(dataset_root).resolve()),
                    "split_seed": int(split_seed),
                    "train_ratio": float(train_ratio),
                    "val_ratio": float(val_ratio),
                    "test_ratio": float(test_ratio),
                    "use_all_training_pairs_for_training": bool(use_all_training_pairs_for_training),
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return bundle


class LineartColorizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[PairRecord] | tuple[PairRecord, ...],
        image_width: int,
        image_height: int,
        controlnet_model: str,
        enable_horizontal_flip: bool = False,
    ) -> None:
        self.records = list(records)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.controlnet_model = controlnet_model
        self.enable_horizontal_flip = enable_horizontal_flip
        self.image_transform = T.Compose(
            [
                T.Resize((self.image_height, self.image_width), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.control_transform = T.Compose(
            [
                T.Resize((self.image_height, self.image_width), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        color_image = Image.open(record.color_path).convert("RGB")
        lineart_image = Image.open(record.lineart_path).convert("L")

        if self.enable_horizontal_flip and torch.rand(1).item() > 0.5:
            color_image = T.functional.hflip(color_image)
            lineart_image = T.functional.hflip(lineart_image)

        color_tensor = self.image_transform(color_image)

        lineart_gray = np.array(lineart_image, dtype=np.uint8)
        control_gray = build_control_image(lineart_gray, self.controlnet_model)
        control_image = Image.fromarray(control_gray).convert("RGB")
        lineart_tensor = self.control_transform(control_image)

        return {
            "color": color_tensor,
            "lineart": lineart_tensor,
            "record": record.__dict__,
        }
