from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .data import IMAGE_EXTENSIONS, normalize_lineart_image


@dataclass(frozen=True)
class LineartInputRecord:
    image_id: str
    path: str


@dataclass(frozen=True)
class LineartFeatureRecord:
    image_id: str
    path: str
    height: int
    width: int
    line_density: float
    edge_density: float
    blank_ratio: float
    edge_entropy: float
    component_count: int
    component_density: float
    component_area_mean: float
    stroke_width_mean: float
    stroke_width_std: float
    orientation_entropy: float
    endpoint_count: int
    junction_count: int
    junction_density: float
    complexity_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_gray_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    return image


def _resolve_candidate_path(base_dir: Path | None, value: str) -> str:
    candidate = Path(str(value).strip())
    if candidate.is_absolute():
        return str(candidate.resolve())
    if base_dir is not None:
        resolved = (base_dir / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    return str(candidate.resolve())


def _parse_file_list_payload(payload: Any, base_dir: Path | None) -> list[LineartInputRecord]:
    records: list[LineartInputRecord] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str) and item.strip():
                path = _resolve_candidate_path(base_dir, item)
                records.append(LineartInputRecord(image_id=Path(path).stem, path=path))
            elif isinstance(item, dict):
                lineart_path = item.get("lineart_path") or item.get("path") or item.get("file")
                if not lineart_path:
                    continue
                path = _resolve_candidate_path(base_dir, str(lineart_path))
                image_id = str(item.get("image_id") or Path(path).stem)
                records.append(LineartInputRecord(image_id=image_id, path=path))
        return records

    if isinstance(payload, dict):
        for key in ("metric_records", "preview_records", "records", "items"):
            if isinstance(payload.get(key), list):
                records.extend(_parse_file_list_payload(payload[key], base_dir))
        if records:
            return records
        if isinstance(payload.get("image_ids"), list):
            for item in payload["image_ids"]:
                text = str(item).strip()
                if not text:
                    continue
                resolved = _resolve_candidate_path(base_dir, text)
                records.append(LineartInputRecord(image_id=Path(text).stem, path=resolved))
    return records


def load_explicit_lineart_records(file_list_path: str | Path) -> list[LineartInputRecord]:
    path = Path(file_list_path)
    if not path.exists():
        raise FileNotFoundError(f"File list not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = _parse_file_list_payload(payload, path.parent)
    elif path.suffix.lower() == ".csv":
        import pandas as pd

        frame = pd.read_csv(path)
        candidate_column = next((column for column in ("lineart_path", "path", "file") if column in frame.columns), None)
        if candidate_column is None:
            candidate_column = frame.columns[0]
        image_id_column = next((column for column in ("image_id", "id") if column in frame.columns), None)
        records = []
        for row in frame.to_dict(orient="records"):
            raw_path = row.get(candidate_column)
            if raw_path is None or not str(raw_path).strip():
                continue
            resolved = _resolve_candidate_path(path.parent, str(raw_path))
            image_id = str(row.get(image_id_column) or Path(resolved).stem) if image_id_column else Path(resolved).stem
            records.append(LineartInputRecord(image_id=image_id, path=resolved))
    else:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                resolved = _resolve_candidate_path(path.parent, text)
                records.append(LineartInputRecord(image_id=Path(resolved).stem, path=resolved))

    deduplicated: list[LineartInputRecord] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record.image_id, str(Path(record.path).resolve()))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(LineartInputRecord(image_id=record.image_id, path=str(Path(record.path).resolve())))
    if not deduplicated:
        raise RuntimeError(f"No lineart records parsed from {path}")
    return deduplicated


def discover_lineart_records(
    *,
    lineart_dir: str | Path,
    file_list_path: str | Path | None = None,
    recursive: bool = False,
) -> list[LineartInputRecord]:
    if file_list_path:
        return load_explicit_lineart_records(file_list_path)

    root = Path(lineart_dir)
    if not root.exists():
        raise FileNotFoundError(f"Lineart directory not found: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    records: list[LineartInputRecord] = []
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        records.append(LineartInputRecord(image_id=path.stem, path=str(path.resolve())))
    if not records:
        raise RuntimeError(f"No lineart images found in {root}")
    return records


def _compute_edge_mask(normalized_gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(normalized_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (cv2.Canny(255 - binary, 50, 150) > 0).astype(np.uint8)


def _compute_grid_entropy(mask: np.ndarray, grid_rows: int = 8, grid_cols: int = 8) -> float:
    height, width = mask.shape
    if height == 0 or width == 0:
        return 0.0
    values: list[float] = []
    for row_index in range(grid_rows):
        row_start = int(round(row_index * height / grid_rows))
        row_end = int(round((row_index + 1) * height / grid_rows))
        for col_index in range(grid_cols):
            col_start = int(round(col_index * width / grid_cols))
            col_end = int(round((col_index + 1) * width / grid_cols))
            patch = mask[row_start:row_end, col_start:col_end]
            values.append(float(patch.mean()) if patch.size else 0.0)
    distribution = np.array(values, dtype=np.float32)
    distribution = distribution[distribution > 0]
    if distribution.size == 0:
        return 0.0
    distribution = distribution / float(distribution.sum())
    entropy = -np.sum(distribution * np.log(distribution + 1e-12))
    return float(entropy / np.log(float(grid_rows * grid_cols)))


def _thin_binary_mask(line_mask: np.ndarray) -> np.ndarray:
    binary = (line_mask.astype(np.uint8) * 255).copy()
    if binary.max() == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(binary)
    working = binary.copy()
    while True:
        eroded = cv2.erode(working, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(working, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        working = eroded
        if cv2.countNonZero(working) == 0:
            break
    return skeleton


def _compute_orientation_entropy(line_signal: np.ndarray, edge_mask: np.ndarray, bins: int = 12) -> float:
    if edge_mask.max() == 0:
        return 0.0
    gx = cv2.Sobel(line_signal, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(line_signal, cv2.CV_32F, 0, 1, ksize=3)
    angles = np.mod(np.arctan2(gy, gx), np.pi)
    sampled = angles[edge_mask > 0]
    if sampled.size == 0:
        return 0.0
    hist, _ = np.histogram(sampled, bins=bins, range=(0.0, float(np.pi)))
    hist = hist.astype(np.float32)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    hist = hist / float(hist.sum())
    entropy = -np.sum(hist * np.log(hist + 1e-12))
    return float(entropy / np.log(float(bins)))


def _compute_skeleton_topology(skeleton_mask: np.ndarray) -> tuple[int, int]:
    if skeleton_mask.max() == 0:
        return 0, 0
    skeleton_uint8 = (skeleton_mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighborhood_sum = cv2.filter2D(skeleton_uint8, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
    neighbor_count = neighborhood_sum - skeleton_uint8
    endpoints = int(np.logical_and(skeleton_uint8 == 1, neighbor_count == 1).sum())
    junctions = int(np.logical_and(skeleton_uint8 == 1, neighbor_count >= 3).sum())
    return endpoints, junctions


def extract_lineart_features(image_id: str, image_path: str | Path) -> LineartFeatureRecord:
    gray = _read_gray_image(image_path)
    normalized = normalize_lineart_image(gray)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    line_mask = (binary < 128).astype(np.uint8)
    edge_mask = _compute_edge_mask(normalized)
    skeleton = _thin_binary_mask(line_mask)

    height, width = normalized.shape[:2]
    total_pixels = max(height * width, 1)
    total_pixels_per_10k = max(total_pixels / 10_000.0, 1e-6)

    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
    component_areas = component_stats[1:, cv2.CC_STAT_AREA].astype(np.float32) if component_count > 1 else np.array([], dtype=np.float32)
    real_component_count = int(component_areas.size)

    line_density = float(line_mask.mean())
    edge_density = float(edge_mask.mean())
    blank_ratio = float(1.0 - line_density)
    edge_entropy = _compute_grid_entropy(edge_mask)
    component_density = float(real_component_count / total_pixels_per_10k)
    component_area_mean = float(component_areas.mean()) if component_areas.size else 0.0

    distance_input = (line_mask * 255).astype(np.uint8)
    distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 5)
    skeleton_mask = skeleton > 0
    stroke_width_samples = distance[skeleton_mask] * 2.0
    stroke_width_mean = float(stroke_width_samples.mean()) if stroke_width_samples.size else 0.0
    stroke_width_std = float(stroke_width_samples.std(ddof=0)) if stroke_width_samples.size else 0.0

    orientation_entropy = _compute_orientation_entropy(255 - normalized, edge_mask)
    endpoint_count, junction_count = _compute_skeleton_topology(skeleton_mask.astype(np.uint8))
    junction_density = float(junction_count / total_pixels_per_10k)

    line_term = min(1.0, line_density / 0.25)
    edge_term = min(1.0, edge_density / 0.15)
    component_term = min(1.0, float(np.log1p(component_density) / np.log1p(60.0)))
    junction_term = min(1.0, float(np.log1p(junction_density) / np.log1p(700.0)))
    stroke_term = min(1.0, stroke_width_mean / 4.0)
    complexity_score = float(
        np.clip(
            (0.24 * line_term)
            + (0.12 * edge_term)
            + (0.14 * edge_entropy)
            + (0.08 * (1.0 - blank_ratio))
            + (0.18 * component_term)
            + (0.10 * orientation_entropy)
            + (0.10 * junction_term)
            + (0.04 * stroke_term),
            0.0,
            1.0,
        )
    )

    return LineartFeatureRecord(
        image_id=str(image_id),
        path=str(Path(image_path).resolve()),
        height=int(height),
        width=int(width),
        line_density=line_density,
        edge_density=edge_density,
        blank_ratio=blank_ratio,
        edge_entropy=edge_entropy,
        component_count=real_component_count,
        component_density=component_density,
        component_area_mean=component_area_mean,
        stroke_width_mean=stroke_width_mean,
        stroke_width_std=stroke_width_std,
        orientation_entropy=orientation_entropy,
        endpoint_count=endpoint_count,
        junction_count=junction_count,
        junction_density=junction_density,
        complexity_score=complexity_score,
    )


def extract_lineart_features_from_records(records: list[LineartInputRecord]) -> list[LineartFeatureRecord]:
    return [extract_lineart_features(record.image_id, record.path) for record in records]
