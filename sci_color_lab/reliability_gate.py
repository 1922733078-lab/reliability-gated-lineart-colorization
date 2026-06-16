from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CALIBRATION_FEATURES = (
    "line_density",
    "edge_entropy",
    "blank_ratio",
    "component_density",
    "complexity_score",
)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=32)
def load_json_cached(path: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    with candidate.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def apply_quantile_clip(value: float, stats: dict[str, Any]) -> float:
    lower = float(stats.get("q10", 0.0))
    upper = float(stats.get("q90", 1.0))
    if upper <= lower:
        return 0.5
    return float(np.clip((float(value) - lower) / max(upper - lower, 1e-12), 0.0, 1.0))


def augment_with_calibrated_features(
    features: dict[str, Any],
    calibration_payload: dict[str, Any] | None,
    *,
    feature_names: tuple[str, ...] = DEFAULT_CALIBRATION_FEATURES,
) -> dict[str, float]:
    augmented: dict[str, float] = {}
    for key, value in features.items():
        numeric = _to_float(value)
        if numeric is not None:
            augmented[str(key)] = numeric

    if not calibration_payload:
        return augmented

    feature_stats = calibration_payload.get("features") or {}
    for feature_name in feature_names:
        raw_value = _to_float(features.get(feature_name))
        stats = feature_stats.get(feature_name) or {}
        if raw_value is None or not stats:
            continue
        calibrated = apply_quantile_clip(raw_value, stats)
        augmented[f"calibrated::{feature_name}"] = calibrated
        augmented[f"calibrated__{feature_name}"] = calibrated
    return augmented


def build_reliability_feature_view(
    *,
    features: dict[str, Any],
    raw_decision: dict[str, Any],
    calibration_payload: dict[str, Any] | None = None,
) -> dict[str, float]:
    feature_view = augment_with_calibrated_features(features, calibration_payload)
    for key in ("raw_best_score", "confidence_margin"):
        numeric = _to_float(raw_decision.get(key))
        if numeric is not None:
            feature_view[key] = numeric

    label_scores = raw_decision.get("label_scores") or {}
    for label, value in label_scores.items():
        numeric = _to_float(value)
        if numeric is not None:
            feature_view[f"score::{label}"] = numeric

    default_label = str(raw_decision.get("default_label") or "")
    raw_best_label = str(raw_decision.get("raw_best_label") or "")
    default_score = _to_float(label_scores.get(default_label))
    raw_best_score = _to_float(raw_decision.get("raw_best_score"))
    if raw_best_score is not None and default_score is not None:
        feature_view["score_margin"] = float(raw_best_score - default_score)
    elif _to_float(raw_decision.get("score_margin")) is not None:
        feature_view["score_margin"] = float(raw_decision["score_margin"])

    if raw_best_label:
        raw_best_label_score = _to_float(label_scores.get(raw_best_label))
        if raw_best_label_score is not None:
            feature_view[f"score::{raw_best_label}"] = raw_best_label_score
    return feature_view


def _condition_matches(condition: dict[str, Any], feature_view: dict[str, float]) -> bool:
    feature_name = str(condition.get("feature") or "")
    if not feature_name:
        return False
    value = feature_view.get(feature_name)
    if value is None:
        return False
    lower = _to_float(condition.get("min"))
    upper = _to_float(condition.get("max"))
    if lower is not None and value < lower - 1e-12:
        return False
    if upper is not None and value > upper + 1e-12:
        return False
    return True


def evaluate_reliability_gate(
    gate_payload: dict[str, Any] | None,
    *,
    features: dict[str, Any],
    raw_decision: dict[str, Any],
    calibration_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = gate_payload or {}
    target_label = str(payload.get("target_label", "E0"))
    raw_best_label = str(raw_decision.get("raw_best_label") or "")
    default_label = str(raw_decision.get("default_label") or "")

    feature_view = build_reliability_feature_view(
        features=features,
        raw_decision=raw_decision,
        calibration_payload=calibration_payload,
    )

    result: dict[str, Any] = {
        "accepted": False,
        "rule_name": "",
        "reliability_score": 0.0,
        "reason": "no_matching_rule",
        "target_label": target_label,
        "default_label": default_label,
    }
    if raw_best_label != target_label:
        result["reason"] = "raw_best_not_target_label"
        return result

    for rule in payload.get("rules") or []:
        conditions = rule.get("conditions") or []
        if not conditions:
            continue
        if all(_condition_matches(condition, feature_view) for condition in conditions):
            result["accepted"] = True
            result["rule_name"] = str(rule.get("name") or "matched_rule")
            result["reliability_score"] = float(rule.get("reliability_score", 1.0))
            result["reason"] = "matched_rule"
            return result
    return result

