from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sci_color_lab.reliability_gate import evaluate_reliability_gate, load_json_cached


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class MembershipFunction:
    function_type: str
    params: tuple[float, ...]

    def evaluate(self, value: float) -> float:
        x = float(value)
        if self.function_type == "trimf":
            a, b, c = self.params
            if x <= a or x >= c:
                return 0.0
            if x == b:
                return 1.0
            if x < b:
                return _clamp01((x - a) / max(b - a, 1e-12))
            return _clamp01((c - x) / max(c - b, 1e-12))
        if self.function_type == "trapmf":
            a, b, c, d = self.params
            if x <= a or x >= d:
                return 0.0
            if b <= x <= c:
                return 1.0
            if x < b:
                return _clamp01((x - a) / max(b - a, 1e-12))
            return _clamp01((d - x) / max(d - c, 1e-12))
        if self.function_type == "gaussmf":
            sigma, mean = self.params
            sigma = max(float(sigma), 1e-12)
            return float(np.exp(-0.5 * ((x - mean) / sigma) ** 2))
        raise ValueError(f"Unsupported membership function type: {self.function_type}")


@dataclass(frozen=True)
class FuzzyRule:
    name: str
    antecedents: dict[str, str]
    consequent: str
    weight: float = 1.0
    operator: str = "min"


@dataclass(frozen=True)
class SelectionPolicy:
    mode: str = "legacy_raw_best"
    default_label: str = ""
    min_margin: float = 0.0
    min_expected_gain: float = 0.0
    fallback_to_default: bool = True
    cost_sensitive_mode: bool = False
    override_requirements: dict[str, dict[str, str]] | None = None
    extras: dict[str, Any] | None = None


@dataclass(frozen=True)
class FuzzyConfig:
    labels: tuple[str, ...]
    variables: dict[str, dict[str, MembershipFunction]]
    rules: tuple[FuzzyRule, ...]
    priors: dict[str, float]
    fallback_label: str
    selection_policy: SelectionPolicy


def load_fuzzy_config(path: str | Path) -> FuzzyConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    labels = tuple(str(item) for item in payload.get("labels", []))
    if not labels:
        raise ValueError(f"No labels configured in {config_path}")

    variables: dict[str, dict[str, MembershipFunction]] = {}
    for variable_name, memberships in (payload.get("variables") or {}).items():
        if not isinstance(memberships, dict):
            continue
        variables[str(variable_name)] = {}
        for membership_name, membership_payload in memberships.items():
            if not isinstance(membership_payload, dict):
                continue
            function_type = str(membership_payload.get("type", "trimf"))
            params = tuple(float(item) for item in membership_payload.get("params", []))
            variables[str(variable_name)][str(membership_name)] = MembershipFunction(function_type=function_type, params=params)

    rules: list[FuzzyRule] = []
    for item in payload.get("rules", []) or []:
        if not isinstance(item, dict):
            continue
        rules.append(
            FuzzyRule(
                name=str(item.get("name", f"rule_{len(rules) + 1}")),
                antecedents={str(key): str(value) for key, value in (item.get("if") or {}).items()},
                consequent=str(item.get("then")),
                weight=float(item.get("weight", 1.0)),
                operator=str(item.get("operator", "min")).lower(),
            )
        )
    if not rules:
        raise ValueError(f"No fuzzy rules configured in {config_path}")

    priors = {str(key): float(value) for key, value in (payload.get("priors") or {}).items()}
    fallback_label = str(payload.get("fallback_label") or labels[0])
    policy_payload = payload.get("selection_policy") or {}
    known_policy_keys = {
        "mode",
        "default_label",
        "min_margin",
        "min_expected_gain",
        "fallback_to_default",
        "cost_sensitive_mode",
        "override_requirements",
    }
    selection_policy = SelectionPolicy(
        mode=str(policy_payload.get("mode", "legacy_raw_best")),
        default_label=str(policy_payload.get("default_label", "E5" if "E5" in labels else fallback_label)),
        min_margin=float(policy_payload.get("min_margin", 0.0)),
        min_expected_gain=float(policy_payload.get("min_expected_gain", 0.0)),
        fallback_to_default=bool(policy_payload.get("fallback_to_default", True)),
        cost_sensitive_mode=bool(policy_payload.get("cost_sensitive_mode", False)),
        override_requirements={
            str(label): {str(key): str(value) for key, value in requirements.items()}
            for label, requirements in (policy_payload.get("override_requirements") or {}).items()
            if isinstance(requirements, dict)
        },
        extras={str(key): value for key, value in policy_payload.items() if key not in known_policy_keys},
    )
    return FuzzyConfig(
        labels=labels,
        variables=variables,
        rules=tuple(rules),
        priors=priors,
        fallback_label=fallback_label,
        selection_policy=selection_policy,
    )


def evaluate_memberships(config: FuzzyConfig, features: dict[str, Any]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for variable_name, membership_lookup in config.variables.items():
        feature_value = float(features.get(variable_name, 0.0) or 0.0)
        summary[variable_name] = {
            membership_name: function.evaluate(feature_value)
            for membership_name, function in membership_lookup.items()
        }
    return summary


def _combine_memberships(values: list[float], operator: str) -> float:
    if not values:
        return 0.0
    if operator == "product":
        result = 1.0
        for value in values:
            result *= float(value)
        return float(result)
    if operator == "mean":
        return float(np.mean(values))
    return float(min(values))


def compute_raw_fuzzy_decision(config: FuzzyConfig, features: dict[str, Any]) -> dict[str, Any]:
    memberships = evaluate_memberships(config, features)
    label_scores = {label: float(config.priors.get(label, 0.0)) for label in config.labels}
    rule_scores: dict[str, float] = {}

    for rule in config.rules:
        antecedent_scores = [float(memberships.get(variable_name, {}).get(membership_name, 0.0)) for variable_name, membership_name in rule.antecedents.items()]
        firing_strength = _combine_memberships(antecedent_scores, rule.operator) * float(rule.weight)
        rule_scores[rule.name] = float(firing_strength)
        label_scores[rule.consequent] = label_scores.get(rule.consequent, 0.0) + float(firing_strength)

    sorted_labels = sorted(label_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    raw_best_label = sorted_labels[0][0] if sorted_labels and sorted_labels[0][1] > 0 else config.fallback_label
    raw_best_score = float(sorted_labels[0][1]) if sorted_labels else 0.0
    second_score = float(sorted_labels[1][1]) if len(sorted_labels) > 1 else 0.0
    return {
        "memberships": memberships,
        "rule_scores": rule_scores,
        "label_scores": label_scores,
        "raw_best_label": raw_best_label,
        "raw_best_score": raw_best_score,
        "confidence_margin": float(raw_best_score - second_score),
        "sorted_labels": sorted_labels,
    }


def apply_selection_policy(
    config: FuzzyConfig,
    raw_decision: dict[str, Any],
    *,
    features: dict[str, Any] | None = None,
    expected_gain_by_label: dict[str, float] | None = None,
) -> dict[str, Any]:
    policy = config.selection_policy
    default_label = policy.default_label if policy.default_label in config.labels else config.fallback_label
    label_scores = {str(label): float(score) for label, score in (raw_decision.get("label_scores") or {}).items()}
    raw_best_label = str(raw_decision.get("raw_best_label") or default_label)
    raw_best_score = float(raw_decision.get("raw_best_score", 0.0) or 0.0)
    default_score = float(label_scores.get(default_label, 0.0))
    score_margin = float(raw_best_score - default_score)
    memberships = raw_decision.get("memberships") or {}
    expected_gain_lookup = {str(key): float(value) for key, value in (expected_gain_by_label or {}).items() if value is not None}
    if not expected_gain_lookup:
        override_requirements = policy.override_requirements or {}
        expected_gain_lookup = {default_label: 0.0}
        for label in config.labels:
            if label == default_label:
                continue
            if raw_best_label != label:
                expected_gain_lookup[label] = float("-1e-6")
                continue
            requirements = override_requirements.get(label, {})
            if not requirements:
                expected_gain_lookup[label] = score_margin
                continue
            requirement_scores = [float(memberships.get(variable_name, {}).get(membership_name, 0.0)) for variable_name, membership_name in requirements.items()]
            minimum_requirement_score = min(requirement_scores) if requirement_scores else 0.0
            if minimum_requirement_score > 0.0:
                expected_gain_lookup[label] = score_margin * float(minimum_requirement_score)
            else:
                expected_gain_lookup[label] = float("-1e-6")
    raw_best_expected_gain = expected_gain_lookup.get(raw_best_label)

    predicted_label = raw_best_label
    fallback_applied = False
    fallback_reason = "raw_best_selected"
    reliability_gate_passed = False
    reliability_gate_rule = ""
    reliability_gate_score = 0.0
    reliability_gate_reason = "not_used"

    if policy.mode == "e5_default_conservative":
        predicted_label = default_label
        fallback_reason = "default_selected"
        if raw_best_label == default_label:
            fallback_reason = "default_best"
        else:
            margin_ok = score_margin >= float(policy.min_margin)
            gain_ok = raw_best_expected_gain is not None and raw_best_expected_gain >= float(policy.min_expected_gain)
            if margin_ok and gain_ok:
                predicted_label = raw_best_label
                fallback_reason = "override_passed"
            elif not policy.fallback_to_default:
                predicted_label = raw_best_label
                fallback_reason = "fallback_disabled"
            else:
                fallback_applied = True
                if raw_best_expected_gain is None:
                    fallback_reason = "missing_expected_gain"
                elif not margin_ok:
                    fallback_reason = "margin_below_threshold"
                else:
                    fallback_reason = "expected_gain_below_threshold"
    elif policy.mode == "e5_default_reliability_aware":
        predicted_label = default_label
        fallback_reason = "default_selected"
        gate_extras = policy.extras or {}
        supported_override_labels = tuple(str(label) for label in (gate_extras.get("supported_override_labels") or ["E0"]))
        gate_path = str(gate_extras.get("reliability_gate_artifact") or "")
        calibration_path = str(gate_extras.get("feature_calibration_path") or "")
        calibration_payload = load_json_cached(calibration_path) if calibration_path else {}
        gate_payload = load_json_cached(gate_path) if gate_path else {}

        if raw_best_label == default_label:
            fallback_reason = "default_best"
            reliability_gate_reason = "default_best"
        elif raw_best_label not in supported_override_labels:
            fallback_applied = True
            fallback_reason = "unsupported_override_label"
            reliability_gate_reason = "unsupported_override_label"
        elif not gate_payload:
            fallback_applied = True
            fallback_reason = "missing_reliability_gate"
            reliability_gate_reason = "missing_reliability_gate"
        else:
            gate_result = evaluate_reliability_gate(
                gate_payload,
                features=features or {},
                raw_decision={
                    **raw_decision,
                    "default_label": default_label,
                    "score_margin": score_margin,
                },
                calibration_payload=calibration_payload or None,
            )
            reliability_gate_passed = bool(gate_result.get("accepted"))
            reliability_gate_rule = str(gate_result.get("rule_name") or "")
            reliability_gate_score = float(gate_result.get("reliability_score", 0.0) or 0.0)
            reliability_gate_reason = str(gate_result.get("reason") or "")

            margin_ok = score_margin >= float(policy.min_margin)
            if reliability_gate_passed and margin_ok:
                predicted_label = raw_best_label
                fallback_reason = "reliability_gate_passed"
            elif not policy.fallback_to_default:
                predicted_label = raw_best_label
                fallback_reason = "fallback_disabled"
            else:
                fallback_applied = True
                fallback_reason = "reliability_gate_failed" if not reliability_gate_passed else "margin_below_threshold"
    elif policy.mode not in {"legacy_raw_best", "raw_best"}:
        raise ValueError(f"Unsupported selection policy mode: {policy.mode}")

    return {
        "predicted_label": predicted_label,
        "raw_best_label": raw_best_label,
        "default_label": default_label,
        "raw_best_score": raw_best_score,
        "default_score": default_score,
        "score_margin": score_margin,
        "fallback_applied": bool(fallback_applied),
        "fallback_reason": fallback_reason,
        "expected_gain_raw_best_label": raw_best_expected_gain,
        "expected_gain_default_label": expected_gain_lookup.get(default_label),
        "expected_gain_by_label": expected_gain_lookup,
        "reliability_gate_passed": reliability_gate_passed,
        "reliability_gate_rule": reliability_gate_rule,
        "reliability_gate_score": reliability_gate_score,
        "reliability_gate_reason": reliability_gate_reason,
    }


def score_fuzzy_rules(config: FuzzyConfig, features: dict[str, Any], *, expected_gain_by_label: dict[str, float] | None = None) -> dict[str, Any]:
    raw_decision = compute_raw_fuzzy_decision(config, features)
    final_decision = apply_selection_policy(config, raw_decision, features=features, expected_gain_by_label=expected_gain_by_label)
    return {
        **raw_decision,
        **final_decision,
        "top_score": float(raw_decision.get("raw_best_score", 0.0)),
    }
