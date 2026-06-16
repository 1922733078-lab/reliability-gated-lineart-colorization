from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ModuleFlags:
    dynamic_weight: bool
    adaptive_threshold: bool
    adaptive_rf: bool
    adaptive_norm: bool

    @property
    def code(self) -> str:
        return (
            f"DW{int(self.dynamic_weight)}_"
            f"AT{int(self.adaptive_threshold)}_"
            f"ARF{int(self.adaptive_rf)}_"
            f"AN{int(self.adaptive_norm)}"
        )


@dataclass(frozen=True)
class AblationGroup:
    group_id: str
    display_name: str
    description: str
    flags: ModuleFlags
    category: str


@dataclass(frozen=True)
class AblationSummary:
    single_contributions: dict[str, float]
    pair_interactions: dict[str, float]


ABALATION_GROUPS: dict[str, AblationGroup] = {
    "E0": AblationGroup(
        group_id="E0",
        display_name="Baseline",
        description="Pure SDXL + ControlNet + LoRA baseline without the adaptive adapter features.",
        flags=ModuleFlags(False, False, False, False),
        category="baseline",
    ),
    "E_FULL": AblationGroup(
        group_id="E_FULL",
        display_name="Full",
        description="All adaptive adapter modules enabled.",
        flags=ModuleFlags(True, True, True, True),
        category="full",
    ),
    "E1": AblationGroup(
        group_id="E1",
        display_name="w/o DW",
        description="Disable dynamic weighting only.",
        flags=ModuleFlags(False, True, True, True),
        category="single",
    ),
    "E2": AblationGroup(
        group_id="E2",
        display_name="w/o AT",
        description="Disable adaptive threshold only.",
        flags=ModuleFlags(True, False, True, True),
        category="single",
    ),
    "E3": AblationGroup(
        group_id="E3",
        display_name="w/o ARF",
        description="Disable adaptive receptive field only.",
        flags=ModuleFlags(True, True, False, True),
        category="single",
    ),
    "E4": AblationGroup(
        group_id="E4",
        display_name="w/o AN",
        description="Disable adaptive normalization only.",
        flags=ModuleFlags(True, True, True, False),
        category="single",
    ),
    "E5": AblationGroup(
        group_id="E5",
        display_name="w/o DW + AT",
        description="Disable dynamic weighting and adaptive threshold.",
        flags=ModuleFlags(False, False, True, True),
        category="pair",
    ),
    "E6": AblationGroup(
        group_id="E6",
        display_name="w/o DW + ARF",
        description="Disable dynamic weighting and adaptive receptive field.",
        flags=ModuleFlags(False, True, False, True),
        category="pair",
    ),
    "E7": AblationGroup(
        group_id="E7",
        display_name="w/o DW + AN",
        description="Disable dynamic weighting and adaptive normalization.",
        flags=ModuleFlags(False, True, True, False),
        category="pair",
    ),
    "E8": AblationGroup(
        group_id="E8",
        display_name="w/o AT + ARF",
        description="Disable adaptive threshold and adaptive receptive field.",
        flags=ModuleFlags(True, False, False, True),
        category="pair",
    ),
    "E9": AblationGroup(
        group_id="E9",
        display_name="w/o AT + AN",
        description="Disable adaptive threshold and adaptive normalization.",
        flags=ModuleFlags(True, False, True, False),
        category="pair",
    ),
    "E10": AblationGroup(
        group_id="E10",
        display_name="w/o ARF + AN",
        description="Disable adaptive receptive field and adaptive normalization.",
        flags=ModuleFlags(True, True, False, False),
        category="pair",
    ),
}


def list_groups() -> Iterable[AblationGroup]:
    ordered = ["E0", "E_FULL", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]
    return (ABALATION_GROUPS[key] for key in ordered)


def get_group(group_id: str) -> AblationGroup:
    key = group_id.strip().upper()
    try:
        return ABALATION_GROUPS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown ablation group: {group_id}") from exc


def build_ablation_summary(metric_by_group: dict[str, float]) -> AblationSummary:
    full = metric_by_group.get("E_FULL")
    if full is None:
        return AblationSummary(single_contributions={}, pair_interactions={})

    single = {}
    for module_key, group_id in {
        "DW": "E1",
        "AT": "E2",
        "ARF": "E3",
        "AN": "E4",
    }.items():
        value = metric_by_group.get(group_id)
        if value is not None:
            single[module_key] = float(full - value)

    pair_mapping = {
        "DW-AT": ("E5", "DW", "AT"),
        "DW-ARF": ("E6", "DW", "ARF"),
        "DW-AN": ("E7", "DW", "AN"),
        "AT-ARF": ("E8", "AT", "ARF"),
        "AT-AN": ("E9", "AT", "AN"),
        "ARF-AN": ("E10", "ARF", "AN"),
    }
    interactions: dict[str, float] = {}
    for pair_name, (group_id, left, right) in pair_mapping.items():
        pair_value = metric_by_group.get(group_id)
        if pair_value is None or left not in single or right not in single:
            continue
        delta_pair = float(full - pair_value)
        interactions[pair_name] = float(delta_pair - (single[left] + single[right]))

    return AblationSummary(single_contributions=single, pair_interactions=interactions)
