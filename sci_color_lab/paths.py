from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL_REPOS = {
    "base": "stabilityai/stable-diffusion-xl-base-1.0",
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "scribble": "xinsir/controlnet-scribble-sdxl-1.0",
}

MODEL_ENV_VARS = {
    "base": ("SCI_BASE_MODEL", "SDXL_BASE_MODEL"),
    "canny": ("SCI_CONTROLNET_CANNY_MODEL",),
    "scribble": ("SCI_CONTROLNET_SCRIBBLE_MODEL",),
}

MODEL_REPO_CANDIDATES = {
    "base": [
        Path("models/base"),
        Path("models/stable-diffusion-xl-base-1.0"),
    ],
    "canny": [
        Path("models/controlnet-canny-sdxl-1.0"),
        Path("models/canny"),
    ],
    "scribble": [
        Path("models/controlnet-scribble-sdxl-1.0"),
        Path("models/scribble"),
    ],
}


def _repo_root_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _standard_huggingface_cache_candidates(role: str) -> list[Path]:
    repo_id = DEFAULT_MODEL_REPOS[role]
    repo_root_name = _repo_root_name(repo_id)
    candidates: list[Path] = []

    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub_cache:
        candidates.append(Path(hub_cache) / repo_root_name)

    huggingface_home = os.environ.get("HF_HOME", "").strip()
    if huggingface_home:
        candidates.append(Path(huggingface_home) / "hub" / repo_root_name)

    candidates.append(Path.home() / ".cache" / "huggingface" / "hub" / repo_root_name)
    return candidates


def _env_candidates(role: str) -> list[Path]:
    candidates: list[Path] = []
    for env_name in MODEL_ENV_VARS.get(role, ()):
        value = os.environ.get(env_name, "").strip()
        if value:
            candidates.append(Path(value).expanduser())
    return candidates


def is_model_snapshot(path: Path) -> bool:
    return (path / "model_index.json").exists() or (path / "config.json").exists() or (path / "unet").exists()


def find_snapshot(repo_root: Path) -> Path | None:
    if not repo_root.exists():
        return None
    if is_model_snapshot(repo_root):
        return repo_root
    snapshots_dir = repo_root / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted([item for item in snapshots_dir.iterdir() if item.is_dir()], key=lambda item: item.stat().st_mtime, reverse=True)
    for snapshot in snapshots:
        if is_model_snapshot(snapshot):
            return snapshot
    return None


def auto_resolve_model_path(role: str) -> Path:
    for repo_root in _env_candidates(role) + MODEL_REPO_CANDIDATES[role] + _standard_huggingface_cache_candidates(role):
        snapshot = find_snapshot(repo_root)
        if snapshot is not None:
            return snapshot.resolve()

    repo_id = DEFAULT_MODEL_REPOS[role]
    try:
        cached = snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception:
        cached = snapshot_download(repo_id=repo_id)
    return Path(cached).resolve()


def resolve_repo_or_path(value: str, default_role: str) -> str:
    value = (value or "").strip()
    if not value:
        return str(auto_resolve_model_path(default_role))

    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())

    try:
        cached = snapshot_download(repo_id=value, local_files_only=True)
        return str(Path(cached).resolve())
    except Exception:
        return value
