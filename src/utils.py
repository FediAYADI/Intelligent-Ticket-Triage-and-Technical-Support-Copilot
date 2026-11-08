"""Shared utilities: config loading and project paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict:
    """Load config/config.yaml once and cache it."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against the project root
    for key in ("raw_dir", "processed_dir", "kb_dir", "models_dir", "reports_dir"):
        cfg["paths"][key] = str(PROJECT_ROOT / cfg["paths"][key])
    return cfg


def project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))
