"""Configuration loading and access.

Single source of truth for runtime settings. Load order (lowest to highest
priority):

    config.yaml  ->  environment variables  ->  .env file

Environment overrides use a double-underscore path syntax that mirrors the YAML
nesting, e.g. ``RETRIEVAL__TOP_K=8`` overrides ``retrieval.top_k``. Flat keys
such as ``OPENAI_API_KEY`` are also read directly for secrets.

Usage
-----
    from src.utils.config import get_config
    cfg = get_config()
    cfg.retrieval.top_k         # -> int
    cfg.generation.provider     # -> "openai"
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

try:  # dotenv is optional; env vars still work without it
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - best effort
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class AttrDict(dict):
    """Dict that also supports attribute access (cfg.retrieval.top_k)."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc
        return AttrDict(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover
        self[key] = value


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Override nested keys from ``SECTION__KEY`` environment variables."""
    for env_key, raw in os.environ.items():
        if "__" not in env_key:
            continue
        parts = [p.lower() for p in env_key.split("__")]
        node = cfg
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node = None
                break
            node = node[part]
        if node is None:
            continue
        node[parts[-1]] = _coerce(raw)
    return cfg


def _coerce(value: str) -> Any:
    """Best-effort string -> python scalar."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> AttrDict:
    """Load, merge and cache the configuration object."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data = _apply_env_overrides(data)

    # Resolve relative paths against project root so the app can be launched
    # from any working directory.
    for key, rel in data.get("paths", {}).items():
        data["paths"][key] = str((PROJECT_ROOT / rel).resolve())

    # Attach commonly used secrets so callers never touch os.environ directly.
    data.setdefault("secrets", {})
    data["secrets"].update(
        {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        }
    )
    return AttrDict(data)


def project_root() -> Path:
    return PROJECT_ROOT
