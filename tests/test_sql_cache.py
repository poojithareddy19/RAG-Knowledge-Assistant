"""The SQL cache must hit on repeats, miss on variation, and expire."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.utils import cache


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    path = tmp_path / "sql_cache.json"
    monkeypatch.setattr(
        cache,
        "_settings",
        lambda: (path, 3600, True),
    )
    return path


def test_roundtrip(tmp_cache):
    cache.put(
        "mean temp per year",
        "llama3.1:8b",
        "SELECT 1",
    )
    assert (
        cache.get(
            "mean temp per year",
            "llama3.1:8b",
        )
        == "SELECT 1"
    )


def test_key_ignores_case_and_surrounding_space(tmp_cache):
    cache.put(
        "Mean Temp Per Year",
        "llama3.1:8b",
        "SELECT 1",
    )
    assert (
        cache.get(
            "  mean temp per year  ",
            "llama3.1:8b",
        )
        == "SELECT 1"
    )


def test_different_model_is_a_different_entry(tmp_cache):
    cache.put(
        "mean temp per year",
        "llama3.1:8b",
        "SELECT 1",
    )
    assert (
        cache.get(
            "mean temp per year",
            "llama3.2:3b",
        )
        is None
    )


def test_entries_expire(tmp_path, monkeypatch):
    path = tmp_path / "sql_cache.json"

    monkeypatch.setattr(
        cache,
        "_settings",
        lambda: (path, 0, True),
    )

    cache.put("q", "m", "SELECT 1")

    time.sleep(0.01)

    assert cache.get("q", "m") is None


def test_corrupt_cache_does_not_raise(tmp_cache):
    Path(tmp_cache).write_text(
        "{not json",
        encoding="utf-8",
    )

    assert cache.get("q", "m") is None


def test_disabled_cache_never_hits(tmp_path, monkeypatch):
    path = tmp_path / "sql_cache.json"

    monkeypatch.setattr(
        cache,
        "_settings",
        lambda: (path, 3600, False),
    )

    cache.put("q", "m", "SELECT 1")

    assert cache.get("q", "m") is None
    assert not path.exists()