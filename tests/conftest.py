"""Shared test setup."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider

from src.monitoring import tracing


@pytest.fixture(autouse=True, scope="session")
def _no_trace_file():
    """Spans from the suite go nowhere rather than into logs/traces.jsonl.

    A provider with no processors records nothing, and installing it here
    stops the first traced call from setting up the real file exporter.
    Tests that look at spans install their own in-memory provider.
    """
    tracing.use_provider(TracerProvider())
    yield
