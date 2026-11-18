"""Tuner discovery refuses missing and incompatible components."""

import pytest

from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.tuners import get_tuner


def test_missing_component_fails_instead_of_falling_back_to_another_tuner(monkeypatch):
    monkeypatch.setattr("src.app.core.provider_loader.resolve_provider", lambda kind, name: None)
    with pytest.raises(ConfigurationError, match="not installed"):
        get_tuner({"tuning": {"component": "missing"}})


def test_incompatible_component_contract_is_rejected(monkeypatch):
    class OldTuner:
        api_version = 0
    monkeypatch.setattr("src.app.core.provider_loader.resolve_provider", lambda kind, name: OldTuner if (kind, name) == ("tuner", "old") else None)
    with pytest.raises(ConfigurationError, match="api_version = 1"):
        get_tuner(name="old")


