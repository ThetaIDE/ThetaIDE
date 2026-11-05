"""An edit invalidates in-flight validation before the debounce timer fires."""
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
from frontend.app import Window


def test_edit_invalidates_previous_validation_before_debounce():
    calls = []
    window = SimpleNamespace(
        compose_serial=4, compose_valid=True, composed_config={"old": True},
        compose_timer=SimpleNamespace(start=lambda: calls.append("timer")),
        update_launch_state=lambda: calls.append("launch"),
    )
    Window.update_config(window)
    assert window.compose_serial == 5
    assert not window.compose_valid
    assert window.composed_config is None
    assert calls == ["launch", "timer"]
    # A callback from the previous request must not restore the old result.
    Window.compose_finished(window, 4, None, {"valid": True}, None)
    assert not window.compose_valid


def test_preview_submits_yaml_draft_instead_of_reconstructed_overrides():
    captured = {}
    class Backend:
        def compose(self, experiment, overrides, callback, draft_yaml=None):
            captured.update(experiment=experiment, overrides=overrides, draft=draft_yaml)
    viewer = SimpleNamespace(is_dirty=True, _preamble="# @package _global_\n",
                             raw_data={"methods": {"m": {"model": "example", "hidden_sizes": [17, 29]}}})
    window = SimpleNamespace(compose_serial=0, config=lambda: None, current_experiment="g/e",
                             config_viewer=viewer, update_launch_state=lambda: None, backend=Backend())
    Window.request_compose(window)
    assert captured["overrides"] == []
    assert "hidden_sizes" in captured["draft"] and "17" in captured["draft"]
