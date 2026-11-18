"""An edit invalidates in-flight validation before the debounce timer fires."""
from types import SimpleNamespace
from unittest.mock import Mock

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
                             config_viewer=viewer, update_launch_state=lambda: None, backend=Backend(),
                             compose_timer=Mock())
    Window.request_compose(window)
    assert captured["overrides"] == []
    assert "hidden_sizes" in captured["draft"] and "17" in captured["draft"]


@pytest.mark.parametrize("action,previously_valid", [("install", False), ("uninstall", True)])
def test_component_change_revalidates_selected_experiment(action, previously_valid, tmp_path):
    from src.app.workspace import WorkspacePaths
    timer = Mock()
    launch = Mock()
    component = SimpleNamespace(kind="model", id="example", extra={})
    client = SimpleNamespace(get_component=lambda _: component)
    window = SimpleNamespace(
        current_experiment="g/e", compose_serial=7, compose_valid=previously_valid,
        composed_config={"model": "example"},
        compose_timer=SimpleNamespace(start=timer), update_launch_state=launch,
        hub_client=client, statusBar=lambda: SimpleNamespace(showMessage=Mock()),
        workspace_paths=WorkspacePaths(tmp_path), findChildren=lambda _kind: [],
    )
    window.update_config = lambda: Window.update_config(window)

    Window._on_hub_component_changed(window, "example", action)

    assert window.compose_serial == 8
    assert not window.compose_valid
    assert window.composed_config is None
    timer.assert_called_once_with()
    launch.assert_called_once_with()
    # A validation started before the install/uninstall cannot restore its
    # obsolete success or missing-component result while the refresh is queued.
    Window.compose_finished(window, 7, None, {"valid": previously_valid}, None)
    assert not window.compose_valid


def _saving_window(serial, draft):
    viewer = SimpleNamespace(current_rel_path="experiment/g/e.yaml", _preamble="# @package _global_\n",
                             raw_data={"seed": 3})
    timer = Mock()
    window = SimpleNamespace(config_viewer=viewer, compose_serial=serial, _compose_request=(4, draft),
                             compose_timer=timer, statusBar=lambda: SimpleNamespace(showMessage=Mock()))
    return window, viewer, timer


def test_saving_the_checked_draft_does_not_check_it_again():
    import yaml
    window, viewer, timer = _saving_window(4, "# @package _global_\n" + yaml.safe_dump({"seed": 3}, sort_keys=False))
    Window.on_config_saved(window, viewer)
    timer.start.assert_not_called()


@pytest.mark.parametrize("case", ["edited since", "normalized differently", "another file saved", "never checked"])
def test_saving_new_content_checks_it(case):
    import yaml
    checked = "# @package _global_\n" + yaml.safe_dump({"seed": 3}, sort_keys=False)
    window, viewer, timer = _saving_window(5 if case == "edited since" else 4, None if case == "never checked" else checked)
    if case == "normalized differently":
        viewer.raw_data = {"seed": 4}
    saved = SimpleNamespace() if case == "another file saved" else viewer  # e.g. an inherited _base.yaml
    Window.on_config_saved(window, saved)
    timer.start.assert_called_once_with()
