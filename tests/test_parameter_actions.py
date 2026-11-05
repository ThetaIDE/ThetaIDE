from pathlib import Path

import pytest

from frontend.parameter_actions import opening_targets


def test_component_handler_supplies_exact_target_and_current_context(tmp_path):
    target = tmp_path / 'custom.data'
    target.write_text('hello')
    provider = tmp_path / 'actions.py'
    provider.write_text('def targets(*, value, context):\n return context["workspace_root"] + "/" + value\n')
    action = {'type': 'open-in-editor', 'handler': {'path': 'actions.py', 'function': 'targets'}}
    assert opening_targets(action, value='custom.data', context={'workspace_root': str(tmp_path)}, workspace=tmp_path) == [('custom.data', target)]


def test_static_file_action(tmp_path):
    target = tmp_path / 'notes.txt'
    target.touch()
    assert opening_targets({'type': 'open-in-editor', 'path': 'notes.txt'}, value=None, context={}, workspace=tmp_path) == [('notes.txt', target)]


def test_missing_target_does_not_fall_back_to_resource_guessing(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        opening_targets({'type': 'open-in-editor', 'path': 'missing'}, value='default', context={}, workspace=tmp_path)
