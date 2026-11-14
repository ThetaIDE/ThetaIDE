"""Unit tests for plugin and hub dependency inspection and management."""
from pathlib import Path
import pytest

from src.app.components.dependencies import (
    check_missing_dependencies,
    is_module_installed,
    load_requirements_from_file,
    parse_package_name,
)


class TestDependencies:
    def test_parse_package_name_variations(self):
        pkg, mod = parse_package_name("gymnasium[atari]>=0.29.0")
        assert pkg == "gymnasium"
        assert mod == "gymnasium"

        pkg, mod = parse_package_name("ale-py>=0.8.1")
        assert pkg == "ale-py"
        assert mod == "ale_py"

        pkg, mod = parse_package_name("scikit-learn")
        assert pkg == "scikit-learn"
        assert mod == "sklearn"

        pkg, mod = parse_package_name("stable-baselines3==2.1.0 # trailing comment")
        assert pkg == "stable-baselines3"
        assert mod == "stable_baselines3"

    def test_is_module_installed_for_stdlib(self):
        assert is_module_installed("sys") is True
        assert is_module_installed("json") is True
        assert is_module_installed("non_existent_fake_module_xyz_123") is False

    def test_check_missing_dependencies(self):
        reqs = ["sys", "non_existent_pkg_abc_123>=1.0.0"]
        missing = check_missing_dependencies(reqs)
        assert len(missing) == 1
        assert "non_existent_pkg_abc_123" in missing[0]

    def test_load_requirements_from_file(self, tmp_path):
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("gymnasium\n# comment\nale-py>=0.8.1\n\n", encoding="utf-8")

        loaded = load_requirements_from_file(req_file)
        assert loaded == ["gymnasium", "ale-py>=0.8.1"]

    def test_load_requirements_missing_file(self, tmp_path):
        missing_file = tmp_path / "does_not_exist.txt"
        assert load_requirements_from_file(missing_file) == []
