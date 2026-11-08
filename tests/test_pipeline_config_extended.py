import os

import pytest

from src.app.pipeline.config import parse_method_list, resolve_experiment_config_name


def test_parse_method_list_empty():
    assert parse_method_list(None) == []
    assert parse_method_list("") == []
    assert parse_method_list([]) == []


def test_parse_method_list_string():
    assert parse_method_list("ppo") == ["ppo"]


def test_parse_method_list_comma_separated():
    assert parse_method_list("ppo, cql, dqn") == ["ppo", "cql", "dqn"]
    assert parse_method_list("ppo,cql,dqn") == ["ppo", "cql", "dqn"]


def test_parse_method_list_list_input():
    assert parse_method_list(["ppo", "cql"]) == ["ppo", "cql"]


def test_parse_method_list_omegaconf_list():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(["ppo", "cql"])
    assert parse_method_list(cfg) == ["ppo", "cql"]


def test_resolve_experiment_config_name_direct_match(tmp_path, monkeypatch):
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    directory = tmp_path / "in/config/experiment"
    directory.mkdir(parents=True)
    (directory / "my_exp.yaml").write_text("{}\n")
    assert resolve_experiment_config_name("my_exp") == "my_exp"
    assert resolve_experiment_config_name("my_exp.yaml") == "my_exp"


def test_resolve_experiment_config_name_exp_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    assert resolve_experiment_config_name("my_exp") == "my_exp"


def test_resolve_experiment_config_name_glob_single(tmp_path, monkeypatch):
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    directory = tmp_path / "in/config/experiment/group"
    directory.mkdir(parents=True)
    (directory / "my_exp.yaml").write_text("{}\n")
    assert resolve_experiment_config_name("my_exp") == "group/my_exp"


def test_resolve_experiment_config_name_glob_multiple(tmp_path, monkeypatch):
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    for group in ("first", "second"):
        directory = tmp_path / "in/config/experiment" / group
        directory.mkdir(parents=True)
        (directory / "my_exp.yaml").write_text("{}\n")
    with pytest.raises(ValueError, match="Ambiguous experiment name"):
        resolve_experiment_config_name("my_exp")


def test_resolve_experiment_config_name_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    (tmp_path / "in/config/experiment").mkdir(parents=True)
    with pytest.raises(ValueError, match="not found in"):
        resolve_experiment_config_name("my_exp")
