"""Component schemas opt into the same validation used by preview and launch."""
import json

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.component_validation import schemas_for_method, validate_method_components
from src.app.pipeline.exceptions import ConfigurationError


def test_profile_schema_validates_selected_model_and_leaves_legacy_schemas_optional(tmp_path):
    component = tmp_path / "src/usr/models/example"
    component.mkdir(parents=True)
    path = component / ".theta_component.json"
    schema = {"type": "object", "properties": {"width": {"type": "integer", "minimum": 1}},
              "required": ["width"]}
    path.write_text(json.dumps({"id": "example", "config_path": "in/config/model/example.yaml",
                                "parameter_schema": schema}))
    method = {"model": "example"}
    assert schemas_for_method(tmp_path, method) == {}
    schema["x-theta-validate"] = True
    path.write_text(json.dumps({"id": "example", "config_path": "in/config/model/example.yaml",
                                "parameter_schema": schema}))
    schemas = schemas_for_method(tmp_path, method)
    validate_method_components(OmegaConf.create({"model": {"width": 16}}), schemas, "m")
    with pytest.raises(ConfigurationError, match="model.width"):
        validate_method_components(OmegaConf.create({"model": {"width": 0}}), schemas, "m")
    with pytest.raises(ConfigurationError, match="required property"):
        validate_method_components(OmegaConf.create({"model": {}}), schemas, "m")


def test_sweep_checks_every_choice_and_bounds_against_field_schema():
    from src.app.pipeline.component_validation import validate_sweep_dimensions
    schemas = {"model": {"properties": {"width": {"type": "integer", "minimum": 1, "maximum": 100}}}}
    validate_sweep_dimensions({"tune": {"model.width": {"type": "choice", "values": [1, 100]}}}, schemas, "m")
    with pytest.raises(ConfigurationError, match="model.width"):
        validate_sweep_dimensions({"tune": {"model.width": {"type": "choice", "values": [1, 101]}}}, schemas, "m")
    with pytest.raises(ConfigurationError, match="model.width"):
        validate_sweep_dimensions({"tune": {"model.width": {"type": "int", "low": 0, "high": 5}}}, schemas, "m")


def test_declared_model_value_routes_to_model_even_if_yaml_is_old(tmp_path, monkeypatch):
    from src.app.pipeline.commands import build_method_overrides
    component = tmp_path / "src/usr/models/example"
    component.mkdir(parents=True)
    component.joinpath(".theta_component.json").write_text(json.dumps({
        "config_path": "in/config/model/example.yaml",
        "parameter_schema": {"type": "object", "properties": {"logic_lr": {"type": "number"}}},
    }))
    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({"paradigm": "online_rl", "experiment_name": "group/run"})
    overrides = build_method_overrides("m", {"agent": "ppo", "model": "example", "logic_lr": 0.01}, cfg=cfg)
    assert "++model.logic_lr=0.01" in overrides
    assert "++agent.logic_lr=0.01" not in overrides
