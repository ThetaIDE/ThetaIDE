"""Component schemas opt into the same validation used by preview and launch."""

import pytest

from src.app.pipeline.exceptions import ConfigurationError


def test_sweep_checks_every_choice_and_bounds_against_field_schema():
    from src.app.pipeline.component_validation import validate_sweep_dimensions
    schemas = {"model": {"properties": {"width": {"type": "integer", "minimum": 1, "maximum": 100}}}}
    validate_sweep_dimensions({"tune": {"model.width": {"type": "choice", "values": [1, 100]}}}, schemas, "m")
    with pytest.raises(ConfigurationError, match="model.width"):
        validate_sweep_dimensions({"tune": {"model.width": {"type": "choice", "values": [1, 101]}}}, schemas, "m")
    with pytest.raises(ConfigurationError, match="model.width"):
        validate_sweep_dimensions({"tune": {"model.width": {"type": "int", "low": 0, "high": 5}}}, schemas, "m")


