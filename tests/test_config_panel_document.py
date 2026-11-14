"""Deterministic document contracts; no Qt or installed provider prerequisites."""
import pytest

from frontend.config_panel.bindings import ParameterBinding, component_identity
from frontend.config_panel.document import ConfigDocument, MISSING, ChangeKind


@pytest.mark.parametrize("value", ["123", "false", "null", "", None, 0, False, 1e-9, 1e100, 2**80])
def test_typed_set_preserves_type_and_value(value):
    raw = {"unknown": {"keep": True}}
    doc = ConfigDocument(raw)
    assert doc.set(("value",), value).changed
    assert raw["value"] == value and type(raw["value"]) is type(value)
    assert raw["unknown"] == {"keep": True}


def test_container_commits_distinguish_boolean_integer_and_float():
    doc = ConfigDocument({"values": [False], "mapping": {"value": 0}})
    assert doc.set(("values",), [0]).changed
    assert type(doc.explicit["values"][0]) is int
    assert doc.set(("mapping",), {"value": 0.0}).changed
    assert type(doc.explicit["mapping"]["value"]) is float


def test_inherited_presence_null_mask_and_remove_override():
    raw = {}
    doc = ConfigDocument(raw, {"env": {"name": "demo", "count": 3}, "flag": False})
    binding = ParameterBinding(doc, ("flag",))
    assert binding.inherited and not binding.can_remove and binding.value is False
    assert raw == {}
    doc.set(("env",), None)
    assert doc.get(("env",)) is None and doc.get(("env", "count")) is MISSING
    doc.remove(("env",))
    assert doc.get(("env", "count")) == 3
    assert doc.get(("missing",)) is MISSING


@pytest.mark.parametrize("value,schema", [("false", {"type": "boolean"}),
    ([1, "bad"], {"type": "array", "items": {"type": "integer"}}),
    (0, {"minimum": 1}), (False, {"type": "integer"}), (3.0, {"type": "integer"})])
def test_invalid_set_is_atomic(value, schema):
    raw = {}
    changes = []
    doc = ConfigDocument(raw, on_change=changes.append)
    assert doc.set(("missing", "value"), value, schema=schema).error
    assert raw == {} and changes == []


def test_selector_child_edit_preserves_identity_and_inheritance():
    raw = {"env": "demo"}
    doc = ConfigDocument(raw, selector_identity=component_identity)
    result = doc.set(("env", "count"), 0)
    assert result.kind is ChangeKind.STRUCTURE
    assert raw == {"env": {"name": "demo", "count": 0}}
    inherited = ConfigDocument({}, {"model": "dnn"}, selector_identity=component_identity)
    inherited.set(("model", "depth"), 4)
    assert inherited.explicit == {"model": {"name": "dnn", "depth": 4}}


def test_read_only_required_and_noop_edits_do_not_notify():
    changes = []
    raw = {"value": 0}
    doc = ConfigDocument(raw, on_change=changes.append)
    assert not doc.set(("value",), 0).changed
    assert doc.remove(("value",), required=True).error
    doc.read_only = True
    assert doc.set(("value",), 1).error
    assert doc.remove(("value",)).error
    assert not doc.save(lambda _: pytest.fail("write called"))
    assert raw == {"value": 0} and not changes


def test_save_preserves_normalization_order_and_preamble_and_propagates_failure():
    raw = {"description": "123"}
    doc = ConfigDocument(raw)
    calls = []
    doc.save(calls.append, normalize=(lambda: calls.append("clean"), lambda: calls.append("reconcile")),
             preamble="# @package _global_\n")
    assert calls[:2] == ["clean", "reconcile"]
    assert calls[2].startswith("# @package _global_\n")
    import yaml
    assert yaml.safe_load(calls[2])["description"] == "123"
    with pytest.raises(OSError):
        doc.save(lambda _: (_ for _ in ()).throw(OSError("disk full")))
