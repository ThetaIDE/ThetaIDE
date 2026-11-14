"""Tests for the config-driven architecture the visual model builder writes.

These cover the contract the editor depends on: plan_shapes never raises and
names the row that broke, and a saved spec instantiates through build_model
the same way a hand-written architecture does.
"""

from __future__ import annotations

import pytest
import torch

from src.app.core.model_registry import build_model, get_model_class
from src.plugins.components.bundled.sequential_spec.sequential_spec import (
    LAYER_KINDS,
    LayerSpecError,
    build_sequential,
    output_shape,
    plan_shapes,
)

DENSE = [
    {"type": "linear", "out_features": 64},
    {"type": "relu"},
    {"type": "dropout", "p": 0.1},
    {"type": "linear", "out_features": 1},
]

CONV = [
    {"type": "conv2d", "out_channels": 16, "kernel_size": 3, "padding": 1},
    {"type": "relu"},
    {"type": "maxpool2d", "kernel_size": 2},
    {"type": "flatten"},
    {"type": "linear", "out_features": 10},
]


# ── Shape planning ────────────────────────────────────────────────────────


def test_dense_stack_shapes():
    plans = plan_shapes(DENSE, (30,))
    assert [p.out_shape for p in plans] == [(64,), (64,), (64,), (1,)]
    assert all(p.ok for p in plans)


def test_conv_stack_shapes():
    plans = plan_shapes(CONV, (1, 8, 8))
    assert [p.out_shape for p in plans] == [(16, 8, 8), (16, 8, 8), (16, 4, 4), (256,), (10,)]


def test_strided_conv_halves_spatial_dims():
    plans = plan_shapes(
        [{"type": "conv2d", "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1}], (3, 32, 32)
    )
    assert plans[0].out_shape == (8, 16, 16)


def test_output_shape_matches_a_real_forward_pass():
    model = build_model({"architecture": "sequential_spec", "input_shape": [1, 8, 8], "layers": CONV}, input_dim=64)
    declared = output_shape(CONV, (1, 8, 8))
    assert tuple(model(torch.randn(4, 64)).shape) == (4, *declared)


def test_asymmetric_kernel_accepted_as_a_pair():
    plans = plan_shapes([{"type": "conv2d", "out_channels": 4, "kernel_size": [3, 5]}], (1, 20, 20))
    assert plans[0].out_shape == (4, 18, 16)


# ── Errors the editor renders ─────────────────────────────────────────────


def test_dense_layer_on_an_image_explains_the_fix():
    plan = plan_shapes([{"type": "linear", "out_features": 10}], (1, 8, 8))[0]
    assert not plan.ok
    assert "flatten" in plan.error


def test_oversized_kernel_is_reported_not_raised():
    plan = plan_shapes([{"type": "conv2d", "out_channels": 4, "kernel_size": 11}], (1, 8, 8))[0]
    assert not plan.ok
    assert "too small" in plan.error


def test_unknown_layer_names_the_type():
    plan = plan_shapes([{"type": "attention"}], (30,))[0]
    assert not plan.ok
    assert "attention" in plan.error


def test_missing_required_parameter_is_named():
    plan = plan_shapes([{"type": "linear"}], (30,))[0]
    assert "out_features" in plan.error


def test_later_layers_are_marked_skipped_not_wrong():
    plans = plan_shapes([{"type": "nope"}, {"type": "relu"}], (30,))
    assert "unknown" in plans[0].error
    assert "earlier layer" in plans[1].error


def test_non_mapping_layer_is_reported():
    plans = plan_shapes(["relu"], (30,))
    assert not plans[0].ok


def test_plan_shapes_never_raises_on_hostile_input():
    # The editor calls this on every keystroke, so it must not throw.
    for layers in ([{"type": "linear", "out_features": "wide"}], [{}], [{"type": ""}], [{"type": "conv2d"}]):
        assert all(not p.ok for p in plan_shapes(layers, (30,)))


def test_build_sequential_raises_with_a_one_based_layer_number():
    with pytest.raises(LayerSpecError, match="layer 2"):
        build_sequential([{"type": "relu"}, {"type": "linear"}], (30,))


# ── Instantiation through the registry ────────────────────────────────────


def test_registered_under_its_aliases():
    for alias in ("sequential_spec", "custom_sequential", "visual"):
        assert get_model_class(alias).__name__ == "SequentialSpecModel"


def test_single_logit_is_squeezed_for_binary_tasks():
    model = build_model({"architecture": "sequential_spec", "layers": DENSE}, input_dim=30)
    assert tuple(model(torch.randn(8, 30)).shape) == (8,)


def test_multiclass_head_keeps_its_width():
    layers = DENSE[:-1] + [{"type": "linear", "out_features": 3}]
    model = build_model({"architecture": "sequential_spec", "layers": layers}, input_dim=30)
    assert tuple(model(torch.randn(8, 30)).shape) == (8, 3)


def test_squeeze_can_be_disabled():
    model = build_model(
        {"architecture": "sequential_spec", "layers": DENSE, "squeeze_single_logit": False}, input_dim=30
    )
    assert tuple(model(torch.randn(8, 30)).shape) == (8, 1)


def test_empty_layer_list_is_rejected():
    with pytest.raises(LayerSpecError, match="at least one layer"):
        build_model({"architecture": "sequential_spec", "layers": []}, input_dim=30)


def test_missing_input_shape_and_dim_is_rejected():
    with pytest.raises(LayerSpecError, match="input_shape or input_dim"):
        build_model({"architecture": "sequential_spec", "layers": DENSE})


def test_extra_env_kwargs_are_ignored():
    # build_model forwards n_actions/obs_dim to every architecture it builds.
    model = build_model({"architecture": "sequential_spec", "layers": DENSE}, input_dim=30, n_actions=4)
    assert tuple(model(torch.randn(2, 30)).shape) == (2,)


def test_the_stack_is_trainable():
    model = build_model({"architecture": "sequential_spec", "layers": DENSE}, input_dim=30)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = torch.randn(32, 30), torch.randint(0, 2, (32,)).float()
    first = last = None
    for _ in range(25):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert last < first


# ── Vocabulary ────────────────────────────────────────────────────────────


def test_every_layer_kind_declares_what_the_editor_needs():
    for name, kind in LAYER_KINDS.items():
        assert kind.name == name
        assert kind.doc, f"{name} has no description to show in the editor"
        assert kind.ranks, f"{name} declares no accepted input ranks"


def test_every_default_parameter_set_builds():
    # A layer dropped in from the palette with untouched defaults must work.
    shape_for_rank = {1: (30,), 2: (4, 16), 3: (3, 16, 16)}
    for name, kind in LAYER_KINDS.items():
        layers = [{"type": name, **kind.params}]
        shape = shape_for_rank[kind.ranks[0]]
        plans = plan_shapes(layers, shape)
        assert plans[0].ok, f"{name} with default params failed: {plans[0].error}"
        build_sequential(layers, shape)
