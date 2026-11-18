import math
import pytest
import torch

from src.app.core.provider_catalog import get_catalog
from src.app.core.provider_loader import load_module

_package = get_catalog().package("blendrl-dev")
monitoring = load_module(_package, "monitoring.py")


def test_routing_catalog_explains_dynamic_experts_pairs_and_soft_gates():
    catalog = load_module(_package, "monitor_catalog.py")
    rows = catalog.describe({"modules": [{"type": "neural"}, {"type": "nsfr"}, {"type": "nsfr"}],
                             "blend_function": "gumbel_softmax"}, {})
    descriptions = {row["name"]: row["description"] for row in rows}
    assert all(descriptions.values())
    assert "Jensen-Shannon" in descriptions["blend/eval/agreement/logic_1__logic_2/js_divergence"]
    assert "all 2 logic experts" in descriptions["blend/train/soft_gate/logic/weight_mean"]
    assert "90%" in descriptions["blend/train/logic_2/exclusive_share"]
    assert "symbolic expert" in descriptions["blend/train/logic_2/exclusive_share"]
    labels = {row["name"]: row["label"] for row in rows}
    assert labels["blend/train/neural/weight_mean"] == "Neural expert · mean weight"
    assert labels["blend/train/agreement/neural__logic_1/top_action"] == "Neural vs logic 1 · top-action agreement"
    assert "blend/train/neural_0/weight_mean" not in descriptions


def test_single_expert_per_type_is_named_by_type():
    catalog = load_module(_package, "monitor_catalog.py")
    assert catalog.expert_names(["logic", "neural"]) == ["logic", "neural"]
    assert catalog.expert_names(["neural", "logic", "logic"]) == ["neural", "logic_1", "logic_2"]
    w = torch.tensor([[.25, .75], [.75, .25]])
    m = monitoring.routing_metrics(w, ["logic", "neural"])
    assert m["blend/logic/weight_mean"] == pytest.approx(.5)
    assert m["blend/neural/dominant_share"] == pytest.approx(.5)
    declared = {row["name"] for row in catalog.describe({"symbolic": {"type": "nsfr"}, "neural": {"name": "dnn"}}, {})}
    assert {key.replace("blend/", "blend/train/", 1) for key in m} <= declared


def test_multi_expert_weights_and_agreement():
    w = torch.tensor([[1., 0., 0.], [0., .5, .5]])
    p = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]], [[0., 1.], [1., 0.], [1., 0.]]])
    m = monitoring.routing_metrics(w, ['neural', 'logic', 'logic'], expert_probs=p)
    assert m['blend/neural/weight_mean'] == pytest.approx(.5)
    assert m['blend/logic/weight_mean'] == pytest.approx(.5)
    assert m['blend/mixed_share'] == pytest.approx(.5)
    assert m['blend/entropy_normalized'] == pytest.approx(math.log(2) / (2 * math.log(3)))
    assert m['blend/agreement/neural__logic_1/top_action'] == pytest.approx(.5)
    assert m['blend/agreement/neural__logic_1/js_divergence'] == pytest.approx(math.log(2) / 2)


def test_single_module_and_invalid_inputs():
    assert monitoring.routing_metrics(torch.ones(2, 1), ['logic'])['blend/entropy_normalized'] == 0
    with pytest.raises(ValueError):
        monitoring.routing_metrics(torch.tensor([[.5, .6]]), ['logic', 'neural'])
    with pytest.raises(ValueError):
        monitoring.routing_metrics(torch.tensor([[1., 0.]]), ['logic', 'neural'], threshold=.5)


BlendRoutingPlotter = load_module(_package, "plot/blendrl/routing.py").BlendRoutingPlotter
import numpy as np

def test_sample_plots_with_optional_metadata(tmp_path):
    rng = np.random.default_rng(3)
    states = rng.normal(size=(50, 4))
    weight = rng.uniform(size=(50, 2))
    weight /= weight.sum(axis=1, keepdims=True)
    path = tmp_path / 'sample.npz'
    np.savez(path, states=states, weights=weight, module_types=['logic', 'neural'], expert_names=['logic', 'neural'],
             cohort=np.where(states[:, 0] > 0, 'high', 'low'), distance=np.abs(states[:, 0]))
    BlendRoutingPlotter.plot_samples(path, tmp_path / 'plots', {})
    for name in ('authority.png', 'weights.png', 'state_0.png', 'state_1.png', 'cohort.png', 'distance.png'):
        assert (tmp_path / 'plots' / name).stat().st_size > 0


def test_sample_plots_reject_files_without_expert_names(tmp_path):
    path = tmp_path / 'old.npz'
    np.savez(path, states=np.zeros((4, 2)), weights=np.full((4, 2), .5), module_types=['logic', 'neural'])
    with pytest.raises(ValueError, match='expert_names'):
        BlendRoutingPlotter.plot_samples(path, tmp_path / 'plots', {})
