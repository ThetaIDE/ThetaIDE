import math
import pytest
import torch

from src.usr.models.blendrl import monitoring


def test_multi_expert_weights_and_agreement():
    w = torch.tensor([[1., 0., 0.], [0., .5, .5]])
    p = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]], [[0., 1.], [1., 0.], [1., 0.]]])
    m = monitoring.routing_metrics(w, ['neural', 'logic', 'logic'], expert_probs=p)
    assert m['blend/neural/weight_mean'] == pytest.approx(.5)
    assert m['blend/logic/weight_mean'] == pytest.approx(.5)
    assert m['blend/mixed_share'] == pytest.approx(.5)
    assert m['blend/entropy_normalized'] == pytest.approx(math.log(2) / (2 * math.log(3)))
    assert m['blend/agreement/neural_0__logic_1/top_action'] == pytest.approx(.5)
    assert m['blend/agreement/neural_0__logic_1/js_divergence'] == pytest.approx(math.log(2) / 2)


def test_single_module_and_invalid_inputs():
    assert monitoring.routing_metrics(torch.ones(2, 1), ['logic'])['blend/entropy_normalized'] == 0
    with pytest.raises(ValueError):
        monitoring.routing_metrics(torch.tensor([[.5, .6]]), ['logic', 'neural'])
    with pytest.raises(ValueError):
        monitoring.routing_metrics(torch.tensor([[1., 0.]]), ['logic', 'neural'], threshold=.5)


from plot.blendrl.routing import BlendRoutingPlotter
import numpy as np

def test_sample_plots_with_optional_metadata(tmp_path):
    rng = np.random.default_rng(3)
    states = rng.normal(size=(50, 4))
    weight = rng.uniform(size=(50, 2))
    weight /= weight.sum(axis=1, keepdims=True)
    path = tmp_path / 'sample.npz'
    np.savez(path, states=states, weights=weight, module_types=['logic', 'neural'],
             cohort=np.where(states[:, 0] > 0, 'high', 'low'), distance=np.abs(states[:, 0]))
    BlendRoutingPlotter.plot_samples(path, tmp_path / 'plots', {})
    for name in ('authority.png', 'weights.png', 'state_0.png', 'state_1.png', 'cohort.png', 'distance.png'):
        assert (tmp_path / 'plots' / name).stat().st_size > 0
