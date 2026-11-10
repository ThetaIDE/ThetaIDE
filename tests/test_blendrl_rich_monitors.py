"""Named BlendRL diagnostics, intervention semantics and method-scoped execution."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.app.core.provider_catalog import get_catalog
from src.app.monitoring.api import Context, ModelAccess, ProbeSet, write_snapshot


class KnownBlend(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.module_types = ['logic', 'neural']
        self.env = SimpleNamespace(get_action_meanings=lambda: ['left', 'noop', 'right'])
        self.response_calls = 0
        self.intervention_calls = 0

    def responses(self, obs):
        weight = obs[:, 0].sigmoid()
        weights = torch.stack((weight, 1-weight), 1)
        experts = obs.new_tensor([[[.9, .05, .05], [.1, .2, .7]]]).expand(len(obs), -1, -1)
        return weights, experts

    def get_action_probs(self, obs):
        weights, experts = self.responses(obs)
        return (weights[:, :, None]*experts).sum(1)

    def monitoring_interventions(self, obs):
        self.intervention_calls += 1
        _, experts = self.responses(obs)
        return {'uniform_gate': experts.mean(1), 'neural_removed_renormalized': experts[:, 0]}, {'gate': 'softmax'}

    def collect_monitoring(self, obs, *, include_experts=False):
        assert include_experts
        self.response_calls += 1
        weights, experts = self.responses(obs)
        return {}, {'weights': weights, 'expert_probabilities': experts,
                    'action_probabilities': self.get_action_probs(obs)}


def context(outputs=None):
    obs = np.array([[-2.], [-1.], [1.], [2.]], np.float32)
    return Context('blend', {'target': ''}, ModelAccess(KnownBlend(), 'get_action_probs', 'log_probabilities'),
                   ProbeSet(obs, np.array([11, 22, 33, 44]), {'sampling': 'fixed'}), 64, 4, 0,
                   outputs=outputs, lineage='known-blend')


def monitor():
    return get_catalog().load('monitor', 'blendrl-dev/reliance')(action=0, reference_action=2)


def test_routing_probabilities_and_signed_intervention_effects_have_named_axes():
    ctx = context()
    result = monitor().collect(ctx)
    assert set(result.visuals) == {'routing', 'intervention_effects', 'interventions'}
    np.testing.assert_allclose(result.arrays['routing_weights'].sum(1), 1.)
    np.testing.assert_allclose(result.arrays['policy_probabilities'].sum(1), 1.)
    assert result.visuals['routing']['layers'][0]['feature_ids'] == ['logic_0', 'neural_1']
    assert result.visuals['routing']['layers'][1]['feature_ids'] == ['left', 'noop', 'right']
    assert all(layer['value_range'] == [0., 1.] for layer in result.visuals['routing']['layers'])
    reference = ctx.access.scores(ctx.observations)
    variant = ctx.access.model.monitoring_interventions(ctx.observations)[0]['uniform_gate'].log()
    expected = (reference[:, 0]-reference[:, 2] - variant[:, 0]+variant[:, 2]).numpy()
    np.testing.assert_allclose(result.arrays['intervention_effects'][:, 0], expected, rtol=1e-6)
    graph = result.visuals['interventions']['graph']
    assert graph['edges'][0]['effect'] == pytest.approx(np.abs(expected).mean())
    assert all(edge['baseline'] == 'unmodified policy' for edge in graph['edges'])
    assert 'no held-out split' in graph['cohort']
    assert 'heldout_effect' not in graph['nodes'][0]
    assert 'left' in graph['nodes'][-1]['label'] and 'right' in graph['nodes'][-1]['label']


def test_scalar_only_collection_skips_extra_expert_capture_and_visual_arrays():
    ctx = context(frozenset({'mean_absolute_effect', 'max_action_flip_rate'}))
    result = monitor().collect(ctx)
    assert ctx.access.model.response_calls == 0
    assert not result.visuals
    assert 'routing_weights' not in result.arrays and 'intervention_effects' not in result.arrays
    assert result.metrics['mean_absolute_effect'] > 0


def test_recipe_preserves_dnn_and_scopes_blend_monitors_to_its_worker():
    from src.app.monitoring.config import instances, plain, validate_monitors
    from src.app.pipeline.commands import build_method_overrides
    from src.app.pipeline.compose import compose_experiment
    from src.app.pipeline.config import deep_merge, parse_methods_dict
    from src.app.telemetry.monitors import available_monitors
    cfg = compose_experiment('newMonitors/onlineRL').cfg
    validate_monitors(cfg)
    methods = parse_methods_dict(cfg)
    assert methods['PPO']['agent'] == 'sb3-dev/dqn'
    assert methods['PPO']['model'] == 'basics/dnn'
    blend = methods['BlendRL']
    assert blend['agent']['name'] == 'sb3-dev/ppo'
    args = build_method_overrides('BlendRL', blend, cfg=cfg)
    assert not any('agent.monitors' in arg for arg in args)
    worker = compose_experiment('newMonitors/onlineRL', [a for a in args if not a.startswith('+experiment=')]).cfg
    active = instances(worker)
    assert set(active) == {'blend_reliance', 'gate_activations', 'gate_network', 'neural_activations', 'neural_network'}
    assert worker.symbolic_state_loader.params.feature_count == 2
    dnn = instances(deep_merge(plain(cfg), methods['PPO']))
    assert 'blend_reliance' not in dnn and 'network' in dnn
    declared = {s.name: s.kind for s in available_monitors(plain(cfg))}
    assert declared['interp/blend_reliance/routing'] == 'matrix'
    assert declared['interp/blend_reliance/interventions'] == 'intervention_graph'


def test_rich_blend_snapshots_render_and_export_with_probability_units(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from frontend.live_run import LiveRun
    from frontend.monitor.visual_view import VisualsView
    from src.app.monitoring.report import write_report
    from src.app.telemetry.monitors import _catalog_outputs
    app = QApplication.instance() or QApplication([])
    ctx = context(); result = monitor().collect(ctx)
    package, capability = get_catalog().resolve('monitor', 'blendrl-dev/reliance')
    declarations = _catalog_outputs(package, capability.metadata['monitoring']['catalog'], {'instance': 'blend'}, {})
    by_key = {item['name'].rsplit('/', 1)[-1]: item for item in declarations}
    result.visuals = {key: {**by_key[key], 'data': value} for key, value in result.visuals.items()}
    path = write_snapshot(tmp_path, ctx, result, model_hash='known', provider={}, elapsed=0.)
    live = LiveRun()
    live.add([{'type': 'artifact', 'kind': 'monitor_snapshot', 'path': str(path), 'data': {
        'instance': 'blend', 'optimizer_updates': 4, 'transitions': 64,
        'visuals': [{k:v for k,v in visual.items() if k != 'data'} for visual in result.visuals.values()]}}])
    view = VisualsView(); view.show_lives({'BlendRL': live})
    view.output.setCurrentIndex(view.output.findData('interp/blend/routing'))
    text = '\n'.join(getattr(item, 'toPlainText', lambda: '')() for item in view.canvas.scene().items())
    assert 'probability' in text and 'fixed scale' in text
    assert 'logic_0' in text and 'neural_1' in text
    view.output.setCurrentIndex(view.output.findData('interp/blend/interventions'))
    assert 'unavailable' not in view.caption.text().lower()
    view.close()
    report = write_report(tmp_path).read_text()
    assert 'Expert routing and action probabilities' in report
    assert 'no held-out split' in report and 'unmodified policy' in report


def test_mountaincar_rules_own_their_valuation_and_use_two_observation_objects():
    from src.app.services.rulesets import resolve_ruleset
    from src.app.workspace import WorkspacePaths
    import runpy
    resource = resolve_ruleset(WorkspacePaths.for_process().root, 'mountaincar', 'mountaincar/monitoring_default')
    functions = runpy.run_path(str(resource.valuation_path))
    states = torch.tensor([[-.5, -.02], [-.5, 0.], [-.5, .02]])
    torch.testing.assert_close(functions['moving_left'](states), torch.tensor([.99, .01, .01]))
    torch.testing.assert_close(functions['moving_right'](states), torch.tensor([.01, .01, .99]))
    torch.testing.assert_close(functions['not_moving'](states), torch.tensor([.01, .99, .01]))
    from src.app.core.data_loaders import create_state_loader
    loader = create_state_loader({'symbolic_state_loader': {'name': 'blendrl-dev/symbolic_state',
                                 'params': {'object_count': 2, 'feature_count': 2}}})
    symbolic = loader(states)
    assert symbolic.shape == (3, 2, 2)
    torch.testing.assert_close(symbolic[:, 0], states)
    torch.testing.assert_close(symbolic[:, 1], states)


def test_actual_blendrl_training_is_identical_with_and_without_rich_monitors(tmp_path):
    import copy
    import gymnasium as gym
    from src.app.core.provider_loader import load_module
    from src.app.monitoring.runtime import MonitorManager
    from src.app.monitoring.sb3 import monitored_learn
    from src.app.pipeline.compose import compose_experiment
    from src.app.pipeline.config import parse_methods_dict
    from src.app.telemetry.run import Run
    method = parse_methods_dict(compose_experiment('newMonitors/onlineRL').cfg)['BlendRL']
    configuration = {'model': method['model'], 'symbolic_state_loader': method['symbolic_state_loader']}
    module = load_module(get_catalog().package('sb3-dev'), 'theta_sb3_policy.py')
    results = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for enabled in (False, True):
            env = gym.make('MountainCar-v0')
            try:
                agent = module.ThetaPPO(module.ThetaActorCriticPolicy, env, seed=19, n_steps=8, batch_size=4,
                    n_epochs=1, policy_kwargs={'theta_config': configuration, 'theta_model': 'blendrl-dev',
                    'env_name': 'mountaincar', 'action_meanings': ['left', 'noop', 'right']})
                monitors = {key: {**value, 'enabled': enabled, 'max_snapshots': 2, 'interval_updates': 1,
                            'probes': {'count': 4, 'steps': 8, 'seed': 0}}
                            for key, value in method['monitors'].items() if value.get('component')}
                cfg = {**configuration, 'monitors': monitors}
                with Run(tmp_path/str(enabled), [], config=cfg, system_interval=0) as run:
                    manager = MonitorManager(cfg, run, agent.policy.theta_model_module, 'ppo', lambda: gym.make('MountainCar-v0'))
                    monitored_learn(agent, manager, total_timesteps=16)
                    assert manager.closed
                results.append(copy.deepcopy(agent.policy.state_dict()))
            finally:
                env.close()
    finally:
        torch.set_num_threads(previous_threads)
    for key, value in results[0].items():
        assert torch.equal(value, results[1][key]), key
    assert len(list((tmp_path/'True/monitors').glob('*/snapshot_*.json'))) == 10


def test_routing_only_selection_skips_counterfactual_work():
    ctx = context(frozenset({'routing'}))
    result = monitor().collect(ctx)
    assert ctx.access.model.response_calls == 1
    assert ctx.access.model.intervention_calls == 0
    assert set(result.visuals) == {'routing'}
    assert not result.metrics
    assert not any(key.startswith('effect_') for key in result.arrays)
