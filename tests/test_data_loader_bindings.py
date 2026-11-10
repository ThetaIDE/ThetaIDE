"""Input ownership, per-method binding, and inference-time state conversion."""
import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from src.app.core.data_loaders import (
    create_state_loader, data_loader_errors, input_requirements, loader_selection,
    merge_loader_selection, state_loader_choices,
)
from src.app.pipeline.config import deep_merge, parse_methods_dict
from src.app.workspace import WorkspacePaths

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/usr/models/blendrl-dev'


@pytest.fixture
def loader_workspace(tmp_path, monkeypatch):
    paths = WorkspacePaths(tmp_path)
    package = paths.user('model') / 'state-model'
    shutil.copytree(SOURCE, package, ignore=shutil.ignore_patterns('__pycache__'))
    # Converters can be loaded without importing any reasoner or model module.
    meta = json.loads((package / '.theta_component.json').read_text())
    model_cap = next(c for c in meta['capabilities'] if c['kind'] == 'model')
    model_cap['id'] = 'state-model'
    model_cap['entry_point'] = 'never_construct.py:Model'
    meta['id'] = 'state-model'
    meta['payload']['implementations'].append('never_construct.py')
    meta['payload']['configs']['in/config/model/state-model.yaml'] = 'configs/model/state-model.yaml'
    shutil.copyfile(package / 'configs/model/blendrl-dev.yaml', package / 'configs/model/state-model.yaml')
    meta['config_paths'] = list(meta['payload']['configs'])
    (package / '.theta_component.json').write_text(json.dumps(meta))
    (package / 'never_construct.py').write_text('raise AssertionError("Model code must not be imported by a state loader")\n')
    monkeypatch.setenv('THETA_WORKSPACE', str(tmp_path))
    monkeypatch.setenv('THETA_DATA_DIR', str(tmp_path / '.thetaide'))
    return paths


def cfg(**extra):
    return {'paradigm': 'online_rl', 'model': {'name': 'state-model', 'actor_mode': 'hybrid'}, **extra}


def test_method_bindings_replace_provider_params_and_preserve_shared_data_policy(loader_workspace):
    base = cfg(data_module={'name': 'one', 'params': {'old': 3}},
               symbolic_state_loader={'name': 'blendrl-dev/cartpole_state', 'params': {'feature_count': 4}},
               dataset_path='/shared/data', seed=12, val_split=.2)
    method = {'data_module': {'name': 'two', 'params': {'new': 5}},
              'symbolic_state_loader': {'params': {'object_count': 3}}}
    selection, source = loader_selection(base, 'data_module', method, paths=loader_workspace)
    assert selection == {'name': 'two', 'params': {'new': 5}} and source == 'method'
    state, _ = loader_selection(base, 'symbolic_state_loader', method, paths=loader_workspace)
    assert state == {'name': 'blendrl-dev_symbolic_state', 'params': {'object_count': 3, 'feature_count': 4, 'builder': None}}
    assert base['symbolic_state_loader']['params'] == {'feature_count': 4}
    assert (base['dataset_path'], base['seed'], base['val_split']) == ('/shared/data', 12, .2)
    assert deep_merge(base, method)['data_module'] == method['data_module']
    assert merge_loader_selection(base['data_module'], None) is None


def test_requirements_follow_selected_model_mode_and_reject_wrong_role(loader_workspace):
    settings = cfg()
    assert any('requires symbolic_state_loader' in e for e in data_loader_errors(settings, paths=loader_workspace))
    settings['model']['actor_mode'] = 'neural'
    assert input_requirements(settings, paths=loader_workspace) == []
    settings['model']['blender'] = {'mode': 'symbolic'}
    assert input_requirements(settings, paths=loader_workspace)
    settings['symbolic_state_loader'] = 'SimulatorDataModule'
    assert data_loader_errors(settings, paths=loader_workspace)
    settings['symbolic_state_loader'] = 'blendrl-dev/cartpole_state'
    assert data_loader_errors(settings, paths=loader_workspace) == []
    settings['data_module'] = 'blendrl-dev/cartpole_state'
    assert any('supplies symbolic_state' in e for e in data_loader_errors(settings, paths=loader_workspace))
    settings.pop('data_module')
    settings['symbolic_state_loader'] = {'name': 'blendrl-dev/cartpole_state', 'params': {'object_count': 0}}
    assert any('minimum' in e for e in data_loader_errors(settings, paths=loader_workspace))


def test_state_loader_preserves_values_gradients_device_and_never_imports_model(loader_workspace):
    selected = cfg(symbolic_state_loader='blendrl-dev/cartpole_state')
    loader = create_state_loader(selected, paths=loader_workspace)
    state = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
    symbolic = loader(state)
    assert symbolic.shape == (1, 2, 4) and symbolic.device == state.device
    torch.testing.assert_close(symbolic[:, 0], state)
    symbolic.sum().backward()
    torch.testing.assert_close(state.grad, torch.full_like(state, 2.))
    torch.testing.assert_close(loader(symbolic), symbolic)
    with pytest.raises(ValueError, match='4 symbolic features'):
        loader(torch.zeros(2, 3))
    with pytest.raises(ValueError, match='batched'):
        loader(torch.zeros(4))
    with pytest.raises(ValueError, match='2 symbolic objects'):
        loader(torch.zeros(2, 1, 4))
    choices = state_loader_choices(paths=loader_workspace)
    assert {'blendrl-dev_symbolic_state', 'blendrl-dev/cartpole_state', 'blendrl-dev/symbolic_state'} <= choices.keys()


def test_environment_does_not_select_model_inputs(loader_workspace):
    selected = cfg(env={'data_module': 'bad', 'symbolic_observation_builder': 'bad:builder'})
    selection, source = loader_selection(selected, 'data_module', paths=loader_workspace)
    assert selection['name'] == 'SimulatorDataModule' and source == 'paradigm'
    assert any('requires symbolic_state_loader' in e for e in data_loader_errors(selected, paths=loader_workspace))
    with pytest.raises(ValueError, match='Select symbolic_state_loader'):
        create_state_loader(selected, paths=loader_workspace)


def test_method_inputs_reach_run_and_hydra_replaces_old_parameters(fixture_workspace, monkeypatch):
    from src.app.pipeline.commands import build_method_overrides
    from src.app.pipeline.compose import compose_experiment
    paths = WorkspacePaths(fixture_workspace)
    monkeypatch.setenv('THETA_WORKSPACE', str(fixture_workspace))
    (paths.config / 'agent/loader_probe.yaml').write_text('name: loader_probe\n')
    recipe = paths.config / 'experiment/thetaide/_loader_test.yaml'
    recipe.write_text('''# @package _global_
defaults:
  - thetaide/_base
data_module:
  name: first
  params: {old_parameter: 9}
symbolic_state_loader:
  name: state_first
  params: {old_state_parameter: 7}
methods:
  demo:
    agent: loader_probe
    model: basics/dnn
    data_module: {name: second, params: {batch_size: 16}}
    symbolic_state_loader: {name: state_second, params: {object_count: 2}}
''')
    composed = compose_experiment('thetaide/_loader_test', workspace=paths)
    method = parse_methods_dict(composed.cfg, paths=paths)['demo']
    args = build_method_overrides('demo', method, cfg=composed.cfg, workspace=paths, config_dir=paths.config)
    assert not any('agent.data_module' in a or 'agent.symbolic_state_loader' in a for a in args)
    trial = compose_experiment('thetaide/_loader_test', [a for a in args if not a.startswith('+experiment=')], workspace=paths)
    assert OmegaConf.to_container(trial.cfg.data_module) == {'name': 'second', 'params': {'batch_size': 16}}
    assert OmegaConf.to_container(trial.cfg.symbolic_state_loader) == {'name': 'state_second', 'params': {'object_count': 2}}
    # Explicit launch overrides retain the final say.
    args = build_method_overrides('demo', method, cfg=composed.cfg, workspace=paths, config_dir=paths.config,
                                 extra_args=['++symbolic_state_loader.params.object_count=3'])
    trial = compose_experiment('thetaide/_loader_test', [a for a in args if not a.startswith('+experiment=')], workspace=paths)
    assert trial.cfg.symbolic_state_loader.params.object_count == 3
    cli = compose_experiment('thetaide/_loader_test', [
        '++symbolic_state_loader.name=third', '++symbolic_state_loader.params.object_count=4',
    ], workspace=paths)
    assert OmegaConf.to_container(cli.cfg.symbolic_state_loader) == {'name': 'third', 'params': {'object_count': 4}}


@pytest.mark.parametrize('algorithm,actor_mode', [('ppo', 'hybrid'), ('ppo', 'logic'), ('ppo', 'neural'), ('dqn', 'hybrid'), ('dqn', 'neural'), ('dqn', 'logic')])
def test_actual_blendrl_keeps_construction_training_and_checkpoint_behavior(tmp_path, algorithm, actor_mode):
    import gymnasium as gym
    from stable_baselines3 import PPO, DQN
    from src.app.core.provider_loader import resolve_provider

    from types import SimpleNamespace
    from src.app.core.provider_loader import provider_object
    adapter = SimpleNamespace(
        ThetaActorCriticPolicy=provider_object('sb3-suite', 'theta_sb3_policy.py:ThetaActorCriticPolicy'),
        ThetaDQNPolicy=provider_object('sb3-suite', 'theta_sb3_policy.py:ThetaDQNPolicy'))
    selected_cfg = {
        'model': {'architecture': 'blendrl-dev', 'actor_mode': actor_mode, 'blend_function': 'softmax',
                  'neural': {'type': 'dnn', 'hidden_sizes': [16, 16]},
                  'symbolic': {'type': 'nsfr', 'rules': 'cartpole/variant2'},
                  'blender': {'mode': 'neural', 'model': {'name': 'basics/dnn', 'hidden_sizes': [8]}}},
        'symbolic_state_loader': 'blendrl-dev/cartpole_state', 'env': {},
    }
    if actor_mode == 'neural':
        selected_cfg.pop('symbolic_state_loader')
    env = gym.make('CartPole-v1')
    kwargs = {'theta_config': selected_cfg, 'theta_model': 'blendrl-dev', 'env_name': 'cartpole',
              'action_meanings': ['left', 'right']}
    if algorithm == 'ppo':
        model = PPO(adapter.ThetaActorCriticPolicy, env, n_steps=8, batch_size=4, n_epochs=1,
                    policy_kwargs=kwargs, seed=4)
        selected = model.policy.theta_model_module
    else:
        model = DQN(adapter.ThetaDQNPolicy, env, buffer_size=40, learning_starts=0, batch_size=4,
                    train_freq=1, policy_kwargs=kwargs, seed=4)
        selected = model.policy.q_net.theta_model_module
    try:
        assert isinstance(selected, resolve_provider('model', 'blendrl-dev'))
        assert selected.module_types == ['logic', 'neural']
        assert selected.actor.blender is selected.blender
        before = {name: value.detach().clone() for name, value in selected.named_parameters()}
        model.learn(total_timesteps=16)
        assert any(not torch.equal(before[name], value) for name, value in selected.named_parameters())
        state = torch.tensor([[0., 0., -.1, 0.]])
        expected = selected.get_action_probs(state).detach()
        checkpoint = tmp_path / 'policy.zip'
        model.save(checkpoint)
        restored = type(model).load(checkpoint, env=env)
        actual = restored.policy.theta_model_module if algorithm == 'ppo' else restored.policy.q_net.theta_model_module
        torch.testing.assert_close(actual.get_action_probs(state), expected)
        if actor_mode != 'neural':
            torch.testing.assert_close(actual.build_logic_observation(state), state.unsqueeze(1).expand(-1, 2, -1))
    finally:
        env.close()


def test_migration_is_explicit_and_keeps_model_construction_settings():
    from src.app.core.loader_migration import migrate_loader_config
    original = {'env': {'data_module': 'replay', 'symbolic_observation_builder': 'domain:convert'},
                'methods': {'blend': {'model': {'name': 'blendrl', 'blender': {'mode': 'neural'},
                                                'symbolic': {'type': 'nsfr', 'rules': 'custom'}}}}}
    with pytest.raises(ValueError, match='Choose a symbolic state provider'):
        migrate_loader_config(original)
    migrated, changed = migrate_loader_config(original, state_provider='test_state')
    assert migrated['env'] == {} and migrated['data_module'] == 'replay'
    assert migrated['symbolic_state_loader'] == {'name': 'test_state', 'params': {'builder': 'domain:convert', 'object_count': None}}
    assert migrated['methods'] == original['methods']
    assert len(changed) == 2
    assert original['env']['data_module'] == 'replay'
    assert merge_loader_selection('one', {'params': {'batch_size': 5}}) == {'name': 'one', 'params': {'batch_size': 5}}


def test_authoring_exposes_required_loader_and_uses_shared_preflight_status(loader_workspace, monkeypatch):
    from PyQt6.QtWidgets import QApplication, QLabel
    from frontend.config_model import ConfigTree
    from frontend.config_viewer import ConfigViewer
    import frontend.config_viewer as viewer_module

    app = QApplication.instance() or QApplication([])
    paths = loader_workspace
    paths.config.mkdir(parents=True, exist_ok=True)
    tree = ConfigTree(paths.config)
    monkeypatch.setattr(viewer_module, 'config_tree', lambda *a, **k: tree)
    recipe = paths.config / 'recipe.yaml'
    values = {'paradigm': 'online_rl', 'methods': {'policy': {'agent': 'ppo', 'model': 'state-model'}}}
    recipe.write_text(yaml.safe_dump(values))
    widget = ConfigViewer()
    widget.load_file(recipe, 'experiment/probe/recipe.yaml')
    try:
        assert any('requires symbolic_state_loader' in w.text()
                   for w in widget.findChildren(QLabel, 'dataLoaderStatus'))
        assert widget.raw_data == values and not widget.is_dirty
        combo = widget.field_widgets['methods.policy.symbolic_state_loader']
        combo.setCurrentIndex(combo.findData('blendrl-dev/cartpole_state'))
        app.processEvents()
        assert not widget.findChildren(QLabel, 'dataLoaderStatus')
        assert widget.raw_data['methods']['policy']['symbolic_state_loader']['name'] == 'blendrl-dev/cartpole_state'
        assert widget.save_to_disk()
        assert yaml.safe_load(recipe.read_text())['methods']['policy']['symbolic_state_loader']['params'] == {
            'object_count': 2, 'feature_count': 4, 'builder': None}
    finally:
        widget.close()


def test_algorithms_can_require_different_training_contracts(loader_workspace):
    paths = loader_workspace
    manifest = paths.user('model') / 'state-model/.theta_component.json'
    meta = json.loads(manifest.read_text())
    for provider, output in [('transition_loader', 'rl_transitions/v1'), ('sequence_loader', 'trajectories/v1')]:
        meta['capabilities'].append({'id': provider, 'kind': 'datamodule', 'role': 'training',
                                     'entry_point': 'never_construct.py:Model', 'output_contracts': [output]})
    meta['capabilities'].append({'id': 'sequence_method', 'kind': 'algorithm', 'entry_point': 'never_construct.py:Model',
                                'input_requirements': [{'slot': 'data_module', 'contract': 'trajectories/v1'}]})
    manifest.write_text(json.dumps(meta))
    settings = {'paradigm': 'offline_rl', 'data_module': 'transition_loader'}
    method = {'agent': 'sequence_method'}
    assert any('does not produce' in e for e in data_loader_errors(settings, method, paths=paths))
    method['data_module'] = 'sequence_loader'
    assert data_loader_errors(settings, method, paths=paths) == []


def test_export_traces_method_state_loader_profiles(loader_workspace):
    from src.app.services.experiment_exporter import ExperimentExporter
    files = {}
    ExperimentExporter(loader_workspace.root)._trace_data_modules('probe', None, files, raw_cfg={
        'methods': {'hybrid': {'symbolic_state_loader': 'blendrl-dev/cartpole_state'}}})
    assert any(path.name == 'symbolic_state.py' for path in files.values())
    assert any(path.name == 'cartpole_state.yaml' for path in files.values())


def test_neural_actor_passes_loaded_states_to_its_symbolic_gate():
    from types import SimpleNamespace
    import inspect
    from src.app.core.provider_loader import resolve_provider
    actor_class = inspect.getmodule(resolve_provider('model', 'blendrl-dev')).BlenderActor
    actor = actor_class.__new__(actor_class)
    torch.nn.Module.__init__(actor)
    actor.actor_mode = 'neural'
    actor.blender = torch.nn.Identity()
    actor.env = SimpleNamespace(n_actions=2)
    actor.module_types = ['neural']

    class Expert(torch.nn.Module):
        def get_action_probs(self, state):
            return state.new_tensor([[.4, .6]]).expand(state.shape[0], -1)

    actor.policy_modules = torch.nn.ModuleList([Expert()])
    seen = []
    actor.to_blender_policy_distribution = lambda observation, symbolic: seen.append(symbolic) or observation.new_ones((observation.shape[0], 1))
    symbolic = torch.randn(2, 2, 4)
    probabilities, _ = actor(torch.zeros(2, 4), symbolic)
    assert seen[0] is symbolic
    torch.testing.assert_close(probabilities, torch.tensor([[.4, .6], [.4, .6]]))
