"""Fresh-workspace execution uses explicit editable providers, without Hub builds."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(importlib.util.find_spec('stable_baselines3') is None, reason='SB3 is optional')
def test_fresh_editable_sb3_runs_selected_model_and_reloads(tmp_path):
    project = Path(__file__).resolve().parents[1]
    from src.app.core.provider_catalog import get_catalog
    dnn = get_catalog().package('dnn')
    if dnn is None:
        pytest.skip('DNN source fixture is not installed')
    workspace = tmp_path / 'workspace'
    package = workspace / 'src/usr/algorithms/sb3-dev'
    shutil.copytree(project / 'src/usr/algorithms/sb3-dev', package, ignore=shutil.ignore_patterns('__pycache__'))
    config = workspace / 'in/config'
    shutil.copytree(project / 'src/app/config_defaults', config)
    model = workspace / 'src/usr/models/dnn-test-dev'
    model.mkdir(parents=True)
    # Read source to create a distinct editable fixture; never patch an installed payload.
    source = (dnn.root / 'dnn_model.py').read_text().replace('@register_model("dnn")', '@register_model("dnn-test-dev")')
    (model / 'model.py').write_text(source)
    (model / '.theta_component.json').write_text(json.dumps({
        'id': 'dnn-test-dev', 'name': 'DNN test fixture', 'kind': 'model', 'version': '0.1.0',
        'storage_version': 2, 'api_version': 1, 'origin': 'user',
        'parent': {'id': dnn.id, 'version': dnn.version, 'origin': dnn.origin},
        'capabilities': [{'id': 'dnn-test-dev', 'kind': 'model', 'entry_point': 'model.py:DNNModel'}],
        'payload': {'implementations': ['model.py'], 'resources': [], 'configs': {}}}))
    (config / 'model').mkdir(exist_ok=True)
    (config / 'model/dnn-test-dev.yaml').write_text('architecture: dnn-test-dev\nhidden_sizes: [16,16]\n')
    (config / 'experiment').mkdir(exist_ok=True)
    (config / 'experiment/local.yaml').write_text(
        '# @package _global_\nparadigm: online_rl\ngroup: g\nexperiment_id: e\n'
        'total_timesteps: 16\neval_episodes: 1\nenv: {env_id: CartPole-v1, num_envs: 1}\n')
    environment = {**os.environ, 'PYTHONPATH': str(project), 'THETA_WORKSPACE': str(workspace),
                   'THETA_DATA_DIR': str(tmp_path / 'state'), 'OMP_NUM_THREADS': '1'}
    environment.pop('THETA_PROVIDER_SNAPSHOT', None)
    command = [sys.executable, '-m', 'src.app.pipeline.provider_worker', 'src/app/train.py',
               '+experiment=local', 'agent=sb3-dev/ppo', 'model=dnn-test-dev',
               '++agent.name=m',
               'agent.n_steps=8', 'agent.batch_size=4', 'agent.n_epochs=1',
               '++model.architecture=dnn-test-dev', '++trainer.accelerator=cpu']
    result = subprocess.run(command, cwd=workspace, env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(next(workspace.rglob('execution.json')).read_text())
    assert record['runner'] == 'OnlineRLRunner' and record['status'] == 'completed'
    assert record['metadata']['algorithm']['id'] == 'sb3-dev/ppo'
    assert record['metadata']['backend'] == 'sb3'
    checkpoint = next(workspace.rglob('latest_model.zip'))
    reload_code = (
        'from src.app.core.provider_loader import install_namespace_finder; install_namespace_finder(); '
        'from stable_baselines3 import PPO; import sys; '
        'model=PPO.load(sys.argv[1]); '
        'assert model.policy.theta_model_module.network[0].out_features == 16; '
        'import numpy as np; model.predict(np.zeros(4,dtype="float32"))')
    reload = subprocess.run([sys.executable, '-c', reload_code, str(checkpoint)], cwd=workspace,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert reload.returncode == 0, reload.stdout + reload.stderr
    assert next(workspace.glob('results/logs/g/e/m/*/metrics.csv')).is_file()
