import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("jobs", [1, 2])
def test_real_managed_trials_overlap_and_keep_separate_artifacts(tmp_path, jobs):
    optuna = pytest.importorskip('optuna')
    project = Path(__file__).resolve().parents[1]
    settings = tmp_path / 'settings.toml'
    # Resource estimates allow two workers. The legacy count of one is ignored.
    # Hold both large-model reservations through completion, including the
    # portion this synthetic worker never allocates.
    settings.write_text('[training]\ncpu_percent=100\nmemory_percent=20\nmax_parallel=1\n')
    script = tmp_path / 'objective.py'
    script.write_text(
        'import hydra, time, os, json\nfrom pathlib import Path\n'
        'from src.app.services import resource_budget\n'
        'resource_budget.host_capacity=lambda: (8,32*resource_budget.GIB,32*resource_budget.GIB)\n'
        f'@hydra.main(version_base=None, config_path={str(project / "in/config")!r}, config_name="config")\n'
        'def main(cfg):\n'
        '    assert os.environ["THETA_TEST_ENV"] == "passed"\n'
        '    assert os.environ["OMP_NUM_THREADS"] == "1"\n'
        '    start=time.time()\n'
        '    time.sleep(7)\n'
        '    Path(f"trial_{cfg.theta_tuning.trial_number}.json").write_text(json.dumps('
        '{"pid":os.getpid(),"start":start,"end":time.time(),"version":cfg.theta_tuning.log_version}))\n'
        '    return float(cfg.x)\n'
        'if __name__ == "__main__": main()\n')
    storage = f'sqlite:///{tmp_path}/study.db'
    env = {**os.environ, 'PYTHONPATH': str(project), 'THETA_RESOURCE_POOL': str(tmp_path / 'resources.db'),
           'THETAIDE_SETTINGS_FILE': str(settings), 'THETA_MANAGED_SERIAL': '1', 'THETA_MANAGED_THREADS': '1',
           'OMP_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, str(script), '--multirun', 'hydra/sweeper=optuna_offline',
        f'hydra.sweeper.storage={storage}', 'hydra.sweeper.study_name=parallel', 'hydra.sweeper.n_trials=3',
        f'hydra.sweeper.n_jobs={jobs}', f'hydra.sweep.dir={tmp_path}/hydra', '++trainer.accelerator=cpu',
        '++model.hidden_sizes=[8000,8000]',
        '++hydra.job.env_set.THETA_TEST_ENV=passed', '++hydra.job.env_set.OMP_NUM_THREADS=64',
        '+x=interval(1.0,2.0)'], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    first = json.loads((tmp_path / 'trial_0.json').read_text())
    second = json.loads((tmp_path / 'trial_1.json').read_text())
    third = json.loads((tmp_path / 'trial_2.json').read_text())
    assert first['pid'] != second['pid']
    if jobs == 1:
        assert first['end'] <= second['start']
        assert second['end'] <= third['start']
    else:
        assert max(first['start'], second['start']) < min(first['end'], second['end'])
    assert first['version'] != second['version']
    # A free slot must be refilled while a slower sibling is still running.
    if jobs == 2:
        assert first['end'] <= third['start'] < second['end']
    study = optuna.load_study(study_name='parallel', storage=storage)
    assert len(study.trials) == 3
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert (tmp_path / 'hydra/0/.hydra/config.yaml').exists()
    assert (tmp_path / 'hydra/1/.hydra/config.yaml').exists()


def test_ordinary_training_uses_shared_pool_and_releases_lease(tmp_path, monkeypatch):
    import sqlite3
    from src.app.pipeline.datasets import run_experiment
    from src.app.services.resource_budget import GIB
    settings = tmp_path / 'settings.toml'
    settings.write_text('[training]\ncpu_percent=100\nmemory_percent=95\nmax_parallel=2\n')
    pool = tmp_path / 'resources.db'
    monkeypatch.setenv('THETA_RESOURCE_POOL', str(pool))
    monkeypatch.setenv('THETAIDE_SETTINGS_FILE', str(settings))
    monkeypatch.setenv('THETA_RESOURCE_SPEC', json.dumps({'key': 'ordinary', 'memory': .1*GIB, 'cpu': 1}))
    monkeypatch.setenv('THETA_RESOURCE_ACCELERATOR', '0')
    monkeypatch.setenv('THETA_MANAGED_THREADS', '1')
    script = tmp_path / 'ordinary.py'
    script.write_text('import os,sqlite3,time\ntime.sleep(.1)\n'
                      'db=sqlite3.connect(os.environ["THETA_RESOURCE_POOL"])\n'
                      'assert db.execute("SELECT COUNT(*) FROM tasks WHERE pid=?",(os.getpid(),)).fetchone()[0]==1\n'
                      'print("training finished",flush=True)\n')
    output = tmp_path / 'output.log'
    run_experiment(['agent=sb3/ppo', 'model=basics/mlp', '++model.hidden_sizes=[32,32]'],
                   script_entrypoint=str(script), python_executable=sys.executable, output_log=output)
    assert 'training finished' in output.read_text()
    with sqlite3.connect(pool) as db:
        assert db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0] == 0
