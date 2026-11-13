import time

import pytest
from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication

from frontend.runner_manager import JobManager
from src.app.services.resource_budget import GIB, ResourceBudget, managed_environment


def capacity(**changes):
    return dict(cores=8, total=32 * GIB, available=24 * GIB, active_memory=[], **changes)


def test_ram_admission_reserves_growth_of_existing_jobs():
    budget = ResourceBudget(memory_gb_per_job=8)
    # An active job uses 16 GiB, with up to 20 GiB estimated demand.
    values = capacity()
    values.update(available=16 * GIB, active_memory=[16 * GIB], reservations=[20 * GIB])
    assert 'RAM' in budget.admission_reason(**values)
    values['reservations'] = [16 * GIB]
    assert budget.admission_reason(**values) == ''


def test_cpu_admission_respects_allocations_before_a_settings_reduction():
    budget = ResourceBudget(cpu_percent=50, threads_per_job=1)
    values = capacity()
    values.update(active_memory=[GIB], active_threads=[4])
    assert 'CPU' in budget.admission_reason(**values)


def test_accelerator_workers_share_capacity_instead_of_using_a_count_limit():
    budget = ResourceBudget()
    assert budget.admission_reason(**capacity(), accelerator=True, active_accelerators=True) == ''
    assert budget.admission_reason(**capacity(), accelerator=False, active_accelerators=True) == ''


def test_bad_settings_are_bounded_and_small_cpu_targets_keep_one_thread():
    budget = ResourceBudget.from_settings({'cpu_percent': -1, 'memory_percent': 200,
                                         'threads_per_job': 'bad', 'memory_gb_per_job': float('nan')})
    assert budget.cpu_percent == 1 and budget.memory_percent == 95
    assert budget.memory_gb_per_job == .75 and budget.threads(1) == 1


@pytest.fixture
def runner(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    manager = JobManager(store_path=tmp_path / 'jobs.db', project_root=tmp_path,
                         logs_dir=tmp_path / 'logs', settings_path=tmp_path / 'settings.toml')
    monkeypatch.setattr('frontend.runner_manager.host_capacity', lambda: (8, 32 * GIB, 24 * GIB))
    monkeypatch.setattr('frontend.runner_manager.tree_memory', lambda pid: GIB)
    monkeypatch.setattr('src.app.services.resource_budget.psutil.cpu_percent', lambda *a: 0)
    manager.max_parallel = 1
    yield manager
    manager.queue_running = False
    for process in list(manager._processes.values()):
        process.kill()
        process.waitForFinished(3000)


def plan(name, accelerator=False):
    return {'experiment': name, 'group': 'g', 'experiment_id': name, 'agents': ['m'],
            'total_timesteps': 1, 'effective_timesteps': 1, 'run_in_parallel': True,
            'local': True, 'accelerator': accelerator}


def fake_start(manager, job):
    job.update(status='running', started=time.time())
    manager._processes[job['job_id']] = QProcess(manager)


def test_coordinators_do_not_consume_training_worker_slots(runner, monkeypatch):
    runner.settings_path.write_text('[training]\nmax_parallel=1\n')
    monkeypatch.setattr(runner, '_start', lambda job: fake_start(runner, job))
    first = runner.launch('a', _plan=plan('a'))
    second = runner.launch('b', _plan=plan('b'))
    assert first['status'] == 'running'
    assert second['status'] == 'running'
    assert runner.running_count() == 2


def test_queue_waits_for_ram_then_retries_without_completion(runner, monkeypatch):
    monkeypatch.setattr(runner, '_start', lambda job: fake_start(runner, job))
    runner.settings_path.write_text('[training]\nmemory_percent=25\n')
    monkeypatch.setattr(runner.resource_pool, '_sample', lambda db: [{'rss': 9.55 * GIB, 'memory': 9.55 * GIB, 'cpu': 1, 'accelerator': False}])
    runner.launch('a', queue=True, _plan=plan('a'))
    runner.start_queue()
    assert runner.running_count() == 0 and 'RAM' in runner.queue_state()['waiting_reason']
    monkeypatch.setattr(runner.resource_pool, '_sample', lambda db: [])
    runner.resource_tick()
    assert runner.running_count() == 1 and not runner.queue_order


def test_multiple_accelerator_workers_are_admitted_when_estimates_fit(runner, monkeypatch):
    spec = {'key': 'gpu', 'memory': .1 * GIB, 'cpu': 1}
    token, reason = runner.resource_pool.acquire(spec, accelerator=True, capacity=(8,32*GIB,24*GIB))
    assert token and not reason
    # Warmup is separate from the accelerator restriction.
    with runner.resource_pool.connect() as db:
        db.execute('UPDATE tasks SET created=created-10')
    token2, reason = runner.resource_pool.acquire(spec, accelerator=True, capacity=(8,32*GIB,24*GIB))
    assert token2 and not reason
    runner.resource_pool.release(token2)
    runner.resource_pool.release(token)


def test_persisted_budget_changes_are_loaded_by_detached_runner(runner, monkeypatch):
    runner.settings_path.write_text('[training]\nmax_parallel = 3\ncpu_percent = 50\nthreads_per_job = 4\n')
    runner.resource_tick()
    assert runner.max_parallel == 3 and runner.resource_budget.threads(8) == 4


def test_host_memory_pressure_does_not_stop_research_or_pause_queue(runner, monkeypatch):
    monkeypatch.setattr(runner, '_start', lambda job: fake_start(runner, job))
    first = runner.launch('a', _plan=plan('a'))
    runner.launch('b', queue=True, _plan=plan('b'))
    runner.queue_running = True
    monkeypatch.setattr('frontend.runner_manager.host_capacity', lambda: (8, 32 * GIB, .5 * GIB))
    stopped = []
    monkeypatch.setattr(runner, 'cancel', lambda key: stopped.append(key))
    for _ in range(3):
        runner._last_resource_tick = 0
        runner.resource_tick()
    assert stopped == [] and runner.queue_running
    assert runner.running_count() == 2


def test_real_child_receives_managed_thread_environment(runner, monkeypatch):
    monkeypatch.setattr('src.app.workspace.PROJECT_ROOT', runner.project_root)
    (runner.project_root / 'run_pipeline.py').write_text(
        'import os\nprint(os.environ["OMP_NUM_THREADS"], os.environ["THETA_MANAGED_SERIAL"], flush=True)\n')
    job = runner.launch('a', _plan=plan('a'))
    process = runner._processes[job['job_id']]
    process.waitForFinished(5000)
    QApplication.processEvents()
    assert runner.jobs[job['job_id']]['status'] == 'completed'
    assert '2 1' in (runner.jobs_dir / (job['job_id'] + '.log')).read_text()
    assert managed_environment(2)['OPENBLAS_NUM_THREADS'] == '2'


def test_settings_controls_persist_and_update_runner(runner):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QWidget
    from frontend.settings import SettingsManager
    from frontend.settings_view import SettingsView

    settings = SettingsManager(runner.settings_path.parent / 'preferences')
    runner.settings_path = settings.workspace_settings_path
    view = SettingsView.__new__(SettingsView)
    QWidget.__init__(view)
    view.window = SimpleNamespace(settings_manager=settings)
    card = view._build_resource_budget_card()
    view.resource_controls['cpu_percent'].setValue(50)
    assert 'memory_gb_per_job' not in view.resource_controls
    view.resource_controls['threads_per_job'].setValue(3)
    view.resource_controls['memory_percent'].setValue(60)
    view.resource_controls['max_parallel'].setValue(2)
    runner.resource_tick()
    assert runner.resource_budget.cpu_percent == 50
    assert runner.resource_budget.memory_percent == 60
    assert runner.resource_budget.threads_per_job == 3 and runner.max_parallel == 2
    assert 'Theta RAM target' in view.resource_budget_summary.text()
    card.close()


def test_site_environment_cannot_override_managed_thread_limits(monkeypatch):
    from types import SimpleNamespace
    from src.app.pipeline.runtime import get_subprocess_env
    monkeypatch.setenv('THETA_MANAGED_THREADS', '2')
    env = get_subprocess_env(SimpleNamespace(extra_env_vars={'OMP_NUM_THREADS': '64', 'THETA_MANAGED_SERIAL': '0'}))
    assert env['OMP_NUM_THREADS'] == '2' and env['THETA_MANAGED_SERIAL'] == '1'


def test_managed_method_pool_ignores_old_count_limit(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from omegaconf import OmegaConf
    from src.app.pipeline import local_runner
    cfg = OmegaConf.create({'run_in_parallel': True})
    ctx = {'methods': {str(i): {} for i in range(5)}}
    calls = []
    sizes = []
    settings = tmp_path / 'settings.toml'
    settings.write_text('[training]\nmax_parallel=2\n')
    monkeypatch.setenv('THETA_RESOURCE_POOL', str(tmp_path / 'pool.db'))
    monkeypatch.setenv('THETAIDE_SETTINGS_FILE', str(settings))
    monkeypatch.setattr(local_runner, 'require_runnable_tuning', lambda *a: None)
    monkeypatch.setattr(local_runner, '_run_methods_sequential', lambda cfg, context: calls.append(list(context['methods'])))
    def pool(**kwargs):
        sizes.append(kwargs['max_workers'])
        return ThreadPoolExecutor(**kwargs)
    monkeypatch.setattr(local_runner, 'ThreadPoolExecutor', pool)
    local_runner.run_methods(cfg, ctx)
    assert sizes == [5] and sorted(calls) == [[str(i)] for i in range(5)]


def test_managed_optuna_batches_are_single_trials(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    optuna = pytest.importorskip('optuna')
    from src.app.tuners.optuna.sweeper import ThetaOptunaSweeper
    batches = []
    def launch(jobs, initial_job_idx):
        batches.append(len(jobs))
        return [SimpleNamespace(return_value=.5) for job in jobs]
    sweeper = ThetaOptunaSweeper(optuna.samplers.RandomSampler(), 'minimize',
                                 f'sqlite:///{tmp_path}/study.db', 'managed', 3, 8, None)
    sweeper.impl.config = OmegaConf.create({'hydra': {'sweep': {'dir': str(tmp_path)}}})
    sweeper.impl.launcher = SimpleNamespace(launch=launch)
    monkeypatch.setenv('THETA_MANAGED_SERIAL', '1')
    sweeper.sweep(['+x=interval(1.0,2.0)'])
    assert batches == [1, 1, 1]


def test_local_distributed_launch_is_rejected_before_process_start(monkeypatch):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from src.app.services import config_service
    composed = SimpleNamespace(cfg=OmegaConf.create({'trainer': {'devices': 2}}))
    monkeypatch.setattr(config_service, 'compose_experiment', lambda *a: composed)
    monkeypatch.setattr(config_service, 'validate_composed', lambda c: None)
    monkeypatch.setattr(config_service, 'method_plans', lambda c: {})
    with pytest.raises(config_service.LaunchError, match='one training device'):
        config_service.prepare_launch('x')


def test_five_percentage_point_ram_overshoot_does_not_cancel_work(runner, monkeypatch):
    monkeypatch.setattr(runner, '_start', lambda job: fake_start(runner, job))
    runner.launch('a', _plan=plan('a'))
    runner.queue_running = True
    runner.settings_path.write_text('[training]\nmemory_percent=50\n')
    monkeypatch.setattr('frontend.runner_manager.host_capacity', lambda: (8, 32 * GIB, 14.4 * GIB))
    monkeypatch.setattr(runner, 'cancel', lambda key: pytest.fail('A soft target must not kill work'))
    for _ in range(5):
        runner._last_resource_tick = 0
        runner.resource_tick()
    assert runner.queue_running


def test_estimates_vary_by_model_and_learn_observed_peak(tmp_path, monkeypatch):
    from src.app.services.resource_budget import ResourcePool, workload_estimate
    monkeypatch.setattr('src.app.services.resource_budget.psutil.cpu_percent', lambda *a: 0)
    small = workload_estimate({'model': {'hidden_sizes': [32, 32]}})
    large = workload_estimate({'model': {'hidden_sizes': [4096, 4096]}})
    assert large['memory'] > small['memory'] and large['key'] != small['key']
    pool = ResourcePool(tmp_path / 'usage.db')
    with pool.connect() as db:
        db.execute('INSERT INTO profiles VALUES(?,?,?,?)', (small['key'], 2*GIB, .7, 10))
    memory, cpu, samples = pool.estimate(small)
    assert memory == pytest.approx(2.3 * GIB) and cpu == pytest.approx(.77) and samples == 10


def test_old_manual_memory_setting_is_ignored():
    a = ResourceBudget.from_settings({'memory_gb_per_job': 1})
    b = ResourceBudget.from_settings({'memory_gb_per_job': 64})
    assert a == b


def test_other_apps_do_not_consume_theta_budget_or_gate_admission():
    budget = ResourceBudget(memory_percent=50)
    values = capacity()
    values.update(available=8 * GIB, next_memory=GIB)
    # Other apps use 75% of RAM; Theta can still use physically available room.
    assert budget.admission_reason(**values) == ''
    values['available'] = GIB
    assert budget.admission_reason(**values) == ''


def test_shared_pool_admission_is_atomic_across_callers(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import psutil
    from src.app.services.resource_budget import ResourcePool
    monkeypatch.setattr(psutil, 'cpu_percent', lambda *a: 0)
    monkeypatch.setattr(psutil.Process, 'children', lambda *a, **kw: [])
    settings = tmp_path / 'settings.toml'
    settings.write_text('[training]\nmax_parallel=1\ncpu_percent=100\n')
    pool = ResourcePool(tmp_path / 'resources.db', settings)
    spec = {'key': 'same-workload', 'memory': 20 * GIB, 'minimum_memory': 20 * GIB, 'cpu': 1}
    with pool.connect() as db:
        db.execute('INSERT INTO profiles VALUES(?,?,?,?)', (spec['key'], 20*GIB, 1, 10))
    def acquire():
        return ResourcePool(pool.path, settings).acquire(spec, capacity=(8,32*GIB,31*GIB))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: acquire(), range(2)))
    tokens = [token for token, _ in results if token]
    assert len(tokens) == 1
    pool.release(tokens[0])


def test_small_tasks_respect_workspace_worker_limit(tmp_path, monkeypatch):
    import psutil
    from src.app.services.resource_budget import ResourcePool
    monkeypatch.setattr(psutil, 'cpu_percent', lambda *a: 0)
    monkeypatch.setattr(psutil.Process, 'children', lambda *a, **kw: [])
    settings = tmp_path / 'settings.toml'
    settings.write_text('[training]\nmax_parallel=1\ncpu_percent=75\n')
    pool = ResourcePool(tmp_path / 'resources.db', settings)
    spec = {'key': 'small', 'memory': .1*GIB, 'cpu': .25}
    with pool.connect() as db:
        db.execute('INSERT INTO profiles VALUES(?,?,?,?)', (spec['key'], .1*GIB, .25, 10))
    tokens = []
    for i in range(5):
        token, reason = pool.acquire(spec, capacity=(8,32*GIB,31*GIB))
        if i == 0:
            assert token and not reason
            tokens.append(token)
        else:
            assert token is None and 'worker limit' in reason
    for token in tokens:
        pool.release(token)


def test_waiting_for_capacity_can_be_cancelled(tmp_path, monkeypatch):
    from src.app.services.resource_budget import ResourcePool
    from src.app.services.cancellation import TrainingCancelled
    pool = ResourcePool(tmp_path / 'resources.db')
    monkeypatch.setattr(pool, 'acquire', lambda *a, **kw: (None, 'Waiting for RAM'))
    monkeypatch.setattr('src.app.services.resource_budget.time.sleep', lambda duration: None)
    calls = []
    def check():
        calls.append(1)
        if len(calls) == 2:
            raise TrainingCancelled()
    with pytest.raises(TrainingCancelled):
        with pool.slot({}, check_cancel=check):
            pytest.fail('A cancelled task must never start')


def test_pause_is_shared_with_trial_and_method_admission(tmp_path, monkeypatch):
    import psutil
    from src.app.services.resource_budget import ResourcePool
    monkeypatch.setattr(psutil, 'cpu_percent', lambda *a: 0)
    monkeypatch.setattr(psutil.Process, 'children', lambda *a, **kw: [])
    pool = ResourcePool(tmp_path / 'usage.db')
    worker = ResourcePool(pool.path)
    spec = {'key': 'trial', 'memory': .1 * GIB, 'cpu': 1}
    pool.set_paused(True)
    token, reason = worker.acquire(spec, capacity=(8,32*GIB,24*GIB))
    assert token is None and 'paused' in reason
    pool.set_paused(False)
    token, reason = worker.acquire(spec, capacity=(8,32*GIB,24*GIB))
    assert token and not reason
    worker.release(token)


def test_runs_do_not_inherit_the_mcp_servers_token(runner, monkeypatch):
    import frontend.runner_manager as runner_module

    monkeypatch.setenv('THETA_MCP_URL', 'http://127.0.0.1:1/mcp')
    monkeypatch.setenv('THETA_MCP_TOKEN', 'secret')
    seen = []

    class Recording(runner_module.QProcess):
        def setProcessEnvironment(self, env):  # noqa: N802
            seen.append(env)

        def start(self, *args):  # nothing is launched: only the environment matters here
            pass

        def waitForStarted(self, *args):  # noqa: N802
            return True

    monkeypatch.setattr(runner_module, 'QProcess', Recording)
    runner.python = 'true'
    runner._start({'job_id': 'mcp-env', 'experiment': 'x', 'group': 'g', 'experiment_id': 'x', 'request': {'local': False}})
    runner._processes.pop('mcp-env', None)
    for files in [runner._log_files.pop('mcp-env', None), *runner._stream_files.pop('mcp-env', {}).values()]:
        if files is not None:
            files.close()
    assert seen and not seen[0].contains('THETA_MCP_TOKEN') and not seen[0].contains('THETA_MCP_URL')
    assert seen[0].contains('PYTHONUNBUFFERED')
