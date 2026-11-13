"""One immutable experiment snapshot per SLURM allocation, with portable receipts."""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import shutil
import tempfile
import time
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.app.pipeline.result_paths import saved_run_id

from . import site_name
from .snapshot import freeze, extract
from .storage import digest, read_json, record_lock, safe_path, transfer_lock, write_json

TERMINAL = {'completed', 'failed', 'cancelled', 'timeout', 'out_of_memory', 'node_fail', 'preempted', 'boot_fail', 'deadline', 'revoked'}


def worker_threads(config, environment):
    """Allocation size is a ceiling, not the thread count of each serial worker."""
    site = config.get('site') or {}
    extra = site.get('extra_env_vars') or {}
    requested = int(site.get('threads_per_worker') or extra.get('THETA_MANAGED_THREADS')
                    or extra.get('OMP_NUM_THREADS') or 1)
    allocated = int(environment.get('SLURM_CPUS_PER_TASK') or
                    (config.get('resources') or {}).get('cores') or 1)
    return max(1, min(requested, allocated))


def accounting_details(fields):
    names = ('scheduler_state_raw', 'scheduler_exit_code', 'scheduler_reason',
             'scheduler_elapsed', 'scheduler_time_limit', 'scheduler_start', 'scheduler_end')
    return {key: value.strip() for key, value in zip(names, fields[1:])}


def close_unfinished_runs(runtime, record, status):
    """Called only after allocation-owned producers have stopped, before publication."""
    directory = safe_path(runtime, 'results/executions/' + record['group'] + '/' + record['result_experiment_id'])
    for path in directory.glob('*/manifest.json'):
        manifest = read_json(path)
        for run in manifest.get('runs', []):
            if run.get('status') in ('queued', 'pending', 'running'):
                run.update(status=status, error='Allocation ended before this seed completed')
        write_json(path, manifest)
    if record['config'].get('tuning') is not None:
        from src.app.pipeline.tuners import get_tuner
        hook = getattr(get_tuner(record['config']), 'finalize_interrupted', None)
        if callable(hook):
            hook(Path(runtime), record)


def receipt_path(root, execution_id):
    if not re.fullmatch(r'[a-f0-9]{32}', execution_id): raise ValueError('Invalid execution ID')
    return Path(root) / 'results/slurm' / execution_id / 'submission.json'


def validate_allocation(cfg):
    from src.app.pipeline.exceptions import ConfigurationError
    site = cfg.get('site') or {}
    from .resources import normalize
    resources = normalize(cfg)
    trainer = cfg.get('trainer') or {}
    if int(resources.get('nodes') or 1) != 1 or int(trainer.get('num_nodes') or 1) != 1:
        raise ConfigurationError('SLURM experiment allocations currently support one node')
    from omegaconf import OmegaConf
    devices = trainer.get('devices', 1)
    if devices not in (1, 'auto') and not ((isinstance(devices, (list, tuple)) or OmegaConf.is_list(devices)) and len(devices) == 1):
        raise ConfigurationError('SLURM experiment allocations currently support one training device')
    if resources.get('gpus') is not None and (isinstance(resources['gpus'], bool) or not isinstance(resources['gpus'], int) or resources['gpus'] not in (0, 1)):
        raise ConfigurationError('Request at most one GPU per experiment allocation')
    if cfg.get('plot_only'):
        raise ConfigurationError('Slurm plot_only is not supported yet; plot synced prior results locally (wipe=false)')
    raw_config = OmegaConf.to_container(OmegaConf.masked_copy(cfg, [key for key in cfg if key != 'hydra']), resolve=False) if OmegaConf.is_config(cfg) else cfg
    from urllib.parse import urlsplit
    def check_secrets(value):
        if isinstance(value, dict):
            for child in value.values(): check_secrets(child)
        elif isinstance(value, list):
            for child in value: check_secrets(child)
        elif isinstance(value, str) and '://' in value and urlsplit(value).password:
            raise ConfigurationError('Database credentials must use an environment reference, not a captured URL')
    check_secrets(raw_config)
    options = ((cfg.get('tuning') or {}).get('options') or {}).get('optuna') or {}
    jobs = options.get('n_jobs', (cfg.get('hydra', {}).get('sweeper') or {}).get('n_jobs', 1))
    if jobs != 1:
        raise ConfigurationError('Slurm supports n_jobs=1 only; allocation-scoped parallel trials are not qualified. Local parallel trials remain supported.')
    if (cfg.get('variability') or {}).get('parallel_runs', 1) != 1:
        raise ConfigurationError('Slurm supports variability.parallel_runs=1 only')
    if cfg.get('workflow'):
        raise ConfigurationError('SLURM workflow execution is not supported yet; submit individual experiments')
    if cfg.get('dash') or cfg.get('dash_only'):
        raise ConfigurationError('Remote dashboards are not supported; use Sync Metrics to inspect the study')
    if cfg.get('recover'):
        raise ConfigurationError('Remote recovery requires an explicit prior execution and is not supported yet; original checkpoints are retained')
    for key in ('cores', 'nodes'):
        if resources.get(key) is not None and (isinstance(resources[key], bool) or not isinstance(resources[key], int) or resources[key] < 1):
            raise ConfigurationError(f'resources.{key} must be positive')
    return resources


def submit(root, experiment, overrides, cfg):
    # The same lock protects source references, continuation ownership, and GC.
    with transfer_lock(root):
        return _submit(root, experiment, overrides, cfg)


def _submit(root, experiment, overrides, cfg):
    """Called on the cluster. A lost sbatch response is recorded as unknown, never retried."""
    from omegaconf import OmegaConf
    from src.app.pipeline.slurm import generate_sbatch_header
    root = Path(root).resolve()
    from .policy import resolve
    policy = resolve(cfg, root)
    resources = validate_allocation(cfg)
    from .resources import submission_environment
    submit_env = submission_environment()
    site = cfg.get('site') or {}
    python = str(site.get('python_executable') or 'auto')
    if python == 'auto':
        venv = Path(str(site.get('venv_dir') or 'venv'))
        python = str((root / venv / 'bin/python3').absolute())
    if not Path(python).is_absolute(): python = str(root / python)
    if not Path(python).is_file(): raise ValueError(f'Cluster Python does not exist: {python}')
    execution_id = uuid.uuid4().hex
    path = receipt_path(root, execution_id)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=False)
    try:
        snapshot = directory / 'workspace'
        if policy:
            source = read_json(root / '.thetaide/source.json')
            bundle = root / '.thetaide/sources' / (str(source.get('sha256')) + '.zip')
            if not source.get('archive_sha256') or not bundle.is_file():
                raise ValueError('Scratch execution requires a source archive from site push; upload this revision first')
            if sum(item.stat().st_size for item in (root / '.thetaide/sources').glob('*.zip')) > policy['max_source_bytes']:
                raise ValueError('Source archive store exceeds max_source_bytes; preview managed maintenance')
            from .policy import stage_inputs, own_work_root
            own_work_root(policy['work_root'], source['project_id'])
            inputs = Path(policy['work_root']) / 'inputs' / execution_id
            inputs.mkdir(parents=True, exist_ok=False)
            try:
                stage_inputs(root, inputs, policy)
            except BaseException:
                shutil.rmtree(inputs)
                raise
        else:
            source = freeze(root, snapshot)
            (snapshot / 'results').symlink_to(root / 'results', target_is_directory=True)
            (snapshot / 'in').mkdir(exist_ok=True)
            if (root / 'in/datasets').exists():
                (snapshot / 'in/datasets').symlink_to(root / 'in/datasets', target_is_directory=True)
        result_id = saved_run_id(str(cfg.experiment_id), f'__{execution_id}')
        command_overrides = [arg for arg in overrides if not arg.lstrip('+').startswith(('experiment_id=', 'group='))]
        command_overrides += [f'++experiment_id={json.dumps(result_id)}', f'++group={json.dumps(str(cfg.group))}']
        # Each allocation owns its SQLite study, including its WAL/checkpoint lifecycle.
        storage = ''
        tuning = cfg.get('tuning')
        if tuning is not None:
            from src.app.pipeline.tuners import tuner_name
            if tuner_name(cfg) == 'optuna':
                options = tuning.get('options', {}).get('optuna', {})
                storage = options.get('storage') or tuning.get('storage') or cfg.get('hydra', {}).get('sweeper', {}).get('storage') or ''
                if not policy and (not storage or str(storage).startswith('sqlite:')):
                    command_overrides.append(f'++tuning.options.optuna.storage=sqlite:///results/optuna/{execution_id}/study.sqlite')
        record = {'schema_version': 1, 'project_id': source['project_id'], 'execution_id': execution_id,
                  'attempt_id': uuid.uuid4().hex, 'site': site_name(cfg), 'experiment': experiment,
                  'group': str(cfg.group), 'result_experiment_id': result_id,
                  'created_at': datetime.now(timezone.utc).isoformat(), 'state': 'prepared',
                  'job_id': None, 'source_sha256': source['sha256'],
                  'overrides': command_overrides, 'resources': resources,
                  'config': OmegaConf.to_container(cfg, resolve=False), 'artifacts_finalized': False,
                  'execution_policy': policy, 'control_root': str(root), 'source_archive_sha256': source.get('archive_sha256'),
                  'publication_state': 'pending' if policy else 'legacy',
                  'tuning_storage_kind': 'external' if storage and not str(storage).startswith('sqlite:') else 'sqlite'}
        request = cfg.get('study_continuation')
        if request:
            if not policy: raise ValueError('Study continuation requires scratch execution')
            if cfg.get('remake') or cfg.get('recover'): raise ValueError('Continuation cannot remake or resume a training trial')
            from .continuation import pin
            record['continuation'] = pin(root, record, dict(request))
            write_json(path, record)
            command_overrides += ['++tuning.options.optuna.study_mode=resume',
                                  '++tuning.options.optuna.study_name=' + json.dumps(request['study_name'])]
        # Hydra metadata contains job-only interpolations and isn't needed in a receipt.
        record['config'].pop('hydra', None)
        write_json(path, record)
        write_json(directory / 'config.json', record['config'])
        header = generate_sbatch_header('theta_' + execution_id[:12], directory.relative_to(root), cfg, resources=resources)
        allowance = int(policy['stage_out_seconds']) if policy else 60
        header += f'#SBATCH --no-requeue\n#SBATCH --signal=B:TERM@{allowance}\n'
        script = header + '\nset -eo pipefail\n'
        script += 'cd ' + shlex.quote(str(root)) + '\n'
        script += 'unset THETA_RESOURCE_POOL THETA_RESOURCE_QUEUE_PARENT THETAIDE_SETTINGS_FILE\n'
        script += 'export THETA_WORKSPACE=' + shlex.quote(str(snapshot)) + '\n'
        script += 'export THETA_DATA_DIR=' + shlex.quote(str(snapshot / '.thetaide')) + '\n'
        bootstrap_root = root if policy else snapshot
        pythonpath = [str(bootstrap_root), *(str(snapshot / str(extra)) for extra in site.get('extra_pythonpath') or [])]
        script += 'export PYTHONPATH=' + shlex.quote(':'.join(pythonpath)) + '\n'
        if site.get('module_loads'):
            script += 'if ! type module >/dev/null 2>&1; then\n'
            script += '  if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; '
            script += 'elif [ -f /usr/share/modules/init/bash ]; then source /usr/share/modules/init/bash; fi\nfi\n'
        for module in site.get('module_loads') or []:
            script += 'module load ' + shlex.quote(str(module)) + '\n'
        for key, value in (site.get('extra_env_vars') or {}).items():
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', str(key)): raise ValueError('Invalid environment variable')
            if str(key).startswith('THETA_') or key == 'PYTHONPATH': raise ValueError(f'{key} is managed by Theta')
            script += 'export ' + str(key) + '=' + shlex.quote(str(value)) + '\n'
        script += 'set -u\n'
        # Slurm signals the batch PID; the supervisor must own that PID to finalize.
        script += 'exec ' + shlex.join([python, '-m', 'src.app.sites', '_run', '--root', str(root), execution_id]) + '\n'
        (directory / 'job.sbatch').write_text(script)
        record['state'] = 'submitting'
        write_json(path, record)
        try:
            result = subprocess.run(['sbatch', '--parsable', '--comment=theta:' + source['project_id'] + ':' + execution_id,
                                     str(directory / 'job.sbatch')], cwd=root, env=submit_env, capture_output=True, text=True, timeout=60)
            if result.returncode:
                record.update(state='submission_failed', error=result.stderr.strip())
            else:
                match = re.fullmatch(r'(\d+)(?:;([A-Za-z0-9_.-]+))?', result.stdout.strip())
                if not match: raise ValueError('Unrecognized sbatch response; reconcile before resubmitting')
                record.update(state='pending', job_id=match[1], cluster=match[2])
        except FileNotFoundError:
            record.update(state='submission_failed', error='sbatch is unavailable; submit from a cluster login node')
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            record.update(state='submission_unknown', error=str(exc))
        # The batch worker can start before sbatch returns. Do not overwrite its state.
        with record_lock(path):
            current = read_json(path)
            if current.get('started_at'):
                record = {**record, **current, 'job_id': record.get('job_id') or current.get('job_id')}
            write_json(path, record)
        print(json.dumps({'execution_id': execution_id, 'state': record['state'], 'job_id': record['job_id'],
                          'receipt': str(path)}, indent=2))
        if record['state'] in ('submission_failed', 'submission_unknown'):
            raise RuntimeError(record.get('error', record['state']))
        return record
    except BaseException:
        if path.exists():
            failed = read_json(path)
            if failed.get('state') == 'prepared':
                failed.update(state='submission_failed', error='Submission preparation failed; no sbatch call made')
                write_json(path, failed)
            if failed.get('state') == 'submission_failed':
                payload = Path(policy['work_root']) / 'inputs' / execution_id if policy else directory / 'workspace'
                if payload.exists() and not payload.is_symlink(): shutil.rmtree(payload)
        else:
            if policy:
                inputs = Path(policy['work_root']) / 'inputs' / execution_id
                if inputs.exists(): shutil.rmtree(inputs)
            shutil.rmtree(directory)
        raise


def _regular_output(path):
    if path.name.endswith(('-wal', '-shm', '-journal')): return False
    try: return path.is_file() and not path.is_symlink()
    except FileNotFoundError: return False


def owned_files(root, record):
    """Select only output trees owned by this UUID, including nested seed executions."""
    root = Path(root)
    prefix = record['result_experiment_id']
    for category in ('logs', 'checkpoints', 'plots', 'datasets', 'executions'):
        parent = safe_path(root, f'results/{category}/{record["group"]}')
        if not parent.is_dir(): continue
        # Preserve recipe subdirectories and allow variability's stamped suffix.
        parent = parent / str(Path(prefix).parent)
        if not parent.is_dir(): continue
        for directory in parent.iterdir():
            if directory.name != Path(prefix).name and not directory.name.startswith(Path(prefix).name + '_'): continue
            if directory.is_dir() and not directory.is_symlink():
                for path in directory.rglob('*'):
                    if _regular_output(path):
                        yield safe_path(root, path.relative_to(root).as_posix())
    database = Path(root) / 'results/optuna' / record['execution_id']
    if database.is_dir() and not database.is_symlink():
        for path in database.rglob('*'):
            if _regular_output(path):
                yield safe_path(root, path.relative_to(root).as_posix())
    directory = receipt_path(root, record['execution_id']).parent
    for path in directory.iterdir():
        if _regular_output(path) and path.suffix != '.lock' and path.name not in ('submission.json', 'artifacts.json'):
            yield path


def finalize(root, execution_id):
    path = receipt_path(root, execution_id)
    record = read_json(path)
    if record.get('state') not in TERMINAL or record.get('scheduler_state') not in TERMINAL:
        raise ValueError('Finalization requires confirmed scheduler termination')
    if record.get('execution_policy'):
        publication = read_json(Path(record['execution_policy']['work_root']) / 'results/slurm' / execution_id / 'publication.json')
        if not publication.get('sealed'):
            raise ValueError('Payload is unsealed; explicit scheduled repair is required')
        write_json(path.parent / 'artifacts.json', {'execution_id': execution_id, 'files': publication['files']})
        record.update(artifacts_finalized=True)
        write_json(path, record)
        return record
    artifacts = []
    for item in sorted(set(owned_files(root, record))):
        artifacts.append({'path': item.relative_to(root).as_posix(), 'size': item.stat().st_size,
                          'sha256': digest(item)})
    write_json(path.parent / 'artifacts.json', {'execution_id': execution_id, 'files': artifacts})
    record.update(artifacts_finalized=True)
    write_json(path, record)
    return record


def run(root, execution_id):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('The allocation worker requires SLURM_JOB_ID; never run it on a login node')
    root = Path(root).resolve()
    path = receipt_path(root, execution_id)
    with record_lock(path):
        record = read_json(path)
        if record.get('started_at'): raise ValueError('Execution already started; use a new attempt')
    policy = record.get('execution_policy')
    runtime = root
    snapshot = path.parent / 'workspace'
    if policy:
        from .policy import assert_local_scratch
        scratch = Path(policy['scratch_root'])
        assert_local_scratch(scratch)
        runtime = Path(tempfile.mkdtemp(prefix='theta-' + execution_id + '-', dir=scratch))
        snapshot = runtime
        write_json(runtime / '.theta-owner.json', {'project_id': record['project_id'],
                   'execution_id': execution_id, 'job_id': os.environ.get('SLURM_JOB_ID'),
                   'host': __import__('socket').gethostname(), 'pid': os.getpid()})
        bundle = root / '.thetaide/sources' / (record['source_sha256'] + '.zip')
        if digest(bundle) != record['source_archive_sha256']: raise ValueError('Source archive changed')
        import zipfile
        with zipfile.ZipFile(bundle) as archive:
            required = sum(item.file_size for item in archive.infolist())
        required += read_json(Path(policy['work_root']) / 'inputs' / execution_id / '.theta-inputs.json').get('bytes', 0)
        if shutil.disk_usage(runtime).free < required + policy['scratch_reserve_bytes']:
            raise ValueError('Insufficient job-local scratch for source/inputs and reserve')
        source = extract(bundle, runtime)
        if source['sha256'] != record['source_sha256']: raise ValueError('Source identity mismatch')
        inputs = Path(policy['work_root']) / 'inputs' / execution_id
        if inputs.is_dir(): shutil.copytree(inputs, runtime, dirs_exist_ok=True, copy_function=shutil.copyfile)
    else:
        source = read_json(snapshot / '.thetaide/source.json')
        for relative, expected in source['files'].items():
            if digest(safe_path(snapshot, relative)) != expected:
                raise ValueError(f'Submission source changed: {relative}')
    with record_lock(path):
        record = read_json(path)
        if record.get('started_at'): raise ValueError('Execution already started; use a new attempt')
        record.update(state='running', started_at=datetime.now(timezone.utc).isoformat(),
                      job_id=os.environ.get('SLURM_JOB_ID', record.get('job_id')))
        write_json(path, record)
    environment = os.environ.copy()
    environment.update(THETA_IN_ALLOCATION='1', THETA_WORKSPACE=str(snapshot),
                       THETA_DATA_DIR=str(snapshot / '.thetaide'),
                       PYTHONPATH=str(snapshot) + os.pathsep + environment.get('PYTHONPATH', ''),
                       THETA_EXECUTION_ID=execution_id, THETA_OUTPUT_ROOT=str(runtime),
                       THETA_PYTHON_EXECUTABLE=sys.executable,
                       THETA_MANAGED_THREADS=str(worker_threads(record['config'], environment)),
                       THETA_MANAGED_SERIAL='1')
    for key in ('THETA_RESOURCE_POOL', 'THETA_RESOURCE_QUEUE_PARENT', 'THETAIDE_SETTINGS_FILE', 'THETA_PROVIDER_SNAPSHOT'):
        environment.pop(key, None)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[key] = environment['THETA_MANAGED_THREADS']
    overrides = list(record['overrides'])
    publisher = None
    if policy:
        from .publication import Publisher
        database = runtime / 'results/optuna' / execution_id / 'study.sqlite'
        options = ((record['config'].get('tuning') or {}).get('options') or {}).get('optuna') or {}
        storage = options.get('storage') or ''
        if record['config'].get('tuning') and record.get('tuning_storage_kind') == 'sqlite':
            overrides = [arg for arg in overrides if not arg.lstrip('+').startswith('tuning.options.optuna.storage=')]
            overrides.append('++tuning.options.optuna.storage=' + json.dumps('sqlite:///' + str(database)))
        methods = record['config'].get('methods') or {'run': {}}
        method_count = sum(name != 'params' and (not isinstance(value, dict) or value.get('enabled', True)) for name, value in methods.items())
        variability = record['config'].get('variability') or {}
        seeds = len(variability.get('training_seeds') or []) or variability.get('training_runs', 1)
        planned_runs = max(1, method_count * seeds * (options.get('n_trials', 40) if record['config'].get('tuning') else 1))
        environment.update(THETA_SQLITE_ROOT=str(database.parent),
                           THETA_COMMIT_DIR=str(runtime / '.thetaide/commits'),
                           THETA_DB_EXPORT_REQUEST=str(runtime / '.thetaide/db-export-request.json'),
                           THETA_PUBLICATION_STOP=str(runtime / '.thetaide/publication-stop'),
                           THETA_DIAGNOSTIC_BUDGET=json.dumps({**{key: policy[key] for key in ('diagnostic_bytes', 'diagnostic_files', 'snapshot_bytes')}, 'planned_runs': planned_runs}),
                           THETA_STREAM_FLUSH_SECONDS=str(policy['stream_flush_seconds']),
                           THETA_SYSTEM_INTERVAL=str(policy['system_interval']),
                           THETA_DIAGNOSTIC_STRICT='1' if policy['diagnostic_strict'] else '0',
                           THETA_DEFER_MONITOR_REPORTS='1')
        if record.get('continuation'):
            from .continuation import restore
            restore(runtime, record)
        publisher = Publisher(runtime, policy['work_root'], record, policy)
        controls = runtime / 'results/slurm' / execution_id
        controls.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'job.sbatch'):
            shutil.copyfile(path.parent / name, controls / name)
        runtime_log = controls / 'pipeline.log'
    else:
        runtime_log = path.parent / 'pipeline.log'
    process = None
    interrupted = []
    stop_started = []
    def stop(signum, _frame):
        interrupted.append(signum)
        if not stop_started: stop_started.append(time.monotonic())
        if process and process.poll() is None:
            try: os.killpg(process.pid, signum)
            except ProcessLookupError: pass
    deadline = float(os.environ.get('SLURM_JOB_END_TIME') or 0)
    allowance = int(policy['stage_out_seconds']) if policy else 60
    termination_reason = None
    previous = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
    code = 1
    try:
        from src.app.services.process_output import ProcessOutput
        import traceback
        error_log = runtime_log.with_suffix('.err')
        with runtime_log.open('w', encoding='utf-8', errors='replace') as output, error_log.open('w', encoding='utf-8', errors='replace') as errors:
            output.write(f"[Theta] Allocation {record.get('job_id')}: worker_threads={environment['THETA_MANAGED_THREADS']}, requested_time={record.get('resources', {}).get('time') or 'partition default'}, deadline={datetime.fromtimestamp(deadline, timezone.utc).isoformat() if deadline else 'unknown'}.\n")
            output.flush()
            process = subprocess.Popen([sys.executable, '-u', str(snapshot / 'run_pipeline.py'), record['experiment'],
                                        *overrides], cwd=runtime, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       encoding='utf-8', errors='replace', bufsize=1, start_new_session=True)
            capture = ProcessOutput(process, output, errors, stderr=sys.stderr)
            try:
                while process.poll() is None:
                    if interrupted: break
                    if publisher:
                        try:
                            publisher.tick()
                        except Exception as exc:
                            record['publication_error'] = str(exc)
                            capture.write_error('[Theta] Result publication failed:\n' + traceback.format_exc())
                            (runtime / '.thetaide/publication-stop').touch()
                            stop(signal.SIGTERM, None)
                            break
                    try: process.wait(timeout=1)
                    except subprocess.TimeoutExpired: pass
                if interrupted:
                    near_deadline = deadline and 0 <= deadline - time.time() <= allowance + 60
                    termination_reason = ('Result publication failed: ' + record['publication_error'] if record.get('publication_error') else
                                          'Likely Slurm pre-timeout warning: allocation deadline is near' if near_deadline else
                                          'External termination signal; inspect scheduler accounting for attribution')
                    capture.write_error(f'[Theta] Received {signal.Signals(interrupted[-1]).name}: {termination_reason}. Finalizing results.\n')
                if interrupted and process.poll() is None:
                    try: process.wait(timeout=int(policy['stage_out_seconds']) // 2 if policy else 30)
                    except subprocess.TimeoutExpired:
                        capture.write_error('[Theta] Training did not stop in time; sending SIGKILL before final stage-out.\n')
                        os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
            except BaseException:
                capture.write_error('[Theta] Execution supervisor failed:\n' + traceback.format_exc())
                if process.poll() is None:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
                raise
            finally:
                capture.finish()
    finally:
        if interrupted or code != 0:
            try:
                close_unfinished_runs(runtime, record, 'cancelled' if interrupted else 'failed')
            except Exception:
                import traceback
                print('[Theta] Could not finalize interrupted trial statuses:\n' + traceback.format_exc(), file=sys.stderr, flush=True)
        if publisher:
            try:
                allowance = policy['stage_out_seconds']
                if stop_started: allowance = max(1, allowance - (time.monotonic() - stop_started[0]))
                publisher.finish(seconds=allowance)
                record['publication_state'] = 'sealed'
            except Exception as exc:
                record.update(publication_state='incomplete', publication_error=str(exc))
                import traceback
                diagnostic = '[Theta] Final result publication failed:\n' + traceback.format_exc()
                print(diagnostic, file=sys.stderr, flush=True)
                for diagnostic_log in (runtime_log, runtime_log.with_suffix('.err')):
                    try:
                        with diagnostic_log.open('a', encoding='utf-8', errors='replace') as stream:
                            stream.write(diagnostic)
                    except OSError as log_error:
                        print(f'[Theta] Could not save {diagnostic_log.name}: {log_error}', file=sys.stderr, flush=True)
            finally:
                record['last_durable'] = publisher.manifest['durable']
                publisher.close()
        for number, handler in previous.items(): signal.signal(number, handler)
        with record_lock(path):
            record = {**read_json(path), **{key: value for key, value in record.items() if key in ('publication_state', 'publication_error', 'last_durable')}}
            training_state = 'cancelled' if interrupted else ('completed' if code == 0 else 'failed')
            record.update(state=training_state, training_state=training_state,
                          termination_reason=termination_reason, termination_signal=interrupted[-1] if interrupted else None,
                          exit_code=code, ended_at=datetime.now(timezone.utc).isoformat())
            write_json(path, record)
    if publisher and record.get('publication_state') == 'sealed':
        shutil.rmtree(runtime)  # allocation-owned temporary tree only, after durable sealing
    return (code if code >= 0 else 128 - code) or (1 if record.get('publication_state') == 'incomplete' else 0)


def refresh_status(root, record):
    if record.get('scheduler_state') in TERMINAL or record.get('state') == 'submission_failed': return record
    job_id = record.get('job_id')
    try:
        if not job_id:
            # Reconcile the case where sbatch accepted the job but its response was lost.
            result = subprocess.run(['squeue', '--noheader', '--user', str(os.getuid()), '--format=%i|%k'],
                                    capture_output=True, text=True, timeout=20, check=True)
            token = 'theta:' + record['project_id'] + ':' + record['execution_id']
            matches = [line.split('|')[0].strip() for line in result.stdout.splitlines() if line.partition('|')[2].strip() == token]
            if len(matches) == 1: job_id = record['job_id'] = matches[0]
        if not job_id: return record
        result = subprocess.run(['squeue', '--noheader', '--jobs', job_id, '--format=%T'],
                                capture_output=True, text=True, timeout=20, check=True)
        value = result.stdout.strip().splitlines()
        if value:
            record['scheduler_state'] = value[0].lower()
        else:
            record['scheduler_state'] = 'unknown'
            result = subprocess.run(['sacct', '--noheader', '--parsable2', '--allocations', '--jobs', job_id,
                                     '--format=JobIDRaw,State%80,ExitCode,Reason%120,Elapsed,Timelimit,Start,End'], capture_output=True, text=True, timeout=20, check=True)
            for line in result.stdout.splitlines():
                parts = line.split('|')
                if len(parts) >= 3 and parts[0] == job_id:
                    state = parts[1].split()[0].rstrip('+').lower()
                    record['scheduler_state'] = state
                    record.update(accounting_details(parts))
                    if state in TERMINAL:
                        training = record.get('training_state') or (record.get('state') if record.get('state') in TERMINAL else 'unknown')
                        record['training_state'] = training
                        record['state'] = training if state == 'completed' and training in ('failed', 'cancelled') else state
                        record['scheduler_exit_code'] = parts[2]
                        record.setdefault('exit_code', parts[2])
                    break
        if record.get('scheduler_state') == 'unknown':
            record['status_error'] = 'No scheduler accounting record is available yet'
        else:
            record['scheduler_checked_at'] = datetime.now(timezone.utc).isoformat()
            record.pop('status_error', None)
    except (OSError, subprocess.SubprocessError) as exc:
        record['status_error'] = str(exc)
    # Re-read so a worker finishing during the scheduler query wins.
    with record_lock(receipt_path(root, record['execution_id'])):
        current = read_json(receipt_path(root, record['execution_id']))
        if current.get('state') in TERMINAL and record.get('scheduler_state') not in TERMINAL:
            record = {**record, **current, 'scheduler_state': record.get('scheduler_state', current.get('scheduler_state'))}
        write_json(receipt_path(root, record['execution_id']), record)
    return record


def refresh_many(root, records, *, minimum_seconds=10):
    """Two batched scheduler RPCs per 200 unresolved jobs; terminal state is cached."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    pending = []
    for record in records:
        if record.get('scheduler_state') in TERMINAL or record.get('state') == 'submission_failed': continue
        checked = record.get('scheduler_checked_at')
        if checked and now - datetime.fromisoformat(checked) < timedelta(seconds=minimum_seconds): continue
        pending.append(record)
    for offset in range(0, len(pending), 200):
        batch = pending[offset:offset + 200]
        try:
            known = [str(item['job_id']) for item in batch if item.get('job_id')]
            command = ['squeue', '--noheader', '--format=%i|%T|%k']
            command += ['--jobs', ','.join(known)] if len(known) == len(batch) else ['--user', str(os.getuid())]
            queue = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
            states = {}
            for line in queue.stdout.splitlines():
                fields = line.split('|')
                if len(fields) >= 2:
                    states[fields[0].strip()] = fields[1].strip().lower()
                    if len(fields) >= 3:
                        for record in batch:
                            if fields[2].strip() == f'theta:{record["project_id"]}:{record["execution_id"]}': record['job_id'] = fields[0].strip()
            missing = [str(item['job_id']) for item in batch if item.get('job_id') and str(item['job_id']) not in states]
            exits = {}
            details = {}
            if missing:
                accounting = subprocess.run(['sacct', '--noheader', '--parsable2', '--allocations', '--jobs', ','.join(missing),
                                             '--format=JobIDRaw,State%80,ExitCode,Reason%120,Elapsed,Timelimit,Start,End'], capture_output=True, text=True, timeout=20, check=True)
                for line in accounting.stdout.splitlines():
                    fields = line.split('|')
                    if len(fields) >= 3 and fields[0] in missing:
                        states[fields[0]] = fields[1].split()[0].rstrip('+').lower()
                        exits[fields[0]] = fields[2]
                        details[fields[0]] = accounting_details(fields)
            for record in batch:
                path = receipt_path(root, record['execution_id'])
                with record_lock(path):
                    current = read_json(path)
                    state = states.get(str(record.get('job_id')), 'unknown')
                    current.update(job_id=record.get('job_id'), scheduler_state=state, scheduler_checked_at=now.isoformat())
                    current.update(details.get(str(record.get('job_id')), {}))
                    if state in TERMINAL:
                        training = current.get('training_state') or (current.get('state') if current.get('state') in TERMINAL else 'unknown')
                        current.update(training_state=training,
                                       state=training if state == 'completed' and training in ('failed', 'cancelled') else state,
                                       scheduler_exit_code=exits.get(str(record.get('job_id'))))
                    if state == 'unknown': current['status_error'] = 'Scheduler/accounting does not confirm termination'
                    else: current.pop('status_error', None)
                    write_json(path, current)
                    record.update(current)
        except (OSError, subprocess.SubprocessError) as exc:
            for record in batch: record['status_error'] = str(exc)
    return records
