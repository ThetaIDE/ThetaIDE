"""One immutable experiment snapshot per SLURM allocation, with portable receipts."""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import site_name
from .snapshot import freeze
from .storage import digest, read_json, record_lock, safe_path, transfer_lock, write_json

TERMINAL = {'completed', 'failed', 'cancelled', 'timeout', 'out_of_memory', 'node_fail', 'preempted', 'boot_fail', 'deadline', 'revoked'}


def receipt_path(root, execution_id):
    if not re.fullmatch(r'[a-f0-9]{32}', execution_id): raise ValueError('Invalid execution ID')
    return Path(root) / 'results/slurm' / execution_id / 'submission.json'


def validate_allocation(cfg):
    from src.app.pipeline.exceptions import ConfigurationError
    site = cfg.get('site') or {}
    resources = {**dict(site.get('resources') or {}),
                 **{k: v for k, v in (cfg.get('resources') or {}).items() if v is not None}}
    trainer = cfg.get('trainer') or {}
    if int(resources.get('nodes') or 1) != 1 or int(trainer.get('num_nodes') or 1) != 1:
        raise ConfigurationError('SLURM experiment allocations currently support one node')
    from omegaconf import OmegaConf
    devices = trainer.get('devices', 1)
    if devices not in (1, 'auto') and not ((isinstance(devices, (list, tuple)) or OmegaConf.is_list(devices)) and len(devices) == 1):
        raise ConfigurationError('SLURM experiment allocations currently support one training device')
    if resources.get('gpus') is not None and (isinstance(resources['gpus'], bool) or not isinstance(resources['gpus'], int) or resources['gpus'] not in (0, 1)):
        raise ConfigurationError('Request at most one GPU per experiment allocation')
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
    """Called on the cluster. A lost sbatch response is recorded as unknown, never retried."""
    from omegaconf import OmegaConf
    from src.app.pipeline.slurm import generate_sbatch_header
    root = Path(root).resolve()
    resources = validate_allocation(cfg)
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
    snapshot = directory / 'workspace'
    with transfer_lock(root):
        source = freeze(root, snapshot)
    (snapshot / 'results').symlink_to(root / 'results', target_is_directory=True)
    # Data is deliberately shared, while source/config/provider selection is frozen.
    (snapshot / 'in').mkdir(exist_ok=True)
    if (root / 'in/datasets').exists():
        (snapshot / 'in/datasets').symlink_to(root / 'in/datasets', target_is_directory=True)
    result_id = f'{cfg.experiment_id}__{execution_id}'
    command_overrides = [arg for arg in overrides if not arg.lstrip('+').startswith(('experiment_id=', 'group='))]
    command_overrides += [f'++experiment_id={json.dumps(result_id)}', f'++group={json.dumps(str(cfg.group))}']
    # Each allocation owns its SQLite study, including its WAL/checkpoint lifecycle.
    tuning = cfg.get('tuning')
    if tuning is not None:
        from src.app.pipeline.tuners import tuner_name
        if tuner_name(cfg) == 'optuna':
            options = tuning.get('options', {}).get('optuna', {})
            storage = options.get('storage') or tuning.get('storage') or cfg.get('hydra', {}).get('sweeper', {}).get('storage') or ''
            if not storage or str(storage).startswith('sqlite:'):
                command_overrides.append(f'++tuning.options.optuna.storage=sqlite:///results/optuna/{execution_id}/study.sqlite')
    record = {'schema_version': 1, 'project_id': source['project_id'], 'execution_id': execution_id,
              'attempt_id': uuid.uuid4().hex, 'site': site_name(cfg), 'experiment': experiment,
              'group': str(cfg.group), 'result_experiment_id': result_id,
              'created_at': datetime.now(timezone.utc).isoformat(), 'state': 'prepared',
              'job_id': None, 'source_sha256': source['sha256'],
              'overrides': command_overrides, 'resources': resources,
              'config': OmegaConf.to_container(cfg, resolve=False), 'artifacts_finalized': False}
    # Hydra metadata contains job-only interpolations and isn't needed in a receipt.
    record['config'].pop('hydra', None)
    write_json(path, record)
    write_json(directory / 'config.json', record['config'])
    header = generate_sbatch_header('theta_' + execution_id[:12], directory.relative_to(root), cfg)
    header += '#SBATCH --no-requeue\n#SBATCH --signal=B:TERM@60\n'
    script = header + '\nset -eo pipefail\n'
    script += 'cd ' + shlex.quote(str(root)) + '\n'
    script += 'unset THETA_RESOURCE_POOL THETA_RESOURCE_QUEUE_PARENT THETAIDE_SETTINGS_FILE\n'
    script += 'export THETA_WORKSPACE=' + shlex.quote(str(snapshot)) + '\n'
    script += 'export THETA_DATA_DIR=' + shlex.quote(str(snapshot / '.thetaide')) + '\n'
    pythonpath = [str(snapshot), *(str(snapshot / str(extra)) for extra in site.get('extra_pythonpath') or [])]
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
    script += shlex.join([python, '-m', 'src.app.sites', '_run', '--root', str(root), execution_id]) + '\n'
    (directory / 'job.sbatch').write_text(script)
    record['state'] = 'submitting'
    write_json(path, record)
    try:
        result = subprocess.run(['sbatch', '--parsable', '--comment=theta:' + source['project_id'] + ':' + execution_id,
                                 str(directory / 'job.sbatch')], cwd=root, capture_output=True, text=True, timeout=60)
        if result.returncode:
            record.update(state='submission_failed', error=result.stderr.strip())
        else:
            match = re.fullmatch(r'(\d+)(?:;([A-Za-z0-9_.-]+))?', result.stdout.strip())
            if not match: raise ValueError('Unrecognized sbatch response; reconcile before resubmitting')
            record.update(state='pending', job_id=match[1], cluster=match[2])
    except FileNotFoundError:
        record.update(state='submission_failed', error='sbatch is unavailable; submit from a cluster login node')
    except (subprocess.TimeoutExpired, ValueError) as exc:
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
                    if path.is_file() and not path.is_symlink():
                        yield safe_path(root, path.relative_to(root).as_posix())
    database = Path(root) / 'results/optuna' / record['execution_id']
    if database.is_dir() and not database.is_symlink():
        for path in database.rglob('*'):
            if path.is_file() and not path.is_symlink():
                yield safe_path(root, path.relative_to(root).as_posix())
    directory = receipt_path(root, record['execution_id']).parent
    for path in directory.iterdir():
        if path.is_file() and path.suffix != '.lock' and path.name not in ('submission.json', 'artifacts.json'):
            yield path


def finalize(root, execution_id):
    path = receipt_path(root, execution_id)
    record = read_json(path)
    if record.get('state') not in TERMINAL or record.get('scheduler_state') not in TERMINAL:
        raise ValueError('Finalization requires confirmed scheduler termination')
    artifacts = []
    for item in sorted(set(owned_files(root, record))):
        artifacts.append({'path': item.relative_to(root).as_posix(), 'size': item.stat().st_size,
                          'sha256': digest(item)})
    write_json(path.parent / 'artifacts.json', {'execution_id': execution_id, 'files': artifacts})
    record.update(artifacts_finalized=True)
    write_json(path, record)
    return record


def run(root, execution_id):
    root = Path(root).resolve()
    path = receipt_path(root, execution_id)
    with record_lock(path):
        record = read_json(path)
        if record.get('started_at'): raise ValueError('Execution already started; use a new attempt')
    snapshot = path.parent / 'workspace'
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
                       THETA_EXECUTION_ID=execution_id, THETA_OUTPUT_ROOT=str(root),
                       THETA_MANAGED_THREADS=str(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
                       THETA_MANAGED_SERIAL='1')
    for key in ('THETA_RESOURCE_POOL', 'THETA_RESOURCE_QUEUE_PARENT', 'THETAIDE_SETTINGS_FILE', 'THETA_PROVIDER_SNAPSHOT'):
        environment.pop(key, None)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[key] = environment['THETA_MANAGED_THREADS']
    process = None
    interrupted = []
    def stop(signum, _frame):
        interrupted.append(signum)
        if process and process.poll() is None: os.killpg(process.pid, signum)
    previous = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
    code = 1
    try:
        with (path.parent / 'pipeline.log').open('w') as output:
            process = subprocess.Popen([sys.executable, str(snapshot / 'run_pipeline.py'), record['experiment'],
                                        *record['overrides']], cwd=root, env=environment,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait()
    finally:
        for number, handler in previous.items(): signal.signal(number, handler)
        with record_lock(path):
            record = read_json(path)
            record.update(state='cancelled' if interrupted else ('completed' if code == 0 else 'failed'),
                          exit_code=code, ended_at=datetime.now(timezone.utc).isoformat())
            write_json(path, record)
    return code if code >= 0 else 128 - code


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
                                     '--format=JobIDRaw,State,ExitCode'], capture_output=True, text=True, timeout=20, check=True)
            for line in result.stdout.splitlines():
                parts = line.split('|')
                if len(parts) >= 3 and parts[0] == job_id:
                    state = parts[1].split()[0].rstrip('+').lower()
                    record['scheduler_state'] = state
                    if state in TERMINAL:
                        record['state'] = state
                        record['exit_code'] = parts[2]
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
