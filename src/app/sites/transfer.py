"""Explicit SSH/rsync operations. No networking happens during discovery or imports."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .execution import TERMINAL, finalize, owned_files, receipt_path, refresh_status, refresh_many
from .snapshot import archive, freeze, source_files
from .transport import rsync_shell
from . import transfer_output
from .storage import digest, read_json, safe_path, transfer_lock, workspace_id, write_json


def configure(root, site, host, remote_root, python='venv/bin/python3', *, scheduler=None):
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', host) or host.startswith('-'):
        raise ValueError('Use an SSH config alias or user@hostname')
    if remote_root == '/' or not remote_root.startswith('/') or any(part in ('.', '..') for part in remote_root.split('/')):
        raise ValueError('Remote project directory must be an absolute path without traversal')
    if any(c in remote_root + python for c in '\x00\n\r') or not python or python.startswith('-'):
        raise ValueError('Invalid remote path or Python command')
    from .catalog import configure as configure_component
    return configure_component(root, site, {'host': host, 'remote_root': remote_root.rstrip('/'), 'python': python},
                               scheduler=scheduler)


def connections(root):
    from . import profiles
    from .catalog import all_profiles, load
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths(root)
    available = all_profiles(paths)
    return {selector: load(paths, selector, available=available).get('connection', {})
            for selector in profiles(paths, available=available)}


def connection(root, site):
    from .catalog import load
    config = load(root, site)
    values = config.get('connection') or {}
    if not values.get('host') or not values.get('remote_root'):
        raise ValueError(f'Edit the connection in the {site!r} site component in Components first')
    values = dict(values)
    if config.get('execution') and not Path(values.get('python', 'venv/bin/python3')).is_absolute():
        values['python'] = str(Path(values['remote_root']) / values.get('python', 'venv/bin/python3'))
    return {**values, '_root': str(Path(root).resolve()), '_shortcuts': config.get('shortcuts', {})}


def ssh(config, arguments, *, input=None, log_stdout=True):
    command = 'cd ' + shlex.quote(config['remote_root']) + ' && ' + shlex.join(arguments)
    from .transport import ssh_arguments
    arguments = [*ssh_arguments(config['_root'], config), '--', config['host'], command]
    transfer_output.command(arguments)
    result = subprocess.run(arguments, input=input, capture_output=True, text=True, timeout=180)
    if log_stdout: transfer_output.emit(result.stdout)
    transfer_output.emit(result.stderr)
    transfer_output.emit(f'Process exited with code {result.returncode}')
    if result.returncode: raise RuntimeError(result.stderr.strip() or 'Remote command failed')
    return result.stdout


def remote(root, site, command, *arguments, input=None):
    config = connection(root, site)
    return json.loads(ssh(config, [config['python'], '-m', 'src.app.sites', command, '--root', '.', *arguments], input=input, log_stdout=False))


def require_local_rsync(config):
    executable = config.get('rsync_executable', 'rsync')
    try:
        transfer_output.command([executable, '--version'])
        result = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f'Cannot run configured rsync executable {executable!r}: {exc}') from exc
    output = (result.stdout or '') + (result.stderr or '')
    transfer_output.emit(output)
    match = re.search(r'\brsync\s+version\s+(\d+)\.', output)
    if result.returncode or not match or int(match[1]) < 3:
        resolved = shutil.which(executable) or executable
        detected = output.strip().splitlines()[0] if output.strip() else 'version unavailable'
        raise ValueError(
            f'Theta requires rsync 3 or newer. Configured {executable!r} resolves to {resolved}: {detected}. '
            'On macOS, install with brew install rsync, then set connection.rsync_executable '
            'in the site component to the absolute installed path (usually /opt/homebrew/bin/rsync '
            'on Apple Silicon or /usr/local/bin/rsync on Intel).'
        )
    return executable


def remote_requirements(root, site):
    """Executables the selected scheduler needs remotely, as (command, missing-note) pairs.

    A `None` note marks a required command; a string is printed when an optional one is absent.
    """
    from .catalog import load
    from .scheduler_api import get_scheduler
    from src.app.workspace import WorkspacePaths
    site_config = load(root, site)
    if (site_config.get('scheduler') or 'local') == 'local':
        return []
    provider = get_scheduler(cfg={'site': site_config}, paths=WorkspacePaths(root))
    hook = getattr(provider, 'remote_requirements', None)
    requirements = list(hook(site_config)) if callable(hook) else []
    for item in requirements:
        if (not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[0], str)
                or not re.fullmatch(r'[A-Za-z0-9_.+-]+', item[0]) or not (item[1] is None or isinstance(item[1], str))):
            raise ValueError(f'Scheduler remote requirement {item!r} must be (command, None | note)')
    return requirements


def check(root, site):
    config = connection(root, site)
    require_local_rsync(config)
    # This check works before the first project upload and does not require Theta remotely.
    requirements = remote_requirements(root, site)
    required = [f'command -v {command}' for command, note in requirements if note is None]
    optional = [f"{{ command -v {command} || printf '%s\\n' {shlex.quote(note)}; }}" for command, note in requirements if note is not None]
    rsync = shlex.quote(config.get('remote_rsync', 'rsync'))
    check_script = ' && '.join([*required, f'{rsync} --version', 'test -w .', *optional])
    output = ssh(config, ['bash', '-lc', check_script])
    if not re.search(r'version\s+[3-9]\.', output): raise ValueError('Remote rsync must be version 3 or newer')
    python_version = ssh(config, [config['python'], '--version']).strip()
    return {'site': site, 'checks': output.strip(), 'python': python_version}


def push(root, site, *, datasets=False, live_output=False):
    root = Path(root).resolve()
    config = connection(root, site)
    def announce(message):
        transfer_output.emit(message)
        if live_output: print(message, flush=True)
    announce('Checking local rsync...')
    require_local_rsync(config)
    with transfer_lock(root), tempfile.TemporaryDirectory(prefix='theta-push-') as temporary:
        stage = Path(temporary)
        announce('Preparing a frozen project snapshot...')
        source = freeze(root, stage)
        # Transfer a frozen list, not a tree that the user may still be editing.
        write_json(stage / '.theta-managed.json', {'project_id': source['project_id'], 'expires_at': __import__('time').time() + 7200})
        files = [*source['files'], '.thetaide/source.json', '.theta-managed.json']
        if datasets and (root / 'in/datasets').is_dir():
            for path in (root / 'in/datasets').rglob('*'):
                if path.is_symlink(): raise ValueError('Dataset symlinks need a cluster-side dataset mapping')
                if path.is_file():
                    relative = path.relative_to(root).as_posix()
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    files.append(relative)
        listing = stage / 'files.list'
        listing.write_bytes(b''.join(name.encode() + b'\0' for name in files))
        announce(f'Prepared {len(source["files"])} source files. Checking {config["host"]}:{config["remote_root"]}...')
        # Refuse an existing workspace belonging to a different local project.
        identity = shlex.quote(config['remote_root'] + '/.thetaide/workspace.json')
        probe = 'if [ -f ' + identity + ' ]; then cat ' + identity + '; else printf "{}"; fi'
        existing = json.loads(ssh(config, ['bash', '-c', probe], log_stdout=False))
        if existing.get('id') and existing['id'] != source['project_id']:
            raise ValueError('Remote directory belongs to another project; choose a new directory')
        upload = config['remote_root'] + '/.thetaide/uploads/' + uuid.uuid4().hex
        ssh(config, ['mkdir', '-p', upload])
        announce('Uploading changed files with rsync (unchanged files are reused remotely)...')
        progress_flags = ['--progress', '--human-readable'] if live_output else transfer_output.progress_flags()
        run_transfer([config.get('rsync_executable', 'rsync'), '-r', *progress_flags, '--protect-args', '--from0', '--files-from=' + str(listing),
                        '--checksum', '--partial-dir=.theta-partial', '--link-dest=' + config['remote_root'], '--rsync-path=' + config.get('remote_rsync', 'rsync'), '-e', rsync_shell(root, config),
                        str(stage) + '/', config['host'] + ':' + upload + '/'], live_output=live_output)
        announce('Verifying uploaded source and activating the project...')
        ssh(config, [config['python'], upload + '/run_pipeline.py', 'site', '_activate',
                     '--root', config['remote_root'], '--source', upload], log_stdout=False)
        state = read_json(root / '.thetaide/site_sync.json')
        state.setdefault(site, {}).update(last_push=now(), source_sha256=source['sha256'])
        write_json(root / '.thetaide/site_sync.json', state)
        return {'site': site, 'source_sha256': source['sha256'], 'files': len(files),
                'command': 'cd ' + shlex.quote(config['remote_root']) + ' && ' +
                shlex.join([config['python'], 'run_pipeline.py', '<group/experiment>', 'site=' + site])}


def now(): return datetime.now(timezone.utc).isoformat()


def digest_name(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def index(root, mode, execution_id=None, diagnostics=False):
    from .maintenance import collect
    collect(root, apply=True, temporary_only=True)
    with transfer_lock(root):
        return _index(root, mode, execution_id, diagnostics)


def _index(root, mode, execution_id=None, diagnostics=False):
    """Login helper: read committed manifests; legacy DBs require explicit migration."""
    root = Path(root).resolve()
    records, files = [], []
    generation = uuid.uuid4().hex
    cache = root / 'results/slurm/.sync' / generation
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / '.theta-managed.json', {'project_id': workspace_id(root), 'expires_at': __import__('time').time() + 7200})
    try:
        selected = [read_json(receipt) for receipt in sorted((root / 'results/slurm').glob('*/submission.json'))]
        selected = [record for record in selected if not execution_id or record['execution_id'] == execution_id]
        managed = refresh_many(root, [record for record in selected if record.get('execution_policy')])
        by_id = {record['execution_id']: record for record in managed}
        for record in selected:
            receipt = receipt_path(root, record['execution_id'])
            record = by_id[record['execution_id']] if record['execution_id'] in by_id else refresh_status(root, record)
            if record.get('execution_policy'):
                publication = read_json(Path(record['execution_policy']['work_root']) / 'results/slurm' / record['execution_id'] / 'publication.json')
                if publication and publication.get('schema') != 'theta.publication/v1': raise ValueError('Unknown publication schema')
                if publication.get('sealed') and record.get('scheduler_state') in TERMINAL and not record.get('artifacts_finalized'):
                    record = finalize(root, record['execution_id'])
                record['last_durable'] = publication.get('durable', {})
                record['publication_state'] = ('sealed' if publication.get('sealed') else
                    'incomplete' if record.get('scheduler_state') in TERMINAL else
                    'publishing' if publication else 'pending')
                record['publication_manifest'] = publication
                # Scheduler-owned late stdout is a bounded control preview, not
                # part of the compute-sealed scientific payload/cleanup proof.
                previews = []
                for suffix in ('out', 'err'):
                    name = f'theta_{record["execution_id"][:12]}_{record.get("job_id")}.{suffix}'
                    log = receipt.parent / name
                    if not log.is_file() or log.is_symlink(): continue
                    with log.open('rb') as stream:
                        stream.seek(max(0, log.stat().st_size - 65536))
                        data = stream.read(65536)
                    relative = f'results/slurm/{record["execution_id"]}/scheduler.{suffix}.preview.log'
                    target = cache / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    sha = __import__('hashlib').sha256(data).hexdigest()
                    files.append({'path': relative, 'source': target.relative_to(root).as_posix(),
                                  'size': len(data), 'sha256': sha, 'remote_root': str(root),
                                  'execution_id': record['execution_id'], 'finalized': False})
                    previews.append({'path': relative, 'truncated_bytes': max(0, log.stat().st_size - len(data))})
                record['scheduler_log_previews'] = previews
                records.append(record)
                if not record.get('remote_cleaned_at'):
                    for item in publication.get('files', []):
                        if mode == 'metrics' and item['category'] not in (('metrics', 'diagnostics') if diagnostics else ('metrics',)): continue
                        files.append({**item, 'execution_id': record['execution_id'],
                                      'remote_root': record['execution_policy']['work_root'],
                                      'finalized': publication.get('sealed', False)})
                continue
            if record['state'] in TERMINAL and record.get('scheduler_state') in TERMINAL and not record.get('artifacts_finalized'):
                record = finalize(root, record['execution_id'])
            records.append(record)
            finalized = record.get('artifacts_finalized')
            artifacts = read_json(receipt.parent / 'artifacts.json').get('files', []) if finalized else None
            paths = [safe_path(root, item['path']) for item in artifacts] if artifacts is not None else list(set(owned_files(root, record)))
            if mode == 'results' and finalized and not record.get('remote_cleaned_at'):
                paths += list(owned_files(root, record))  # include newly generated post-run outputs
            # The receipt is transferred too, so manually submitted runs are discovered.
            paths += [receipt]
            if record.get('remote_cleaned_at'): paths = [receipt]
            if finalized: paths.append(receipt.parent / 'artifacts.json')
            known = {item['path']: item for item in (artifacts or [])}
            for path in sorted(set(paths)):
                relative = path.relative_to(root).as_posix()
                if mode == 'metrics' and (relative.startswith('results/checkpoints/') or
                        (path.suffix.lower() not in ('.json', '.jsonl', '.csv', '.log', '.out', '.err', '.yaml', '.yml', '.sbatch')
                         and not ('/monitors/' in relative and path.name.startswith('snapshot_') and path.suffix == '.npz'))):
                    continue
                if path.name.endswith(('-wal', '-shm', '-journal')): continue
                source = relative
                if relative not in known:
                    # A transfer sees a fixed snapshot, even while training keeps appending.
                    backup = cache / relative
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix.lower() in ('.db', '.sqlite', '.sqlite3'):
                        raise ValueError('Legacy live SQLite cannot be synced on the login node; export on its owning compute node or migrate a closed study')
                    else:
                        length = path.stat().st_size
                        with path.open('rb') as src, backup.open('wb') as dest:
                            remaining = length
                            while remaining:
                                block = src.read(min(1024 * 1024, remaining))
                                if not block: break
                                dest.write(block)
                                remaining -= len(block)
                        # Readers only receive complete records from active append-only files.
                        if path.suffix.lower() in ('.jsonl', '.csv'):
                            with backup.open('rb+') as stream:
                                data = stream.read()
                                if data and not data.endswith(b'\n'):
                                    stream.truncate(data.rfind(b'\n') + 1)
                    source = backup.relative_to(root).as_posix()
                    path = backup
                info = path.stat()
                expected = known.get(relative, {}).get('sha256') or digest(path)
                files.append({'path': relative, 'source': source, 'size': info.st_size,
                              'sha256': expected, 'execution_id': record['execution_id'],
                              'finalized': bool(finalized and relative in known)})
        return {'project_id': workspace_id(root), 'executions': records, 'files': files, 'generation': generation}
    except BaseException:
        shutil.rmtree(cache)
        raise


def sync(root, site, mode='metrics', *, execution_id=None, experiment=None, diagnostics=False, cancel_event=None, progress=None):
    if mode not in ('metrics', 'results'): raise ValueError('Unknown sync mode')
    root = Path(root).resolve()
    config = connection(root, site)
    from .transport import ssh_session
    with ssh_session(root, config):
        return _sync(root, site, config, mode, execution_id=execution_id, experiment=experiment,
                     diagnostics=diagnostics, cancel_event=cancel_event, progress=progress)


def _sync(root, site, config, mode, *, execution_id=None, experiment=None, diagnostics=False, cancel_event=None, progress=None):
    state_path = root / '.thetaide/site_sync.json'
    with transfer_lock(root):
        state = read_json(state_path)
        status = state.setdefault(site, {})
        listing = None
        try:
            options = ['--mode', mode]
            if execution_id: options += ['--execution', execution_id]
            if diagnostics: options += ['--diagnostics']
            listing = remote(root, site, '_index', *options)
            if listing['project_id'] != workspace_id(root): raise ValueError('Remote project identity does not match')
            if experiment:
                # Filter the returned manifest without requiring new cluster CLI flags.
                records = [record for record in listing['executions'] if record.get('experiment') == experiment]
                selected_ids = {record['execution_id'] for record in records}
                listing = {**listing, 'executions': records,
                           'files': [item for item in listing['files'] if item.get('execution_id') in selected_ids]}
            synced_executions = listing['executions']
            if execution_id or experiment:
                # A scoped pull must retain discovery/status for other experiments.
                synced_executions = [record for record in status.get('executions', [])
                                     if record['execution_id'] != execution_id and (not experiment or record.get('experiment') != experiment)] + listing['executions']
            transfer_output.emit(f'Found {len(listing["executions"])} executions and {len(listing["files"])} published files.')
            for record in listing['executions']:
                job_state = record.get('scheduler_state') or record.get('state', 'unknown')
                transfer_output.emit(f'Job {record.get("job_id") or record["execution_id"]}: {job_state}')
                detail = {key: record[key] for key in ('scheduler_state_raw', 'scheduler_reason', 'scheduler_exit_code', 'scheduler_elapsed', 'scheduler_time_limit', 'termination_reason') if record.get(key)}
                if detail:
                    transfer_output.emit('Job details: ' + json.dumps(detail))
                if record.get('status_error'):
                    transfer_output.emit('Scheduler status unavailable: ' + record['status_error'])
            if any(record.get('execution_policy') for record in listing['executions']):
                result = sync_objects(root, site, config, listing, cancel_event=cancel_event, progress=progress)
                status.update(last_sync=now(), mode=mode, executions=synced_executions, error=None)
                write_json(state_path, state)
                return {'site': site, 'last_sync': status['last_sync'], **result, 'executions': listing['executions']}
            ledger_path = root / '.thetaide/site_transfers.json'
            ledger = read_json(ledger_path)
            pending = []
            for item in listing['files']:
                path = safe_path(root, item['path'])
                safe_path(root, item['source'])
                if not item['path'].startswith('results/') or not item['source'].startswith('results/'):
                    raise ValueError('Remote artifact lies outside results')
                previous = ledger.get(site + ':' + item['path'])
                stamp = [path.stat().st_size, path.stat().st_mtime_ns] if path.is_file() else None
                unchanged = previous and previous.get('sha256') == item['sha256'] and previous.get('local_stamp') == stamp
                # The ledger avoids rereading unchanged checkpoints on every metrics refresh.
                if path.is_file() and (unchanged or digest(path) == item['sha256']):
                    ledger[site + ':' + item['path']] = {**item, 'local_stamp': stamp}
                else:
                    previous = ledger.get(site + ':' + item['path'])
                    if path.exists() and previous is None:
                        raise ValueError(f'Local output collision: {item["path"]}')
                    pending.append(item)
            write_json(ledger_path, ledger)
            stage = root / '.thetaide/site-staging' / digest_name(site)
            stage.mkdir(parents=True, exist_ok=True)
            names = stage / 'files.list'
            names.write_bytes(b''.join(item['source'].encode() + b'\0' for item in pending))
            if pending:
                run_transfer([config.get('rsync_executable', 'rsync'), '-r', *transfer_output.progress_flags(), '--protect-args', '--from0', '--files-from=' + str(names),
                                '--partial-dir=.theta-partial', '--rsync-path=' + config.get('remote_rsync', 'rsync'), '-e', rsync_shell(root, config),
                                config['host'] + ':' + config['remote_root'] + '/', str(stage) + '/'],
                               cancel_event=cancel_event)
            for item in pending:
                source = safe_path(stage, item['source'])
                if not source.is_file() or digest(source) != item['sha256']:
                    raise ValueError(f'Artifact changed during transfer; sync again: {item["path"]}')
                target = safe_path(root, item['path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                transfer_output.emit(f'Saved {target} ({target.stat().st_size:,} bytes)')
                ledger[site + ':' + item['path']] = {**item, 'local_stamp': [target.stat().st_size, target.stat().st_mtime_ns]}
                write_json(ledger_path, ledger)
            shutil.rmtree(stage)
            status.update(last_sync=now(), mode=mode, executions=synced_executions, error=None)
            write_json(state_path, state)
            return {'site': site, 'last_sync': status['last_sync'], 'transferred': len(pending),
                    'skipped': len(listing['files']) - len(pending), 'bytes': sum(item['size'] for item in pending),
                    'executions': listing['executions']}
        except Exception as exc:
            status['error'] = str(exc)
            write_json(state_path, state)
            raise
        finally:
            if listing and listing.get('generation'):
                try: remote(root, site, '_release', listing['generation'])
                except Exception: pass  # Remote scratch snapshots can be released on the next visit.


def clean(root, site, execution_id, *, preview=False):
    """Explicit cleanup requires a complete, reverified local copy of the final manifest."""
    root = Path(root).resolve()
    with transfer_lock(root):
        path = receipt_path(root, execution_id)
        record = read_json(path)
        if record.get('site') != site or not record.get('artifacts_finalized'):
            raise ValueError('Sync finalized results for this site first')
        manifest = read_json(path.parent / 'artifacts.json')
        if not manifest.get('files'): raise ValueError('Final artifact manifest is missing')
        for item in manifest['files']:
            if digest(safe_path(root, item['path'])) != item['sha256']:
                raise ValueError('Local artifacts are missing or changed; cleanup refused')
        return remote(root, site, '_clean', execution_id, *(['--preview'] if preview else []),
                      input=json.dumps({'project_id': workspace_id(root), 'files': manifest['files']}))


def clean_remote(root, execution_id, proof, *, preview=False):
    with transfer_lock(root):
        return _clean_remote(root, execution_id, proof, preview=preview)


def _clean_remote(root, execution_id, proof, *, preview=False):
    root = Path(root).resolve()
    path = receipt_path(root, execution_id)
    record = refresh_status(root, read_json(path))
    manifest = read_json(path.parent / 'artifacts.json')
    if record.get('remote_cleaned_at'): return {'execution_id': execution_id, 'removed': 0}
    if (proof.get('project_id') != workspace_id(root) or record.get('state') not in TERMINAL
            or record.get('scheduler_state') not in TERMINAL
            or not record.get('artifacts_finalized') or proof.get('files') != manifest.get('files')):
        raise ValueError('Cleanup requires a matching verified terminal execution')
    # Conservatively prevent cleanup while any other allocation is nonterminal.
    for other in (root / 'results/slurm').glob('*/submission.json'):
        value = refresh_status(root, read_json(other))
        if value.get('state') != 'submission_failed' and value.get('scheduler_state') not in TERMINAL:
            raise ValueError('Another execution may depend on these outputs; wait for it to finish')
    for other in (root / 'results/slurm').glob('*/submission.json'):
        value = read_json(other)
        if value.get('continuation', {}).get('parent_execution_id') == execution_id and not value.get('remote_cleaned_at'):
            raise ValueError('Continuation pins these study/checkpoint dependencies')
    if preview:
        return {'execution_id': execution_id, 'preview': True, 'files': len(manifest['files']),
                'payload_bytes': sum(item['size'] for item in manifest['files']),
                'workspace': str(path.parent / 'workspace'), 'release_staged_inputs': bool(record.get('execution_policy')),
                'retained': ['submission.json', 'artifacts.json', 'source identity'],
                'note': 'Active transfer leases and remote integrity are checked again before deletion'}
    if record.get('execution_policy'):
        from .maintenance import active_leases
        if active_leases(root): raise ValueError('Active transfer lease pins publication; retry after release/expiry')
        work = Path(record['execution_policy']['work_root'])
        published = work / 'results/slurm' / execution_id
        if published.exists() and published.is_symlink(): raise ValueError('Invalid publication ownership')
        # All proofs were compared above; immutable objects were verified locally.
        # Delete only this execution's namespace; retained receipt/manifests are tombstones.
        if published.exists(): shutil.rmtree(published)
        inputs = work / 'inputs' / execution_id
        if inputs.exists() and not inputs.is_symlink(): shutil.rmtree(inputs)
        record['remote_cleaned_at'] = now()
        write_json(path, record)
        return {'execution_id': execution_id, 'removed': len(manifest['files'])}
    allowed = {p.relative_to(root).as_posix() for p in owned_files(root, record)}
    for item in manifest['files']:
        if item['path'] not in allowed or digest(safe_path(root, item['path'])) != item['sha256']:
            raise ValueError('Remote artifact changed; cleanup refused')
    for item in manifest['files']:
        safe_path(root, item['path']).unlink()
    workspace = path.parent / 'workspace'
    if workspace.is_dir() and not workspace.is_symlink(): shutil.rmtree(workspace)
    record['remote_cleaned_at'] = now()
    write_json(path, record)
    # Retain the submission receipt and final manifest as provenance/tombstones.
    return {'execution_id': execution_id, 'removed': len(manifest['files'])}


def activate(root, source):
    root, source = Path(root).resolve(), Path(source).resolve()
    manifest = read_json(source / '.thetaide/source.json')
    if not manifest.get('files'): raise ValueError('Uploaded source manifest is missing')
    identity_path = root / '.thetaide/workspace.json'
    if identity_path.exists() and read_json(identity_path).get('id') != manifest['project_id']:
        raise ValueError('Remote directory belongs to another project')
    previous = read_json(root / '.thetaide/source.json')
    with transfer_lock(root):
        allowed = set(source_files(source))
        for relative, expected in manifest['files'].items():
            if relative not in allowed or digest(safe_path(source, relative)) != expected:
                raise ValueError(f'Uploaded source failed verification: {relative}')
        for relative in set(previous.get('files', {})) - set(manifest['files']):
            path = safe_path(root, relative)
            if path.is_file(): path.unlink()
        for relative, expected in manifest['files'].items():
            target = safe_path(root, relative)
            if target.is_file() and digest(target) == expected:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.theta-upload')
            shutil.copyfile(safe_path(source, relative), temporary)
            os.replace(temporary, target)
        # Dataset uploads are explicit and separate from the source manifest.
        if (source / 'in/datasets').is_dir():
            for path in (source / 'in/datasets').rglob('*'):
                if path.is_file():
                    relative = path.relative_to(source).as_posix()
                    target = safe_path(root, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + '.theta-upload')
                    shutil.copyfile(safe_path(source, relative), temporary)
                    os.replace(temporary, target)
        # Build the pinned archive from the verified remote snapshot. Transferring
        # a full ZIP on every push would defeat per-file rsync reuse.
        target = root / '.thetaide/sources' / (manifest['sha256'] + '.zip')
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target.parent / 'ownership.json', {'project_id': manifest['project_id']})
        if not target.exists():
            archive(source, target.with_suffix('.tmp'))
            os.replace(target.with_suffix('.tmp'), target)
        manifest['archive_sha256'] = digest(target)
        write_json(root / '.thetaide/source.json', manifest)
        lease = read_json(source / '.theta-managed.json')
        if lease.get('project_id') == manifest['project_id']:
            lease['expires_at'] = 0
            write_json(source / '.theta-managed.json', lease)
    return {'source_sha256': manifest['sha256']}


def release(root, generation):
    if not re.fullmatch(r'[a-f0-9]{32}', generation): raise ValueError('Invalid transfer generation')
    directory = safe_path(root, 'results/slurm/.sync/' + generation)
    if directory.exists(): shutil.rmtree(directory)
    return {'released': generation}


def status_local(root):
    root = Path(root).resolve()
    return refresh_many(root, [read_json(path)
            for path in sorted((root / 'results/slurm').glob('*/submission.json'))])


def sync_objects(root, site, config, listing, *, cancel_event=None, progress=None):
    """Fetch immutable missing chunks, verify, then materialize the existing result layout."""
    ledger_path = root / '.thetaide/site_transfers.json'
    ledger = read_json(ledger_path)
    cache = root / '.thetaide/site-objects' / digest_name(site)
    cache.mkdir(parents=True, exist_ok=True)
    pending, objects = [], {}
    transferred_bytes = 0
    for item in listing['files']:
        target = safe_path(root, item['path'])
        if not item['path'].startswith('results/'): raise ValueError('Artifact outside results')
        previous = ledger.get(site + ':' + item['path'])
        stamp = [target.stat().st_size, target.stat().st_mtime_ns] if target.is_file() else None
        if previous and previous.get('sha256') == item['sha256'] and previous.get('local_stamp') == stamp: continue
        if target.exists() and previous is None and digest(target) != item['sha256']:
            raise ValueError(f'Local output collision: {item["path"]}')
        chunks = item.get('chunks', [{key: item[key] for key in ('source', 'size', 'sha256')}]) if 'chunks' not in item else item['chunks']
        if sum(chunk.get('length', chunk['size']) for chunk in chunks) != item['size']: raise ValueError('Invalid chunk lengths')
        for chunk in chunks:
            if not re.fullmatch(r'[a-f0-9]{64}', chunk['sha256']): raise ValueError('Invalid object ID')
            safe_path(root, chunk['source'])
            if not chunk['source'].startswith('results/'): raise ValueError('Object outside results')
            objects[(item.get('remote_root', config['remote_root']), chunk['sha256'])] = chunk
        pending.append((item, chunks))
    try:
        total_bytes = sum(chunk['size'] for chunk in objects.values())
        completed_bytes = 0
        batches = {}
        for (remote_root, sha), chunk in objects.items():
            if cancel_event and cancel_event.is_set(): raise RuntimeError('Transfer cancelled; completed objects retained for retry')
            target = cache / sha
            if target.is_file() and digest(target) == sha:
                completed_bytes += chunk['size']
                continue
            batches.setdefault(remote_root, []).append((sha, chunk))
        renewed_at = time.monotonic()

        def renew_lease():
            nonlocal renewed_at
            if listing.get('generation') and time.monotonic() - renewed_at >= 300:
                remote(root, site, '_renew', listing['generation'])
                renewed_at = time.monotonic()

        for remote_root, batch in batches.items():
            if cancel_event and cancel_event.is_set(): raise RuntimeError('Transfer cancelled; completed objects retained for retry')
            renew_lease()
            stage = cache / 'incoming'
            stage.mkdir(exist_ok=True)
            names = stage / 'files.list'
            names.write_bytes(b''.join(chunk['source'].encode() + b'\0' for _, chunk in batch))
            transfer_output.emit(f'Downloading {len(batch)} missing objects from {remote_root} in one rsync batch.')

            def cache_completed(*, strict):
                nonlocal completed_bytes, transferred_bytes
                invalid = False
                for sha, chunk in batch:
                    source = safe_path(stage, chunk['source'])
                    valid = source.is_file() and source.stat().st_size == chunk['size'] and digest(source) == sha
                    if not valid:
                        invalid = True
                        continue
                    os.replace(source, cache / sha)
                    transferred_bytes += chunk['size']
                    completed_bytes += chunk['size']
                    if progress: progress(completed_bytes, total_bytes)
                if strict and invalid:
                    raise ValueError('Artifact changed during transfer; immutable object checksum mismatch')

            try:
                run_transfer([config.get('rsync_executable', 'rsync'), '-r', *transfer_output.progress_flags(), '--protect-args', '--from0',
                            '--files-from=' + str(names), '--partial-dir=.theta-partial',
                            '--rsync-path=' + config.get('remote_rsync', 'rsync'), '-e', rsync_shell(root, config),
                            config['host'] + ':' + remote_root + '/', str(stage) + '/'], cancel_event=cancel_event,
                            **({'on_poll': renew_lease} if listing.get('generation') else {}))
            except Exception:
                cache_completed(strict=False)
                raise
            cache_completed(strict=True)
        for item, chunks in pending:
            target = safe_path(root, item['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name('.' + target.name + '.theta-sync')
            try:
                with temporary.open('wb') as output:
                    for chunk in chunks:
                        with (cache / chunk['sha256']).open('rb') as source:
                            offset, length = chunk.get('offset', 0), chunk.get('length', chunk['size'])
                            if offset < 0 or length < 0 or offset + length > chunk['size']: raise ValueError('Invalid bundle range')
                            source.seek(offset)
                            while length:
                                data = source.read(min(length, 1024 * 1024))
                                if not data: raise ValueError('Truncated object')
                                output.write(data)
                                length -= len(data)
                if digest(temporary) != item['sha256']: raise ValueError('Reconstructed artifact checksum mismatch')
                os.replace(temporary, target)
                transfer_output.emit(f'Saved {target} ({target.stat().st_size:,} bytes)')
            finally:
                temporary.unlink(missing_ok=True)
            ledger[site + ':' + item['path']] = {**item, 'local_stamp': [target.stat().st_size, target.stat().st_mtime_ns]}
            write_json(ledger_path, ledger)
        for record in listing['executions']:
            if not record.get('execution_policy'): continue
            local_record = dict(record)
            publication = local_record.pop('publication_manifest', {})
            path = receipt_path(root, record['execution_id'])
            old = read_json(path)
            if old and old.get('project_id') != listing['project_id']: raise ValueError('Receipt ownership collision')
            write_json(path, local_record)
            write_json(path.parent / 'publication.json', publication)
            if record.get('artifacts_finalized'):
                write_json(path.parent / 'artifacts.json', {'execution_id': record['execution_id'], 'files': publication.get('files', [])})
        report_errors = []
        directories = {safe_path(root, item['path']).parent for item, _ in pending
                       if '/monitors/' in item['path'] and Path(item['path']).name.startswith('snapshot_') and item['path'].endswith('.json')}
        if directories:
            from src.app.monitoring.report import write_report
            for directory in directories:
                try:
                    if all(path.with_suffix('.npz').exists() for path in directory.glob('snapshot_*.json')):
                        write_report(directory)
                except Exception as exc: report_errors.append(str(exc))
        # The cache is transfer scratch, not a second permanent results archive.
        # Keep only latest append prefixes for incremental growth. Failed attempts
        # retain verified objects above; collection happens after complete success.
        retained = {chunk['sha256'] for key, value in ledger.items() if key.startswith(site + ':')
                    and Path(value['path']).suffix in ('.csv', '.jsonl', '.log', '.out', '.err')
                    for chunk in value.get('chunks', [])}
        for cached in cache.iterdir():
            if re.fullmatch(r'[a-f0-9]{64}', cached.name) and cached.name not in retained and not cached.is_symlink(): cached.unlink()
        incoming = cache / 'incoming'
        if incoming.exists(): shutil.rmtree(incoming)
        return {'transferred': len(pending), 'bytes': transferred_bytes, 'skipped': len(listing['files']) - len(pending), 'diagnostic_report_errors': report_errors}
    finally:
        # Successful chunks remain resumable; partial rsync objects retain their names.
        pass


def renew(root, generation):
    if not re.fullmatch(r'[a-f0-9]{32}', generation): raise ValueError('Invalid transfer generation')
    with transfer_lock(root):
        path = safe_path(root, 'results/slurm/.sync/' + generation + '/.theta-managed.json')
        lease = read_json(path)
        if lease.get('project_id') != workspace_id(root): raise ValueError('Transfer lease is missing or owned by another project')
        lease['expires_at'] = __import__('time').time() + 7200
        write_json(path, lease)
    return {'renewed': generation}


def run_transfer(arguments, *, cancel_event=None, live_output=False, on_poll=None):
    if transfer_output.streaming():
        return transfer_output.run_streamed(arguments, cancel_event, on_poll=on_poll)
    if live_output:
        # The caller is already attached to the floating terminal PTY. Inherit
        # its streams so progress and the original rsync error remain visible.
        return subprocess.run(arguments, check=True, timeout=3600)
    if cancel_event is None and on_poll is None:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(arguments, stdout=output, stderr=output, timeout=3600)
            if result.returncode:
                output.seek(0, os.SEEK_END)
                output.seek(max(0, output.tell() - 65536))
                detail = output.read().decode(errors='replace').strip()
                raise RuntimeError(f'rsync failed (exit {result.returncode}): {detail or "no error output"}')
            return result
    import signal
    import time
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(arguments, stdout=output, stderr=output, start_new_session=True)
        deadline = time.monotonic() + 3600
        try:
            while process.poll() is None:
                if (cancel_event and cancel_event.is_set()) or time.monotonic() > deadline:
                    raise RuntimeError('Transfer cancelled or timed out; verified objects retained for retry')
                if on_poll: on_poll()
                try: process.wait(timeout=.2)
                except subprocess.TimeoutExpired: pass
            if process.returncode:
                output.seek(0, os.SEEK_END)
                output.seek(max(0, output.tell() - 65536))
                detail = output.read().decode(errors='replace').strip()
                raise RuntimeError(f'rsync failed (exit {process.returncode}): {detail or "no error output"}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
