"""Explicit SSH/rsync operations. No networking happens during discovery or imports."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .execution import TERMINAL, finalize, owned_files, receipt_path, refresh_status
from .snapshot import freeze, source_files
from .slurm import rsync_shell
from .storage import digest, read_json, safe_path, transfer_lock, workspace_id, write_json


def configure(root, site, host, remote_root, python='venv/bin/python3'):
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', host) or host.startswith('-'):
        raise ValueError('Use an SSH config alias or user@hostname')
    if remote_root == '/' or not remote_root.startswith('/') or any(part in ('.', '..') for part in remote_root.split('/')):
        raise ValueError('Remote project directory must be an absolute path without traversal')
    if any(c in remote_root + python for c in '\x00\n\r') or not python or python.startswith('-'):
        raise ValueError('Invalid remote path or Python command')
    from .catalog import configure as configure_component
    return configure_component(root, site, {'host': host, 'remote_root': remote_root.rstrip('/'), 'python': python})


def connections(root):
    from . import profiles
    from .catalog import load
    from src.app.workspace import WorkspacePaths
    return {selector: load(root, selector).get('connection', {}) for selector in profiles(WorkspacePaths(root))}


def connection(root, site):
    from .catalog import load
    config = load(root, site)
    values = config.get('connection') or {}
    if not values.get('host') or not values.get('remote_root'):
        raise ValueError(f'Edit the connection in the {site!r} site component in Components first')
    return {**values, '_root': str(Path(root).resolve()), '_shortcuts': config.get('shortcuts', {})}


def ssh(config, arguments, *, input=None):
    command = 'cd ' + shlex.quote(config['remote_root']) + ' && ' + shlex.join(arguments)
    from .slurm import ssh_arguments
    result = subprocess.run([*ssh_arguments(config.get('_root', Path.cwd()), config), '--', config['host'], command],
                            input=input, capture_output=True, text=True, timeout=180)
    if result.returncode: raise RuntimeError(result.stderr.strip() or 'Remote command failed')
    return result.stdout


def remote(root, site, command, *arguments, input=None):
    config = connection(root, site)
    return json.loads(ssh(config, [config['python'], '-m', 'src.app.sites', command, '--root', '.', *arguments], input=input))


def check(root, site):
    config = connection(root, site)
    result = subprocess.run([config.get('rsync_executable', 'rsync'), '--version'], capture_output=True, text=True, check=True)
    if not re.search(r'version\s+[3-9]\.', result.stdout):
        raise ValueError('Project transfers require rsync 3 or newer locally and on the cluster')
    # This check works before the first project upload and does not require Theta remotely.
    check_script = "command -v sbatch && command -v squeue && {rsync} --version && test -w . && { command -v sacct || printf '%s\\n' 'sacct unavailable: accounting will be unknown'; }"
    check_script = check_script.replace('{rsync}', shlex.quote(config.get('remote_rsync', 'rsync')))
    output = ssh(config, ['bash', '-lc', check_script])
    if not re.search(r'version\s+[3-9]\.', output): raise ValueError('Remote rsync must be version 3 or newer')
    python_version = ssh(config, [config['python'], '--version']).strip()
    return {'site': site, 'checks': output.strip(), 'python': python_version}


def push(root, site, *, datasets=False):
    root = Path(root).resolve()
    config = connection(root, site)
    with transfer_lock(root), tempfile.TemporaryDirectory(prefix='theta-push-') as temporary:
        stage = Path(temporary)
        source = freeze(root, stage)
        # Transfer a frozen list, not a tree that the user may still be editing.
        files = [*source['files'], '.thetaide/source.json']
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
        # Refuse an existing workspace belonging to a different local project.
        identity = shlex.quote(config['remote_root'] + '/.thetaide/workspace.json')
        probe = 'if [ -f ' + identity + ' ]; then cat ' + identity + '; else printf "{}"; fi'
        existing = json.loads(ssh(config, ['bash', '-c', probe]))
        if existing.get('id') and existing['id'] != source['project_id']:
            raise ValueError('Remote directory belongs to another project; choose a new directory')
        upload = config['remote_root'] + '/.thetaide/uploads/' + source['sha256']
        ssh(config, ['mkdir', '-p', upload])
        subprocess.run([config.get('rsync_executable', 'rsync'), '-rt', '--protect-args', '--from0', '--files-from=' + str(listing),
                        '--partial-dir=.theta-partial', '--link-dest=' + config['remote_root'], '--rsync-path=' + config.get('remote_rsync', 'rsync'), '-e', rsync_shell(root, config),
                        str(stage) + '/', config['host'] + ':' + upload + '/'], check=True, timeout=3600)
        ssh(config, [config['python'], upload + '/run_pipeline.py', 'site', '_activate',
                     '--root', config['remote_root'], '--source', upload])
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


def index(root, mode):
    """Runs on the cluster. Snapshot mutable SQLite files using SQLite's backup API."""
    root = Path(root).resolve()
    records, files = [], []
    generation = uuid.uuid4().hex
    cache = root / 'results/slurm/.sync' / generation
    for receipt in sorted((root / 'results/slurm').glob('*/submission.json')):
        record = refresh_status(root, read_json(receipt))
        if record['state'] in TERMINAL and record.get('scheduler_state') in TERMINAL and not record.get('artifacts_finalized'):
            record = finalize(root, record['execution_id'])
        records.append(record)
        finalized = record.get('artifacts_finalized')
        artifacts = read_json(receipt.parent / 'artifacts.json').get('files', []) if finalized else None
        paths = [safe_path(root, item['path']) for item in artifacts] if artifacts is not None else list(set(owned_files(root, record)))
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
                    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=10) as src:
                        with sqlite3.connect(backup) as dest: src.backup(dest)
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


def sync(root, site, mode='metrics'):
    if mode not in ('metrics', 'results'): raise ValueError('Unknown sync mode')
    root = Path(root).resolve()
    config = connection(root, site)
    state_path = root / '.thetaide/site_sync.json'
    with transfer_lock(root):
        state = read_json(state_path)
        status = state.setdefault(site, {})
        listing = None
        try:
            listing = remote(root, site, '_index', '--mode', mode)
            if listing['project_id'] != workspace_id(root): raise ValueError('Remote project identity does not match')
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
                subprocess.run([config.get('rsync_executable', 'rsync'), '-rt', '--protect-args', '--from0', '--files-from=' + str(names),
                                '--partial-dir=.theta-partial', '--rsync-path=' + config.get('remote_rsync', 'rsync'), '-e', rsync_shell(root, config),
                                config['host'] + ':' + config['remote_root'] + '/', str(stage) + '/'],
                               check=True, timeout=3600)
            for item in pending:
                source = safe_path(stage, item['source'])
                if not source.is_file() or digest(source) != item['sha256']:
                    raise ValueError(f'Artifact changed during transfer; sync again: {item["path"]}')
                target = safe_path(root, item['path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                ledger[site + ':' + item['path']] = {**item, 'local_stamp': [target.stat().st_size, target.stat().st_mtime_ns]}
                write_json(ledger_path, ledger)
            shutil.rmtree(stage)
            status.update(last_sync=now(), mode=mode, executions=listing['executions'], error=None)
            write_json(state_path, state)
            return {'site': site, 'last_sync': status['last_sync'], 'transferred': len(pending),
                    'skipped': len(listing['files']) - len(pending), 'executions': listing['executions']}
        except Exception as exc:
            status['error'] = str(exc)
            write_json(state_path, state)
            raise
        finally:
            if listing and listing.get('generation'):
                try: remote(root, site, '_release', listing['generation'])
                except Exception: pass  # Remote scratch snapshots can be released on the next visit.


def clean(root, site, execution_id):
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
        return remote(root, site, '_clean', execution_id,
                      input=json.dumps({'project_id': workspace_id(root), 'files': manifest['files']}))


def clean_remote(root, execution_id, proof):
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
        if value.get('state') not in TERMINAL | {'submission_failed'}:
            raise ValueError('Another execution may depend on these outputs; wait for it to finish')
    allowed = {p.relative_to(root).as_posix() for p in owned_files(root, record)}
    for item in manifest['files']:
        if item['path'] not in allowed or digest(safe_path(root, item['path'])) != item['sha256']:
            raise ValueError('Remote artifact changed; cleanup refused')
    for item in manifest['files']:
        safe_path(root, item['path']).unlink()
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
        for relative in manifest['files']:
            target = safe_path(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.theta-upload')
            shutil.copy2(safe_path(source, relative), temporary)
            os.replace(temporary, target)
        # Dataset uploads are explicit and separate from the source manifest.
        if (source / 'in/datasets').is_dir():
            for path in (source / 'in/datasets').rglob('*'):
                if path.is_file():
                    relative = path.relative_to(source).as_posix()
                    target = safe_path(root, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + '.theta-upload')
                    shutil.copy2(safe_path(source, relative), temporary)
                    os.replace(temporary, target)
        write_json(root / '.thetaide/source.json', manifest)
    return {'source_sha256': manifest['sha256']}


def release(root, generation):
    if not re.fullmatch(r'[a-f0-9]{32}', generation): raise ValueError('Invalid transfer generation')
    directory = safe_path(root, 'results/slurm/.sync/' + generation)
    if directory.exists(): shutil.rmtree(directory)
    return {'released': generation}


def status_local(root):
    root = Path(root).resolve()
    return [refresh_status(root, read_json(path))
            for path in sorted((root / 'results/slurm').glob('*/submission.json'))]
