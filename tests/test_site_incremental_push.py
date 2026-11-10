"""Exercise project pushes with real local rsync, replacing only the SSH transport."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from src.app.sites import transfer
from src.app.sites.snapshot import extract
from src.app.sites.storage import read_json


def test_push_transfers_only_changed_files_and_retains_pinned_archives(tmp_path, monkeypatch):
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir()
    remote.mkdir()
    (local / 'src').mkdir()
    component = local / 'src/component.py'
    component.write_text('version = 1\n')
    yaml = local / 'in/config/group/run.yaml'
    yaml.parent.mkdir(parents=True)
    yaml.write_text('seed: 1\n')
    (local / 'run_pipeline.py').write_text('# launcher\n')
    config = {'host': 'fake', 'remote_root': str(remote), 'python': 'python3'}
    monkeypatch.setattr(transfer, 'connection', lambda *_: config)
    try:
        executable = transfer.require_local_rsync(config)
    except ValueError as exc:
        pytest.skip(str(exc))
    batches = []

    def ssh(_config, arguments, **kwargs):
        if arguments[:2] == ['bash', '-c']:
            return json.dumps(read_json(remote / '.thetaide/workspace.json'))
        if arguments[:2] == ['mkdir', '-p']:
            Path(arguments[2]).mkdir(parents=True)
            return ''
        assert '_activate' in arguments
        return json.dumps(transfer.activate(remote, Path(arguments[-1])))

    def rsync(arguments, **kwargs):
        # Keep the real checksums/files list/link-dest behavior; replace SSH with
        # a local destination so the test never needs credentials or a server.
        assert '--checksum' in arguments
        listing = Path(next(arg.split('=', 1)[1] for arg in arguments if arg.startswith('--files-from=')))
        assert b'.thetaide/source.zip\0' not in listing.read_bytes()
        command = [executable, '--itemize-changes']
        i = 1
        while i < len(arguments) - 2:
            arg = arguments[i]
            if arg == '-e':
                i += 2
                continue
            if not arg.startswith('--rsync-path='):
                command.append(arg)
            i += 1
        command.extend([arguments[-2], arguments[-1].removeprefix('fake:')])
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        batches.append([line.split(' ', 1)[1] for line in result.stdout.splitlines() if line.startswith('>f')])

    monkeypatch.setattr(transfer, 'ssh', ssh)
    monkeypatch.setattr(transfer, 'run_transfer', rsync)
    first = transfer.push(local, 'cluster')
    original_inode = (remote / 'src/component.py').stat().st_ino
    # Even same-size edits with an unchanged mtime must upload.
    stamp = yaml.stat()
    yaml.write_text('seed: 2\n')
    os.utime(yaml, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    second = transfer.push(local, 'cluster')
    payloads = lambda batch: [name for name in batch if not name.startswith('.theta')]
    assert payloads(batches[1]) == ['in/config/group/run.yaml']
    assert (remote / 'src/component.py').stat().st_ino == original_inode
    assert (remote / 'in/config/group/run.yaml').read_text() == 'seed: 2\n'
    transfer.push(local, 'cluster')
    assert payloads(batches[2]) == []
    component.write_text('version = 2\n')
    transfer.push(local, 'cluster')
    assert payloads(batches[3]) == ['src/component.py']
    component.unlink()
    transfer.push(local, 'cluster')
    assert not (remote / 'src/component.py').exists()
    # Source archives still reproduce old and new experiments independently.
    for result, seed in [(first, 'seed: 1\n'), (second, 'seed: 2\n')]:
        destination = tmp_path / result['source_sha256']
        extract(remote / '.thetaide/sources' / (result['source_sha256'] + '.zip'), destination)
        assert (destination / 'in/config/group/run.yaml').read_text() == seed
        assert (destination / 'src/component.py').read_text() == 'version = 1\n'
