"""Explicit site operations, usable locally and from a cluster login node."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('list', 'cancel', 'configure', 'check', 'ssh', 'push', 'launch', 'sync', 'status', 'clean', '_index', '_run', '_clean', 'finalize', '_activate', '_release', '_renew', '_status', 'maintenance', 'inventory'):
        command = commands.add_parser(name)
        command.add_argument('--root', type=Path, default=Path.cwd())
        if name in ('configure', 'check', 'ssh', 'push', 'launch', 'sync', 'status', 'clean'): command.add_argument('site')
        if name == 'configure':
            command.add_argument('--host', required=True)
            command.add_argument('--remote-root', required=True)
            command.add_argument('--python', default='venv/bin/python3')
            command.add_argument('--scheduler', help='Installed scheduler capability for a newly created site')
        if name in ('clean', '_clean'): command.add_argument('--preview', action='store_true')
        if name == 'maintenance': command.add_argument('--apply', action='store_true')
        if name == 'push': command.add_argument('--include-datasets', action='store_true')
        if name == 'launch': command.add_argument('--experiment', required=True)
        if name in ('sync', '_index'):
            command.add_argument('--mode', choices=('metrics', 'results'), default='metrics')
            command.add_argument('--execution')
            command.add_argument('--diagnostics', action='store_true')
        if name in ('clean', 'cancel', '_run', '_clean', 'finalize', '_release', '_renew'): command.add_argument('execution_id')
        if name == '_activate': command.add_argument('--source', type=Path, required=True)
    args = parser.parse_args(argv)
    from . import transfer
    from .execution import finalize, run, receipt_path, refresh_status
    from .storage import read_json
    try:
        if args.command in ('maintenance', 'inventory'):
            from .maintenance import collect, inventory
            result = collect(args.root, apply=args.apply) if args.command == 'maintenance' else inventory(args.root)
        elif args.command == 'cancel':
            from .execution import cancel
            result = cancel(args.root, read_json(receipt_path(args.root, args.execution_id)))
        elif args.command == 'list': result = transfer.connections(args.root)
        elif args.command == 'configure': result = transfer.configure(args.root, args.site, args.host, args.remote_root, args.python, scheduler=args.scheduler)
        elif args.command == 'check': result = transfer.check(args.root, args.site)
        elif args.command == 'ssh':
            # Interactive login using the site's managed SSH options; replaces this process.
            import os
            from .transport import ssh_arguments
            config = transfer.connection(args.root, args.site)
            os.execvp('ssh', [*ssh_arguments(args.root, config, True), '--', config['host']])
        elif args.command == 'launch':
            from .catalog import load
            from .scheduler_api import get_scheduler
            from src.app.workspace import WorkspacePaths
            cfg = {'site': load(args.root, args.site)}
            return get_scheduler(cfg=cfg, paths=WorkspacePaths(args.root)).launch_in_terminal(args.root, args.site, args.experiment)
        elif args.command == 'push': result = transfer.push(args.root, args.site, datasets=args.include_datasets)
        elif args.command == 'sync': result = transfer.sync(args.root, args.site, args.mode, execution_id=args.execution, diagnostics=args.diagnostics)
        elif args.command == 'status': result = transfer.remote(args.root, args.site, '_status')
        elif args.command == 'clean': result = transfer.clean(args.root, args.site, args.execution_id, preview=args.preview)
        elif args.command == '_status': result = transfer.status_local(args.root)
        elif args.command == '_index': result = transfer.index(args.root, args.mode, args.execution, args.diagnostics)
        elif args.command == '_clean': result = transfer.clean_remote(args.root, args.execution_id, json.load(sys.stdin), preview=args.preview)
        elif args.command == '_run': return run(args.root, args.execution_id)
        elif args.command == '_renew': result = transfer.renew(args.root, args.execution_id)
        elif args.command == '_release': result = transfer.release(args.root, args.execution_id)
        elif args.command == '_activate': result = transfer.activate(args.root, args.source)
        else:
            refresh_status(args.root, read_json(receipt_path(args.root, args.execution_id)))
            result = finalize(args.root, args.execution_id)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        if args.command == '_run':
            import traceback
            traceback.print_exc()
        else:
            print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
