"""API-1 site-command adapter. SSH primitives live in transport; scheduler commands are optional."""
from .transport import ssh_arguments, ssh_session, rsync_shell
from .scheduler_api import scheduler_object


def __getattr__(name):
    if name not in {'submission_command', 'launch_in_terminal', 'bootstrap_command', 'shortcuts', 'verify_environment'}:
        raise AttributeError(name)
    return scheduler_object('slurm', 'commands.py:' + name)


if __name__ == '__main__':
    import argparse, os
    from pathlib import Path
    from .transfer import connection
    parser = argparse.ArgumentParser()
    parser.add_argument('site', nargs='?')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        __getattr__('verify_environment')(args.root)
    else:
        config = connection(args.root, args.site)
        os.execvp('ssh', [*ssh_arguments(args.root, config, True), '--', config['host']])
