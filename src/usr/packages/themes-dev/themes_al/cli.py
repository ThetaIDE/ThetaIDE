"""Standalone entry point; shares fit/export implementation with future adapters."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description='TICC + EM-EDM reconstruction (local research)')
    parser.add_argument('command', choices=['run', 'synthetic', 'verify'])
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return
    command = parser.parse_args(argv[:1]).command
    if command == 'run':
        from .demo import main as run_main
        return run_main(argv[1:])
    if command == 'verify':
        verify_parser = argparse.ArgumentParser(description='Verify persisted inference and source lineage')
        verify_parser.add_argument('--output', type=Path, required=True)
        verify_parser.add_argument('--without-source', action='store_true', help='Skip original source checks')
        args = verify_parser.parse_args(argv[1:])
        from .artifacts import verify
        print(json.dumps(verify(args.output, source=not args.without_source), indent=2))
        return
    synthetic_parser = argparse.ArgumentParser(description='Generate artificial input and fit a small smoke')
    synthetic_parser.add_argument('--output', type=Path, required=True)
    synthetic_parser.add_argument('--seed', type=int, default=42)
    synthetic_parser.add_argument('--visits', type=int, default=40)
    synthetic_parser.add_argument('--rows', type=int, default=12)
    args = synthetic_parser.parse_args(argv[1:])
    if args.output.exists() and any(args.output.iterdir()):
        synthetic_parser.error('Output directory must be empty')
    from .synthetic import generate
    dataset = args.output.with_name(args.output.name + '_synthetic.npz')
    generate(dataset, visits=args.visits, rows=args.rows, seed=args.seed)
    from .demo import main as run_main
    return run_main(['--dataset', str(dataset), '--output', str(args.output), '--patients', '0',
                     '--stages', '2', '--groups', '2', '--min-group-segments', '1',
                     '--ticc-iterations', '2', '--em-iterations', '2', '--updates', '3',
                     '--sgld-steps', '1', '--seed', str(args.seed)])
