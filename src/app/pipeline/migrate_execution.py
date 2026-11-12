"""Preview/apply migration of explicitly selected editable first-party YAML files.

Never visits installed packages or guesses replacements for arbitrary scripts.
"""
import argparse
import copy
import difflib
from pathlib import Path
import yaml

from src.app.workspace import WorkspacePaths
from src.app.core.execution import REMOVED_SELECTORS


class ManualResolution(ValueError):
    pass


def migrate(data):
    result = copy.deepcopy(data)
    def selection(value):
        if isinstance(value, str) and value.startswith(('sb3/', 'd3rlpy/')):
            head, tail = value.split('/', 1)
            return head + '-dev/' + tail
        return value
    def node(value, *, profile=False):
        if not isinstance(value, dict):
            return
        old = [key for key in REMOVED_SELECTORS if key in value]
        if len(old) > 1:
            raise ManualResolution('Conflicting execution selectors: ' + ', '.join(old))
        if old:
            field = old[0]
            source = value[field]
            if field == 'engine' and source in ('pytorch', 'lightning'):
                value.pop(field)
            elif field == 'entrypoint' and source in (
                'package://sb3-suite/sb3_runner.py', 'package://sb3-dev/sb3_runner.py',
                'package://d3rlpy-suite/d3rlpy_runner.py'):
                value['backend'] = 'd3rlpy' if 'd3rlpy' in source else 'sb3'
                if profile and value.get('algorithm'):
                    algorithm = str(value['algorithm']).rsplit('/', 1)[-1].lower()
                    value['algorithm'] = value['backend'] + '-dev/' + algorithm
                value.pop(field)
            else:
                raise ManualResolution(f'Unknown execution declaration {field}={source!r}; author a registered adapter')
        aliases = [key for key in ('python', 'venv', 'venv_python') if key in value]
        if len(aliases) > 1 or 'venv' in aliases:
            raise ManualResolution('Resolve interpreter declarations manually; use python: /path/to/python')
        if 'venv_python' in value:
            value['python'] = value.pop('venv_python')
        if 'agent' in value:
            if isinstance(value['agent'], str):
                value['agent'] = selection(value['agent'])
            else:
                for key in ('name', 'algorithm'):
                    if key in value['agent']:
                        value['agent'][key] = selection(value['agent'][key])
                node(value['agent'])
        if profile and isinstance(value.get('name'), str):
            value['name'] = selection(value['name'])
        for method in (value.get('methods') or {}).values():
            node(method)
    node(result, profile=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    paths = WorkspacePaths.for_process()
    proposals = []
    for value in args.files:
        if value.is_symlink():
            raise ManualResolution(f"{value}: select an editable file directly, not a symlink")
        file = value.resolve()
        editable = (file.is_relative_to(paths.research) or file.is_relative_to(paths.config))
        if not editable or file.suffix not in ('.yaml', '.yml') or file.is_symlink():
            raise ManualResolution(f'{file}: select a local editable YAML file under src/usr or in/config')
        original = file.read_text()
        updated = yaml.safe_dump(migrate(yaml.safe_load(original)), sort_keys=False)
        proposals.append((file, updated))
        print(''.join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                          fromfile=str(file), tofile=str(file))), end='')
    if args.apply:
        for file, updated in proposals:
            file.write_text(updated)


if __name__ == '__main__':
    main()
