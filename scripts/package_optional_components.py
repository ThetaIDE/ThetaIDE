"""Build installable native archives from explicitly maintained optional sources."""
import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build(source, output):
    from src.app.components.archive import validate_native_package
    metadata = json.loads((source / '.theta_component.json').read_text())
    validate_native_package(metadata, source)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{metadata['id']}-{metadata['version']}.zip"
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob('*')):
            if path.is_file() and not any(part in ('__pycache__', '.git', 'node_modules') for part in path.parts):
                archive.write(path, path.relative_to(source))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('components', nargs='*')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'optional_components'
    for name in args.components or sorted(p.name for p in root.iterdir() if (p / '.theta_component.json').is_file()):
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Component names must be a single path segment')
        print(build(root / name, args.output))


if __name__ == '__main__':
    # Script entry point only; native package code never depends on a checkout.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
