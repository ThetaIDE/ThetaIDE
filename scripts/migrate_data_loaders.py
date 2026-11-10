"""Preview or apply explicit input-binding migration to authored YAML recipes."""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.core.loader_migration import migrate_loader_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--state-provider", help="Registered state provider ID, e.g. blendrl_symbolic_state")
    parser.add_argument("--write", action="store_true", help="Apply the displayed YAML changes")
    args = parser.parse_args()
    for path in args.files:
        before = path.read_text()
        updated, changes = migrate_loader_config(yaml.safe_load(before) or {}, state_provider=args.state_provider)
        if not changes:
            continue
        after = ("# @package _global_\n" if "# @package _global_" in before else "") + yaml.safe_dump(updated, sort_keys=False)
        print("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=str(path), tofile=str(path))), end="")
        if args.write:
            path.write_text(after)


if __name__ == "__main__":
    main()
