"""Experiment identities and discovery within the results directory tree."""

from pathlib import Path


def result_identity(experiment: str, group=None, experiment_id=None) -> tuple[str, str]:
    """Keep recipe subdirectories below its top-level group, including renamed runs."""
    recipe = Path(experiment).with_suffix("") if experiment.endswith(".yaml") else Path(experiment)
    parts = recipe.parts
    group = str(group or "ungrouped")
    if group == "ungrouped" and len(parts) > 1:
        group = parts[0]
    name = str(experiment_id or "default_exp")
    if name == "default_exp":
        name = recipe.name
    # A qualified ID is already relative to the group. This makes repeated
    # composition and the resolved overrides passed to providers idempotent.
    if "/" not in name and len(parts) > 2:
        name = (Path(*parts[1:-1]) / name).as_posix()
    check_result_path(name)
    check_result_path(group, nested=False)
    return group, name


def check_result_path(value: str, *, nested: bool = True) -> None:
    """Allow relative subdirectories without allowing traversal or absolute paths."""
    if (not value or any(c in value for c in "\\\x00:")
            or any(part in ("", ".", "..") for part in value.split("/"))
            or (not nested and "/" in value)):
        raise ValueError(f"Invalid result path: {value!r}")


def iter_run_dirs(root: Path):
    """Find run roots at any depth, stopping before their agent/version artifacts."""
    if not root.is_dir():
        return

    def visit(directory, depth):
        children = sorted(directory.iterdir())
        dirs = [path for path in children if path.is_dir() and not path.is_symlink()]
        if depth >= 2 and (
            any(path.is_file() and (path.name in {"config.yaml", "experiment.yaml", "runtime.json"}
                                   or path.name.endswith((".combined.log", ".stderr.log"))) for path in children)
            or any(next(path.glob("version_*"), None) is not None for path in dirs)
        ):
            yield directory
            return
        for path in dirs:
            yield from visit(path, depth + 1)

    yield from visit(root, 0)
