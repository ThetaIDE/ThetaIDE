"""Experiment identities and discovery within the results directory tree."""

import re
from pathlib import Path, PurePosixPath


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
    """Find run roots at any depth, excluding their agent/version artifacts."""
    if not root.is_dir():
        return

    def visit(directory, depth):
        children = sorted(directory.iterdir())
        dirs = [path for path in children if path.is_dir() and not path.is_symlink()]
        is_run = depth >= 2 and (
            any(path.is_file() and (path.name in {"config.yaml", "experiment.yaml", "runtime.json"}
                                   or path.name.endswith((".combined.log", ".stderr.log"))) for path in children)
            or any(next(path.glob("version_*"), None) is not None for path in dirs)
        )
        if is_run:
            yield directory
        nested = nested_run_names(directory) if is_run else None
        for path in dirs:
            # Agent directories belong to the current run; nested saved runs
            # are siblings of those agents and must still be discovered.
            if nested is not None and path.name not in nested:
                continue
            yield from visit(path, depth + 1)

    yield from visit(root, 0)


SAVED_RUN_NAME = re.compile(r"(?:\d{8}-\d{6}(?:(?:-\d+)|(?:_[0-9a-f]{8}))?|[0-9a-f]{32})$")
RUN_SUFFIX = re.compile(r"(?:_\d{8}-\d{6}(?:(?:-\d+)|(?:_[0-9a-f]{8}))?|__[0-9a-f]{32})$")


def result_experiment_id(run_id: str) -> str:
    """Group both legacy sibling runs and nested runs under their recipe identity."""
    path = PurePosixPath(run_id)
    if SAVED_RUN_NAME.fullmatch(path.name) and path.parent != PurePosixPath("."):
        return path.parent.as_posix()
    name = RUN_SUFFIX.sub("", path.name)
    if name != path.name and path.parent.name == name:
        return path.parent.as_posix()
    return path.with_name(name).as_posix()


def saved_run_id(experiment_id: str, suffix: str) -> str:
    """Place unique saved executions inside the experiment folder."""
    experiment = PurePosixPath(result_experiment_id(experiment_id))
    return (experiment / suffix.lstrip("_")).as_posix()


def nested_run_names(log_dir: Path) -> set[str]:
    """Saved runs directly below a legacy run root, excluding its agents."""
    if not log_dir.is_dir():
        return set()
    return {child.name for child in log_dir.iterdir() if child.is_dir() and not child.is_symlink()
            and (any((child / name).is_file() for name in ("config.yaml", "experiment.yaml", "runtime.json"))
                 or any(next(agent.glob("version_*"), None) is not None
                        for agent in child.iterdir() if agent.is_dir()))}
