"""Resolve Discussion data selections against saved metric files."""

from dataclasses import dataclass
from pathlib import Path

from src.app.pipeline.result_paths import check_result_path, iter_result_versions


FIGURE_SOURCES = "x-theta-sources"


@dataclass(frozen=True)
class MetricFile:
    method: str
    version: str
    path: Path


def metric_files(paths, target: str) -> list[MetricFile]:
    """Find metric files directly owned by a result, including legacy unversioned files."""
    check_result_path(target)
    logs = paths.results / "logs"
    root = logs / target
    if any(directory.is_symlink() for directory in (root, *root.parents) if logs in directory.parents):
        return []
    try:
        methods = sorted(root.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return []
    files = []
    for method in methods:
        if not method.is_dir() or method.is_symlink():
            continue
        for directory in [method, *sorted(iter_result_versions(method))]:
            path = directory / "metrics.csv"
            if path.is_file() and not path.is_symlink():
                version = directory.relative_to(method).as_posix() if directory != method else "unversioned"
                files.append(MetricFile(method.name, version, path))
    return files


def figure_source_targets(target, sources):
    """Return the explicit result targets, or the figure's own target by default."""
    return tuple(dict.fromkeys(source["target"] for source in sources)) if sources else (target,)


def selected_metric_files(paths, target, sources, methods=(), *, allow_missing=False):
    """Yield each selected log once, even when parent and child selections overlap."""
    for result in figure_source_targets(target, sources):
        entries = metric_files(paths, result)
        if not sources:
            yield from ((result, entry) for entry in entries if not methods or entry.method in methods)
            continue
        selected = set()
        for source in sources:
            if source["target"] != result:
                continue
            matches = [entry for entry in entries
                       if ("method" not in source or source["method"] == entry.method)
                       and ("version" not in source or source["version"] == entry.version)]
            if not matches and not allow_missing:
                label = " / ".join(source[key] for key in ("target", "method", "version") if key in source)
                raise ValueError(f"No saved metrics for {label}. Check that the result still exists and is synced.")
            selected.update(matches)
        yield from ((result, entry) for entry in entries if entry in selected)


def load_selected_metrics(paths, sources) -> dict:
    """Load explicit selections as {result/method: {version: DataFrame}}."""
    import pandas as pd

    data = {}
    if not sources:
        return data
    for result, entry in selected_metric_files(paths, "", sources):
        try:
            frame = pd.read_csv(entry.path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"Cannot read metrics for {result} / {entry.method} / {entry.version}: {exc}") from exc
        if not frame.empty:
            data.setdefault(f"{result}/{entry.method}", {})[entry.version] = frame
    return data
