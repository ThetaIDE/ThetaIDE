"""Discussion storage: the experiment tree, figure documents and discussion text. No Qt.

    src/usr/discussion/<group>/<subdirs>/<experimentID>/<figure>.yaml   plotter + params
    src/usr/discussion/<group>/<subdirs>/<experimentID>/<figure>.md     discussion text

A figure file has exactly two keys: ``plotter`` (a plotter capability selector, resolved
through the provider catalog) and ``params`` (explicit parameter values only). Unset
parameters take the capability schema's default and are never written. The figure spec is
derived from these by the plot component and never stored.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

import yaml

FIGURE_KEYS = ("plotter", "params")
LIVE = "x-theta-live"


class FigureError(ValueError):
    """A figure file or edit that does not satisfy its plot component's schema."""


# ── Experiment tree ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TreeNode:
    """A group, subfolder or experiment of the Experiment panel's tree."""

    name: str
    rel: str
    kind: str  # "group", "folder" or "experiment"
    children: tuple = field(default=())


def experiment_root(paths) -> Path:
    return paths.config / "experiment"


def experiment_tree(paths, runs=()) -> list[TreeNode]:
    """Current recipes plus saved runs from the Results catalog, grouped by experiment.

    Historical experiments remain visible without a recipe. Group defaults are omitted.
    """
    root = experiment_root(paths)

    def visit(directory: Path, prefix: str, depth: int) -> list[TreeNode]:
        dirs = sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith("."))
        files = sorted(p for p in directory.iterdir()
                       if p.is_file() and p.suffix in (".yaml", ".yml") and p.stem != "_base")
        nodes = []
        for sub in dirs:
            rel = f"{prefix}{sub.name}"
            nodes.append(TreeNode(sub.name, rel, "group" if depth == 0 else "folder", tuple(visit(sub, rel + "/", depth + 1))))
        for file in files:
            nodes.append(TreeNode(file.stem, f"{prefix}{file.stem}", "experiment"))
        return nodes

    nodes = visit(root, "", 0) if root.is_dir() else []
    from src.app.pipeline.result_paths import result_experiment_id, saved_run_label

    def insert(siblings, parts, prefix, run):
        rel = "/".join([*prefix, parts[0]])
        index = next((i for i, node in enumerate(siblings) if node.rel == rel), None)
        kind = "experiment" if len(parts) == 1 else ("group" if not prefix else "folder")
        node = siblings[index] if index is not None else TreeNode(parts[0], rel, kind)
        children = list(node.children)
        if len(parts) > 1:
            insert(children, parts[1:], [*prefix, parts[0]], run)
        else:
            node = replace(node, kind="experiment")
            if run is not None:
                children.append(run)
        node = replace(node, children=tuple(children))
        if index is None:
            siblings.append(node)
        else:
            siblings[index] = node

    for entry in sorted(runs, key=lambda e: (e["group"], e["experiment_id"])):
        if not entry.get("agents") and not entry.get("has_run_logs"):
            continue
        group, run_id = entry["group"], entry["experiment_id"]
        experiment = (PurePosixPath(run_id).parent.as_posix() if (entry.get("config") or {}).get("saved_run")
                      else result_experiment_id(run_id))
        label = saved_run_label(run_id)
        if not entry.get("finished"):
            label += " · in progress"
        if entry.get("incomplete_results"):
            label += " · incomplete results"
        run = TreeNode(label, f"{group}/{run_id}", "run") if run_id != experiment else None
        insert(nodes, [group, *experiment.split("/")], [], run)

    def ordered(items):
        return [replace(node, children=tuple(ordered(node.children))) for node in
                sorted(items, key=lambda node: (node.kind not in ("group", "folder"), node.name))]
    return ordered(nodes)


def experiment_ids(nodes) -> list[str]:
    """Every experiment ``rel`` in a tree, depth first."""
    result = []
    for node in nodes:
        if node.kind == "experiment":
            result.append(node.rel)
        result.extend(experiment_ids(node.children))
    return result


# ── Figures ──────────────────────────────────────────────────────────────────


def _check_rel(experiment_rel: str) -> str:
    rel = str(experiment_rel).strip("/")
    if not rel or any(part in ("", ".", "..") for part in rel.split("/")) or "\\" in rel:
        raise FigureError(f"Invalid experiment: {experiment_rel!r}")
    return rel


def figure_dir(paths, experiment_rel: str) -> Path:
    return paths.discussion / _check_rel(experiment_rel)


def figures(paths, experiment_rel: str) -> list[Path]:
    """The experiment's figure files, sorted by name."""
    folder = figure_dir(paths, experiment_rel)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix == ".yaml")


def figure_experiment(paths, figure_path: Path) -> str:
    """The experiment ``rel`` a figure file belongs to."""
    return Path(figure_path).parent.relative_to(paths.discussion).as_posix()


def resolve_plotter(plotter: str, *, paths):
    """The (package, capability) a figure's ``plotter`` selector names."""
    from src.app.core.provider_catalog import DuplicateProvider, get_catalog
    try:
        found = get_catalog(paths).resolve("plotter", plotter)
    except DuplicateProvider as exc:
        raise FigureError(str(exc)) from exc
    if found is None:
        raise FigureError(f"No installed plot component provides {plotter!r}")
    return found


def capability_schema(capability) -> dict:
    """A figure plotter capability's ``parameter_schema``."""
    from src.app.core.plotting.figures import FIGURE_CONTRACT
    if FIGURE_CONTRACT not in (capability.metadata.get("provides") or []):
        raise FigureError(f"Plot component {capability.id!r} does not provide {FIGURE_CONTRACT}")
    schema = capability.metadata.get("parameter_schema")
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        raise FigureError(f"Plot component {capability.id!r} declares no figure parameter_schema")
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict) or not all(isinstance(v, dict) for v in properties.values()):
        raise FigureError(f"Plot component {capability.id!r} declares invalid figure parameters")
    return schema


def schema_defaults(schema: dict) -> dict:
    return {key: copy.deepcopy(node["default"]) for key, node in (schema.get("properties") or {}).items()
            if "default" in node}


def live_parameters(schema: dict) -> list[str]:
    return [key for key, node in (schema.get("properties") or {}).items() if node.get(LIVE) is True]


def _validation_error(schema: dict, params: dict) -> str | None:
    """Explicit values: known keys only, each valid, and the merged values satisfy the schema."""
    import jsonschema
    properties = schema.get("properties") or {}
    unknown = sorted(set(params) - set(properties))
    if unknown:
        return f"Unknown parameter{'s' if len(unknown) > 1 else ''}: {', '.join(unknown)}"
    merged = {**schema_defaults(schema), **params}
    try:
        jsonschema.validate(merged, {**schema, "additionalProperties": False})
    except jsonschema.ValidationError as exc:
        where = ".".join(str(p) for p in exc.absolute_path)
        return f"{where}: {exc.message}" if where else exc.message
    return None


class FigureDoc:
    """One figure file: its plotter selection and explicit parameter values."""

    def __init__(self, path: Path, plotter: str, params: dict | None = None, *, paths):
        self.path = Path(path)
        self.plotter = plotter
        self.params = dict(params or {})
        self.paths = paths
        self._capability = None

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def experiment(self) -> str:
        return figure_experiment(self.paths, self.path)

    @classmethod
    def load(cls, path: Path, *, paths) -> FigureDoc:
        path = Path(path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise FigureError(f"Cannot read {path.name}: {exc}") from exc
        if not isinstance(data, dict):
            raise FigureError(f"{path.name} must be a mapping with keys {', '.join(FIGURE_KEYS)}")
        unknown = sorted(set(data) - set(FIGURE_KEYS))
        if unknown:
            raise FigureError(f"{path.name}: unknown key{'s' if len(unknown) > 1 else ''} {', '.join(unknown)}")
        missing = [key for key in FIGURE_KEYS if key not in data]
        if missing:
            raise FigureError(f"{path.name}: missing {', '.join(missing)}")
        if not isinstance(data["plotter"], str) or not data["plotter"].strip():
            raise FigureError(f"{path.name}: plotter must name a plot component")
        params = data["params"] if data["params"] is not None else {}
        if not isinstance(params, dict):
            raise FigureError(f"{path.name}: params must be a mapping")
        doc = cls(path, data["plotter"], params, paths=paths)
        error = _validation_error(doc.schema(), doc.params)
        if error:
            raise FigureError(f"{path.name}: {error}")
        return doc

    def capability(self):
        """The resolved (package, capability) of ``plotter``."""
        if self._capability is None:
            self._capability = resolve_plotter(self.plotter, paths=self.paths)
        return self._capability

    def schema(self) -> dict:
        return capability_schema(self.capability()[1])

    def defaults(self) -> dict:
        return schema_defaults(self.schema())

    def values(self) -> dict:
        """The parameters the component receives: schema defaults merged with explicit values."""
        return {**self.defaults(), **copy.deepcopy(self.params)}

    def live_params(self) -> list[str]:
        return live_parameters(self.schema())

    def required(self) -> list[str]:
        return list(self.schema().get("required") or [])

    def set_param(self, name: str, value) -> bool:
        """Set an explicit value; raises FigureError and leaves the document unchanged when invalid."""
        candidate = {**self.params, name: copy.deepcopy(value)}
        error = _validation_error(self.schema(), candidate)
        if error:
            raise FigureError(error)
        if name in self.params and self.params[name] == value and type(self.params[name]) is type(value):
            return False
        self.params = candidate
        return True

    def unset_param(self, name: str) -> bool:
        """Remove an explicit value so the schema default applies again."""
        if name not in self.params:
            return False
        if name in self.required() and "default" not in (self.schema().get("properties") or {}).get(name, {}):
            raise FigureError(f"{name} is required")
        candidate = {k: v for k, v in self.params.items() if k != name}
        error = _validation_error(self.schema(), candidate)
        if error:
            raise FigureError(error)
        self.params = candidate
        return True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump({"plotter": self.plotter, "params": self.params}, sort_keys=False)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self.path)


def _figure_name(capability_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", capability_id.split("/")[-1]).strip("_")
    return name or "figure"


def create_figure(paths, experiment_rel: str, capability: str) -> Path:
    """Write ``<capability>.yaml`` (``_2``, ``_3``… when taken) with empty params."""
    folder = figure_dir(paths, experiment_rel)
    resolve_plotter(capability, paths=paths)
    base = _figure_name(capability)
    name, index = base, 1
    while (folder / f"{name}.yaml").exists() or (folder / f"{name}.md").exists():
        index += 1
        name = f"{base}_{index}"
    doc = FigureDoc(folder / f"{name}.yaml", capability, {}, paths=paths)
    doc.save()
    return doc.path


def figure_candidates(paths, experiment_rel: str) -> list:
    """Candidates against the current recipe or the selected run's saved configuration."""
    from src.app.core.plotting.figures import compatible_figure_plotters
    from src.app.pipeline.compose import compose_experiment
    target = _check_rel(experiment_rel)
    if is_saved_run(paths, target):
        from omegaconf import OmegaConf
        config = paths.results / "logs" / target / "config.yaml"
        if not config.is_file():
            raise FigureError(f"Saved run {target} has no config.yaml; sync its results first.")
        cfg = OmegaConf.load(config)
    else:
        cfg = compose_experiment(target, workspace=paths).cfg
    return compatible_figure_plotters(cfg, paths=paths)


def is_saved_run(paths, target: str) -> bool:
    """A target identifies a saved execution rather than the current recipe."""
    from src.app.pipeline.result_paths import result_experiment_id
    target = _check_rel(target)
    # A saved snapshot also identifies renamed runs whose names are not timestamps.
    return (result_experiment_id(target) != target
            or (not (experiment_root(paths) / f"{target}.yaml").is_file()
                and (paths.results / "logs" / target / "config.yaml").is_file()))


# ── Discussion text ──────────────────────────────────────────────────────────


def discussion_path(figure_path: Path) -> Path:
    return Path(figure_path).with_suffix(".md")


def read_discussion(figure_path: Path) -> str:
    path = discussion_path(figure_path)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def write_discussion(figure_path: Path, text: str) -> bool:
    """Write the discussion text. The file is created by the first non-empty text."""
    path = discussion_path(figure_path)
    if not path.is_file():
        if not text:
            return False
    elif path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True
