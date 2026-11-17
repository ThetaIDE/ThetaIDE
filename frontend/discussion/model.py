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
from dataclasses import dataclass, field
from pathlib import Path

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


def experiment_tree(paths) -> list[TreeNode]:
    """Groups, subfolders and experiment IDs under the config experiment directory, in the
    Experiment tree's order: folders before experiments, each sorted. Group defaults
    (``_base.yaml``) are not experiments."""
    root = experiment_root(paths)
    if not root.is_dir():
        return []

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

    return visit(root, "", 0)


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
    """A plotter capability's figure ``parameter_schema``."""
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
