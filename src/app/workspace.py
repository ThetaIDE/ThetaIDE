"""Path policy for the existing Theta checkout layout; no workspace migration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPONENT_DIRECTORIES = {
    "model": "src/usr/models",
    "algorithm": "src/usr/algorithms",
    "method": "src/usr/algorithms",
    "agent": "src/usr/algorithms",
    "datamodule": "src/usr/data",
    "data": "src/usr/data",
    "evaluator": "src/usr/eval",
    "eval": "src/usr/eval",
    "env": "in/envs",
    "plotter": "plot",
    "plot": "plot",
    "experiment": ".thetaide/experiments",
    "suite": ".thetaide/suites",
    "package": ".thetaide/suites",
    "plugin": ".thetaide/plugins",
}


@dataclass(frozen=True)
class WorkspacePaths:
    """Explicit paths shared by headless services and desktop adapters.

    This describes today's layout. Selecting a root is useful for isolated tests
    and existing custom paths; it does not create a new workspace lifecycle.
    """

    root: Path = PROJECT_ROOT
    data_dir: Path | None = None

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser().resolve()
                           if self.data_dir is not None else self.root / ".thetaide")

    @property
    def config(self):
        return self.root / "in/config"

    @property
    def research(self):
        return self.root / "src/usr"

    @property
    def results(self):
        return self.root / "results"

    @property
    def collection(self):
        return self.root / ".thetaide/components"

    @property
    def component_roots(self):
        roots = [self.root / p for p in dict.fromkeys(COMPONENT_DIRECTORIES.values())]
        roots.extend((self.root / "src/usr/methods", self.root / "in/config/experiment",
                      self.root / "plugins", self.data_dir / "plugins"))
        return tuple(dict.fromkeys(roots))

    def manifests(self):
        """Installed manifests only, excluding retained packages and caches."""
        return sorted({p.resolve() for root in self.component_roots if root.is_dir()
                       for p in root.rglob(".theta_component.json")})

    def config_file(self, value: str | Path) -> Path:
        """Accept config-root-relative or workspace-relative paths, never escapes."""
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path if path.parts[:2] == ("in", "config") else self.config / path
        if not path.suffix:
            path = path.with_suffix(".yaml")
        path = path.resolve()
        if not path.is_relative_to(self.config.resolve()) or path.suffix not in (".yaml", ".yml"):
            raise ValueError("Configuration paths must be YAML files inside in/config/.")
        return path

    @classmethod
    def from_config_root(cls, config_root: Path):
        return cls(Path(config_root).resolve().parent.parent)

    @classmethod
    def for_process(cls):
        """Preserve legacy terminal use from an existing checkout's directory."""
        current = Path.cwd()
        return cls(current) if (current / "in/config").is_dir() or (current / "src/usr").is_dir() else cls()
