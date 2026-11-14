"""Configuration catalogs and domain read adapters, independent of Qt presentation."""

from pathlib import Path

import yaml

from src.app.core.provider_discovery import discovery_cached

from ..config_model import ConfigTree


def _get_available_paradigms() -> list[str]:
    """Return all paradigm names defined in in/config/paradigms/*.yaml."""
    from src.app.pipeline.validation import list_paradigms

    return list_paradigms()


def _get_component_kind(rel_path: str) -> tuple[str, str]:
    """Return (Category Display Name, Pill Accent Role)."""
    p = (rel_path or "").lower()
    if p.startswith("agent/"):
        return ("Algorithm", "secondary")
    elif p.startswith("env/"):
        return ("Environment", "primary")
    elif p.startswith("model/"):
        return ("Model", "accent")
    elif p.startswith("data/"):
        return ("Data Loader", "number")
    elif p.startswith("eval/"):
        return ("Evaluator", "focus")
    elif p.startswith(("plot/", "plots/")):
        return ("Figure", "primary_hover")
    elif p.startswith(("paradigms/", "paradigm/")):
        return ("Paradigm", "accent")
    elif p.startswith("site/"):
        return ("Site Profile", "secondary")
    elif p.startswith("workflow/"):
        return ("Workflow", "primary")
    elif p.startswith(("hydra/", "sweeper/")):
        return ("Hydra / Sweeper", "muted")
    parts = Path(rel_path).parts
    return (parts[0].capitalize() if parts else "Component", "secondary")


TRAINING_HYPERPARAMETERS = (
    "lr",
    "batch_size",
    "gamma",
    "epochs",
    "eval_every_n_epochs",
    "weight_decay",
    "from_study",
)


def get_agent_candidate_hyperparameters(
    agent_name: str | None,
    tree: ConfigTree | None,
    current_keys: set[str] | None = None,
) -> list[str]:
    """Candidate hyperparameters specific to an agent algorithm."""
    if not agent_name:
        return []
    current_keys = current_keys or set()
    candidates: list[str] = []
    if tree:
        agent_params = tree.agent_hyperparameters(agent_name)
        agent_schema = tree.agents.get(agent_name).parameter_schema if agent_name in tree.agents else {}
        agent_keys = list(agent_params)
        for key in agent_schema.get("properties", {}):
            if key not in agent_keys:
                agent_keys.append(key)
        for k in agent_keys:
            if k not in ("algorithm", "name", "type", "enabled") and k not in TRAINING_HYPERPARAMETERS:
                if k not in current_keys and k not in candidates:
                    candidates.append(k)
    return candidates


def get_model_candidate_hyperparameters(
    model_name: str | None,
    tree: ConfigTree | None,
    current_keys: set[str] | None = None,
) -> list[str]:
    """Candidate hyperparameters specific to a model architecture (or composite top-level)."""
    if not model_name:
        return []
    current_keys = current_keys or set()
    candidates: list[str] = []
    if tree:
        for k in tree.model_parameter_keys(model_name):
            if k not in ("architecture", "type", "lightning_module", "name", "enabled"):
                if k not in current_keys and k not in candidates:
                    candidates.append(k)
    return candidates


def get_training_candidate_hyperparameters(current_keys: set[str] | None = None) -> list[str]:
    """Candidate general training hyperparameters for a method."""
    current_keys = current_keys or set()
    return [k for k in TRAINING_HYPERPARAMETERS if k not in current_keys]


def get_method_candidate_hyperparameters(
    agent_name: str | None,
    model_name: str | None,
    tree: ConfigTree | None,
    current_keys: set[str] | None = None,
) -> list[str]:
    """Dynamically determine valid hyperparameters for a method based on its agent and model."""
    current_keys = current_keys or set()
    candidates: list[str] = []

    # 1. Base training parameters
    agent_defaults = tree.agent_hyperparameters(agent_name) if tree and agent_name else {}
    for p in get_training_candidate_hyperparameters(current_keys):
        if agent_defaults and p in ("lr", "batch_size", "gamma") and p not in agent_defaults:
            continue
        if p not in candidates:
            candidates.append(p)

    # 2. Agent-specific parameters
    for k in get_agent_candidate_hyperparameters(agent_name, tree, current_keys):
        if k not in candidates:
            candidates.append(k)

    # 3. Model-specific parameters
    for k in get_model_candidate_hyperparameters(model_name, tree, current_keys):
        if k not in candidates:
            candidates.append(k)

    return candidates


def _dataset_workspace():
    from src.app.workspace import WorkspacePaths

    tree = config_tree()
    return tree.workspace if tree else WorkspacePaths.for_process()


def discover_datasets(env_name: str | None = None, *, paths=None) -> list[str]:
    """Discover available dataset files and folders in in/datasets and results/datasets."""
    found = []
    paths = paths or _dataset_workspace()
    base_dirs = [paths.datasets, paths.results / "datasets"]
    for b in base_dirs:
        try:
            if not b.is_dir():
                continue
            for p in b.glob("**/*"):
                try:
                    if p.name.startswith(".") or p.name.startswith("__"):
                        continue
                    if p.is_file() and p.suffix in (".pkl", ".csv", ".parquet", ".h5", ".pt"):
                        found.append(p.relative_to(paths.root).as_posix())
                    elif p.is_dir():
                        try:
                            has_files = any(p.iterdir())
                        except OSError:
                            has_files = False
                        if has_files:
                            found.append(p.relative_to(paths.root).as_posix())
                except OSError:
                    continue
        except OSError:
            continue
    return sorted(list(set(found)))


def _recorded_metrics(data):
    """The experiment's telemetry.metrics list, or None when it records every metric."""
    telemetry = data.get("telemetry") if isinstance(data, dict) else None
    metrics = telemetry.get("metrics") if isinstance(telemetry, dict) else None
    return [str(m) for m in metrics] if isinstance(metrics, (list, tuple)) else None


def _monitor_settings(data):
    telemetry = data.get("telemetry") if isinstance(data, dict) else None
    return telemetry.get("monitors") if isinstance(telemetry, dict) else None


CHECKPOINT_KEYS = ("checkpoint_every_steps", "num_checkpoints")


DEFAULT_CHECKPOINTS = 10


FIXED_BOX_KEYS = frozenset(
    {
        "defaults",
        "paradigm",
        "experiment_id",
        "seed",
        "total_timesteps",
        "eval_episodes",
        "epochs",
        "eval_every_n_epochs",
        "methods",
        "data_module",
        "symbolic_state_loader",
        "env",
        "site",
        "trainer",
        "resources",
        "tensorboard",
        "no_plot",
        "sweep",  # legacy activation flag; composed tuning determines sweep mode
        "save_dataset",
        "recover",
        # Dataset values are rendered in the inputs card.  Unsupported legacy
        # resource/sweeper blocks stay hidden rather than being surfaced as
        # generic editable settings.
        "hydra",
        "dataset_path",
        "offline_datasets",
        "description",
        "eval_freq_steps",
        "eval_freq",
        "checkpoint_every_steps",
        "num_checkpoints",
        "checkpoint",  # the Checkpoint Settings box
        "experiment_id",
    }
)


_CONFIG_TREE = None


_CONFIG_TREE_LOADED = False


def invalidate_config_tree():
    """Clear cached in/config tree so subsequent lookups re-scan the filesystem."""
    global _CONFIG_TREE, _CONFIG_TREE_LOADED
    _CONFIG_TREE = None
    _CONFIG_TREE_LOADED = False


def set_config_tree(tree: ConfigTree | None):
    """Explicitly set or override the active ConfigTree (useful for tests and dependency injection)."""
    global _CONFIG_TREE, _CONFIG_TREE_LOADED
    _CONFIG_TREE = tree
    _CONFIG_TREE_LOADED = True


def config_tree(force_reload: bool = False):
    """The parsed in/config tree, or None if it cannot be found.

    Callers must handle None: the viewer falls back to free-text entry so it
    still works when run outside a checkout.
    """
    global _CONFIG_TREE, _CONFIG_TREE_LOADED
    if force_reload or not _CONFIG_TREE_LOADED:
        _CONFIG_TREE_LOADED = True
        try:
            _CONFIG_TREE = ConfigTree.discover()
        except (FileNotFoundError, OSError):
            _CONFIG_TREE = None
    return _CONFIG_TREE


def _read_group_base(tree, group):
    """The group's _base.yaml as plain data, or {} when there is none."""
    if tree is None or not group:
        return {}
    path = tree.workspace.resolve_config(f"experiment/{group}/_base.yaml")
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


@discovery_cached
def _component_profiles(config_root: Path, directory: str, selector: str, *, workspace=None) -> dict[str, dict]:
    """Installed YAML profiles, keyed by the runtime component name."""
    from src.app.core.provider_metadata import profile_values

    profiles = {}
    from src.app.workspace import WorkspacePaths
    paths = workspace or WorkspacePaths.from_config_root(config_root)
    resolver = paths.config_resolver()
    for key in resolver.selectors(directory):
        path = resolver.resolve(key, required=True).file
        if path.name.startswith(("_", ".")):
            continue
        try:
            values = profile_values(path)
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(values, dict):
            name = str(values.get(selector) or path.stem)
            profiles.setdefault(name, values)
    return profiles
