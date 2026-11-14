"""Keep the experiment recipe used for each launch alongside its results."""

from time import time_ns
from src.app.workspace import WorkspacePaths


def save_experiment_snapshot(cfg, context):
    recipe = context.get("experiment_yaml")
    if recipe is None:
        return
    paths = context.get("workspace") or WorkspacePaths.for_process()
    directory = paths.results / "logs" / cfg.group / cfg.experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "experiment.yaml"
    if target.exists():
        target = directory / f"experiment_{time_ns()}.yaml"
    target.write_bytes(recipe)


def save_run_config(cfg, *, paths=None):
    """Save the composed launch configuration before any training begins."""
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError

    # Hydra's internal configuration is launcher metadata, not part of the
    # experiment. In particular, it may contain job-only interpolations.
    snapshot = {}
    for key, value in cfg.items_ex(resolve=False):
        if key == "hydra":
            continue
        try:
            if OmegaConf.is_config(value):
                snapshot[key] = OmegaConf.to_container(value, resolve=True, enum_to_str=True)
            elif OmegaConf.is_missing(cfg, key):
                snapshot[key] = value
            else:
                snapshot[key] = cfg[key]
        except InterpolationResolutionError:
            # Preserve an application section if it needs runtime-only values.
            snapshot[key] = (OmegaConf.to_container(value, resolve=False, enum_to_str=True)
                             if OmegaConf.is_config(value) else value)

    paths = paths or WorkspacePaths.for_process()
    directory = paths.results / "logs" / cfg.group / cfg.experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "config.yaml"
    temporary = directory / "config.yaml.tmp"
    temporary.write_text(OmegaConf.to_yaml(OmegaConf.create(snapshot), resolve=False), encoding="utf-8")
    temporary.replace(target)

    from src.app.pipeline.provider_worker import provider_snapshot
    from src.app.components.transactions import write_json
    providers = directory / "providers.json"
    if providers.exists(): providers = directory / f"providers_{time_ns()}.json"
    write_json(providers, provider_snapshot(paths))
    import os
    os.environ["THETA_PROVIDER_SNAPSHOT"] = str(providers.resolve())
