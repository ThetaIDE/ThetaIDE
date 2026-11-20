"""Launch seeds: drawn once per experiment launch unless the config gives them to reproduce a run.

A launch draws every seed its config leaves open: the top-level ``seed``, and with
``variability`` the training seeds (``training_runs`` of them) and, when every method
controls evaluation seeding, the evaluation seed. Every method, sweep trial and replicate
of the launch receives the same drawn values as explicit overrides, so methods stay matched.

Draws come from the operating system's secure generator and stay below 2^32, the limit
of NumPy's legacy seeding that Stable-Baselines3 uses. Runners derive further streams by
adding small offsets (vector environment ``seed + i``, evaluation ``base + event``), so each
draw keeps ``SPACING`` away from the other draws and from every seed recorded in the
results: no two runs can then share an environment or evaluation stream.
"""
from __future__ import annotations

import json
import secrets

import yaml
from omegaconf import OmegaConf, open_dict

from src.app.pipeline.exceptions import ConfigurationError

SPACING = 2**16
"""Minimum distance between drawn seeds; covers runners' small seed offsets."""
SEED_LIMIT = 2**32
DRAWN_KEY = "theta_drawn_seeds"
"""Records which seed settings the launch drew rather than read from the config."""


def draw_seeds(count, taken=()):
    """``count`` secure random seeds, each SPACING from the others and from ``taken``."""
    chosen = []
    near = sorted(int(seed) for seed in taken)
    while len(chosen) < count:
        candidate = secrets.randbelow(SEED_LIMIT - SPACING)
        if all(abs(candidate - other) >= SPACING for other in (*chosen, *_neighbours(near, candidate))):
            chosen.append(candidate)
    return chosen


def _neighbours(ordered, value):
    from bisect import bisect_left
    index = bisect_left(ordered, value)
    return ordered[max(0, index - 1):index + 1]


def recorded_seeds(paths):
    """Training and evaluation seeds of every run recorded under the workspace results."""
    seeds = set()
    executions = paths.results / "executions"
    for manifest in executions.rglob("manifest.json"):
        plan = _read_seed_plan(manifest).get("plan") or {}
        seeds.update(seed for seed in plan.get("training_seeds") or () if isinstance(seed, int))
        if isinstance(plan.get("evaluation_seed"), int):
            seeds.add(plan["evaluation_seed"])
    for path in executions.glob("*/_study_seeds/*.json"):
        plan = _read_seed_plan(path)
        seeds.update(seed for seed in plan.get("training_seeds") or () if type(seed) is int)
        if type(plan.get("evaluation_seed")) is int:
            seeds.add(plan["evaluation_seed"])
    for config in (paths.results / "logs").rglob("config.yaml"):
        try:
            saved = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(saved, dict):
            continue
        panel = saved.get("variability") or {}
        if isinstance(panel, dict):
            seeds.update(seed for seed in panel.get("training_seeds") or () if type(seed) is int)
            if type(panel.get("evaluation_seed")) is int:
                seeds.add(panel["evaluation_seed"])
        for key in ("seed", "variability_evaluation_seed"):
            if type(saved.get(key)) is int:
                seeds.add(saved[key])
    return seeds


def _read_seed_plan(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _single_trial_seeds(cfg):
    if cfg.get("theta_variability_sample") or cfg.get("variability_run_id"):
        return None
    runner = cfg.get("agent") or cfg.get("model") or {}
    seed = runner.get("seed", cfg.get("seed")) if isinstance(runner, dict) else cfg.get("seed")
    return ((seed,), cfg.get("variability_evaluation_seed")) if type(seed) is int else None


def record_study_seed_plan(paths, group, study, training, evaluation):
    """Record the intended seeds when creating a study with queued configurations."""
    path = _study_seed_plan_path(paths, group, study)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"training_seeds": training, "evaluation_seed": evaluation}), encoding="utf-8")
    temporary.replace(path)


def _study_seed_plan_path(paths, group, study):
    from hashlib import sha256

    key = sha256(study.encode()).hexdigest()
    return paths.results / "executions" / str(group) / "_study_seeds" / f"{key}.json"


def study_seeds(paths, group, studies, *, require_records=False):
    """The training and evaluation seeds recorded by trials of ``studies``, or None for new studies.

    Trials of one study must share seeds, so a launch that continues a study reuses them.
    """
    found = {}
    for manifest in (paths.results / "executions" / str(group)).rglob("manifest.json"):
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        study = (record.get("parent") or {}).get("study")
        if study in studies:
            plan = record.get("plan") or {}
            training = tuple(plan.get("training_seeds") or ())
            if training:
                found.setdefault(study, set()).add((training, plan.get("evaluation_seed")))

    # Single-run trials can bypass the repeated-task coordinator and have only
    # their worker's effective config. Prefer the manifest when both exist.
    missing = studies - found.keys()
    if missing:
        for path in (paths.results / "checkpoints" / str(group)).rglob("execution.json"):
            try:
                cfg = json.loads(path.read_text(encoding="utf-8")).get("effective_config") or {}
            except (OSError, ValueError):
                continue
            study = (cfg.get("theta_tuning") or {}).get("study_name")
            panel = _single_trial_seeds(cfg) if study in missing else None
            if panel is not None:
                found.setdefault(study, set()).add(panel)
        missing -= found.keys()
    if missing:
        for path in (paths.results / "logs" / str(group)).rglob(".hydra/config.yaml"):
            try:
                cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(cfg, dict):
                continue
            study = (cfg.get("theta_tuning") or {}).get("study_name")
            panel = _single_trial_seeds(cfg) if study in missing else None
            if panel is not None:
                found.setdefault(study, set()).add(panel)

    for study in studies - found.keys():
        plan = _read_seed_plan(_study_seed_plan_path(paths, group, study))
        if plan.get("training_seeds"):
            found[study] = {(tuple(plan["training_seeds"]), plan.get("evaluation_seed"))}
    unrecorded = studies - found.keys()
    if require_records and unrecorded:
        raise ConfigurationError(f"Could not recover the original seeds for {sorted(unrecorded)}. "
                                 "Restore their trial records or start a new study.")

    if not found:
        return None
    panels = set().union(*found.values())
    if len(panels) != 1:
        raise ConfigurationError(f"Studies {sorted(found)} recorded different seeds. Start a new study "
                                 "to compare trials on one training and evaluation seed set.")
    training, evaluation = panels.pop()
    return {"training_seeds": list(training), "evaluation_seed": evaluation}


def pin_study_seeds(cfg, recorded, *, methods=None):
    """Restore a continued study's seed settings and reject changes to its protocol."""
    if recorded is None:
        return []
    variability = cfg.get("variability")
    training = recorded["training_seeds"]
    evaluation = recorded["evaluation_seed"]
    if variability is None:
        if len(training) != 1:
            raise ConfigurationError(f"The study used {len(training)} training seeds. "
                                     "Set variability.training_runs to that count to resume it.")
        for key in ("agent.seed", "model.seed"):
            configured = OmegaConf.select(cfg, key)
            method_seeds = [OmegaConf.select(OmegaConf.create(method), key)
                            for method in (methods or {}).values()]
            if any(seed is not None and seed != training[0] for seed in [configured, *method_seeds]):
                raise ConfigurationError(f"{key} differs from the study's recorded training seed {training[0]}. "
                                         "Clear the override to resume or start a new study.")
        settings = {"seed": training[0], "variability_evaluation_seed": evaluation}
    else:
        count = variability.get("training_runs")
        if count is not None and count != len(training):
            raise ConfigurationError(f"The study used {len(training)} training seeds; "
                                     f"variability.training_runs is {count}.")
        settings = {"variability.training_seeds": training, "variability.evaluation_seed": evaluation}
    overrides = {}
    for key, value in settings.items():
        configured = OmegaConf.select(cfg, key)
        if configured is not None and configured != value:
            raise ConfigurationError(f"{key} differs from the study's recorded value {value!r}. "
                                     "Clear the override to resume or start a new study.")
        if configured is None and value is not None:
            overrides[key] = value
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    return [f"++{key}={json.dumps(value)}" for key, value in overrides.items()]


def open_seeds(cfg, methods, *, resuming=False):
    """The seed settings ``cfg`` leaves for the launch to draw, with how many seeds each needs."""
    needed = {}
    if cfg.get("seed") is None:
        needed["seed"] = 1
    variability = cfg.get("variability")
    if variability is not None and not resuming:
        if variability.get("training_seeds") is None:
            needed["variability.training_seeds"] = int(variability.get("training_runs") or 1)
        if variability.get("evaluation_seed") is None:
            from src.app.pipeline.variability import method_capabilities
            capabilities = method_capabilities(cfg, methods)
            if capabilities and all(c["evaluation_seed_control"] for c in capabilities.values()):
                needed["variability.evaluation_seed"] = 1
    return needed


def resolve_launch_seeds(cfg, methods, *, taken=(), resuming=False):
    """Draw the open seeds into ``cfg``; return the Hydra overrides that pin them for subprocesses."""
    needed = open_seeds(cfg, methods, resuming=resuming)
    if not needed:
        return []
    configured = (cfg.get("seed"), cfg.get("variability_evaluation_seed"))
    variability = cfg.get("variability")
    if hasattr(variability, "get"):
        training = variability.get("training_seeds")
        configured += (variability.get("evaluation_seed"),)
        if isinstance(training, (list, tuple)) or OmegaConf.is_list(training):
            configured += tuple(training)
    occupied = {seed for seed in configured if type(seed) is int}
    occupied.update(taken)
    values = iter(draw_seeds(sum(needed.values()), occupied))
    drawn = {key: [next(values) for _ in range(count)] for key, count in needed.items()}
    overrides = []
    with open_dict(cfg):
        for key, seeds in drawn.items():
            value = seeds if key == "variability.training_seeds" else seeds[0]
            OmegaConf.update(cfg, key, value, force_add=True)
            overrides.append(f"++{key}={json.dumps(value)}")
        cfg[DRAWN_KEY] = list(drawn)
    overrides.append(f"++{DRAWN_KEY}={json.dumps(list(drawn))}")
    return overrides


def with_launch_seeds(cfg, methods, *, resuming=False):
    """A copy of ``cfg`` with its open seeds drawn, for validation before a launch draws them."""
    if not open_seeds(cfg, methods, resuming=resuming):
        return cfg
    copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    resolve_launch_seeds(copy, methods, resuming=resuming)
    return copy


def validate_seed_ranges(cfg, methods, plan):
    """Check initial training worker seeds and the evaluation base for collisions."""
    workers = []
    for method in (cfg, *methods.values()):
        for key in ("env", "agent"):
            section = method.get(key)
            if hasattr(section, "get"):
                workers.append(section.get("num_envs", 1))
        workers.append(method.get("num_envs", 1))
    width = max((value for value in workers if type(value) is int and value > 0), default=1)
    training = sorted(plan["training_seeds"])
    if any(right - left < width for left, right in zip(training, training[1:])):
        raise ConfigurationError(f"Training seed ranges overlap with {width} vector environments. "
                                 "Use variability.training_runs to draw a fresh set or space the seeds farther apart.")
    evaluation = plan["evaluation_seed"]
    if evaluation is not None and any(seed <= evaluation < seed + width for seed in training):
        raise ConfigurationError("The evaluation seed overlaps a training environment seed. "
                                 "Clear variability.evaluation_seed to draw a separate seed.")


def settle_worker_seed(cfg, paths):
    """Draw the seed of a worker started outside a launch (``python -m src.app.train``); it is its own launch."""
    if cfg.get("seed") is not None:
        return
    seed = draw_seeds(1, recorded_seeds(paths))[0]
    with open_dict(cfg):
        cfg.seed = seed
        cfg[DRAWN_KEY] = [*(cfg.get(DRAWN_KEY) or []), "seed"]
