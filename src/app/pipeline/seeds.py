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

from omegaconf import OmegaConf, open_dict

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
    for manifest in (paths.results / "executions").rglob("manifest.json"):
        try:
            plan = json.loads(manifest.read_text(encoding="utf-8")).get("plan") or {}
        except (OSError, ValueError):
            continue
        seeds.update(seed for seed in plan.get("training_seeds") or () if isinstance(seed, int))
        if isinstance(plan.get("evaluation_seed"), int):
            seeds.add(plan["evaluation_seed"])
    import yaml
    for config in (paths.results / "logs").rglob("config.yaml"):
        try:
            saved = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(saved, dict) and isinstance(saved.get("seed"), int):
            seeds.add(saved["seed"])
    return seeds


def study_seeds(paths, group, studies):
    """The training and evaluation seeds recorded by trials of ``studies``, or None for new studies.

    Trials of one study must share seeds, so a launch that continues a study reuses them.
    """
    found = set()
    for manifest in (paths.results / "executions" / str(group)).rglob("manifest.json"):
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (record.get("parent") or {}).get("study") in studies:
            plan = record.get("plan") or {}
            found.add((tuple(plan.get("training_seeds") or ()), plan.get("evaluation_seed")))
    if not found:
        return None
    if len(found) > 1:
        from src.app.pipeline.exceptions import ConfigurationError
        raise ConfigurationError(f"Studies {sorted(studies)} recorded different seeds {sorted(found)}; "
                                 "pin variability.training_seeds and evaluation_seed to the ones to continue")
    training, evaluation = found.pop()
    return {"training_seeds": list(training), "evaluation_seed": evaluation}


def pin_study_seeds(cfg, recorded):
    """Fill the open variability seeds with a continued study's recorded seeds; return their overrides."""
    variability = cfg.get("variability")
    if variability is None or recorded is None:
        return []
    overrides = []
    with open_dict(cfg):
        for key in ("training_seeds", "evaluation_seed"):
            if variability.get(key) is None and recorded[key] is not None:
                variability[key] = recorded[key]
                overrides.append(f"++variability.{key}={json.dumps(recorded[key])}")
    return overrides


def open_seeds(cfg, methods):
    """The seed settings ``cfg`` leaves for the launch to draw, with how many seeds each needs."""
    needed = {}
    if cfg.get("seed") is None:
        needed["seed"] = 1
    variability = cfg.get("variability")
    if variability is not None:
        if variability.get("training_seeds") is None:
            needed["variability.training_seeds"] = int(variability.get("training_runs") or 1)
        if variability.get("evaluation_seed") is None:
            from src.app.pipeline.variability import method_capabilities
            capabilities = method_capabilities(cfg, methods)
            if capabilities and all(c["evaluation_seed_control"] for c in capabilities.values()):
                needed["variability.evaluation_seed"] = 1
    return needed


def resolve_launch_seeds(cfg, methods, *, taken=()):
    """Draw the open seeds into ``cfg``; return the Hydra overrides that pin them for subprocesses."""
    needed = open_seeds(cfg, methods)
    if not needed:
        return []
    values = iter(draw_seeds(sum(needed.values()), taken))
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


def with_launch_seeds(cfg, methods):
    """A copy of ``cfg`` with its open seeds drawn, for validation before a launch draws them."""
    if not open_seeds(cfg, methods):
        return cfg
    copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    resolve_launch_seeds(copy, methods)
    return copy
