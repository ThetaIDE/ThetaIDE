"""Components and their tags, and the arXiv phrase each tag searches for.

Tags come from three places, merged in this order:
1. Built-in tags for the agents, models and environments in in/config (BUILTIN_TAGS).
2. The Community Hub index (each Hub component's `tags`).
3. Tags the user edits in the panel, which replace the tags above for that component.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KIND_TITLES = {"agent": "Methods", "method": "Methods", "model": "Models", "env": "Environments",
               "plugin": "Plugins", "experiment": "Experiments"}
KIND_ORDER = ["agent", "method", "model", "env", "experiment", "plugin"]

# (config group, name) -> tags. Only components whose literature is known are tagged; others can be
# tagged from the panel.
BUILTIN_TAGS: dict[tuple[str, str], list[str]] = {
    ("agent", "ppo"): ["ppo", "policy-gradient", "reinforcement-learning"],
    ("agent", "cql"): ["cql", "offline-rl", "q-learning"],
    ("agent", "iql"): ["iql", "offline-rl", "q-learning"],
    ("model", "nsfr"): ["nsfr", "neuro-symbolic", "differentiable-logic"],
    ("model", "neumann"): ["neumann", "neuro-symbolic", "differentiable-logic", "graph-neural-network"],
    ("model", "blendrl"): ["blendrl", "neuro-symbolic", "reinforcement-learning"],
    ("model", "cew"): ["cew", "fuzzy", "neuro-symbolic", "interpretability"],
    ("model", "transformer"): ["transformer", "attention"],
    ("model", "lstm"): ["lstm", "recurrent-neural-network"],
    ("model", "autoencoder"): ["autoencoder", "representation-learning"],
    ("model", "dense_autoencoder"): ["autoencoder", "representation-learning"],
    ("model", "dnn"): ["multilayer-perceptron"],
    ("model", "dueling_resnet"): ["dueling-network", "resnet", "dqn"],
    ("env", "cartpole"): ["cartpole", "classic-control"],
    ("env", "mountaincar"): ["mountain-car", "classic-control"],
    ("env", "seaquest"): ["atari", "ale"],
    ("env", "atari"): ["atari", "ale"],
    ("env", "mimic"): ["mimic", "electronic-health-records", "clinical"],
}

# Tags whose words alone search badly on arXiv: abbreviations and very short or generic tags.
TAG_PHRASES = {
    "rl": "reinforcement learning",
    "reinforcement-learning": "reinforcement learning",
    "offline-rl": "offline reinforcement learning",
    "online-rl": "reinforcement learning",
    "ppo": "proximal policy optimization",
    "cql": "conservative q-learning",
    "iql": "implicit q-learning",
    "dqn": "deep q-network",
    "sb3": "stable baselines",
    "baselines": "reinforcement learning baselines",
    "ale": "arcade learning environment",
    "nsfr": "neuro-symbolic forward reasoning",
    "neural-symbolic": "neuro-symbolic",
    "neuro-symbolic": "neuro-symbolic",
    "logic": "logic",
    "q-learning": "q-learning",
    "policy-gradient": "policy gradient",
    "ehr": "electronic health records",
    "fps": "first-person shooter",
    "ai": "artificial intelligence",
    "gym": "gymnasium reinforcement learning",
    "doom": "vizdoom",
}


def tag_phrase(tag: str) -> str:
    """The phrase arXiv is searched for when `tag` is selected."""
    tag = tag.strip().lower()
    return TAG_PHRASES.get(tag, tag.replace("-", " ").replace("_", " ")).strip()


def normalize_tags(tags) -> list[str]:
    out = []
    for tag in tags or []:
        tag = "-".join(str(tag).strip().lower().split())
        if tag and tag not in out:
            out.append(tag)
    return out


@dataclass
class Component:
    key: str  # "agent/cql", "hub/blendrl"
    name: str
    kind: str
    tags: list[str] = field(default_factory=list)
    source: str = "local"  # "local" or "hub"
    description: str = ""
    edited: bool = False


def local_components(config_root: Path) -> list[Component]:
    """Agents, models and environments in in/config, with their built-in tags."""
    components = []
    for group in ("agent", "model", "env"):
        folder = Path(config_root) / group
        if not folder.is_dir():
            continue
        names = sorted({p.stem for p in folder.glob("*.yaml")} | {p.name for p in folder.iterdir() if p.is_dir()})
        for name in names:
            components.append(Component(key=f"{group}/{name}", name=name, kind=group,
                                        tags=list(BUILTIN_TAGS.get((group, name), []))))
    return components


def hub_components(hub_client) -> list[Component]:
    """Community Hub components with their tags: the loaded index, else the cached copy on disk."""
    raw = []
    for comp in getattr(hub_client, "components", None) or []:
        raw.append({"id": comp.id, "name": comp.name, "kind": comp.kind, "tags": comp.tags,
                    "description": comp.description})
    cache_file = getattr(hub_client, "cache_file", None)
    if not raw and cache_file and Path(cache_file).is_file():
        try:
            data = json.loads(Path(cache_file).read_text(encoding="utf-8"))
            raw = data.get("components", []) if isinstance(data, dict) else []
        except (OSError, ValueError):
            raw = []
    out = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        out.append(Component(key=f"hub/{item['id']}", name=item.get("name") or item["id"],
                             kind=item.get("kind", "plugin"), tags=normalize_tags(item.get("tags")),
                             source="hub", description=item.get("description", "")))
    return out


def build_catalog(config_root: Path, hub_client=None, overrides: dict | None = None) -> list[Component]:
    """Every component with its tags; `overrides` maps component keys to tags the user set."""
    components = local_components(config_root) + (hub_components(hub_client) if hub_client else [])
    for comp in components:
        if overrides and comp.key in overrides:
            comp.tags = normalize_tags(overrides[comp.key])
            comp.edited = True
    rank = {kind: i for i, kind in enumerate(KIND_ORDER)}
    return sorted(components, key=lambda c: (rank.get(c.kind, len(rank)), c.source != "local", c.name.lower()))


def tag_counts(components) -> dict[str, int]:
    """How many components carry each tag, most common first."""
    counts: dict[str, int] = {}
    for comp in components:
        for tag in comp.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
