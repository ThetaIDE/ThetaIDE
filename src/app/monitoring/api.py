"""Probe execution and versioned diagnostic artifacts for small PyTorch models.

V1 intentionally analyses an isolated model copy on fixed inputs. It does not
claim to capture activations from the training forward/backward computation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@contextmanager
def preserve_rng():
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        if not torch.is_tensor(tensor):
            digest.update(repr(tensor).encode())
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def framework_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


@dataclass
class ModelAccess:
    """Models may return their own adapter via get_monitoring_adapter()."""
    model: nn.Module
    score_method: str
    score_kind: str
    capabilities: frozenset[str] = frozenset({"probe_forward.v1", "module_interventions.v1"})

    def __post_init__(self):
        score_capability = "policy_probabilities.v1" if self.score_kind == "log_probabilities" else "action_values.v1"
        self.capabilities = frozenset(self.capabilities) | {score_capability}

    def scores(self, observations):
        scores = getattr(self.model, self.score_method)(observations)
        if scores.ndim != 2 or scores.shape[0] != len(observations) or not torch.isfinite(scores).all():
            raise ValueError("Probe scores must be finite [samples, actions] tensors")
        if self.score_kind == "log_probabilities":
            if (scores < 0).any() or not torch.allclose(scores.sum(-1), torch.ones_like(scores[:, 0]), atol=1e-4):
                raise ValueError("Probe action probabilities must be normalized")
            scores = scores.clamp_min(1e-12).log()
        return scores

    def topology_modules(self, target=""):
        """Architecture adapters may override this explicit ordered-path contract."""
        from .topology import sequential_modules
        return sequential_modules(self, target)

    def target(self, path=""):
        return self.model.get_submodule(path) if path else self.model

    def fork(self):
        if any(module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
               for module in self.model.modules()):
            raise ValueError("Models with existing module hooks require an architecture-specific ModelAccess.fork()")
        result = copy.copy(self)
        # Some architectures cache non-leaf tensors (e.g. their latest routing
        # weights). PyTorch deepcopy rejects these; diagnostic copies own detached
        # clones of such caches rather than the training autograd graph.
        memo, seen = {}, set()
        def detach_caches(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if torch.is_tensor(value):
                if not isinstance(value, nn.Parameter):
                    memo[id(value)] = value.detach().clone()
            elif isinstance(value, nn.Module):
                detach_caches(value.__dict__)
            elif isinstance(value, dict):
                for item in value.values():
                    detach_caches(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    detach_caches(item)
        detach_caches(self.model)
        result.model = copy.deepcopy(self.model, memo)
        # Runtime training hooks are not diagnostic targets and may close over
        # live state. V1 providers install their own hooks only on this copy.
        for module in result.model.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()
            module._backward_hooks.clear()
        result.model.eval()
        return result


def model_access(model, algorithm):
    algorithm = str(algorithm).rsplit("/", 1)[-1].lower()
    custom = getattr(model, "get_monitoring_adapter", None)
    if callable(custom):
        adapter = custom(algorithm)
        if not isinstance(adapter, ModelAccess):
            raise TypeError("get_monitoring_adapter() must return ModelAccess")
        return adapter
    method = "get_q_values" if algorithm.lower() == "dqn" else "get_action_probs"
    if not isinstance(model, nn.Module) or not callable(getattr(model, method, None)):
        raise ValueError(f"Model must expose {method}(obs) or get_monitoring_adapter(algorithm)")
    return ModelAccess(model, method, "q_values" if algorithm.lower() == "dqn" else "log_probabilities")


@dataclass
class ProbeSet:
    observations: np.ndarray
    sample_ids: np.ndarray
    metadata: dict

    @property
    def identity(self):
        digest = hashlib.sha256()
        for value in (self.observations, self.sample_ids):
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(json.dumps(self.metadata, sort_keys=True).encode())
        return digest.hexdigest()

    def save(self, path):
        from .budget import estimate, admit, bounded_json
        estimate({'observations': self.observations, 'sample_ids': self.sample_ids})
        path = Path(path)
        temporary = path.with_name('.' + path.stem + '.tmp.npz')
        try:
            np.savez_compressed(temporary, observations=self.observations, sample_ids=self.sample_ids,
                                metadata=np.asarray(bounded_json({"schema": "theta.probes/v1", **self.metadata})))
            admit([(temporary, path)])
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta.pop("schema", None) != "theta.probes/v1":
                raise ValueError("Unsupported probe artifact schema")
            obs, ids = data["observations"].copy(), data["sample_ids"].copy()
        if obs.ndim != 2 or not 4 <= len(obs) <= 4096 or ids.shape != (len(obs),) or len(set(ids.tolist())) != len(ids) or not np.isfinite(obs).all():
            raise ValueError("Probe artifact requires finite flat observations and unique aligned IDs")
        return cls(obs, ids, meta)

    @classmethod
    def collect(cls, make_environment, config):
        """Seeded independent random-policy trajectories, never the training env."""
        rng = np.random.default_rng(config["seed"])
        env = make_environment()
        rows, ids = [], []
        try:
            if not hasattr(env.action_space, "n"):
                raise ValueError("Probe v1 supports discrete actions and flat Box observations")
            observation, _ = env.reset(seed=config["seed"])
            for index in range(config["steps"]):
                row = np.asarray(observation, dtype=np.float32)
                if row.ndim != 1 or not np.isfinite(row).all():
                    raise ValueError("Probe v1 requires finite flat observations")
                if len(rows) < config["count"]:
                    rows.append(row.copy()); ids.append(index)
                else:
                    replace = int(rng.integers(index + 1))
                    if replace < config["count"]:
                        rows[replace], ids[replace] = row.copy(), index
                observation, _, terminated, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
                if terminated or truncated:
                    observation, _ = env.reset()
            order = np.argsort(ids)
            return cls(np.stack(rows)[order], np.asarray(ids, dtype=np.int64)[order],
                       {"source": "independent_random_policy_reservoir", "seed": config["seed"],
                        "steps": config["steps"], "environment": str(getattr(env.unwrapped, "spec", None)),
                        "preprocessing": "runner_environment_observations", "sampling": "fixed"})
        finally:
            env.close()


@dataclass
class Result:
    metrics: dict[str, float] = field(default_factory=dict)
    arrays: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    visuals: dict[str, dict] = field(default_factory=dict)


@dataclass
class Context:
    identity: str
    spec: dict
    access: ModelAccess
    probes: ProbeSet
    transitions: int
    updates: int
    snapshot_index: int
    outputs: frozenset[str] | None = None
    lineage: str = ""

    def wants_output(self, name):
        return self.outputs is None or any(
            not pattern or (name.startswith(pattern) if pattern.endswith("/") else name == pattern)
            for pattern in self.outputs)

    @property
    def observations(self):
        parameter = next(self.access.model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        return torch.as_tensor(self.probes.observations, dtype=torch.float32, device=device)


def write_snapshot(directory, context, result, *, model_hash, provider, elapsed, final=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"snapshot_{context.snapshot_index:05d}"
    arrays_path = directory / f"{stem}.npz"
    values = {k: np.asarray(v) for k, v in result.arrays.items()}
    if any(value.dtype.hasobject for value in values.values()):
        raise ValueError("Monitor artifacts cannot contain object arrays")
    from .budget import estimate, admit, bounded_json
    estimate(values)
    temporary_arrays = directory / f".{stem}.tmp.npz"
    metadata = {"schema": "theta.monitor-snapshot/v1", "instance": context.identity,
                "monitor_api": "theta.monitors/v1", "framework_sha256": framework_digest(),
                "provider": provider, "config": context.spec, "target": context.spec["target"],
                "sampling": "fixed_probes", "boundary": "after_train_call", "transitions": context.transitions,
                "optimizer_updates": context.updates, "snapshot_index": context.snapshot_index,
                "model_state_sha256": model_hash, "probe_set_sha256": context.probes.identity,
                "score_kind": context.access.score_kind, "collection_seconds": elapsed,
                "arrays": arrays_path.name, "metrics": result.metrics, "details": result.details,
                "lineage": context.lineage, "visuals": result.visuals}
    path = directory / f"{stem}.json"
    temporary = directory / f".{stem}.tmp.json"
    try:
        metadata_text = bounded_json(metadata, indent=2)
        np.savez_compressed(temporary_arrays, **values)
        temporary.write_text(metadata_text + "\n")
        admit([(temporary_arrays, arrays_path), (temporary, path)],
              position={'transitions': context.transitions, 'updates': context.updates, 'instance': context.identity}, final=final)
        temporary_arrays.replace(arrays_path)
        temporary.replace(path)
    finally:
        temporary_arrays.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return path


def load_snapshot(path):
    path = Path(path)
    metadata = json.loads(path.read_text())
    if metadata.get("schema") != "theta.monitor-snapshot/v1":
        raise ValueError("Unsupported diagnostic snapshot schema")
    array_path = path.parent / metadata["arrays"]
    if array_path.resolve().parent != path.resolve().parent:
        raise ValueError("Snapshot array path must stay beside its metadata")
    with np.load(array_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    return metadata, arrays


def save_monitor_state(path, state):
    """NPZ + JSON tree; no pickle or executable checkpoint objects."""
    arrays = {}
    def encode(value):
        if isinstance(value, np.ndarray) or torch.is_tensor(value):
            key = f"array_{len(arrays)}"
            arrays[key] = value.detach().cpu().numpy() if torch.is_tensor(value) else value
            if arrays[key].dtype.hasobject:
                raise ValueError("Monitor state cannot contain object arrays")
            return {"__array__": key}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        return value
    tree = encode(state)
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    from .budget import estimate, admit, bounded_json
    estimate(arrays)
    try:
        np.savez_compressed(temporary, metadata=np.asarray(bounded_json({"schema": "theta.monitor-state/v1", "state": tree})), **arrays)
        admit([(temporary, path)])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_monitor_state(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta.get("schema") != "theta.monitor-state/v1":
            raise ValueError("Unsupported monitor-state schema")
        def decode(value):
            if isinstance(value, dict):
                if set(value) == {"__array__"}:
                    return data[value["__array__"]].copy()
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value
        return decode(meta["state"])
