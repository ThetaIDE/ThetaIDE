"""One lifecycle for standalone and model-supplied monitor components."""
from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .api import Context, ProbeSet, framework_digest, model_access, preserve_rng, state_digest, write_snapshot
from .config import instances, model_defaults, plain


class MonitorManager:
    def __init__(self, cfg, run, model, algorithm, make_environment, *, catalog=None):
        from src.app.core.provider_catalog import get_catalog, provider_hash
        from src.app.core.provider_loader import load_capability
        self.run, self.entries, self.closed = run, [], False
        self.cfg = plain(cfg)
        catalog = catalog or get_catalog()
        defaults = model_defaults(self.cfg, catalog)
        factory = getattr(model, "get_monitor_specs", None)
        if callable(factory):
            from src.app.pipeline.config import deep_merge
            defaults = deep_merge(defaults, factory())
        configured = instances(self.cfg, defaults=defaults)
        # No model cloning, probes, ML provider loading or artifact allocation
        # for absent/disabled monitor components.
        if not configured:
            return
        self.access = model_access(model, algorithm)
        banks = {}
        try:
            for identity, spec in configured.items():
                prefix = f"interp/{identity}/"
                if not run.wants_monitor(prefix) or not run.metrics.keeps_section(prefix):
                    continue
                entry = {"identity": identity, "spec": spec, "count": 0, "last": None,
                         "monitor": None, "disabled": False, "lineage": uuid.uuid4().hex}
                self.entries.append(entry)
                try:
                    resolved = catalog.resolve("monitor", spec["component"])
                    if resolved is None:
                        raise ValueError(f"Unknown monitor component {spec['component']!r}")
                    package, capability = resolved
                    from src.app.telemetry.monitors import _catalog_outputs
                    metadata = capability.metadata.get("monitoring", {})
                    declarations = list(metadata.get("provides", []))
                    if metadata.get("catalog"):
                        declarations += _catalog_outputs(package, metadata["catalog"],
                            {**spec["params"], "instance": identity, "target": spec["target"]}, self.cfg)
                    entry["declarations"] = {item["name"].removeprefix(prefix): item for item in declarations}
                    entry["outputs"] = frozenset(key for key, item in entry["declarations"].items()
                        if run.wants_monitor(item["name"]) and (run.metrics.keeps_section(item["name"])
                            if item["name"].endswith("/") else run.metrics.keeps(item["name"])))
                    if declarations and not entry["outputs"]:
                        entry["disabled"] = True
                        continue
                    required = set(capability.metadata.get("requires", ["probe_forward.v1"]))
                    if missing := required - self.access.capabilities - {"post_update.v1"}:
                        raise ValueError(f"Model adapter lacks capabilities: {sorted(missing)}")
                    self.access.target(spec["target"])
                    provider = load_capability(package, capability, paths=catalog.paths)
                    validate = getattr(provider, "validate", None)
                    if callable(validate):
                        validate(spec["params"])
                    monitor = provider(**spec["params"])
                    if not callable(getattr(monitor, "collect", None)):
                        raise TypeError("Monitor must implement collect(context)")
                    entry["monitor"] = monitor
                    entry["provider"] = {"package": package.id, "version": package.version,
                                         "capability": capability.id, "sha256": provider_hash(package)}
                    key = json.dumps(spec["probes"], sort_keys=True)
                    with preserve_rng():
                        if key not in banks:
                            path = spec["probes"].get("path")
                            banks[key] = ProbeSet.load(path) if path else ProbeSet.collect(make_environment, spec["probes"])
                    entry["probes"] = banks[key]
                    directory = run.version_dir.resolve() / "monitors" / identity
                    directory.mkdir(parents=True, exist_ok=True)
                    entry["directory"] = directory
                    path = directory / "probes.npz"
                    banks[key].save(path)
                    run.artifact(path, kind="monitor_probes", instance=identity, sha256=banks[key].identity)
                    if callable(getattr(monitor, "setup", None)):
                        with preserve_rng(), torch.no_grad():
                            monitor.setup(self._context(entry, 0, 0))
                except Exception as exc:
                    self._failure(entry, exc)
        except BaseException:
            self.close()
            raise

    def _context(self, entry, transitions, updates):
        return Context(entry["identity"], copy.deepcopy(entry["spec"]), self.access.fork(),
                       entry["probes"], int(transitions), int(updates), entry["count"],
                       entry.get("outputs"), entry["lineage"])

    def _failure(self, entry, exc):
        entry["disabled"] = True
        self.run.event("monitor_error", f"Monitor {entry['identity']}: {type(exc).__name__}: {exc}",
                       level="error", instance=entry["identity"])
        if entry["spec"]["failure_policy"] == "raise":
            raise exc

    def collect(self, transitions, updates, *, final=False):
        """Called only after train() returned; updates is the runner's real counter."""
        if self.closed:
            raise RuntimeError("Monitor manager is closed")
        for entry in self.entries:
            spec, last = entry["spec"], entry["last"]
            prefix = f"interp/{entry['identity']}/"
            if entry["disabled"] or entry["count"] >= spec["max_snapshots"]:
                continue
            # Reserve one slot for the final trained model. A small artifact
            # budget must not silently omit the end of training.
            if not final and entry["count"] >= spec["max_snapshots"] - 1:
                continue
            if not self.run.wants_monitor(prefix) or not self.run.metrics.keeps_section(prefix):
                continue
            if updates <= 0 or updates == last or (not final and last is not None and updates - last < spec["interval_updates"]):
                continue
            started = time.perf_counter()
            try:
                with self.run.phase("diagnostics", f"Collecting monitor {entry['identity']}"), preserve_rng(), torch.no_grad():
                    model_hash = state_digest(self.access.model)
                    context = self._context(entry, transitions, updates)
                    result = entry["monitor"].collect(context)
                    # Rich payloads live in committed artifacts, never scalar records.
                    visuals = {}
                    for key, payload in result.visuals.items():
                        declaration = entry["declarations"].get(key)
                        if declaration is None or declaration.get("kind", "scalar_series") == "scalar_series":
                            raise ValueError(f"Undeclared rich monitor output: {key}")
                        if key in entry["outputs"]:
                            from src.app.telemetry.visuals import validate_visual_data
                            validate_visual_data(declaration, payload, result.arrays)
                            visuals[key] = {**declaration, "data": payload}
                    result.visuals = visuals
                    result.metrics = {key: value for key, value in result.metrics.items()
                                      if not entry["declarations"] or (context.wants_output(key)
                                          and self.run.wants_monitor(prefix + key) and self.run.metrics.keeps(prefix + key))}
                    # Validate results before either publishing or writing files.
                    if not all(isinstance(k, str) and not k.startswith("/") and np.isfinite(v)
                               for k, v in result.metrics.items()):
                        raise ValueError("Monitor metrics must be named finite scalars")
                    path = write_snapshot(entry["directory"], context, result, model_hash=model_hash,
                                          provider=entry["provider"], elapsed=time.perf_counter() - started)
                entry["count"] += 1
                entry["last"] = int(updates)
                from .api import save_monitor_state
                state_path = self.run.version_dir / "monitors" / "state.npz"
                save_monitor_state(state_path, self.state_dict())
                self.run.log({prefix + key: float(value) for key, value in result.metrics.items()},
                             step=int(transitions), transitions=int(transitions))
                from .report import write_report
                report = write_report(entry["directory"])
                self.run.artifact(report, kind="monitor_report", instance=entry["identity"])
                self.run.artifact(state_path, kind="monitor_state")
                self.run.artifact(path, kind="monitor_snapshot", instance=entry["identity"],
                                  transitions=int(transitions), optimizer_updates=int(updates),
                                  visuals=[{k: v for k, v in visual.items() if k != "data"}
                                           for visual in result.visuals.values()])
            except Exception as exc:
                self._failure(entry, exc)

    def state_dict(self):
        """Restore only alongside the matching model, configuration and probe bank."""
        return {entry["identity"]: {"count": entry["count"], "last": entry["last"],
                "probe_set": entry["probes"].identity, "spec": entry["spec"], "lineage": entry["lineage"],
                "model_state": state_digest(self.access.model),
                "framework": framework_digest(),
                "provider": entry["provider"],
                "monitor": copy.deepcopy(entry["monitor"].state_dict()) if hasattr(entry["monitor"], "state_dict") else {}}
                for entry in self.entries if not entry["disabled"]}

    def load_state_dict(self, state):
        active = {entry["identity"]: entry for entry in self.entries if not entry["disabled"]}
        if set(active) != set(state):
            raise ValueError("Monitor resume instances differ")
        for identity, saved in state.items():
            entry = active[identity]
            if (saved["probe_set"] != entry["probes"].identity or saved["spec"] != entry["spec"]
                    or saved["provider"] != entry["provider"] or saved.get("framework") != framework_digest()
                    or saved["model_state"] != state_digest(self.access.model)):
                raise ValueError(f"Monitor {identity}: resume model, provider, probes or configuration differ")
            entry["count"], entry["last"] = saved["count"], saved["last"]
            entry["lineage"] = saved.get("lineage", entry["lineage"])
            if saved["monitor"]:
                entry["monitor"].load_state_dict(copy.deepcopy(saved["monitor"]))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for entry in self.entries:
            monitor = entry["monitor"]
            if monitor is not None and callable(getattr(monitor, "close", None)):
                try:
                    with preserve_rng():
                        monitor.close()
                except Exception as exc:
                    self.run.event("monitor_cleanup_error", str(exc), level="error", instance=entry["identity"])
