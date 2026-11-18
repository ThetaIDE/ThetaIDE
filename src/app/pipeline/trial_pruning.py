"""Report a running seed panel's objective progress to the tuner; stop the trial when pruned.

A trial that trains several seeds, or scores through an objective component, runs its
seeds in separate workers. This coordinator-side watcher scores their partial results
with the objective's intermediate function, reports each new checkpoint to the
tuner's trial monitor, and cancels every seed of the trial when the tuner prunes it.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

POLL_SECONDS = 5.0


class ProgressPruning:
    def __init__(self, scorer, monitor, cancel_target, *, poll_seconds=POLL_SECONDS):
        self.scorer, self.monitor, self.cancel_target = scorer, monitor, cancel_target
        self.poll_seconds = poll_seconds
        self.pruned = False
        self.error = None
        self.last_step = -1
        self._stop = threading.Event()
        self._thread = None
        self._manifest = None

    def attach(self, manifest_path):
        """Start watching once the execution manifest exists (``run_execution``'s ``on_manifest``)."""
        self._manifest = Path(manifest_path)
        self._thread = threading.Thread(target=self._watch, name="theta-trial-pruning", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def poll(self):
        """Score once; report a new checkpoint. Returns False once the trial is stopped."""
        progress = self.scorer.score(self._manifest)
        if progress is None or progress.step <= self.last_step:
            return True
        self.last_step = progress.step
        print(f"[Pruning] checkpoint {progress.step}: objective progress {progress.value:.6g}", flush=True)
        try:
            self.monitor.report(progress.value, progress.step)
        except Exception as exc:
            if getattr(exc, "theta_trial_status", None) != "pruned":
                raise
            print(f"[Pruning] {exc}; stopping every seed of this trial.", flush=True)
            self.pruned = True
            self._cancel()
            return False
        return True

    def _watch(self):
        while not self._stop.wait(self.poll_seconds):
            try:
                if not self.poll():
                    return
            except Exception as exc:
                # A broken intermediate objective would make pruning silently inert; fail the trial.
                self.error = exc
                self._cancel()
                return

    def _cancel(self):
        from src.app.services.cancellation import request_path
        directory = Path(os.environ["THETA_CANCEL_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        request_path(directory, self.cancel_target).touch()


def progress_pruning(cfg, scoring_cfg, version, output, *, paths):
    """The watcher for an active pruning trial, or None when pruning is off."""
    from omegaconf import OmegaConf
    from src.app.objectives.api import validate_objective
    from src.app.objectives.evaluation import ProgressScorer
    from src.app.pipeline.tuners import trial_monitor
    from src.app.pipeline.tuning_features import validate_tuning_compatibility

    if not validate_tuning_compatibility(cfg, paths=paths):
        return None
    monitor = trial_monitor(cfg)
    if monitor is None:
        return None
    component, params = validate_objective(scoring_cfg, paths=paths)
    config = OmegaConf.to_container(scoring_cfg, resolve=True)
    if not os.environ.get("THETA_CANCEL_DIR"):
        # Launches from the IDE and terminal provide one; a bare Hydra launch gets the trial's own.
        os.environ["THETA_CANCEL_DIR"] = str((Path(output) / "cancellations").resolve())
    return ProgressPruning(ProgressScorer(component, params, config, paths.root), monitor, f"trial:{version}",
                           poll_seconds=POLL_SECONDS)
