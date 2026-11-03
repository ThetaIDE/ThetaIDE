"""Hydra Optuna sweeper that passes Optuna trial identity to each job.

The upstream Hydra adapter uses an independent job counter, which restarts at
zero on resume. Explicit trial identity prevents checkpoint collisions.
"""

import json
import math
import logging
import time
from pathlib import Path

import optuna
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.plugins.sweeper import Sweeper
from hydra_plugins.hydra_optuna_sweeper._impl import (
    OptunaSweeperImpl,
    create_optuna_distribution_from_override,
)
from omegaconf import OmegaConf
from optuna.distributions import BaseDistribution, FloatDistribution

logger = logging.getLogger(__name__)


def _distribution_from_override(override):
    """Handle stepped floats, which the upstream adapter truncates to ints."""
    if override.is_range_sweep():
        value = override.value()
        if not value.shuffle and value.start is not None and value.stop is not None:
            if any(isinstance(part, float) for part in (value.start, value.stop, value.step)):
                return FloatDistribution(low=float(value.start), high=float(value.stop), step=float(value.step))
    return create_optuna_distribution_from_override(override)


def _export_summary(config, study):
    from src.app.tuners.optuna.utils import write_study_summary

    agent = config.get("agent")
    model = config.get("model")
    name = agent.get("name") if hasattr(agent, "get") else None
    name = name or (model.get("name") if hasattr(model, "get") else None)
    group, experiment = config.get("group"), config.get("experiment_id")
    if not name or not group or not experiment:
        return
    root = Path("results/checkpoints") / str(group) / str(experiment) / str(name).replace("/", "_")
    try:
        write_study_summary(study, root)
    except Exception:
        logger.exception("Could not write tuning summary for study %s", study.study_name)


class ThetaOptunaSweeper(Sweeper):
    def __init__(self, sampler, direction, storage, study_name, n_trials, n_jobs, search_space,
                 pruner="none", pruner_startup_trials=5, pruner_warmup_steps=0, timeout_seconds=0,
                 max_failed_trials=0):
        if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError("Optuna n_trials must be a positive integer")
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs <= 0:
            raise ValueError("Optuna n_jobs must be a positive integer")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds < 0:
            raise ValueError("Optuna timeout_seconds must be a nonnegative integer")
        if isinstance(max_failed_trials, bool) or not isinstance(max_failed_trials, int) or max_failed_trials < 0:
            raise ValueError("Optuna max_failed_trials must be a nonnegative integer")
        self.impl = OptunaSweeperImpl(sampler, direction, storage, study_name, n_trials, n_jobs, search_space)
        from src.app.tuners.optuna.policy import make_pruner
        self.pruner_name = pruner
        self.pruner_startup_trials = pruner_startup_trials
        self.pruner_warmup_steps = pruner_warmup_steps
        self.pruner = make_pruner(pruner, pruner_startup_trials, pruner_warmup_steps)
        self.timeout_seconds = timeout_seconds
        self.max_failed_trials = max_failed_trials

    def setup(self, *, hydra_context, task_function, config):
        self.impl.setup(hydra_context=hydra_context, task_function=task_function, config=config)

    def sweep(self, arguments):
        impl = self.impl
        parsed = OverridesParser.create().parse_overrides(arguments)
        search_space = dict(impl.search_space)
        fixed = {}
        for override in parsed:
            value = _distribution_from_override(override)
            (search_space if isinstance(value, BaseDistribution) else fixed)[override.get_key_element()] = value
        for name in fixed:
            search_space.pop(name, None)

        if isinstance(impl.direction, (list, tuple)):
            raise ValueError("Theta's Optuna sweeper currently supports one objective")
        direction = impl.direction if isinstance(impl.direction, str) else impl.direction.name
        study = optuna.create_study(
            study_name=impl.study_name, storage=impl.storage, sampler=impl.sampler,
            pruner=self.pruner, direction=direction, load_if_exists=True,
        )
        tuning = impl.config.get("tuning")
        metric = tuning.get("metric") if hasattr(tuning, "get") else None
        if self.pruner_name != "none" and (not impl.storage or not metric):
            raise ValueError("Optuna pruning requires persistent storage and an objective metric.")
        if metric:
            recorded_metric = study.user_attrs.get("theta_objective_metric")
            if recorded_metric and recorded_metric != metric:
                raise ValueError(f"Study {study.study_name!r} already optimizes {recorded_metric!r}, not {metric!r}")
            study.set_user_attr("theta_objective_metric", str(metric))
        study.set_user_attr("theta_last_stop_reason", "running")
        remaining = impl.n_trials
        failed_count = 0
        start = time.monotonic()
        while remaining > 0 and (not self.timeout_seconds or time.monotonic() - start < self.timeout_seconds):
            trials = [study.ask() for _ in range(min(remaining, impl.n_jobs))]
            jobs = []
            try:
                for trial in trials:
                    for name, distribution in search_space.items():
                        trial._suggest(name, distribution)
                    # Fixed overrides can create or replace parent containers.
                    # Apply them first so sampled nested values exist and cannot
                    # subsequently be overwritten by a fixed parent list.
                    params = dict(fixed)
                    params.update(trial.params)
                    args = [f"{name}={value}" for name, value in params.items()]
                    args.extend([
                        "++theta_tuning.component=optuna",
                        f"++theta_tuning.study_name={json.dumps(study.study_name)}",
                        f"++theta_tuning.trial_number={trial.number}",
                    ])
                    if self.pruner_name != "none":
                        args.extend([
                            f"++theta_tuning.optuna_trial_id={trial._trial_id}",
                            f"++theta_tuning.storage={json.dumps(str(impl.storage))}",
                            f"++theta_tuning.objective_metric={json.dumps(str(metric))}",
                            f"++theta_tuning.pruner={self.pruner_name}",
                            f"++theta_tuning.pruner_startup_trials={self.pruner_startup_trials}",
                            f"++theta_tuning.pruner_warmup_steps={self.pruner_warmup_steps}",
                        ])
                    jobs.append(tuple(args))
            except BaseException as exc:
                for trial in trials:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                study.set_user_attr("theta_last_stop_reason", "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "launch_error")
                _export_summary(impl.config, study)
                raise
            try:
                results = impl.launcher.launch(jobs, initial_job_idx=impl.job_idx)
            except BaseException as exc:
                for trial in trials:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                study.set_user_attr("theta_last_stop_reason", "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "launch_error")
                _export_summary(impl.config, study)
                raise
            impl.job_idx += len(results)
            failure = None
            for trial, result in zip(trials, results):
                try:
                    returned = result.return_value
                    if isinstance(returned, dict) and returned.get("theta_trial_status") == "pruned":
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        continue
                    value = float(returned)
                    if not math.isfinite(value):
                        raise ValueError("Tuning objective must be finite")
                    study.tell(trial, value)
                except optuna.TrialPruned:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                except Exception as exc:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    failed_count += 1
                    failure = failure or exc
            if len(results) != len(trials):
                for trial in trials[len(results):]:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                study.set_user_attr("theta_last_stop_reason", "launch_error")
                _export_summary(impl.config, study)
                raise RuntimeError("Launcher did not return a result for every tuning trial")
            if failure and failed_count > self.max_failed_trials:
                study.set_user_attr("theta_last_stop_reason", "failure_limit")
                _export_summary(impl.config, study)
                raise failure
            _export_summary(impl.config, study)
            remaining -= len(trials)
        study.set_user_attr("theta_last_stop_reason", "time_budget" if remaining else "trial_limit")
        _export_summary(impl.config, study)
        try:
            best = study.best_trial
        except ValueError:
            _export_summary(impl.config, study)
            raise ValueError(f"Optuna study {study.study_name!r} has no completed trials") from None
        OmegaConf.save(
            OmegaConf.create({"name": "optuna", "best_params": best.params, "best_value": best.value}),
            f"{impl.config.hydra.sweep.dir}/optimization_results.yaml",
        )
