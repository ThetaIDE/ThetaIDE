"""Planning and records for opt-in repeated training executions.

Repetition is independent of the runner's evaluation capabilities. A planned
replicate has one attempt; retries and resume are not yet implemented.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from src.app.pipeline.engine import resolve_engine, runner_supports_episode_outcomes
from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.config import normalize_agent_name


def make_plan(cfg, context):
    variability = cfg.get("variability")
    if variability is None:
        return None
    if not hasattr(variability, "get"):
        raise ConfigurationError("variability must be a mapping")
    explicit_seeds = variability.get("training_seeds")
    has_count = variability.get("training_runs") is not None
    has_start = variability.get("starting_seed") is not None
    if explicit_seeds is not None and (has_count or has_start):
        raise ConfigurationError(
            "Choose either variability.training_seeds or variability.training_runs with starting_seed; do not configure both forms."
        )
    if explicit_seeds is not None:
        seeds = explicit_seeds
    else:
        count = variability.get("training_runs", 1)
        start = variability.get("starting_seed", cfg.get("seed", 1))
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ConfigurationError("variability.training_runs must be a positive integer")
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start <= 2**32 - 1:
            raise ConfigurationError("variability.starting_seed must be an integer between 0 and 4294967295")
        if start + count - 1 > 2**32 - 1:
            raise ConfigurationError("variability.starting_seed plus training_runs exceeds the supported seed range")
        seeds = list(range(start, start + count))
    if not isinstance(seeds, (list, tuple)) and not OmegaConf.is_list(seeds):
        raise ConfigurationError("variability.training_seeds must be a nonempty list of unique integer seeds")
    seeds = list(seeds)
    if not seeds or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= 2**32 - 1 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ConfigurationError("variability.training_seeds must contain unique integers between 0 and 4294967295")
    parallel = variability.get("parallel_runs", 1)
    if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1:
        raise ConfigurationError("variability.parallel_runs must be a positive integer")
    evaluation_seed = variability.get("evaluation_seed")
    if evaluation_seed is not None and (isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int) or not 0 <= evaluation_seed <= 2**32 - 1):
        raise ConfigurationError("variability.evaluation_seed must be an integer between 0 and 4294967295")
    episodes = cfg.get("eval_episodes", 0)
    if cfg.get("workflow"):
        raise ConfigurationError("variability does not yet support workflow launches")
    if not context.get("is_interactive", True):
        raise ConfigurationError("variability is not yet supported by the Slurm launcher")
    if cfg.get("recover"):
        raise ConfigurationError("variability cannot yet be combined with recovery")
    capabilities = {}
    from src.app.telemetry.selection import selection_from_config
    selection = selection_from_config(cfg)
    if (cfg.get("paradigm") == "online_rl" and isinstance(episodes, int) and episodes > 0
            and selection.keeps_section("eval") and not selection.keeps("eval/reward")):
        raise ConfigurationError("variability evaluation requires telemetry.metrics to include eval/reward")
    for name, method in context["methods"].items():
        script = resolve_engine(method, cfg)[0]
        runner_supports_eval = runner_supports_episode_outcomes(script)
        episode_events = (runner_supports_eval and cfg.get("paradigm") == "online_rl"
                          and isinstance(episodes, int) and not isinstance(episodes, bool) and episodes > 0
                          and selection.keeps("eval/reward"))
        capabilities[name] = {"runner": script, "episode_events": episode_events,
                              "evaluation_seed_control": episode_events,
                              "training_seed": "passed to runner; component support varies"}
        if evaluation_seed is not None and not episode_events:
            raise ConfigurationError(f"Method {name!r} does not support the requested variability.evaluation_seed; omit it to use the runner's existing evaluation protocol")
    return {"training_seeds": seeds, "evaluation_seed": evaluation_seed, "parallel_runs": parallel,
            "eval_episodes": episodes, "capabilities": capabilities}


def run_execution(cfg, context, plan, *, dispatch=None, parent=None, raise_on_failure=True):
    """Execute a seed panel through one lifecycle for experiments and resolved trials.

    A resolved-task dispatcher receives (config, run, on_start, on_wait). It
    launches the worker; this executor owns identity, status and cancellation.
    """
    from src.app.pipeline.local_runner import _run_methods_sequential, _setup_output_dirs, run_plotting_phase

    execution_id = uuid.uuid4().hex
    result_experiment_id = (str(cfg.experiment_id) if parent else
                            f"{cfg.experiment_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{execution_id[:8]}")
    root = Path("results/executions") / str(cfg.group) / str(cfg.experiment_id) / execution_id
    root.mkdir(parents=True, exist_ok=False)
    runs = []
    logs_root = Path("results/logs") / str(cfg.group) / result_experiment_id
    for method in context["methods"]:
        for version_index, seed in enumerate(plan["training_seeds"]):
            run_id = uuid.uuid4().hex
            version = (f"{parent['log_version']}_sample_{execution_id[:10]}_seed_{version_index}" if parent
                       else f"version_{version_index}")
            agent = str(context["methods"][method].get("name") or normalize_agent_name(method)).replace("/", "_")
            runs.append({"run_id": run_id, "method": method, "training_seed": seed,
                         "status": "queued", "experiment_id": result_experiment_id,
                         "version": version, "agent": agent,
                         "output_log": str(root / "runs" / f"{run_id}.log"),
                         "logs": str(logs_root),
                         "event_path": str(logs_root / agent / version / "evaluation_events.jsonl"),
                         "checkpoints": str(Path("results/checkpoints") / str(cfg.group) / result_experiment_id / agent / version)})
    if OmegaConf.is_config(cfg):
        from src.app.pipeline.compose import effective_config
        config = effective_config(cfg)
    else:
        config = dict(cfg)
    manifest = {"schema_version": 1, "execution_id": execution_id, "group": str(cfg.group),
                "experiment_id": str(cfg.experiment_id), "result_experiment_id": result_experiment_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "owner_pid": int(os.environ.get("THETA_PIPELINE_PID", os.getpid())),
                "parent": parent, "config": config, "plan": plan, "runs": runs}
    path = root / "manifest.json"
    lock = threading.Lock()

    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        temporary.replace(path)

    from src.app.pipeline.experiment_snapshot import save_experiment_snapshot, save_run_config
    run_cfg_base = OmegaConf.create(config)
    run_cfg_base.experiment_id = result_experiment_id
    # Prepare the unique result experiment once. Workers then prepare only their
    # own agent/version subtree, so parallel setup cannot erase sibling output.
    if dispatch is None:
        _setup_output_dirs(run_cfg_base)
        save_experiment_snapshot(run_cfg_base, context)
        save_run_config(run_cfg_base)

    resource_pool = None
    tickets = {}
    if os.environ.get('THETA_RESOURCE_POOL'):
        from src.app.services.resource_budget import ResourcePool
        resource_pool = ResourcePool(os.environ['THETA_RESOURCE_POOL'], os.environ.get('THETAIDE_SETTINGS_FILE'))

    def execute(run):
        from src.app.services.cancellation import requested, TrainingCancelled
        targets = [f"replicate:{run['run_id']}", f"method:{normalize_agent_name(run['method'])}"]
        if parent and parent.get("kind") == "trial":
            targets.append(f"trial:{parent['log_version']}")
        with lock:
            if requested(targets):
                run["status"] = "cancelled"
                save()
                return
            run["status"] = "pending"
            save()

        def worker_started(pid):
            with lock:
                run.update(status="running", worker_pid=pid,
                           execution_started=datetime.now(timezone.utc).isoformat())
                run.pop("waiting_reason", None)
                save()
            print(f"[Training] {run['method']} · seed {run['training_seed']} started (PID {pid}).", flush=True)

        def worker_waiting(reason=""):
            with lock:
                run.update(status="pending", worker_pid=None, execution_started=None)
                run["waiting_reason"] = reason
                save()

        overrides = [arg for arg in context.get("sanitized_extra_args", [])
                     if arg.split("=", 1)[0].lstrip("+") not in {
                         "experiment_id", "seed", "agent.seed", "variability_run_id",
                         "variability_method", "variability_version", "variability_evaluation_seed",
                     }]
        overrides.extend([f"++experiment_id={run['experiment_id']}", f"++seed={run['training_seed']}",
                          f"++variability_run_id={run['run_id']}", f"++variability_method={run['method']}",
                          f"++variability_version={run['version']}"])
        if context["methods"][run["method"]].get("agent"):
            overrides.append(f"++agent.seed={run['training_seed']}")
        if plan["evaluation_seed"] is not None:
            overrides.append(f"++variability_evaluation_seed={plan['evaluation_seed']}")
        try:
            started = time.monotonic()
            if dispatch is not None:
                child = OmegaConf.create(config)
                with open_dict(child):
                    child.experiment_id = result_experiment_id
                    child.seed = run["training_seed"]
                    if child.get("agent") is not None:
                        child.agent.seed = run["training_seed"]
                    child.variability_run_id = run["run_id"]
                    child.variability_method = run["agent"]
                    child.variability_version = run["version"]
                    child.variability_evaluation_seed = plan["evaluation_seed"]
                    child.theta_variability_sample = True
                dispatch(child, dict(run, resource_ticket=tickets.get(run['run_id'])), worker_started, worker_waiting)
            else:
                _run_methods_sequential(cfg, {**context, "methods": {run["method"]: context["methods"][run["method"]]},
                                         "sanitized_extra_args": overrides,
                                         "output_log": run["output_log"],
                                         "propagate_cancellation": True,
                                         "on_worker_start": worker_started,
                                         "on_worker_wait": worker_waiting,
                                         "resource_ticket": tickets.get(run["run_id"])})
        except TrainingCancelled:
            with lock:
                run.update(status="cancelled", elapsed_seconds=time.monotonic() - started)
                save()
        except (Exception, SystemExit) as exc:
            output_path = Path(run["output_log"])
            output_tail = ""
            if output_path.is_file():
                try:
                    output_tail = output_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                except OSError:
                    pass
            error = str(exc)
            if output_tail:
                error = f"{error}\n\nReplicate output (tail):\n{output_tail}"
            with lock:
                run.update(status="failed", error=error, elapsed_seconds=time.monotonic() - started)
                save()
        else:
            with lock:
                run["status"] = "cancelled" if requested(targets) else "completed"
                run["elapsed_seconds"] = time.monotonic() - started
                save()
        print(f"[Training] {run['method']} · seed {run['training_seed']}: {run['status']}.", flush=True)

    save()
    if parent:
        # A portable link lets existing trial consumers discover the same seed
        # manifest that ordinary repeated experiments expose.
        link = logs_root / parent["method"] / parent["log_version"] / "execution.json"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.write_text(json.dumps({"manifest": os.path.relpath(path, link.parent)}) + "\n", encoding="utf-8")
    print(f"Execution manifest: {path}")
    workers = 1 if os.environ.get('THETA_IN_ALLOCATION') == '1' else plan['parallel_runs']
    if os.environ.get('THETA_RESOURCE_POOL'):
        from src.app.services.resource_budget import orchestration_workers
        workers = min(workers, orchestration_workers(len(runs)))
        tickets = dict(zip((run['run_id'] for run in runs), resource_pool.enqueue(
            min(workers, len(runs)), parent=os.environ.get('THETA_RESOURCE_QUEUE_PARENT'))))
    def execute_queued(run):
        # Only ready seeds reserve FIFO positions; undispatched seeds must not
        # block other experiments while waiting for this executor's slots.
        if resource_pool and run['run_id'] not in tickets:
            tickets[run['run_id']] = resource_pool.enqueue()[0]
        try:
            return execute(run)
        finally:
            if resource_pool:
                resource_pool.dequeue(tickets[run['run_id']])

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="theta-replicate") as pool:
            futures = [pool.submit(execute_queued, run) for run in runs]
            for future in as_completed(futures):
                future.result()
    finally:
        if resource_pool:
            for ticket in tickets.values():
                resource_pool.dequeue(ticket)
    # Plotting is still a pipeline phase. Run it once against the shared result
    # experiment after all per-version metrics are available.
    if dispatch is None:
        run_plotting_phase(run_cfg_base, context)
    if raise_on_failure and any(run["status"] != "completed" for run in runs):
        raise RuntimeError(f"One or more replicates failed; see {path}")
    return path


def mark_cancelled(group, experiment_id, owner_pid):
    """Mark a terminated GUI launch's unfinished replicates, preserving completed work."""
    from src.app.services.catalog import check_path_segment

    try:
        check_path_segment(str(group))
        check_path_segment(str(experiment_id))
    except ValueError:
        return
    for path in (Path("results/executions") / str(group) / str(experiment_id)).glob("*/manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("owner_pid") != owner_pid:
                continue
            for run in manifest.get("runs", []):
                if run.get("status") in ("queued", "pending", "running"):
                    run["status"] = "cancelled"
            temporary = path.with_suffix(".cancel.tmp")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        except (OSError, ValueError):
            continue


def finish_trial_runs(group, experiment_id, log_version, status, *, owner_pid=None, method=None):
    """Close unfinished seed records after their owning processes have stopped."""
    if not group or not experiment_id or not (log_version or (owner_pid and method)):
        return
    for path in (Path("results/executions") / str(group) / str(experiment_id)).glob("*/manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            parent = manifest.get("parent") or {}
            if log_version and parent.get("log_version") != log_version:
                continue
            if owner_pid and manifest.get("owner_pid") != owner_pid:
                continue
            if method and parent.get("method") != method:
                continue
            for run in manifest["runs"]:
                if run.get("status") in ("queued", "pending", "running"):
                    run["status"] = status
            temporary = path.with_suffix(".stopped.tmp")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        except (OSError, ValueError, KeyError):
            continue
