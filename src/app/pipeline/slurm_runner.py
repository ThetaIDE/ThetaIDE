"""Slurm training phase runner.

Generates and submits training jobs to Slurm cluster.
"""

import shlex
import sys
from pathlib import Path

from src.app.pipeline.commands import build_method_overrides
from src.app.pipeline.config import normalize_agent_name
from src.app.pipeline.datasets import fast_purge_dir, resolve_dataset_for_method
from src.app.pipeline.engine import resolve_engine
from src.app.pipeline.experiment_snapshot import save_experiment_snapshot, save_run_config
from src.app.pipeline.tuners import get_tuner, tuner_name
from src.app.pipeline.runtime import get_shell_env_block, get_shell_python_cmd
from src.app.pipeline.slurm import generate_sbatch_header, generate_sbatch_script, submit_sbatch
from src.app.pipeline.validation import paradigm_uses_static_dataset
from src.app.pipeline.plot_config import should_run_plotting
from src.app.pipeline.tuning_config import require_runnable_tuning, tuning_disabled


def _winner_command(python_cmd, group, experiment_id, agent_name, storage_url, study_name, component="optuna"):
    arguments = [component, group, experiment_id, agent_name, storage_url or "", study_name]
    return f"{python_cmd} -m src.app.pipeline.tuner_finalize " + " ".join(shlex.quote(str(arg)) for arg in arguments)


def run_slurm_training(cfg, context):
    methods = context.get("methods", {})
    require_runnable_tuning(cfg, methods)
    sanitized_extra_args = context.get("sanitized_extra_args", [])
    storage_url = context.get("storage_url", None)
    is_sweep = context.get("is_sweep", False)
    site_cfg = getattr(cfg, "site", None)
    paradigm = cfg.get("paradigm", "offline_rl")

    log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
    fast_purge_dir(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # A sweep or a disabled-but-retained tuning configuration may refer to
    # existing study artifacts. Do not erase those artifacts during setup.
    has_tuned_method = any(
        method.get("tune") or method.get("search_space")
        for method in methods.values() if hasattr(method, "get")
    )
    if not (cfg.get("recover", False) or cfg.get("tuning") is not None or is_sweep or has_tuned_method):
        ckpt_dir = Path("results/checkpoints") / cfg.group / cfg.experiment_id
        fast_purge_dir(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        exp_log_dir = Path("results/logs") / cfg.group / cfg.experiment_id
        fast_purge_dir(exp_log_dir)
        exp_log_dir.mkdir(parents=True, exist_ok=True)

        if should_run_plotting(cfg):
            clean_exp = Path(cfg.experiment_id).stem
            exp_plot_dir = Path("results/plots") / cfg.group / clean_exp
            fast_purge_dir(exp_plot_dir)
            exp_plot_dir.mkdir(parents=True, exist_ok=True)

    save_experiment_snapshot(cfg, context)
    save_run_config(cfg)
    cfg_consolidate = cfg.get("consolidate", None)
    site_consolidate = getattr(site_cfg, "consolidate", False) if site_cfg else False
    should_consolidate = cfg_consolidate if cfg_consolidate is not None else site_consolidate

    if should_consolidate:
        print(f"\n=== Preparing Consolidated Slurm Job ({cfg.experiment_id}) ===")
        job_name = f"all_{cfg.experiment_id}"
        script_content = generate_sbatch_header(job_name=job_name, log_dir=log_dir, cfg=cfg, is_consolidated=True)
        script_content += "\nset -e\n" + get_shell_env_block(site_cfg) + "\n"
        python_cmd = get_shell_python_cmd(site_cfg)

        for method_name, method_cfg in methods.items():
            agent_name = normalize_agent_name(method_name)
            dataset_path = None

            if paradigm_uses_static_dataset(paradigm):
                try:
                    dataset_path = resolve_dataset_for_method(method_name, method_cfg, cfg)
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

            method_tune = method_cfg.get("tune") or method_cfg.get("search_space") or {}
            method_is_sweep = not tuning_disabled(cfg) and (bool(is_sweep) or bool(method_tune))
            tuner = get_tuner(cfg) if method_is_sweep else None
            study_name = tuner.study_name(cfg, agent_name) if tuner else None

            cmd_args = build_method_overrides(
                method_name=method_name,
                method_cfg=method_cfg,
                dataset_path=dataset_path,
                extra_args=sanitized_extra_args,
                cfg=cfg,
                study_name=study_name,
                is_sweep=method_is_sweep,
            )

            if method_is_sweep:
                tuner.prepare(cfg, storage_url, study_name)

            engine_script, engine_python = resolve_engine(method_cfg, cfg)
            if tuner is not None:
                engine_script, engine_python = tuner.job_entrypoint(cfg, method_cfg, engine_script, engine_python)
            job_python = shlex.quote(engine_python) if engine_python else python_cmd
            train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
            script_content += f'echo "=== [Phase: Training] {method_name} ==="\n'
            script_content += f"{job_python} -m src.app.pipeline.provider_worker {shlex.quote(engine_script)} {train_cmd}\n\n"

            if method_is_sweep:
                script_content += f'echo "=== Promoting Winning Checkpoint for {method_name} ==="\n'
                script_content += _winner_command(job_python, cfg.group, cfg.experiment_id, agent_name, storage_url, study_name, tuner_name(cfg)) + "\n\n"

        if should_run_plotting(cfg):
            plot_cmd = f"{python_cmd} plot/manager.py {cfg.group}/{cfg.experiment_id}"
            if cfg.get("plot_style", None):
                plot_cmd += f" --style {cfg.get('plot_style', None)}"
            if cfg.get("use_cache", False):
                plot_cmd += " --use-cache"
            script_content += 'echo "=== [Generating Final Plots] ==="\n'
            script_content += f"{plot_cmd}\n\n"

        slurm_file = log_dir / f"consolidated_{cfg.experiment_id}.slurm"
        with open(slurm_file, "w") as f:
            f.write(script_content)

        print(f"Submitting Consolidated Slurm Job: {slurm_file}")
        job_id = submit_sbatch(script_content)
        return [job_id] if job_id else [], [], True

    job_ids = []
    submitted_idx = 0
    total_jobs = len(methods)

    for method_name, method_cfg in methods.items():
        agent_name = normalize_agent_name(method_name)
        dataset_path = None

        if paradigm_uses_static_dataset(paradigm):
            try:
                dataset_path = resolve_dataset_for_method(method_name, method_cfg, cfg)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

        method_tune = method_cfg.get("tune") or method_cfg.get("search_space") or {}
        method_is_sweep = not tuning_disabled(cfg) and (bool(is_sweep) or bool(method_tune))
        job_name = f"{agent_name}_{cfg.experiment_id}"
        tuner = get_tuner(cfg) if method_is_sweep else None
        study_name = tuner.study_name(cfg, agent_name) if tuner else None

        cmd_args = build_method_overrides(
            method_name=method_name,
            method_cfg=method_cfg,
            dataset_path=dataset_path,
            extra_args=sanitized_extra_args,
            cfg=cfg,
            study_name=study_name,
            is_sweep=method_is_sweep,
        )

        script_content = generate_sbatch_header(
            job_name=job_name,
            log_dir=log_dir,
            cfg=cfg,
            is_consolidated=False,
        )
        script_content += "\nset -e\n" + get_shell_env_block(site_cfg) + "\n"
        python_cmd = get_shell_python_cmd(site_cfg)

        if method_is_sweep:
            tuner.prepare(cfg, storage_url, study_name)

        engine_script, engine_python = resolve_engine(method_cfg, cfg)
        if tuner is not None:
            engine_script, engine_python = tuner.job_entrypoint(cfg, method_cfg, engine_script, engine_python)
        job_python = shlex.quote(engine_python) if engine_python else python_cmd
        train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
        script_content += f'echo "=== [Phase: Training] {method_name} ==="\n'
        script_content += f"{job_python} -m src.app.pipeline.provider_worker {shlex.quote(engine_script)} {train_cmd}\n\n"

        if method_is_sweep:
            script_content += f'echo "=== Promoting Winning Checkpoint for {method_name} ==="\n'
            script_content += _winner_command(job_python, cfg.group, cfg.experiment_id, agent_name, storage_url, study_name, tuner_name(cfg)) + "\n\n"

        slurm_file = log_dir / f"{job_name}.slurm"
        with open(slurm_file, "w") as f:
            f.write(script_content)

        submitted_idx += 1
        print(f"[{submitted_idx}/{total_jobs}] Submitting [{method_name}] ...", end="", flush=True)
        job_id = submit_sbatch(script_content)
        if job_id:
            job_ids.append(job_id)

    if should_run_plotting(cfg) and job_ids:
        plot_job_name = f"plot_{cfg.experiment_id}"
        dependency_str = ":".join(job_ids)
        plot_header = generate_sbatch_header(
            job_name=plot_job_name,
            log_dir=log_dir,
            cfg=cfg,
            dependency=dependency_str,
        )
        python_cmd = get_shell_python_cmd(site_cfg)
        plot_cmd = f"{python_cmd} plot/manager.py {cfg.group}/{cfg.experiment_id}"
        if cfg.get("plot_style", None):
            plot_cmd += f" --style {cfg.get('plot_style', None)}"
        if cfg.get("use_cache", False):
            plot_cmd += " --use-cache"
        plot_content = (
            plot_header
            + "\n"
            + get_shell_env_block(site_cfg)
            + f'\n\necho "=== [Generating Final Plots] ==="\n{plot_cmd}\n'
        )
        plot_slurm_file = log_dir / f"{plot_job_name}.slurm"
        with open(plot_slurm_file, "w") as f:
            f.write(plot_content)
        print(f"[Post-Process] Submitting Plotting Job (dependent on {len(job_ids)} jobs) ...", end="", flush=True)
        submit_sbatch(plot_content)

    return job_ids, [], False
