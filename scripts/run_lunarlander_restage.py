"""Restart 150k LunarLander tuning: DNN PPO, vanilla BlendRL, then eight features.

--start captures templates and draws shared tuning seeds before detaching. Only
completed studies promote winners. Held-out recipes are prepared, never launched.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RECIPES = ROOT / "src/usr/config/experiment/BlendRL/LunarLander"
LOGS = ROOT / "src/usr/results/sweep_logs/lunarlander_restage"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read(name):
    return yaml.safe_load((RECIPES / name).read_text())


def templates():
    final = read("Bets/Held-out Ablations Template.yaml")
    dnn = copy.deepcopy(final)
    dnn["methods"] = {"DNN": copy.deepcopy(final["methods"]["DNN"])}
    dnn["env"]["num_envs"] = 16
    dnn["methods"]["DNN"]["tune"] = {
        "config.env.num_envs": {"type": "choice", "values": [8, 16]},
        "agent.n_steps": {"type": "choice", "values": [128, 256, 512, 1024]},
        "agent.learning_rate": {"type": "log_float", "low": 3e-5, "high": 1e-3},
        "agent.learning_rate_schedule": {"type": "choice", "values": ["constant", "linear"]},
        "agent.batch_size": {"type": "choice", "values": [64, 128, 256]},
        "agent.n_epochs": {"type": "choice", "values": [4, 10, 20]},
        "agent.gamma": {"type": "choice", "values": [0.99, 0.995, 0.999]},
        "agent.gae_lambda": {"type": "float", "low": 0.90, "high": 0.99},
        "agent.ent_coef": {"type": "float", "low": 0.0, "high": 0.02},
        "agent.clip_range": {"type": "float", "low": 0.1, "high": 0.3},
    }
    blend = read("Full Logic/BlendRL Sweep 150k.yaml")
    features = read("Bets/Feature Sweeps 150k.yaml")
    for method in features["methods"].values():
        assert all(key.startswith("model.") for key in method["tune"])
    for cfg, trials in ((dnn, 48), (blend, 24), (features, 24)):
        cfg["total_timesteps"] = 150000
        cfg["eval_freq_steps"] = 10000
        cfg["eval_episodes"] = 20
        cfg["run_in_parallel"] = False
        cfg["no_plot"] = True
        cfg["timestamp_experiment"] = True
        cfg["objective"] = copy.deepcopy(blend["objective"])
        cfg["tuning"] = copy.deepcopy(blend["tuning"])
        cfg["tuning"]["options"]["optuna"].update(
            n_trials=trials, n_jobs=1, tpe_startup_trials=12,
            pruner_startup_trials=12, pruner_warmup_steps=8,
            study_mode="new", max_failed_trials=0)
    return {"DNN Sweep": dnn, "BlendRL Sweep": blend,
            "Feature Sweeps": features, "Held-out Ablations": final}


def assign(target, dotted, value):
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = copy.deepcopy(value)


def refinement_templates(previous):
    """Expand the promising DNN boundaries, keeping the remaining protocol fixed."""
    plans = copy.deepcopy(previous["templates"])
    winner = previous["stages"][0]["winners"]["DNN"]
    dnn = plans["DNN Sweep"]
    inherit(dnn, winner["params"])
    dnn["methods"]["DNN"]["tune"] = {
        "config.env.num_envs": {"type": "choice", "values": [8, 16, 32]},
        "agent.n_steps": {"type": "choice", "values": [64, 128, 256]},
        "agent.batch_size": {"type": "choice", "values": [32, 64]},
        "agent.n_epochs": {"type": "choice", "values": [10, 20, 30, 40]},
        "agent.learning_rate": {"type": "log_float", "low": 3e-4, "high": 3e-3},
    }
    blend = plans["BlendRL Sweep"]["methods"]["BlendRL"]
    blend["tune"]["model.logic_lr"].update(low=1e-5)
    blend["tune"]["model.blender_lr"].update(low=1e-5)
    plans["Held-out Ablations"].update(total_timesteps=150000, eval_episodes=100, eval_freq_steps=50000)
    return plans


def dnn_protocol(cfg, params):
    """Include fixed PPO settings as well as searched ones when promoting a winner."""
    result = {"env.num_envs": cfg["env"]["num_envs"]}
    result.update({"agent." + key: copy.deepcopy(value)
                   for key, value in cfg["methods"]["DNN"]["agent"].items() if key != "name"})
    result.update(params)
    return result


def inherit(cfg, ppo, blend=None):
    """Promote the whole PPO protocol, including environment count, to every method."""
    for key, value in ppo.items():
        if key == "env.num_envs":
            assign(cfg, key, value)
        elif key.startswith("agent."):
            for method in cfg["methods"].values():
                assign(method, key, value)
        else:
            raise ValueError(f"Unexpected DNN parameter {key}")
    if blend:
        for method in cfg["methods"].values():
            if method["model"]["name"] != "blendrl_bets":
                continue
            for key, value in blend.items():
                if not key.startswith("model."):
                    raise ValueError(f"Unexpected BlendRL parameter {key}")
                assign(method, key, value)


def write_recipe(batch, name, cfg):
    path = RECIPES / "Restaged" / batch / (name + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# @package _global_\n" + yaml.safe_dump(cfg, sort_keys=False))
    return f"BlendRL/LunarLander/Restaged/{batch}/{name}"


def reference_trials():
    # RL Zoo's published LunarLander-v3 PPO settings, adapted only to this
    # experiment's 150k budget, fixed 64x64 model and remaining runner defaults.
    zoo = {"env.num_envs": 16, "agent.n_steps": 1024,
           "agent.learning_rate": 3e-4, "agent.learning_rate_schedule": "constant",
           "agent.batch_size": 64, "agent.n_epochs": 4, "agent.gamma": .999,
           "agent.gae_lambda": .98, "agent.ent_coef": .01, "agent.clip_range": .2}
    incumbent = read("Full Logic/Baselines.yaml")["methods"]["DNN"]["agent"]
    keys = [key for key in zoo if key.startswith("agent.")]
    refs = [zoo]
    for envs in (8, 16):
        refs.append({"env.num_envs": envs, **{key: incumbent[key.split(".")[1]] for key in keys},
                     "agent.n_steps": 2048 // envs})
    return refs


def run(directory):
    import optuna
    from src.app.pipeline.compose import compose_experiment, validate_composed
    state_file = directory / "state.json"
    state = json.loads(state_file.read_text())
    plans, batch = state["templates"], state["batch"]
    state.update(status="running", pid=os.getpid(), stages=[])
    storage = "sqlite:///" + str(ROOT / "src/usr/results/optuna/optuna.db")
    selected = {}
    try:
        for name in ("DNN Sweep", "BlendRL Sweep", "Feature Sweeps"):
            cfg = copy.deepcopy(plans[name])
            cfg["seed"], cfg["variability"] = state["seed"], copy.deepcopy(state["variability"])
            if name != "DNN Sweep":
                inherit(cfg, selected["DNN"], selected.get("BlendRL"))
            study_format = f"LunarLander/restaged/{batch}/{{method}}"
            cfg["tuning"]["options"]["optuna"]["study_name"] = study_format
            references = (state["reference_trials"] if name == "DNN Sweep"
                          else state.get("blend_reference_trials", []) if name == "BlendRL Sweep" else [])
            if references:
                # A new, empty study with queued reference configurations;
                # resume attaches to it without reusing any earlier observations.
                method = next(iter(cfg["methods"]))
                study = optuna.create_study(study_name=study_format.format(method=method),
                                            storage=storage, direction="maximize")
                for params in references:
                    study.enqueue_trial(params, user_attrs={"reference_configuration": True})
                cfg["tuning"]["options"]["optuna"]["study_mode"] = "resume"
            cfg["description"] = f"Restaged 150k LunarLander {batch}: {name}. Four matched tuning seeds; 20 eval episodes/mode every 10k. {state.get('profile', 'initial')} search. DNN protocol then vanilla BlendRL then feature parameters; no automatic held-out training."
            selector = write_recipe(batch, name, cfg)
            composed = compose_experiment(selector)
            validate_composed(composed)
            if references:
                from src.app.pipeline.seeds import record_study_seed_plan

                panel = state["variability"]
                record_study_seed_plan(composed.workspace, composed.cfg.group, study.study_name,
                                       panel["training_seeds"], panel["evaluation_seed"])
            stage = {"name": name, "selector": selector, "status": "running"}
            state["stages"].append(stage)
            state["current_stage"] = name
            save(state_file, state)
            with (directory / (name + ".log")).open("w") as log:
                child = subprocess.Popen([sys.executable, "-u", "run_pipeline.py", selector],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                save(state_file, state)
                code = child.wait()
            if code:
                raise RuntimeError(f"{name} exited {code}; downstream stages will not start")
            stage["winners"] = {}
            for method in cfg["methods"]:
                study_name = study_format.format(method=method)
                study = optuna.load_study(study_name=study_name, storage=storage)
                if study.user_attrs.get("theta_last_stop_reason") != "trial_limit":
                    raise RuntimeError(f"{study_name} did not finish its requested trials")
                winner = study.best_trial
                selected[method] = (dnn_protocol(cfg, winner.params) if method == "DNN"
                                    else dict(winner.params))
                stage["winners"][method] = {"study": study_name, "trial": winner.number,
                                            "value": winner.value, "params": winner.params}
            stage["status"] = "completed"
            save(state_file, state)
        final = copy.deepcopy(plans["Held-out Ablations"])
        inherit(final, selected["DNN"], selected["BlendRL"])
        for name, method in plans["Feature Sweeps"]["methods"].items():
            for key, value in selected[name].items():
                if key not in method["tune"]:
                    raise ValueError(f"Unexpected feature parameter {name}/{key}")
                assign(final["methods"][name], key, value)
        final.update(seed=None, total_timesteps=150000, eval_episodes=100, eval_freq_steps=50000)
        final["variability"] = {"training_runs": 10, "parallel_runs": 4}
        final["description"] = f"Held-out 150k comparison from {batch}. Fresh seeds at launch, 100 episodes/mode/checkpoint; shared DNN-selected PPO protocol, retuned vanilla BlendRL and eight feature variants. Prepared only; not automatically launched."
        selector = write_recipe(batch, "Held-out Ablations", final)
        validate_composed(compose_experiment(selector))
        state.update(status="completed", current_stage=None, selected=selected,
                     held_out_selector=selector)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(state_file, state)


def start(refine_from=None):
    from src.app.pipeline.seeds import draw_seeds, recorded_seeds
    from src.app.workspace import WorkspacePaths
    batch = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = LOGS / batch
    previous = json.loads(Path(refine_from).read_text()) if refine_from else None
    seeds = ([previous["seed"], *previous["variability"]["training_seeds"],
              previous["variability"]["evaluation_seed"]] if previous
             else draw_seeds(6, recorded_seeds(WorkspacePaths())))
    plans = refinement_templates(previous) if previous else templates()
    refs = ([previous["stages"][0]["winners"]["DNN"]["params"]] if previous else reference_trials())
    if previous:
        # Explicit boundary probes complement the replayed incumbent. They share
        # the captured fixed settings and original four seeds for a paired check.
        centre = copy.deepcopy(refs[0])
        for changes in ({"agent.batch_size": 32}, {"agent.n_steps": 64},
                        {"agent.n_epochs": 40}, {"agent.learning_rate": .0015}):
            refs.append({**centre, **changes})
        # Exercise the new environment boundary first, both holding rollout
        # size constant (32x64=2048) and holding per-env horizon constant.
        refs[:0] = [{**centre, "env.num_envs": 32, "agent.n_steps": 64},
                    {**centre, "env.num_envs": 32}]
    state = {"batch": batch, "status": "starting", "templates": plans,
             "seed": seeds[0], "variability": {"training_runs": 4, "parallel_runs": 4,
             "training_seeds": seeds[1:5], "evaluation_seed": seeds[5]},
             "reference_trials": refs,
             "profile": "boundary refinement" if previous else "initial",
             "refines": str(refine_from) if previous else None,
             "reference_source": "https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml",
             "automatic_held_out_training": False,
             "maximum_training_transitions": (48 + 24 + 8 * 24) * 4 * 150000}
    if previous:
        import optuna
        storage = "sqlite:///" + str(ROOT / "src/usr/results/optuna/optuna.db")
        old_blend = optuna.load_study(study_name=f"LunarLander/restaged/{previous['batch']}/BlendRL", storage=storage)
        params = dict(old_blend.best_trial.params)
        state["blend_reference_trials"] = [params, {**params, "model.logic_confidence_floor": 0.0},
            {**params, "model.logic_lr": 1e-5, "model.blender_lr": 1e-5,
             "model.logic_confidence_floor": 0.0}]
    # Validate captured plans before launching any coordinator or creating studies.
    from src.app.pipeline.compose import compose_experiment, validate_composed
    for name, cfg in plans.items():
        candidate = copy.deepcopy(cfg)
        candidate["seed"], candidate["variability"] = seeds[0], copy.deepcopy(state["variability"])
        if name != "DNN Sweep":
            inherit(candidate, dnn_protocol(plans["DNN Sweep"], refs[0]))
        if candidate.get("tuning"):
            candidate["tuning"]["options"]["optuna"]["study_name"] = f"LunarLander/restaged/{batch}/{{method}}"
        validate_composed(compose_experiment(write_recipe(batch, name, candidate)))
    save(directory / "state.json", state)
    with (directory / "coordinator.log").open("w") as log:
        child = subprocess.Popen([sys.executable, "-u", __file__, "--run", str(directory)],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"pid": child.pid, "directory": str(directory)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--refine-from", type=Path, help="Prior batch state.json; replay its seed panel and refine its DNN winner")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".thetaide/matplotlib"))
    if args.start:
        start(args.refine_from)
    elif args.run:
        run(args.run)
    else:
        parser.error("Choose --start or --run")
