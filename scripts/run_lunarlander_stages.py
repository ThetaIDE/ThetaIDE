"""Run DNN -> BlendRL -> feature sweeps, promoting only their selected parameters.

Run with --start to detach a durable coordinator; --check validates the transfer
contract without training. Each stage is a separate Theta launch with private inputs.
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
RESULTS = ROOT / "src/usr/results/sweep_logs/lunarlander_staged"
FEATURES = ("RouteEpisode", "ValueBets", "ValueBetsRouted", "GPIResidual")


def assign(mapping, dotted, value):
    keys = dotted.split(".")
    for key in keys[:-1]:
        mapping = mapping.setdefault(key, {})
    mapping[keys[-1]] = value


def promote(method, params, prefix):
    for key, value in params.items():
        if not key.startswith(prefix):
            raise ValueError(f"Unexpected inherited parameter {key!r}; expected {prefix}")
        assign(method, key, copy.deepcopy(value))


def read(name):
    return yaml.safe_load((RECIPES / (name + ".yaml")).read_text())


def templates():
    baseline, bets, ablations = read("Baseline Sweep"), read("Bets Sweep"), read("Ablations")
    dnn = copy.deepcopy(baseline)
    dnn["methods"] = {"DNN": dnn["methods"]["DNN"]}
    blend = copy.deepcopy(baseline)
    blend["methods"] = {"BlendRL": blend["methods"]["BlendRL"]}
    blend["methods"]["BlendRL"]["tune"] = {
        k: v for k, v in blend["methods"]["BlendRL"]["tune"].items() if k.startswith("model.")}
    blend["methods"]["BlendRL"]["tune"]["model.logic_confidence_floor"] = {"type": "float", "low": 0., "high": .25}
    bets["methods"] = {k: bets["methods"][k] for k in FEATURES}
    for method in bets["methods"].values():
        # These belong to the shared BlendRL base, not to any added feature.
        for key in ("model.blender_lr", "model.blend_ent_coef", "model.logic_confidence_floor"):
            method["tune"].pop(key, None)
    final_names = ("DNN", "Logic", "BlendRL", "RouteEpisode", "FreeSignals", "ValueBets", "ValueBetsRouted", "GPIResidual")
    ablations["methods"] = {k: ablations["methods"][k] for k in final_names}
    for cfg, trials in ((dnn, 32), (blend, 24), (bets, 24)):
        cfg["variability"]["parallel_runs"] = 4
        cfg["tuning"]["options"]["optuna"]["n_trials"] = trials
    return {"DNN Sweep": dnn, "BlendRL Sweep": blend, "Feature Sweeps": bets, "Tuned Ablations": ablations}


def transfer(cfg, ppo, blend=None):
    for method in cfg["methods"].values():
        promote(method, ppo, "agent.")
        if blend is not None and method["model"]["name"] == "blendrl_bets":
            promote(method, blend, "model.")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def check():
    from src.app.pipeline.compose import compose_experiment, validate_composed
    plans = templates()
    ppo = {"agent.learning_rate": .0007, "agent.gamma": .995}
    blend = {"model.logic_lr": .002, "model.blender_lr": .0002, "model.logic_confidence_floor": .1}
    for name in ("BlendRL Sweep", "Feature Sweeps", "Tuned Ablations"):
        transfer(plans[name], ppo, blend if name != "BlendRL Sweep" else None)
        assert all(m["agent"]["learning_rate"] == .0007 for m in plans[name]["methods"].values())
    assert all(k.startswith("model.") for k in plans["BlendRL Sweep"]["methods"]["BlendRL"]["tune"])
    for m in plans["Feature Sweeps"]["methods"].values():
        assert m["model"]["logic_lr"] == .002
        assert all(k.startswith(("model.routing.", "model.confidence.value_heads.", "model.confidence.temperature")) for k in m["tune"])
    folder = RECIPES / "Staged Check"
    folder.mkdir(exist_ok=True)
    for name, cfg in plans.items():
        (folder / (name + ".yaml")).write_text("# @package _global_\n" + yaml.safe_dump(cfg, sort_keys=False))
        notices = validate_composed(compose_experiment("BlendRL/LunarLander/Staged Check/" + name))
        print(name, notices)


def run(directory):
    import optuna
    from src.app.pipeline.seeds import draw_seeds, recorded_seeds
    from src.app.workspace import WorkspacePaths
    from src.app.pipeline.compose import compose_experiment, validate_composed
    directory = directory.resolve()
    state = json.loads((directory / "state.json").read_text())
    batch = state["batch"]
    plans = state["templates"]
    stage_dir = RECIPES / "Staged" / batch
    stage_dir.mkdir(parents=True, exist_ok=True)
    seeds = draw_seeds(6, recorded_seeds(WorkspacePaths()))
    shared = {"training_runs": 4, "parallel_runs": 4, "training_seeds": seeds[1:5], "evaluation_seed": seeds[5]}
    state.update(pid=os.getpid(), status="running", seed=seeds[0], variability=shared, stages=[])
    save(directory / "state.json", state)
    storage = "sqlite:///" + str(ROOT / "src/usr/results/optuna/optuna.db")
    selected = {}
    try:
        for name in ("DNN Sweep", "BlendRL Sweep", "Feature Sweeps"):
            cfg = plans[name]
            cfg["seed"], cfg["variability"] = seeds[0], copy.deepcopy(shared)
            if name != "DNN Sweep":
                transfer(cfg, selected["DNN"], selected.get("BlendRL") if name == "Feature Sweeps" else None)
            cfg["tuning"]["options"]["optuna"]["study_name"] = f"LunarLander/staged/{batch}/{{method}}"
            cfg["description"] = f"Staged LunarLander tuning {batch}: {name}. Shared tuning seeds; inherited parameters frozen. See coordinator state for provenance."
            recipe = stage_dir / (name + ".yaml")
            recipe.write_text("# @package _global_\n" + yaml.safe_dump(cfg, sort_keys=False))
            selector = f"BlendRL/LunarLander/Staged/{batch}/{name}"
            validate_composed(compose_experiment(selector))
            record = {"stage": name, "selector": selector, "status": "running"}
            state["stages"].append(record)
            state["current_stage"] = name
            save(directory / "state.json", state)
            with (directory / (name + ".log")).open("w") as log:
                child = subprocess.Popen([sys.executable, "-u", "run_pipeline.py", selector], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                record["pid"] = child.pid
                save(directory / "state.json", state)
                code = child.wait()
            if code:
                raise RuntimeError(f"{name} exited {code}; see {directory / (name + '.log')}")
            record.update(status="completed", winners={})
            for method in cfg["methods"]:
                study_name = f"LunarLander/staged/{batch}/{method}"
                study = optuna.load_study(study_name=study_name, storage=storage)
                best = study.best_trial  # Fails rather than promoting an empty/failed study.
                selected[method] = dict(best.params)
                record["winners"][method] = {"study": study_name, "trial": best.number, "value": best.value, "params": best.params}
            save(directory / "state.json", state)
        final = plans["Tuned Ablations"]
        transfer(final, selected["DNN"], selected["BlendRL"])
        for method in FEATURES:
            promote(final["methods"][method], selected[method], "model.")
        # Fresh seeds will be drawn only if/when the held-out recipe is launched.
        final["seed"] = None
        final["variability"] = {"training_runs": 10, "parallel_runs": 4}
        final["description"] = f"Held-out staged LunarLander comparison from batch {batch}. PPO chosen on DNN, shared BlendRL parameters chosen next, four feature winners chosen last. FreeSignals has no extra numeric search. Not independently PPO-tuned per method."
        target = stage_dir / "Tuned Ablations.yaml"
        target.write_text("# @package _global_\n" + yaml.safe_dump(final, sort_keys=False))
        validate_composed(compose_experiment(f"BlendRL/LunarLander/Staged/{batch}/Tuned Ablations"))
        state.update(status="completed", current_stage=None, held_out_recipe=str(target), selected=selected)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(directory / "state.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".thetaide/matplotlib"))
    if args.check:
        check()
    elif args.start:
        batch = datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = RESULTS / batch
        save(directory / "state.json", {"batch": batch, "status": "starting", "templates": templates()})
        with (directory / "coordinator.log").open("w") as log:
            child = subprocess.Popen([sys.executable, "-u", __file__, "--run", str(directory)], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "directory": str(directory)}, indent=2))
    elif args.run:
        run(args.run)
    else:
        parser.error("Choose --check or --start")
