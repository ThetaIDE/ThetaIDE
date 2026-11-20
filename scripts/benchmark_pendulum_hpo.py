"""Matched-budget local Pendulum benchmark for Theta's installed tuners."""
from __future__ import annotations

import argparse
import copy
import math
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml

APPLICATION = Path(__file__).resolve().parents[1]
FULL_BUDGET = 4096
SEARCH_CAP = 24576
POPULATION = 3
ROUNDS = 3
SEGMENT = 2048
TUNE = {
    "agent.learning_rate": {"type": "log_float", "low": 0.0001, "high": 0.003},
    "agent.ent_coef": {"type": "log_float", "low": 0.0001, "high": 0.01},
}
AGENT = {"name": "sb3/ppo", "n_steps": 256, "batch_size": 64, "n_epochs": 3}
MODEL = {"name": "basics/dnn", "sb3_native_policy": True}


def recipe(seed, evaluation_seed, *, budget, episodes, agent):
    return {
        "defaults": [{"override /env": "gym_classic_control/pendulum"}],
        "paradigm": "online_rl",
        "seed": seed,
        "total_timesteps": budget,
        "eval_episodes": episodes,
        "variability_evaluation_seed": evaluation_seed,
        "theta_final_evaluation": True,
        "trainer": {"accelerator": "cpu"},
        "no_plot": True,
        "methods": {"ppo": {"agent": dict(agent), "model": dict(MODEL)}},
    }


def worker_result(directory):
    matches = list(directory.rglob("execution.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one worker checkpoint record in {directory}, found {len(matches)}")
    record = json.loads(matches[0].read_text())
    if record["status"] != "completed":
        raise ValueError(f"Worker did not complete: {matches[0]}")
    return record["result"]


class Benchmark:
    def __init__(self, workspace, output, *, calibration, resume=False):
        self.workspace = workspace
        self.output = output
        self.group = f"HPO_Pendulum_{output.name}"
        self.config = workspace / "src/usr/config/experiment" / self.group
        self.results = workspace / "src/usr/results"
        self.config.mkdir(parents=True, exist_ok=resume)
        output.mkdir(parents=True, exist_ok=resume)
        (output / "recipes").mkdir(exist_ok=resume)
        self.calibration = calibration
        self.agent = {**AGENT, **(calibration["ppo_parameters"] if calibration else {})}
        self.full_budget = calibration["qualified_budget"] if calibration else FULL_BUDGET
        self.search_cap = self.full_budget * (SEARCH_CAP // FULL_BUDGET)
        quantum = self.agent["n_steps"]
        self.segment = math.ceil(self.full_budget / (ROUNDS * quantum)) * quantum if calibration else SEGMENT
        self.heldout_episodes = calibration["episodes"] if calibration else 10
        self.selection_episodes = 8 if calibration else 3
        if calibration:
            record = output / "calibration.json"
            if record.exists() and json.loads(record.read_text()) != calibration:
                raise ValueError("Calibration differs from the one saved for this benchmark")
            record.write_text(json.dumps(calibration, indent=2) + "\n")
        self.timings_path = output / "launch_seconds.json"
        self.timings = json.loads(self.timings_path.read_text()) if resume else {}
        self.measurement_path = output / "measurements.jsonl"
        self.measured = {
            (row["comparison"], row["block"], row["arm"])
            for row in map(json.loads, self.measurement_path.read_text().splitlines())
        } if resume and self.measurement_path.is_file() else set()
        self.env = {**os.environ, "THETA_WORKSPACE": str(workspace),
                    "THETA_MANAGED_THREADS": "1", "OMP_NUM_THREADS": "1"}

    def launch(self, name, cfg):
        path = self.config / f"{name}.yaml"
        source = "# @package _global_\n" + yaml.safe_dump(cfg, sort_keys=False)
        if name in self.timings:
            if path.read_text() != source:
                raise ValueError(f"Recipe changed since the completed launch {name}")
            return self.timings[name]
        if path.exists():
            raise ValueError(f"Unfinished launch {name} exists; use a new benchmark output directory")
        path.write_text(source)
        (self.output / "recipes" / path.name).write_text(source)
        with (self.output / f"{name}.log").open("w") as log:
            started = time.monotonic()
            result = subprocess.run([sys.executable, str(APPLICATION / "run_pipeline.py"),
                                     f"{self.group}/{name}"], env=self.env,
                                    stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.monotonic() - started
        if result.returncode:
            raise RuntimeError(f"{name} failed. See {log.name}")
        self.timings[name] = elapsed
        pending = self.timings_path.with_suffix(".tmp")
        pending.write_text(json.dumps(self.timings, indent=2))
        pending.replace(self.timings_path)
        print(f"{name}: {elapsed:.1f}s", flush=True)
        return elapsed

    def checkpoints(self, name):
        return self.results / "checkpoints" / self.group / name

    def study(self, name):
        summaries = list(self.checkpoints(name).rglob("tuning_summary.json"))
        if not summaries:
            raise ValueError(f"Missing study summary for {name}")
        raw = max(summaries, key=lambda path: path.stat().st_mtime_ns).read_text()
        (self.output / f"{name}.study.json").write_text(raw)
        return json.loads(raw)

    def measure(self, data):
        key = (data["comparison"], data["block"], data["arm"])
        if key in self.measured:
            return
        with self.measurement_path.open("a") as stream:
            stream.write(json.dumps(data, allow_nan=False) + "\n")
        self.measured.add(key)

    def search(self, block, arm, seed, eval_seed, optimizer_seed, *, min_fidelity=None, space=TUNE):
        name = f"block{block}_{arm}"
        cfg = recipe(seed, eval_seed, budget=self.full_budget, episodes=self.selection_episodes, agent=self.agent)
        cfg["variability"] = {"training_seeds": [seed], "evaluation_seed": eval_seed}
        cfg["methods"]["ppo"]["tune"] = copy.deepcopy(space)
        component = "dehb" if arm.startswith("dehb") else "optuna"
        options = (
            {"min_transitions": min_fidelity or self.full_budget // 4, "max_transitions": self.full_budget,
             "reduction_factor": 2, "search_cap": self.search_cap, "seed": optimizer_seed}
            if component == "dehb" else
            {"sampler": arm, "seed": optimizer_seed, "n_trials": self.search_cap // self.full_budget,
             "n_jobs": 1, "tpe_startup_trials": 2}
        )
        cfg["tuning"] = {"component": component, "metric": "eval/reward", "direction": "maximize",
                         "options": {component: options}}
        elapsed = self.launch(name, cfg)
        study = self.study(name)
        trials = study["trials"]
        if not trials or any(t["state"] != "complete" for t in trials):
            raise ValueError(f"Study {name} contains incomplete evaluations")
        winner = next(t for t in trials if t["id"] == study["best_trial_id"])
        from src.app.api.tuning import trial_worker_results
        from src.app.workspace import WorkspacePaths
        paths = WorkspacePaths(self.workspace)
        actual = []
        for trial in trials:
            records = trial_worker_results(paths, self.group, name, "ppo", trial["log_version"])
            worker = records.get(seed)
            if not worker or worker["status"] != "completed":
                raise ValueError(f"Missing completed seed record for {name} trial {trial['id']}")
            progress = worker["result"]["progress"]
            assert progress["unit"] == "transitions"
            actual.append(progress["completed"])
        if sum(actual) != self.search_cap:
            raise ValueError(f"Unequal cost for {name}: {sum(actual)}")
        data = {"comparison": "fixed_search", "block": block, "arm": arm, "experiment": name,
                "wall_seconds": elapsed, "selection_return": winner["value"], "parameters": winner["parameters"],
                "transitions": sum(actual), "evaluations": len(actual),
                "full_budget_evaluations": actual.count(self.full_budget)}
        self.measure(data)
        return data

    def retrain(self, name, parameters, seeds, eval_seed, budget, *, schedule=None):
        cfg = recipe(seeds[0], eval_seed, budget=budget, episodes=self.heldout_episodes, agent=self.agent)
        cfg["variability"] = {"training_seeds": seeds, "evaluation_seed": eval_seed}
        cfg["methods"]["ppo"]["agent"].update(
            {key.removeprefix("agent."): value for key, value in parameters.items()})
        if schedule:
            cfg["theta_dynamic"] = {"api_version": 1, "schedule": schedule}
        self.launch(name, cfg)
        directory = self.results / "executions" / self.group / name
        manifests = list(directory.glob("*/manifest.json"))
        if len(manifests) != 1:
            raise ValueError(f"Expected one held-out execution manifest for {name}")
        runs = json.loads(manifests[0].read_text())["runs"]
        scores = {}
        for run in runs:
            if run["status"] != "completed":
                raise ValueError(f"Held-out worker failed in {name}")
            worker = worker_result(self.workspace / run["checkpoints"])
            if worker["progress"]["completed"] != budget:
                raise ValueError(f"Unexpected held-out training cost in {name}")
            scores[str(run["training_seed"])] = worker["metrics"]["eval/reward"]
        return scores

    def population(self, block, seed, eval_seed, optimizer_seed):
        name = f"block{block}_pbt"
        cfg = recipe(seed, eval_seed, budget=self.segment, episodes=self.selection_episodes, agent=self.agent)
        cfg["methods"]["ppo"]["tune"] = copy.deepcopy(TUNE)
        cfg["tuning"] = {
            "component": "pbt", "metric": "eval/reward", "direction": "maximize",
            "options": {"pbt": {"population": POPULATION, "rounds": ROUNDS,
                                "segment_transitions": self.segment, "seed": optimizer_seed}},
        }
        elapsed = self.launch(name, cfg)
        pbt = self.study(name)
        cost = POPULATION * ROUNDS * self.segment
        if pbt.get("completed_rounds") != ROUNDS or pbt["actual_transitions"] != cost:
            raise ValueError(f"PBT did not complete an equal-budget population in block {block}")
        block_seed = pbt["evaluation_seed"]
        fixed_name = f"block{block}_fixed_population"
        fixed = recipe(seed, block_seed, budget=ROUNDS * self.segment, episodes=self.selection_episodes, agent=self.agent)
        fixed["methods"] = {}
        for t in pbt["trials"][:POPULATION]:
            member = t["member"]
            params = {k.removeprefix("agent."): v for k, v in t["parameters"].items()}
            fixed["methods"][f"member{member}"] = {
                "agent": {**self.agent, **params, "seed": pbt["training_seeds"][member]},
                "model": dict(MODEL),
            }
        fixed_elapsed = self.launch(fixed_name, fixed)
        members = [worker_result(self.checkpoints(fixed_name) / member) for member in fixed["methods"]]
        if sum(w["progress"]["completed"] for w in members) != cost:
            raise ValueError(f"Fixed population cost mismatch in block {block}")
        best_member = max(range(POPULATION), key=lambda i: members[i]["metrics"]["eval/reward"])
        fixed_result = members[best_member]
        fixed_path = next(a["path"] for a in fixed_result["artifacts"] if a.get("role") == "final")
        heldout_eval = eval_seed + 500_000
        rows = [
            {"arm": "pbt", "wall_seconds": elapsed, "selection_return": pbt["best_value"],
             "checkpoint": pbt["winner_inference_checkpoint"], "schedule": pbt["winner_schedule"]},
            {"arm": "fixed_population", "wall_seconds": fixed_elapsed,
             "selection_return": fixed_result["metrics"]["eval/reward"], "checkpoint": fixed_path,
             "parameters": pbt["trials"][best_member]["parameters"]},
        ]
        for row in rows:
            scores = evaluate_checkpoint(row["checkpoint"], heldout_eval,
                                         episodes=self.heldout_episodes if self.calibration else 20)
            row.update(comparison="population", block=block, transitions=cost,
                       checkpoint_returns=scores, heldout_checkpoint_return=statistics.mean(scores))
            self.measure(row)
        return rows


def evaluate_checkpoint(path, seed, *, episodes):
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch

    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: Monitor(gym.make("Pendulum-v1"))])
    try:
        env.seed(seed)
        rewards, _ = evaluate_policy(PPO.load(path, device="cpu"), env, n_eval_episodes=episodes,
                                     deterministic=True, return_episode_rewards=True)
        return [float(reward) for reward in rewards]
    finally:
        env.close()


def measure_controls(benchmark, block, training_seeds, evaluation_seed):
    from scripts.calibrate_pendulum_ppo import no_learning_controls

    controls = no_learning_controls(training_seeds, evaluation_seed, benchmark.heldout_episodes,
                                    benchmark.calibration["ppo_parameters"])
    for arm, returns in controls.items():
        benchmark.measure({"comparison": "quality_control", "block": block, "arm": arm,
                           "episode_returns": returns,
                           "heldout_return": statistics.mean(score for series in returns for score in series)})


def summarize(workspace, output):
    """Read the retained records and separate backend work from launch wall time."""
    rows = [json.loads(line) for line in (output / "measurements.jsonl").read_text().splitlines()]
    logs = workspace / "src/usr/results/logs" / f"HPO_Pendulum_{output.name}"
    summarized = {}
    for row in rows:
        if row["comparison"] in {"fixed_search", "population"}:
            name = row.get("experiment") or f"block{row['block']}_{row['arm']}"
            records = [json.loads(p.read_text()) for p in (logs / name).rglob("runtime.json")]
            workers = [record for record in records if "evaluation_protocol" in record]
            expected = row.get("evaluations", POPULATION * ROUNDS if row["arm"] == "pbt" else POPULATION)
            if len(workers) != expected:
                raise ValueError(f"Expected {expected} backend runtimes for {name}, found {len(workers)}")
            row["backend_seconds"] = sum(w["training_time_seconds"] for w in workers)
        comparison = summarized.setdefault(row["comparison"], {})
        samples = comparison.setdefault(row["arm"], [])
        samples.append(row)
    calibration_path = output / "calibration.json"
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text())
        controls = summarized["quality_control"]
        blocks = {row["block"] for row in controls["random_policy"]}
        floors = {block: max(row["heldout_return"] for arm in controls.values()
                             for row in arm if row["block"] == block) for block in blocks}
        for comparison in ("fixed_search_heldout", "population", "population_heldout"):
            for arm in summarized.get(comparison, {}).values():
                for row in arm:
                    reward = row.get("heldout_return", row.get("heldout_checkpoint_return"))
                    floor = floors[row["block"]]
                    improvement = reward - floor
                    weakest = min(row.get("seed_returns", {}).values(), default=reward)
                    row.update(no_learning_floor=floor, improvement_over_control=improvement,
                               quality_gate_passed=weakest - floor >= calibration["margin"])
    (output / "summary.json").write_text(json.dumps(summarized, indent=2, allow_nan=False) + "\n")
    return summarized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="Workspace with the benchmark's compatible tuner, environment and PPO packages")
    parser.add_argument("--output", type=Path, default=APPLICATION / "src/usr/results/analysis/pendulum_hpo")
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--calibration", type=Path,
                        help="Calibration JSON with a qualified PPO training budget")
    parser.add_argument("--resume", type=Path, help="Previous results directory from this workspace")
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error("--blocks must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.resume.resolve() if args.resume else args.output.resolve() / stamp
    calibration_path = args.calibration or (output / "calibration.json" if args.resume else None)
    calibration = json.loads(calibration_path.read_text()) if calibration_path and calibration_path.is_file() else None
    if not args.resume and not calibration:
        parser.error("a completed --calibration is required for new comparisons")
    if args.calibration and not calibration:
        parser.error(f"cannot read calibration {args.calibration}")
    if calibration:
        budget = calibration.get("qualified_budget")
        if not budget:
            parser.error("calibration did not clear the learning threshold")
        if budget % (4 * calibration["ppo_parameters"]["n_steps"]):
            parser.error("the qualified budget must support fourfold rollout-aligned fidelity")
    benchmark = Benchmark(args.workspace.resolve(), output, calibration=calibration, resume=bool(args.resume))
    print(f"Results: {benchmark.output}", flush=True)
    for block in range(1, args.blocks + 1):
        training_seed = 716_000 + block
        evaluation_seed = 927_000 + block
        optimizer_seed = 41 + block
        holdout = [812_000 + block * 10 + i for i in range(2)]
        holdout_eval = evaluation_seed + 500_000
        if calibration:
            training_seed += 1_000_000
            evaluation_seed += 1_000_000
            holdout = [seed + 1_000_000 for seed in holdout]
            holdout_eval += 1_000_000
            measure_controls(benchmark, block, holdout, holdout_eval)
        order = ["dehb", "tpe", "random"]
        if block % 2 == 0:
            order.reverse()
        searches = [
            benchmark.search(block, arm, training_seed, evaluation_seed, optimizer_seed) for arm in order
        ]
        for search in searches:
            name = f"{search['experiment']}_heldout"
            scores = benchmark.retrain(name, search["parameters"], holdout, holdout_eval, benchmark.full_budget)
            benchmark.measure({"comparison": "fixed_search_heldout", "block": block, "arm": search["arm"],
                               "seed_returns": scores, "heldout_return": statistics.mean(scores.values())})
        pbt, fixed = benchmark.population(block, training_seed, evaluation_seed, optimizer_seed)
        for row in (pbt, fixed):
            scores = benchmark.retrain(f"block{block}_{row['arm']}_heldout",
                                       row.get("parameters", {}), holdout, holdout_eval,
                                       ROUNDS * benchmark.segment, schedule=row.get("schedule"))
            benchmark.measure({"comparison": "population_heldout", "block": block, "arm": row["arm"],
                               "seed_returns": scores, "heldout_return": statistics.mean(scores.values())})
    summarize(benchmark.workspace, benchmark.output)


if __name__ == "__main__":
    if str(APPLICATION) not in sys.path:
        sys.path.insert(0, str(APPLICATION))
    main()
