"""Paired comparison of two DEHB fidelity ladders and full-budget random search."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from benchmark_pendulum_hpo import APPLICATION, Benchmark, TUNE, summarize
from calibrate_pendulum_ppo import no_learning_controls

# Choose these bounds from the preceding calibration and exploratory block.
# Apply the same bounds to both DEHB ladders and random search on fresh seeds.
SPACE = {**TUNE, "agent.learning_rate": {**TUNE["agent.learning_rate"], "low": 0.0005}}
TARGET_RETURN = -500
ARMS = {"dehb_8k": 8192, "dehb_16k": 16384, "random": None}


def run_arm(workspace, output, calibration, block, arm, controls):
    bench = Benchmark(workspace, output / f"{output.name}_seed{block}_{arm}", calibration=calibration)
    tuning_seed = 2_216_000 + block
    tuning_eval = 2_427_000 + block
    holdout = [2_312_000 + block * 10 + i for i in range(2)]
    for control, episodes in controls.items():
        bench.measure({"comparison": "quality_control", "block": block, "arm": control,
                       "episode_returns": episodes,
                       "heldout_return": statistics.mean(value for sample in episodes for value in sample)})
    search = bench.search(block, arm, tuning_seed, tuning_eval, 42 + block,
                          min_fidelity=ARMS[arm], space=SPACE)
    scores = bench.retrain(f"{search['experiment']}_heldout", search["parameters"],
                           holdout, tuning_eval + 500_000, bench.full_budget)
    bench.measure({"comparison": "fixed_search_heldout", "block": block, "arm": arm,
                   "seed_returns": scores, "heldout_return": statistics.mean(scores.values())})
    data = summarize(workspace, bench.output)
    heldout = data["fixed_search_heldout"][arm][0]
    study = json.loads((bench.output / f"{search['experiment']}.study.json").read_text())
    spent, first_target = 0, None
    for trial in study["trials"]:
        budget = trial.get("budget", bench.full_budget)
        spent += budget
        if first_target is None and budget == bench.full_budget and trial["value"] >= TARGET_RETURN:
            first_target = spent
    assert spent == bench.search_cap
    return {"block": block, "arm": arm, "search_transitions": spent,
            "transitions_to_tuning_target": first_target,
            "heldout_return": heldout["heldout_return"], "seed_returns": heldout["seed_returns"],
            "quality_gate_passed": heldout["quality_gate_passed"], "results": str(bench.output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=APPLICATION / "src/usr/results/analysis/pendulum_hpo")
    parser.add_argument("--blocks", type=int, default=2, help="Number of new paired seed blocks")
    parser.add_argument("--extend", type=Path, help="Completed comparison directory to extend with fresh seeds")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("qualified_budget") != 32768:
        parser.error("this fidelity comparison requires a calibration qualified at 32,768 transitions")
    if min(args.blocks, args.jobs) < 1:
        parser.error("--blocks and --jobs must be positive")
    protocol = {"search_space": SPACE, "minimum_transitions": ARMS, "full_budget": 32768,
                "search_cap": 196608, "tuning_target": TARGET_RETURN}
    rows, first_block = [], 1
    comparable_wall = args.jobs == 1
    if args.extend:
        previous = json.loads((args.extend / "protocol.json").read_text())
        if any(previous.get(key) != value for key, value in protocol.items()):
            parser.error("completed comparison uses a different protocol")
        rows = json.loads((args.extend / "results.json").read_text())
        keys = {(row["block"], row["arm"]) for row in rows}
        if (not rows or len(rows) != len(keys) or keys != {
            (block, arm) for block in range(1, previous["blocks"] + 1) for arm in ARMS
        }):
            parser.error("completed comparison must contain all arms for every existing seed block")
        if any(json.loads((Path(row["results"]) / "calibration.json").read_text()) != calibration
               for row in rows):
            parser.error("completed comparison used a different PPO calibration")
        first_block = previous["blocks"] + 1
        comparable_wall &= previous["wall_time_comparable"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output.resolve() / f"fidelities_{stamp}"
    output.mkdir(parents=True)
    print(f"Results: {output}", flush=True)
    protocol.update(blocks=first_block - 1 + args.blocks, simultaneous_jobs=args.jobs,
                    wall_time_comparable=comparable_wall)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    panels = {}
    for block in range(first_block, first_block + args.blocks):
        holdout = [2_312_000 + block * 10 + i for i in range(2)]
        panels[block] = no_learning_controls(holdout, 2_927_000 + block, calibration["episodes"],
                                             calibration["ppo_parameters"])
    work = [(block, arm) for block in panels for arm in ARMS]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [pool.submit(run_arm, args.workspace.resolve(), output, calibration, block, arm, panels[block])
                 for block, arm in work]
        for task in as_completed(tasks):
            row = task.result()
            rows.append(row)
            ordered = sorted(rows, key=lambda result: (result["block"], result["arm"]))
            pending = output / "results.tmp"
            pending.write_text(json.dumps(ordered, indent=2) + "\n")
            pending.replace(output / "results.json")
            print(row, flush=True)


if __name__ == "__main__":
    import sys

    if str(APPLICATION) not in sys.path:
        sys.path.insert(0, str(APPLICATION))
    main()
