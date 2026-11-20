"""Analyze the predeclared, matched-seed MountainCar PPO transfer experiment.

Run with the execution manifest path. No training or parameter selection occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/theta-ppo-cross-mpl")

import numpy as np

from src.app.objectives.api import TrialResults
from src.app.objectives.evaluation import read_runs

OUT = ROOT / "src/usr/results/analysis/mountaincar_ppo_cross_test_20261119"
METHODS = ("DNN_PPO_DNN", "DNN_PPO_BlendRL", "BlendRL_PPO_DNN", "BlendRL_PPO_BlendRL")
PARAMS = dict(metric="eval/sampled_reward", budget=120000, floor=-200, ceiling=-100,
              tail_fraction=0.5, tail_weight=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--fresh-seeds", action="store_true",
                        help="Analyze a fresh matched launch in its own output directory")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    output = OUT
    if args.fresh_seeds:
        from uuid import UUID

        plan = manifest["plan"]
        draws = set(plan.get("drawn_seeds") or ())
        training, evaluation = plan["training_seeds"], plan["evaluation_seed"]
        if not {"variability.training_seeds", "variability.evaluation_seed"} <= draws:
            raise ValueError("A fresh cross-test must draw its training and evaluation seeds at launch")
        if len(set(training)) != len(protocol["training_seeds"]) or type(evaluation) is not int:
            raise ValueError("A fresh cross-test needs 12 distinct training seeds and an evaluation seed")
        protocol.update(training_seeds=training, evaluation_seed=evaluation)
        output = OUT / UUID(manifest["execution_id"]).hex
    seeds = protocol["training_seeds"]
    expected = {(method, seed) for method in METHODS for seed in seeds}
    actual = {(r["method"], r["training_seed"]) for r in manifest["runs"]}
    if actual != expected or len(manifest["runs"]) != len(expected):
        raise ValueError("Execution does not match predeclared 4 x 12 design")
    if any(r["status"] != "completed" for r in manifest["runs"]):
        raise ValueError("All 48 runs must finish successfully before analysis")
    if manifest["plan"]["evaluation_seed"] != protocol["evaluation_seed"]:
        raise ValueError("Evaluation seed differs from protocol")
    if set(manifest["plan"]["training_seeds"]) != set(seeds):
        raise ValueError("Training seeds differ from protocol")
    config = manifest["config"]
    if config["total_timesteps"] != 120000 or config["env"]["num_envs"] != 4:
        raise ValueError("Training budget or environment count differs from protocol")
    for method in METHODS:
        model, source_name = method.split("_PPO_")
        effective = config["methods"][method]
        for key, value in protocol["parameters"][source_name].items():
            section, name = key.split(".", 1)
            if section == "agent" and effective[section][name] != value:
                raise ValueError(f"Incorrect PPO coefficient: {method}, {key}")
        if model == "BlendRL":
            for key, value in protocol["parameters"]["BlendRL"].items():
                section, name = key.split(".", 1)
                if section == "model" and effective[section][name] != value:
                    raise ValueError(f"Incorrect fixed BlendRL coefficient: {method}, {key}")
    # Score using the original objective code, retained alongside the results.
    source = ROOT / "src/usr/objectives/return_curve/return_curve.py"
    saved_source = OUT / "return_curve_source.py"
    if not saved_source.exists():
        saved_source.write_bytes(source.read_bytes())
    if hashlib.sha256(saved_source.read_bytes()).hexdigest() != protocol["objective_source_sha256"]:
        raise ValueError("Objective source differs from predeclared snapshot")
    spec = importlib.util.spec_from_file_location("cross_test_return_curve", saved_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = read_runs(manifest, ROOT)
    by_method = {m: [] for m in METHODS}
    for record, run in zip(manifest["runs"], runs):
        by_method[record["method"]].append(run)
    scores, curves, rows, summaries = {}, {}, [], {}
    for method, panel in by_method.items():
        panel.sort(key=lambda run: seeds.index(run.seed))
        result = module.evaluate(TrialResults(panel, manifest["config"]), PARAMS)
        scores[method] = np.array([r["area"] for r in result.diagnostics["per_run"]])
        curves[method] = [run.curve(PARAMS["metric"]) for run in panel]
        greedy = []
        sampled = []
        for run, score in zip(panel, scores[method]):
            final = {}
            for metric in ("eval/sampled_reward", "eval/reward"):
                curve = run.curve(metric)
                if not curve or curve[-1][0] < PARAMS["budget"]:
                    raise ValueError(f"Missing full evaluation: {method}, {run.seed}, {metric}")
                final[metric] = float(np.interp(PARAMS["budget"], *np.array(curve).T))
            sampled.append(final["eval/sampled_reward"])
            greedy.append(final["eval/reward"])
            rows.append(dict(method=method, seed=run.seed, area=float(score),
                             final_sampled=sampled[-1], final_greedy=greedy[-1]))
        summaries[method] = dict(n=len(panel), mean_area=float(scores[method].mean()),
                                 sd_area=float(scores[method].std(ddof=1)),
                                 original_robust_objective=result.value,
                                 final_sampled_mean=float(np.mean(sampled)),
                                 final_greedy_mean=float(np.mean(greedy)))
    differences = {
        "DNN own PPO advantage": scores[METHODS[0]] - scores[METHODS[1]],
        "BlendRL own PPO advantage": scores[METHODS[3]] - scores[METHODS[2]],
    }
    differences["Model by PPO interaction"] = sum(differences.values())
    signs = np.array(list(itertools.product((-1, 1), repeat=len(seeds))))
    rng = np.random.default_rng(20261119)
    # Resample entire seed blocks, retaining pairing across all four conditions.
    indices = rng.integers(0, len(seeds), (50000, len(seeds)))
    tests = {}
    for label, delta in differences.items():
        mean = float(delta.mean())
        p = float(np.mean(np.abs((signs * delta).mean(axis=1)) >= abs(mean) - 1e-14))
        low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
        tests[label] = dict(mean_difference=mean, ci95=[float(low), float(high)],
                            p_exact_two_sided=p, per_seed_difference=delta.tolist())
    ordered = sorted(tests, key=lambda label: tests[label]["p_exact_two_sided"])
    adjusted = 0.
    for index, label in enumerate(ordered):
        adjusted = max(adjusted, min(1., (len(ordered)-index)*tests[label]["p_exact_two_sided"]))
        tests[label]["p_holm"] = adjusted
        tests[label]["significant_at_05"] = adjusted < 0.05
    results = dict(execution_id=manifest["execution_id"], manifest=str(args.manifest.resolve()),
                   protocol=protocol, source_sha256=hashlib.sha256(saved_source.read_bytes()).hexdigest(),
                   summaries=summaries, tests=tests, per_run=rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    with (output / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    grid = np.arange(0, 120001, 5000)
    for ax, model in zip(axes, ("DNN", "BlendRL")):
        for source, color in (("DNN", "#2878b5"), ("BlendRL", "#e87524")):
            method = f"{model}_PPO_{source}"
            values = np.array([np.interp(grid, *np.array(curve).T) for curve in curves[method]])
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(seeds))
            ax.plot(grid / 1000, mean, label=f"{source} PPO settings", color=color)
            ax.fill_between(grid / 1000, mean-sem, mean+sem, color=color, alpha=.18)
        ax.set(title=model, xlabel="Training transitions (thousands)", ylim=(-205, -90))
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Sampled evaluation return")
    fig.suptitle("MountainCar PPO cross-test · 12 held-out seeds · mean ± SEM", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps(dict(summaries=summaries, tests=tests), indent=2))


if __name__ == "__main__":
    main()
