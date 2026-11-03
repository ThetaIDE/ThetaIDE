"""Execute deterministic candidate evaluations for the external-tuner test."""

import json
import subprocess
import sys
from pathlib import Path


def main():
    payload = json.loads(next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--example-tuner=")))
    root = Path("results/checkpoints") / payload["group"] / payload["experiment"] / payload["method"]
    root.mkdir(parents=True, exist_ok=True)
    target, dimension = next(iter(payload["dimensions"].items()))
    trials = []
    for index, value in enumerate(dimension["values"][:payload["count"]]):
        checkpoint = root / f"candidate-{index}.ckpt"
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("trial.py")), str(value), str(checkpoint)],
                                capture_output=True, text=True, check=True)
        trials.append({"id": index, "state": "complete", "value": float(result.stdout.strip()),
                       "parameters": {target: value}, "checkpoint": str(checkpoint)})
    best = min(trials, key=lambda trial: trial["value"]) if payload["direction"] == "minimize" else max(trials, key=lambda trial: trial["value"])
    summary = {"schema_version": 1, "component": "example", "study": payload["study"],
               "objective_metric": payload["metric"], "direction": payload["direction"],
               "best_trial_id": best["id"], "trials": trials}
    (root / "tuning_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "best_params.yaml").write_text(f"{target}: {best['parameters'][target]}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
