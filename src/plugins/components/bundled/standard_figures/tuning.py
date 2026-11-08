"""Backend-neutral tuning tables and objective history from trial summaries."""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from plot.base import BasePlotter


class TuningPlotter(BasePlotter):
    def __init__(self):
        super().__init__("tuning")

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        root = Path("results/checkpoints") / group / Path(exp_id).stem
        summaries = sorted(root.glob("*/tuning_summary.json")) + sorted(root.glob("*/sweeps/*/tuning_summary.json"))
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("schema_version") != 1:
                print(f"Skipping unsupported tuning summary: {summary_path}")
                continue
            trials = sorted(summary.get("trials", []), key=lambda item: item["id"])
            relative = summary_path.relative_to(root)
            method = relative.parts[0]
            subdir = cfg.get("output_subdir", "tuning")
            out = (output_dir / subdir if subdir else output_dir) / method
            if len(relative.parts) == 4:
                out = out / "studies" / relative.parts[2]
            out.mkdir(parents=True, exist_ok=True)
            parameters = sorted({name for trial in trials for name in trial.get("parameters", {})})
            fields = ["trial", "state", "value", "checkpoint"] + [f"param.{name}" for name in parameters]
            with (out / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for trial in trials:
                    row = {"trial": trial["id"], "state": trial.get("state"),
                           "value": trial.get("value"), "checkpoint": trial.get("checkpoint")}
                    row.update({f"param.{key}": value for key, value in trial.get("parameters", {}).items()})
                    writer.writerow(row)
            completed = [trial for trial in trials if trial.get("state") == "complete"
                         and isinstance(trial.get("value"), (int, float)) and math.isfinite(trial["value"])]
            direction = summary.get("direction")
            if completed and direction in ("minimize", "maximize"):
                best = []
                for trial in completed:
                    value = trial["value"]
                    best.append((min if direction == "minimize" else max)(best[-1], value) if best else value)
                fig, ax = plt.subplots(figsize=tuple(cfg.get("figsize", [8, 5])))
                ax.scatter([trial["id"] for trial in completed], [trial["value"] for trial in completed],
                           label="Trial objective", alpha=0.65)
                ax.step([trial["id"] for trial in completed], best, where="post", label="Best so far")
                ax.set(xlabel="Trial", ylabel=summary.get("objective_metric") or "Objective", title=method)
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(out / "objective.png", dpi=cfg.get("dpi", 150))
                plt.close(fig)
            traces = []
            for trial in trials:
                points = []
                for step, value in trial.get("intermediate_values", {}).items():
                    try:
                        step = int(step)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        points.append((step, value))
                if points:
                    traces.append((trial.get("state"), sorted(points)))
            if traces:
                fig, ax = plt.subplots(figsize=tuple(cfg.get("figsize", [8, 5])))
                labels = set()
                for state, points in traces:
                    kind = "Pruned" if state == "pruned" else "Other trials"
                    ax.plot([point[0] for point in points], [point[1] for point in points],
                            color="tab:orange" if state == "pruned" else "tab:blue", alpha=0.7,
                            marker=".", label=kind if kind not in labels else None)
                    labels.add(kind)
                ax.set(xlabel="Validation epoch (zero-based)", ylabel=summary.get("objective_metric") or "Objective",
                       title=method)
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(out / "intermediate.png", dpi=cfg.get("dpi", 150))
                plt.close(fig)
