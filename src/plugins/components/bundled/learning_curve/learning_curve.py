"""Normalized learning-curve area, optionally emphasizing early learning and poor seeds."""
import math
import statistics

from src.app.api.tuning import ObjectiveResult


def evaluate(results, params):
    budget = float(params["curve"].get("budget") or results.config.get("total_timesteps") or 0)
    if budget <= 0:
        raise ValueError("Set curve.budget or total_timesteps to a positive transition budget")
    half_life = params["curve"]["half_life"]
    scale = params["curve"]["reward_scale"]
    window_start = budget * params["curve"].get("start_fraction", 0)
    scores = []
    final_values, window_minima = [], []
    for run in results.runs:
        if run.status != "completed":
            scores.append(float(params["failed_run_score"]))
            continue
        points = run.curve(params["metric"])
        if len(points) < 2 or points[0][0] != 0 or points[-1][0] < budget:
            raise ValueError(f"Seed {run.seed} requires evaluation from transition 0 through budget {budget:g}")
        area = mass = 0.0
        window_values = []
        for (left, y0), (right, y1) in zip(points, points[1:]):
            if left >= budget:
                break
            if right <= window_start:
                continue
            end = min(right, budget)
            slope = (y1 - y0) / (right - left)
            start = max(left, window_start)
            y0 += slope * (start - left)
            left = start
            duration = end - left
            window_values.extend([y0, y0 + slope * duration])
            if half_life:
                decay = math.log(2) / half_life
                z = decay * duration
                interval_mass = -math.expm1(-z) / decay
                # Integral of u*exp(-decay*u); series avoids cancellation near zero.
                moment = (duration**2 * (0.5 - z / 3 + z*z / 8) if z < 1e-4
                          else (-math.expm1(-z) - z * math.exp(-z)) / decay**2)
                weight = math.exp(-decay * (left - window_start))
                area += weight * (y0 * interval_mass + slope * moment) / scale
                mass += weight * interval_mass
            else:
                area += (y0 + y0 + slope * duration) * duration / (2 * scale)
                mass += duration
        scores.append(area / mass)
        final_values.append(window_values[-1])
        window_minima.append(min(window_values))
    aggregation = params["aggregation"]
    if aggregation["method"] == "lower_tail":
        count = max(1, math.ceil(len(scores) * aggregation["fraction"]))
        score = statistics.mean(sorted(scores)[:count])
    elif aggregation["method"] == "mean_lower_tail":
        count = max(1, math.ceil(len(scores) * aggregation["fraction"]))
        weight = aggregation.get("tail_weight", 0.25)
        score = ((1 - weight) * statistics.mean(scores)
                 + weight * statistics.mean(sorted(scores)[:count]))
    elif aggregation["method"] == "mean_minus_std":
        score = statistics.mean(scores) - aggregation["std_penalty"] * (statistics.stdev(scores) if len(scores) > 1 else 0)
    else:
        score = statistics.mean(scores)
    return ObjectiveResult(score, {"per_run_auc": scores, "completed": sum(r.status == "completed" for r in results.runs),
                                   "runs": len(results.runs), "budget": budget,
                                   "window_start": window_start, "completed_final_values": final_values,
                                   "completed_window_minima": window_minima})
