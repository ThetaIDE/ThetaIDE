"""Prefer seeds that solve, then reward early sustained solving and few regressions.

Use sampled evaluation returns to measure the policy used during PPO training.
The training episode moving average is deliberately not part of the score.
"""
import math
import statistics

from src.app.objectives import ObjectiveResult


def evaluate(results, params):
    budget = float(params['budget'] or results.config.get('total_timesteps', 0))
    if budget <= 0 or not results.runs:
        raise ValueError('sustained_solve requires a positive budget and at least one run')
    scale, target = params['reward_scale'], params['target_reward']
    if target > scale:
        raise ValueError('target_reward cannot exceed reward_scale')
    window = budget * (1 - params['stability_fraction'])
    normalize = lambda y: max(0., min(1., y / scale))
    records = []
    for run in results.runs:
        record = dict(seed=run.seed, stable=False, auc=0., sustained_from=None,
                      sustained_speed=0., drawdown=0., late_min=None,
                      late_tail=0., final=None, quality=0.)
        if run.status != 'completed':
            records.append(record)
            continue
        raw = run.curve(params['metric'])
        if len(raw) < 2 or raw[0][0] != 0 or raw[-1][0] < budget:
            raise ValueError(f'Seed {run.seed} requires {params["metric"]} evaluation from transition 0 through budget {budget:g}')
        points = [raw[0]]
        for (left, y0), (right, y1) in zip(raw, raw[1:]):
            end = min(right, budget)
            points.append((end, y0 + (y1 - y0) * (end - left) / (right - left)))
            if right >= budget:
                break
        if params['max_eval_gap'] and any(b - a > params['max_eval_gap'] for (a, _), (b, _) in zip(points, points[1:])):
            raise ValueError(f'Seed {run.seed} has an evaluation gap larger than max_eval_gap')
        if sum(window <= t <= budget for t, _ in raw) < params['min_evaluations']:
            raise ValueError(f'Seed {run.seed} needs more evaluations in the stability window')
        late = [y for t, y in points if t >= window]
        for (a, y0), (b, y1) in zip(points, points[1:]):
            if a < window < b:
                late.insert(0, y0 + (y1 - y0) * (window - a) / (b - a))
        auc = drawdown = 0.
        peak = normalize(points[0][1])
        for (a, y0), (b, y1) in zip(points, points[1:]):
            x, y = normalize(y0), normalize(y1)
            auc += (b - a) * (x + y) / 2 / budget
            # Integral of running-maximum minus the piecewise-linear return.
            if y <= peak:
                gap = peak - (x + y) / 2
            else:
                gap = (peak - x) ** 2 / (2 * (y - x))
            drawdown += (b - a) * gap / budget
            peak = max(peak, y)
        sustained = next((t for i, (t, _) in enumerate(points)
                          if t <= window and len(points) - i >= params['min_evaluations']
                          and all(y >= target for _, y in points[i:])), None)
        speed = 0. if sustained is None else 1 - sustained / budget
        tail_count = max(1, math.ceil(len(late) * params['late_tail_fraction']))
        late_tail = statistics.mean(normalize(y) for y in sorted(late)[:tail_count])
        # Fixed, interpretable shape preference: persistent speed, area, late floor.
        quality = max(0., min(1., .5 * speed + .3 * auc + .2 * late_tail
                             - params['drawdown_weight'] * drawdown))
        record.update(stable=min(late) >= target, auc=auc, sustained_from=sustained,
                      sustained_speed=speed, drawdown=drawdown, late_min=min(late),
                      late_tail=late_tail, final=points[-1][1], quality=quality)
        records.append(record)
    qualities = [r['quality'] for r in records]
    count = max(1, math.ceil(len(records) * params['tail_fraction']))
    robust = ((1 - params['tail_weight']) * statistics.mean(qualities)
              + params['tail_weight'] * statistics.mean(sorted(qualities)[:count]))
    stable = sum(r['stable'] for r in records)
    # As with fast_stable, reliability has priority over curve-shape preferences.
    value = (stable + robust) / (len(records) + 1)
    return ObjectiveResult(value, dict(stable_seeds=stable, runs=len(records),
        stable_fraction=stable / len(records), per_run=records, quality_tiebreak=robust,
        metric=params['metric'], budget=budget, window_start=window, target_reward=target))
