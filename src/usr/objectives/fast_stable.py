"""Rank sustained near-optimal seeds first, then sample efficiency and stability.

This is an editable workspace objective, independent of the model/algorithm.
Scores are comparable between configurations evaluated with the same seed count.
"""
import math
import statistics

from src.app.objectives import ObjectiveResult


def evaluate(results, params):
    budget = float(params['budget'] or results.config.get('total_timesteps', 0))
    if budget <= 0 or not results.runs:
        raise ValueError('fast_stable requires a positive budget and at least one run')
    scale = params['reward_scale']
    target = params['target_reward']
    if target > scale:
        raise ValueError('target_reward cannot exceed reward_scale')
    window_start = budget * (1 - params['stability_fraction'])
    aucs, minima, stable, quality, times, finals = [], [], [], [], [], []
    for run in results.runs:
        if run.status != 'completed':
            aucs.append(0.)
            minima.append(None)
            stable.append(False)
            quality.append(0.)
            times.append(None)
            finals.append(None)
            continue
        raw = run.curve(params['metric'])
        if len(raw) < 2 or raw[0][0] != 0 or raw[-1][0] < budget:
            raise ValueError(f'Seed {run.seed} requires evaluation from transition 0 through budget {budget:g}')
        points = [raw[0]]
        for (left, y0), (right, y1) in zip(raw, raw[1:]):
            end = min(right, budget)
            points.append((end, y0 + (y1 - y0) * (end - left) / (right - left)))
            if right >= budget:
                break
        if sum(window_start <= t <= budget for t, _ in raw) < params['min_evaluations']:
            raise ValueError(f'Seed {run.seed} needs at least {params["min_evaluations"]} evaluations in the stability window')
        normalize = lambda value: max(0., min(1., value / scale))
        area = 0.
        late_values = []
        for (left, y0), (right, y1) in zip(points, points[1:]):
            area += (right - left) * (normalize(y0) + normalize(y1)) / 2
            if right >= window_start:
                start = max(left, window_start)
                late_values.extend([y0 + (y1 - y0) * (start - left) / (right - left), y1])
        auc = area / budget
        late_min = min(late_values)
        aucs.append(auc)
        minima.append(late_min)
        stable.append(late_min >= target)
        # Collapses reduce both the curve area and its late-performance factor.
        quality.append(auc * normalize(late_min))
        finals.append(points[-1][1])
        times.append(next((t for i, (t, _) in enumerate(points)
                           if len(points) - i >= params['min_evaluations']
                           and all(y >= target for _, y in points[i:])), None))
    count = max(1, math.ceil(len(quality) * params['tail_fraction']))
    tail = statistics.mean(sorted(quality)[:count])
    tiebreak = (1 - params['tail_weight']) * statistics.mean(quality) + params['tail_weight'] * tail
    # One extra stable seed outranks any possible speed tiebreak. Unlike a
    # pure late-window objective, equally reliable candidates still separate.
    score = (sum(stable) + tiebreak) / (len(stable) + 1)
    return ObjectiveResult(score, {
        'stable_seeds': sum(stable), 'runs': len(stable), 'stable_fraction': statistics.mean(stable),
        'per_run_stable': stable, 'per_run_auc': aucs, 'per_run_late_min': minima,
        'per_run_quality': quality, 'per_run_sustained_from': times, 'per_run_final': finals,
        'speed_stability_tiebreak': tiebreak, 'budget': budget, 'window_start': window_start,
        'target_reward': target, 'training_seeds': [run.seed for run in results.runs],
    })
