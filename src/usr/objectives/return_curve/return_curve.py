"""Normalized area under an evaluation-return curve, robust to poor seeds.

Each seed's return curve is mapped linearly so that ``floor`` scores 0 and
``ceiling`` scores 1 (clamped), then integrated over transitions 0..budget.
Unlike sustained_solve, returns may be negative, so tasks such as
MountainCar-v0 (-200 means never reaching the goal) are supported. Earlier
improvement and a higher final return both raise the area. The trial value is
(1 - tail_weight) * mean area + tail_weight * mean area of the worst
tail_fraction of seeds; a failed seed scores 0.
"""
import math
import statistics

from src.app.api.tuning import ObjectiveProgress, ObjectiveResult


def evaluate(results, params):
    budget = float(params['budget'] or results.config.get('total_timesteps', 0))
    if budget <= 0 or not results.runs:
        raise ValueError('return_curve requires a positive budget and at least one run')
    floor, ceiling = float(params['floor']), float(params['ceiling'])
    if ceiling <= floor:
        raise ValueError('return_curve requires ceiling above floor')
    normalize = lambda y: max(0., min(1., (y - floor) / (ceiling - floor)))
    records = []
    for run in results.runs:
        if run.status != 'completed':
            records.append(dict(seed=run.seed, area=0., final=None, completed=False))
            continue
        raw = run.curve(params['metric'])
        if len(raw) < 2 or raw[0][0] != 0 or raw[-1][0] < budget:
            raise ValueError(f'Seed {run.seed} requires {params["metric"]} evaluation from transition 0 through budget {budget:g}')
        area = 0.
        final = raw[0][1]
        for (left, y0), (right, y1) in zip(raw, raw[1:]):
            if left >= budget:
                break
            end = min(right, budget)
            y_end = y0 + (y1 - y0) * (end - left) / (right - left)
            # Normalization clamps, so integrate the clamped values at both ends.
            area += (end - left) * (normalize(y0) + normalize(y_end)) / 2 / budget
            final = y_end
        records.append(dict(seed=run.seed, area=area, final=final, completed=True))
    areas = [r['area'] for r in records]
    count = max(1, math.ceil(len(areas) * params['tail_fraction']))
    value = ((1 - params['tail_weight']) * statistics.mean(areas)
             + params['tail_weight'] * statistics.mean(sorted(areas)[:count]))
    return ObjectiveResult(value, dict(per_run=records, runs=len(records),
        completed=sum(r['completed'] for r in records), metric=params['metric'],
        budget=budget, floor=floor, ceiling=ceiling))


def intermediate(results, params):
    """The objective as if the budget ended at the latest evaluation every seed has reached."""
    progress = results.common_progress(params['metric'])
    if progress is None or progress[1] <= 0:
        return None
    step, horizon = progress
    budget = float(params['budget'] or results.config.get('total_timesteps', 0))
    horizon = min(horizon, budget) if budget > 0 else horizon
    result = evaluate(results.snapshot(), {**params, 'budget': horizon})
    return ObjectiveProgress(step, result.value, {'horizon': horizon, **{k: result.diagnostics[k] for k in ('per_run',)}})
