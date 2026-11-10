from pathlib import Path

import pytest

from src.app.objectives.api import RunResult, TrialResults, load_objective


def run(points, metric='eval/sampled_reward'):
    return RunResult(1, 'completed', events=[dict(training_step=t, metric=metric,
                     episode_outcomes=[r]) for t, r in points])


def score(runs, **overrides):
    component = load_objective('sustained_solve', Path(__file__).resolve().parents[1])
    return component.evaluate(TrialResults(runs, {}), component.parameters(dict(budget=100, **overrides)))


def test_sampled_curve_prevents_untrained_greedy_ceiling_from_winning():
    points = [(0, 50), (25, 100), (50, 300), (75, 350), (85, 200), (100, 390)]
    mixed = run(points)
    mixed.events += run([(t, 500) for t, _ in points], 'eval/reward').events
    assert score([mixed]).diagnostics['stable_seeds'] == 0
    assert score([mixed], metric='eval/reward').diagnostics['stable_seeds'] == 1
    assert score([mixed]).value < score([mixed], metric='eval/reward').value


def test_early_sustained_success_beats_recovery_with_similar_area():
    sustained = run([(0, 0), (25, 500), (50, 500), (75, 500), (85, 500), (100, 500)])
    regression = run([(0, 500), (25, 500), (50, 250), (75, 500), (85, 500), (100, 500)])
    result = score([regression])
    assert score([sustained]).value > result.value
    assert result.diagnostics['per_run'][0]['drawdown'] == pytest.approx(.125)
    assert result.diagnostics['per_run'][0]['sustained_from'] == 75


def test_additional_stable_seed_has_priority_and_failures_remain():
    stable = run([(0, 0), (75, 500), (85, 500), (100, 500)])
    fast = run([(0, 500), (75, 500), (85, 500), (100, 500)])
    failed = RunResult(2, 'failed')
    assert score([stable] * 4).value > score([fast] * 3 + [failed]).value
    result = score([stable, failed])
    assert result.diagnostics['runs'] == 2
    assert result.diagnostics['per_run'][1]['quality'] == 0
    assert result.value < score([stable, stable]).value


def test_drawdown_integrates_recovery_crossing_previous_peak():
    result = score([run([(0, 250), (25, 0), (75, 500), (85, 500), (100, 500)])])
    assert result.diagnostics['per_run'][0]['drawdown'] == pytest.approx(.125)


def test_curve_coverage_cadence_and_missing_sampled_metric_fail():
    with pytest.raises(ValueError, match='through budget'):
        score([run([(0, 500), (75, 500), (85, 500)])])
    with pytest.raises(ValueError, match='evaluations'):
        score([run([(0, 500), (100, 500)])])
    with pytest.raises(ValueError, match='max_eval_gap'):
        score([run([(0, 500), (75, 500), (85, 500), (100, 500)])], max_eval_gap=25)
    with pytest.raises(ValueError, match='eval/sampled_reward'):
        score([run([(0, 500), (75, 500), (85, 500), (100, 500)], 'eval/reward')])


def test_budget_is_interpolated_and_scores_are_bounded():
    result = score([run([(0, 0), (75, 375), (80, 400), (90, 450), (120, 600)])])
    assert result.diagnostics['per_run'][0]['final'] == 500
    assert result.diagnostics['per_run'][0]['auc'] == pytest.approx(.5)
    assert 0 <= result.value <= 1
