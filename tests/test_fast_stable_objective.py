from pathlib import Path

import pytest

from src.app.objectives.api import RunResult, TrialResults, load_objective


def sample(points, seed=1):
    return RunResult(seed, 'completed', events=[
        {'training_step': t, 'metric': 'eval/reward', 'episode_outcomes': [reward]}
        for t, reward in points])


def score(runs, **overrides):
    component = load_objective('fast_stable', Path(__file__).resolve().parents[1])
    params = component.parameters({'budget': 100, **overrides})
    return component.evaluate(TrialResults(runs, {}), params)


def test_speed_separates_equally_stable_ceiling_policies():
    fast = sample([(0, 0), (20, 500), (75, 500), (85, 500), (100, 500)])
    slow = sample([(0, 0), (60, 500), (75, 500), (85, 500), (100, 500)])
    assert score([fast]).value > score([slow]).value
    assert score([fast]).diagnostics['per_run_auc'] == pytest.approx([.9])
    assert score([fast]).diagnostics['per_run_sustained_from'] == [20]


def test_stability_has_priority_over_fast_temporary_success():
    stable = sample([(0, 0), (75, 500), (85, 500), (100, 500)])
    collapse = sample([(0, 500), (75, 500), (85, 500), (100, 100)])
    assert score([stable]).value > score([collapse]).value
    fast = sample([(0, 500), (75, 500), (85, 500), (100, 500)])
    assert score([stable] * 4).value > score([fast] * 3 + [collapse]).value


def test_failed_runs_are_not_discarded_and_tail_penalizes_bad_seeds():
    fast = sample([(0, 0), (20, 500), (75, 500), (85, 500), (100, 500)])
    failed = RunResult(2, 'failed', error='worker failed')
    result = score([fast, failed])
    assert result.diagnostics['stable_seeds'] == 1
    assert result.diagnostics['per_run_stable'] == [True, False]
    assert result.diagnostics['per_run_quality'] == pytest.approx([.9, 0])
    assert result.value < score([fast, fast]).value
    assert result.value < score([fast, failed], tail_weight=0).value


def test_near_optimal_threshold_is_explicit_and_sparse_evals_rejected():
    near = sample([(0, 0), (75, 495), (85, 495), (100, 495)])
    assert score([near]).diagnostics['stable_seeds'] == 1
    assert score([near], target_reward=500).diagnostics['stable_seeds'] == 0
    with pytest.raises(ValueError, match='evaluations'):
        score([sample([(0, 0), (100, 500)])])
    with pytest.raises(ValueError, match='through budget'):
        score([sample([(0, 0), (75, 500), (85, 500)])])


def test_overshoot_is_interpolated_and_window_has_a_fixed_budget():
    result = score([sample([(0, 0), (75, 375), (80, 400), (90, 450), (120, 600)])])
    assert result.diagnostics['per_run_final'] == pytest.approx([500])
    assert result.diagnostics['per_run_auc'] == pytest.approx([.5])
    assert result.diagnostics['per_run_late_min'] == pytest.approx([375])
    assert result.value == pytest.approx(.1875)
