import json

import pytest
import yaml
from omegaconf import OmegaConf

from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.seeds import (DRAWN_KEY, SEED_LIMIT, SPACING, draw_seeds, pin_study_seeds, recorded_seeds,
                                    record_study_seed_plan, resolve_launch_seeds, settle_worker_seed, study_seeds,
                                    validate_seed_ranges)
from src.app.workspace import WorkspacePaths


def test_draws_are_spaced_from_each_other_and_from_taken_seeds():
    taken = {SPACING * k for k in range(0, SEED_LIMIT // SPACING, 4)}  # half the seed range is too close
    drawn = draw_seeds(20, taken)
    assert len(set(drawn)) == 20
    assert all(0 <= seed < SEED_LIMIT - SPACING for seed in drawn)
    for seed in drawn:
        assert all(abs(seed - other) >= SPACING for other in (*taken, *(d for d in drawn if d != seed)))


def capabilities(control):
    return lambda cfg, methods: {name: {"evaluation_seed_control": control} for name in methods}


def test_launch_draws_every_open_seed_once_and_pins_it(monkeypatch):
    monkeypatch.setattr("src.app.pipeline.variability.method_capabilities", capabilities(True))
    cfg = OmegaConf.create({"seed": None, "variability": {"training_runs": 3}})
    overrides = resolve_launch_seeds(cfg, {"DNN": {}, "BlendRL": {}})
    training = list(cfg.variability.training_seeds)
    assert len(training) == 3 and cfg.seed not in training
    assert isinstance(cfg.variability.evaluation_seed, int)
    assert list(cfg[DRAWN_KEY]) == ["seed", "variability.training_seeds", "variability.evaluation_seed"]
    assert f"++variability.training_seeds={json.dumps(training)}" in overrides
    assert f"++seed={cfg.seed}" in overrides
    assert resolve_launch_seeds(cfg, {"DNN": {}}) == []  # settled seeds are never redrawn


def test_explicit_seeds_reproduce_and_unsupported_evaluation_stays_with_the_runner(monkeypatch):
    monkeypatch.setattr("src.app.pipeline.variability.method_capabilities", capabilities(False))
    cfg = OmegaConf.create({"seed": 7, "variability": {"training_seeds": [1, 2]}})
    assert resolve_launch_seeds(cfg, {"DNN": {}}) == []
    assert cfg.variability.get("evaluation_seed") is None


def write_manifest(root, name, plan, parent=None):
    path = root / "src/usr/results/executions/g" / name / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"plan": plan, "parent": parent}))


def test_recorded_seeds_cover_manifests_and_launch_configs(tmp_path):
    write_manifest(tmp_path, "e/a", {"training_seeds": [11, 12], "evaluation_seed": 13})
    config = tmp_path / "src/usr/results/logs/g/e/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({"seed": 14, "variability": {"training_seeds": [15], "evaluation_seed": 16}}))
    assert recorded_seeds(WorkspacePaths(tmp_path)) == {11, 12, 13, 14, 15, 16}


def test_a_continued_study_keeps_its_recorded_seeds(tmp_path):
    paths = WorkspacePaths(tmp_path)
    write_manifest(tmp_path, "e/t0", {"training_seeds": [5, 6], "evaluation_seed": 9}, {"study": "e_DNN_v1"})
    write_manifest(tmp_path, "e/other", {"training_seeds": [1], "evaluation_seed": 2}, {"study": "e_DNN_v0"})
    assert study_seeds(paths, "g", {"e_DNN_v2"}) is None
    recorded = study_seeds(paths, "g", {"e_DNN_v1"})
    cfg = OmegaConf.create({"variability": {"training_runs": 2}})
    overrides = pin_study_seeds(cfg, recorded)
    assert list(cfg.variability.training_seeds) == [5, 6] and cfg.variability.evaluation_seed == 9
    assert overrides == ["++variability.training_seeds=[5, 6]", "++variability.evaluation_seed=9"]
    with pytest.raises(ConfigurationError, match="different seeds"):
        study_seeds(paths, "g", {"e_DNN_v1", "e_DNN_v0"})


def test_study_resume_rejects_an_explicit_seed_or_count_change():
    recorded = {"training_seeds": [5, 6], "evaluation_seed": 9}
    for panel, field in [
        ({"training_runs": 1}, "training_runs"),
        ({"training_seeds": [5, 7]}, "training_seeds"),
        ({"evaluation_seed": 10}, "evaluation_seed"),
    ]:
        with pytest.raises(ConfigurationError, match=field):
            pin_study_seeds(OmegaConf.create({"variability": panel}), recorded)


def test_resume_recovers_a_single_seed_from_worker_records(tmp_path):
    paths = WorkspacePaths(tmp_path)
    trial = paths.results / "checkpoints/g/e/m/trial/execution.json"
    trial.parent.mkdir(parents=True)
    trial.write_text(json.dumps({"effective_config": {
        "seed": 1234, "theta_tuning": {"study_name": "study"}, "variability_evaluation_seed": 9876,
    }}))
    recorded = study_seeds(paths, "g", {"study"})
    cfg = OmegaConf.create({"seed": None})
    assert pin_study_seeds(cfg, recorded) == ["++seed=1234", "++variability_evaluation_seed=9876"]
    assert resolve_launch_seeds(cfg, {}, resuming=True) == []
    with pytest.raises(ConfigurationError, match="seed differs"):
        pin_study_seeds(OmegaConf.create({"seed": 1235}), recorded)


def test_resume_preserves_runner_default_evaluation_policy(monkeypatch):
    monkeypatch.setattr("src.app.pipeline.variability.method_capabilities", capabilities(True))
    cfg = OmegaConf.create({"seed": 1, "variability": {"training_runs": 1}})
    pin_study_seeds(cfg, {"training_seeds": [5], "evaluation_seed": None})
    assert resolve_launch_seeds(cfg, {"DNN": {}}, resuming=True) == []
    assert cfg.variability.get("evaluation_seed") is None


def test_queued_study_can_resume_with_a_recorded_seed_plan(tmp_path):
    paths = WorkspacePaths(tmp_path)
    record_study_seed_plan(paths, "g", "queued", [5, 6], 10)
    assert study_seeds(paths, "g", {"queued"}, require_records=True) == {
        "training_seeds": [5, 6], "evaluation_seed": 10,
    }
    with pytest.raises(ConfigurationError, match="original seeds.*started"):
        study_seeds(paths, "g", {"queued", "started"}, require_records=True)


def test_auto_draws_avoid_explicit_pins_and_saved_results(monkeypatch):
    monkeypatch.setattr("src.app.pipeline.variability.method_capabilities", capabilities(True))
    candidates = iter([200, 400_000, 900_000])
    monkeypatch.setattr("src.app.pipeline.seeds.secrets.randbelow", lambda _: next(candidates))
    cfg = OmegaConf.create({"seed": 1, "variability": {"training_seeds": [100]}})
    resolve_launch_seeds(cfg, {"DNN": {}}, taken={400_010})
    assert cfg.variability.evaluation_seed == 900_000


def test_online_rl_seed_ranges_reject_overlapping_vector_worker_ids():
    cfg = OmegaConf.create({"env": {"num_envs": 4}})
    plan = {"training_seeds": [100, 103], "evaluation_seed": 200}
    with pytest.raises(ConfigurationError, match="Training seed ranges overlap"):
        validate_seed_ranges(cfg, {}, plan)
    plan.update(training_seeds=[100, 104], evaluation_seed=106)
    with pytest.raises(ConfigurationError, match="evaluation seed overlaps"):
        validate_seed_ranges(cfg, {}, plan)
    plan.update(evaluation_seed=150)
    validate_seed_ranges(cfg, {}, plan)


def test_a_worker_started_outside_a_launch_draws_and_records_its_seed(tmp_path):
    log = tmp_path / "src/usr/results/logs/g/e/m/version_0/config.yaml"
    log.parent.mkdir(parents=True)
    log.write_text("seed: 5000000\n")
    cfg = OmegaConf.create({"seed": None})
    settle_worker_seed(cfg, WorkspacePaths(tmp_path))
    assert abs(cfg.seed - 5000000) >= SPACING and cfg[DRAWN_KEY] == ["seed"]
    pinned = OmegaConf.create({"seed": 7})
    settle_worker_seed(pinned, WorkspacePaths(tmp_path))
    assert pinned.seed == 7 and DRAWN_KEY not in pinned
