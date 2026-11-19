import json

import pytest
import yaml
from omegaconf import OmegaConf

from src.app.pipeline.seeds import (DRAWN_KEY, SEED_LIMIT, SPACING, draw_seeds, pin_study_seeds, recorded_seeds,
                                    resolve_launch_seeds, study_seeds)
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
    config.write_text(yaml.safe_dump({"seed": 14}))
    assert recorded_seeds(WorkspacePaths(tmp_path)) == {11, 12, 13, 14}


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
    from src.app.pipeline.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError, match="different seeds"):
        study_seeds(paths, "g", {"e_DNN_v1", "e_DNN_v0"})
