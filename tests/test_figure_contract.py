"""The figure/vega-lite-v1 contract: compatibility findings, candidates and returned specs."""

import json

import pytest

from discussion_fixtures import build_discussion_workspace
from frontend.discussion import model
from frontend.discussion.model import FigureDoc
from src.app.core.plotting import figures
from src.app.core.plotting.figures import FIGURE_CONTRACT, build_figure_spec, spec_error
from src.app.core.provider_catalog import get_catalog
from src.app.core.requirements import figure_findings
from src.app.pipeline.compose import compose_experiment
from src.app.workspace import WorkspacePaths


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = build_discussion_workspace(tmp_path_factory.mktemp("discussion") / "workspace")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("THETA_WORKSPACE", str(root))
        yield WorkspacePaths(root)


@pytest.fixture(autouse=True)
def selected(paths, monkeypatch):
    monkeypatch.setenv("THETA_WORKSPACE", str(paths.root))


def plotter(paths, name):
    return get_catalog(paths).resolve("plotter", name)


def findings(paths, experiment):
    cfg = compose_experiment(experiment, workspace=paths).cfg
    return {(f.level, f.message) for f in figure_findings(cfg, plotter(paths, "fixture_curve"), paths=paths)}


def test_sdk_exports_the_contract():
    from src.app.api import SDK_VERSION, plotting
    from src.app.core.plotting import thinning
    assert SDK_VERSION == "3.2"
    assert plotting.FIGURE_CONTRACT == FIGURE_CONTRACT == "figure/vega-lite-v1"
    assert plotting.thin_xy is thinning.thin_xy


def test_config_rules_read_the_composed_experiment(paths):
    assert ("block", "Fixture curves need online runs.") in findings(paths, "figs/sweeps/offline")
    assert not any(level == "block" for level, _ in findings(paths, "figs/online"))


def test_kind_rules_are_satisfied_by_any_enabled_method(paths):
    assert findings(paths, "figs/online") == {("warn", "Fixture curves are tuned for DQN.")}
    assert findings(paths, "other/solo") == set()


def test_figure_findings_never_run_preflight_hooks(paths, monkeypatch):
    import src.app.core.requirements as requirements
    monkeypatch.setattr(requirements, "preflight_findings",
                        lambda *a, **k: pytest.fail("figure findings ran a preflight hook"))
    findings(paths, "figs/online")


def test_candidates_are_the_contract_providers_with_findings(paths):
    candidates = {c.selector: c for c in model.figure_candidates(paths, "figs/sweeps/offline")}
    assert {"convergence", "fixture_curve"} <= set(candidates)
    assert "fixture_table" not in candidates and "losses" not in candidates
    assert candidates["fixture_curve"].blocked
    assert not candidates["convergence"].blocked and not candidates["convergence"].findings
    online = {c.selector: c for c in model.figure_candidates(paths, "figs/online")}
    assert not online["fixture_curve"].blocked
    assert [f.message for f in online["fixture_curve"].warnings] == ["Fixture curves are tuned for DQN."]


@pytest.mark.parametrize("spec, live, message", [
    ({"params": [{"name": "color", "value": "red"}]}, ["color"], None),
    ({"params": []}, ["color"], "live parameter color"),
    ({}, ["color", "size"], "live parameters color, size"),
    ({"params": [{"name": "color", "value": "red", "bind": {"input": "color"}}]}, ["color"], "must not bind"),
    ([], [], "mapping"),
])
def test_spec_errors(spec, live, message):
    error = spec_error(spec, live)
    assert (error is None) if message is None else (message in error)


def test_a_live_parameter_missing_from_the_spec_is_an_error(paths, monkeypatch):
    doc = FigureDoc.load(paths.discussion / "figs/online/fixture_curve.yaml", paths=paths)
    package, capability = doc.capability()
    spec = build_figure_spec(package, capability, "figs/online", doc.values(), doc.live_params(), paths=paths)
    assert spec["params"] == [{"name": "color", "value": "#d62728"}]
    with pytest.raises(ValueError, match="live parameter size"):
        build_figure_spec(package, capability, "figs/online", doc.values(), ["color", "size"], paths=paths)


def test_convergence_returns_a_valid_spec_for_fixture_metrics(paths):
    path = model.create_figure(paths, "figs/online", "convergence")
    doc = FigureDoc.load(path, paths=paths)
    assert doc.live_params() == ["line_width", "opacity", "title"]
    doc.set_param("metric", "train/reward")
    doc.set_param("smoothing_window", 1)
    spec = build_figure_spec(*doc.capability(), "figs/online", doc.values(), doc.live_params(), paths=paths)
    json.dumps(spec, allow_nan=False)  # the view receives strict JSON
    assert spec["$schema"].endswith("/vega-lite/v6.json")
    assert {p["name"]: p["value"] for p in spec["params"]} == {"line_width": 2, "opacity": 1, "title": ""}
    rows = spec["data"]["values"]
    assert {row["method"] for row in rows} == {"a2c", "ppo"}
    assert [row["y"] for row in rows if row["method"] == "ppo"] == [1.0, 2.0, 4.0, 3.0, 5.0, 6.0]
    assert spec["encoding"]["color"]["scale"]["domain"] == ["a2c", "ppo"]

    doc.set_param("methods", ["ppo"])
    spec = build_figure_spec(*doc.capability(), "figs/online", doc.values(), doc.live_params(), paths=paths)
    assert {row["method"] for row in spec["data"]["values"]} == {"ppo"}

    doc.set_param("metric", "never/logged")
    with pytest.raises(ValueError, match="never/logged"):
        build_figure_spec(*doc.capability(), "figs/online", doc.values(), doc.live_params(), paths=paths)


def test_convergence_thins_long_series(paths):
    from src.app.core.plotting.thinning import MAX_DRAWN_POINTS
    metrics = paths.results / "logs/figs/online/ppo/version_0/metrics.csv"
    original = metrics.read_text()
    metrics.write_text("step,train/reward\n" + "".join(f"{i},{(i * 37) % 101}\n" for i in range(1, 6001)))
    try:
        doc = FigureDoc.load(model.create_figure(paths, "figs/online", "convergence"), paths=paths)
        doc.set_param("metric", "train/reward")
        doc.set_param("methods", ["ppo"])
        spec = build_figure_spec(*doc.capability(), "figs/online", doc.values(), doc.live_params(), paths=paths)
        assert 100 < len(spec["data"]["values"]) <= MAX_DRAWN_POINTS + 1
    finally:
        metrics.write_text(original)


def test_convergence_uses_the_selected_stamped_run_only(paths):
    target = "figs/online/20261116-130000"
    doc = FigureDoc.load(model.create_figure(paths, target, "convergence"), paths=paths)
    doc.set_param("metric", "train/reward")
    doc.set_param("smoothing_window", 1)
    spec = build_figure_spec(*doc.capability(), doc.experiment, doc.values(), doc.live_params(), paths=paths)
    assert [row["y"] for row in spec["data"]["values"]] == [201, 202]
