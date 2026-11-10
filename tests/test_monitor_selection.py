"""Monitor contracts across component discovery, producers, sinks and live readers."""
import csv
import json
from unittest.mock import Mock

import pytest

from src.app.telemetry.monitors import (
    MonitorSelection, available_monitors, monitor_errors, monitors_from_config, validate_monitoring_metadata,
)
from src.app.telemetry.run import Run
from src.app.telemetry.sinks import MemorySink


def config(*disabled):
    return {"telemetry": {"monitors": {"disabled": list(disabled)}}}


def records(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def test_live_filter_preserves_archival_metrics_and_objective_summary(tmp_path):
    memory = MemorySink()
    with Run(tmp_path, ["csv", "stream", memory], config=config("eval/reward", "blend/"), system_interval=0) as run:
        run.log({"eval/reward": 4, "blend/train/entropy": .5, "losses/policy": .1}, step=3, transitions=48)
        assert run.summary["eval/reward"] == 4
    streamed = records(tmp_path)
    row = next(rec for rec in streamed if rec["type"] == "metrics")
    assert row["data"] == {"losses/policy": .1, "transitions": 48}
    assert row["step"] == 3
    assert streamed[-1]["summary"] == {"losses/policy": .1, "transitions": 48}
    saved = list(csv.DictReader((tmp_path / "metrics.csv").open()))
    assert saved[0]["eval/reward"] == "4"
    assert memory.records[-1]["summary"]["blend/train/entropy"] == .5


def test_remove_all_still_delivers_lifecycle_errors_and_artifacts(tmp_path):
    with Run(tmp_path, ["stream"], config=config("*"), system_interval=2) as run:
        assert run.sampler is None
        run.log({"eval/reward": 1}, step=1, transitions=5)
        run.system({"rss_mb": 20})
        run.event("error", "A useful error", level="error")
        run.artifact("checkpoint.pt")
    streamed = records(tmp_path)
    assert [rec["type"] for rec in streamed] == ["start", "event", "artifact", "finish"]
    assert streamed[-1]["summary"] == {"transitions": 5}


def test_system_selection_filters_bare_keys_and_stops_unused_sampler(tmp_path):
    with Run(tmp_path / "off", ["csv", "stream"], config=config("system/")) as run:
        assert run.sampler is None
    with Run(tmp_path / "partial", ["stream"], config=config("system/rss_mb"), system_interval=0) as run:
        run.system({"rss_mb": 20, "cpu_percent": 30})
    sample = next(rec for rec in records(tmp_path / "partial") if rec["type"] == "system")
    assert sample["data"] == {"cpu_percent": 30}
    # A custom archival sink remains a consumer even when the live stream is off.
    with Run(tmp_path / "archive", [MemorySink()], config=config("system/")) as run:
        assert run.sampler is not None


def test_lazy_optional_monitors_skip_work_and_validate_output(tmp_path):
    collect = Mock(return_value={"interp/activation_norm": 2})
    with Run(tmp_path, ["stream"], config=config("interp/"), system_interval=0) as run:
        assert not run.monitor("interp/", collect)
        collect.assert_not_called()
    with Run(tmp_path / "on", ["stream"], system_interval=0) as run:
        assert run.monitor("interp/", collect)
        with pytest.raises(ValueError, match="outside"):
            run.monitor("interp/", lambda: {"losses/policy": 4})


@pytest.mark.parametrize("value", [[], "blend/", {"disabled": "blend/"}, {"disabled": [42]},
                                   {"disabled": [" "]}, {"disabled": ["blend/*"]}, {"typo": []}])
def test_invalid_settings_are_rejected(value):
    assert monitor_errors({"telemetry": {"monitors": value}})


def test_unset_and_empty_exclusions_keep_current_and_future_outputs():
    assert MonitorSelection().keeps("future/metric")
    assert monitor_errors(config()) == []
    assert MonitorSelection(["blend/train/"]).keeps("blend/eval/entropy")
    assert not MonitorSelection(["blend/train/"]).keeps("blend/train/neural_0/weight_std")


def test_removed_sets_filter_streams_and_box_bypass_preserves_them(tmp_path):
    cfg = {"telemetry": {"monitors": {"disabled": ["eval/reward"], "removed": ["train/"]}}}
    assert monitor_errors(cfg) == []
    with Run(tmp_path, ["stream"], config=cfg, system_interval=0) as run:
        run.log({"train/loss": 2., "eval/reward": 3., "other/metric": 4.}, step=1)
    row = next(rec for rec in records(tmp_path) if rec["type"] == "metrics")
    assert row["data"] == {"other/metric": 4.}
    cfg["telemetry"]["monitors"]["enabled"] = False
    assert monitors_from_config(cfg).keeps("train/loss")
    assert cfg["telemetry"]["monitors"]["removed"] == ["train/"]


@pytest.mark.parametrize("removed", ["train/", [42], ["train/*"], None])
def test_removed_sets_require_metric_patterns(removed):
    assert monitor_errors({"telemetry": {"monitors": {"removed": removed}}})


def test_disabled_box_ignores_saved_choices_without_mutating_them(tmp_path):
    cfg = {"telemetry": {"monitors": {"enabled": False, "disabled": ["blend/", "eval/reward"]}}}
    assert monitors_from_config(cfg).keeps("blend/train/entropy")
    with Run(tmp_path, ["stream"], config=cfg, system_interval=0) as run:
        run.log({"eval/reward": 3, "blend/train/entropy": .5}, step=7)
    row = next(rec for rec in records(tmp_path) if rec["type"] == "metrics")
    assert row["data"] == {"eval/reward": 3, "blend/train/entropy": .5}
    assert cfg["telemetry"]["monitors"]["disabled"] == ["blend/", "eval/reward"]
    cfg["telemetry"]["monitors"]["enabled"] = True
    assert not monitors_from_config(cfg).keeps("eval/reward")
    assert monitors_from_config({"telemetry": {"monitors": None}}).keeps("anything/new")


@pytest.mark.parametrize("enabled", ["false", 0, None, []])
def test_monitor_enabled_requires_boolean(enabled):
    assert monitor_errors({"telemetry": {"monitors": {"enabled": enabled}}})


def test_restoring_group_under_broad_exclusion_preserves_other_groups():
    known = ["train/a", "train/b", "eval/c"]
    selection = MonitorSelection(["*"])
    restored = selection.with_toggled(["train/a", "train/b"], True, known=known)
    assert restored.keeps("train/a") and restored.keeps("train/b")
    assert not restored.keeps("eval/c")
    assert selection.disabled == ("*",)
    removed = restored.with_toggled(["train/a", "train/b"], False, known=known)
    assert all(not removed.keeps(name) for name in known)


def test_default_stream_keeps_every_original_sample_without_catalog_work(tmp_path, monkeypatch):
    catalog = Mock(side_effect=AssertionError("enabled monitoring must not discover catalogs in the training loop"))
    monkeypatch.setattr("src.app.telemetry.run.available_monitors", catalog)
    filter_record = Mock(side_effect=AssertionError("default delivery must bypass monitor filtering"))
    monkeypatch.setattr(MonitorSelection, "filter_record", filter_record)
    logged = [(7, {"train/reward": 1.}), (13, {"train/reward": 2.}),
              (16, {"blend/train/entropy": .5}), (27, {"train/reward": 3.}),
              (32, {"losses/train/loss": .1}), (64, {"eval/reward": 4.})]
    with Run(tmp_path, ["csv", "stream"], system_interval=0) as run:
        for step, values in logged:
            run.log(values, step=step, transitions=step)
            assert run.wants_monitor("blend/train/")
    streamed = [(rec["step"], {key: value for key, value in rec["data"].items() if key != "transitions"})
                for rec in records(tmp_path) if rec["type"] == "metrics"]
    assert streamed == logged
    catalog.assert_not_called()
    filter_record.assert_not_called()


def test_csv_fallback_honors_monitor_settings(tmp_path):
    from src.app.services.metrics import AgentFeed
    import yaml

    agent = tmp_path / "method"
    version = agent / "version_0"
    version.mkdir(parents=True)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config("eval/")))
    (version / "metrics.csv").write_text("step,eval/reward,losses/policy\n1,4,0.1\n")
    feed = AgentFeed(agent)
    rows, _ = feed.read()
    assert rows[0]["data"] == {"losses/policy": .1}


def test_selected_components_are_intersected_with_actual_runner_hooks(tmp_path):
    from src.app.core.provider_catalog import Capability, Package, ProviderCatalog
    from src.app.workspace import WorkspacePaths

    catalog = ProviderCatalog.__new__(ProviderCatalog)
    catalog.paths = WorkspacePaths(tmp_path)
    catalog.overrides = {}

    def package(identity, kind, metadata):
        root = tmp_path / identity
        root.mkdir()
        # Discovery must never execute the component.
        (root / "component.py").write_text("raise AssertionError('discovery imported code')\n")
        return Package(identity, root, "user", "1.0.0", {"name": identity},
                       [Capability(identity, kind, "component.py:Provider", metadata=metadata)])

    runner = package("runner", "algorithm", {"monitoring": {
        "supports": ["scalars.v1", "model_observations.v1"],
        "provides": [{"name": "train/loss"}]}})
    model = package("model", "model", {"monitoring": {"provides": [
        {"name": "blend/", "requires": ["model_observations.v1"]},
        {"name": "interp/", "requires": ["activations.v1"]}]}})
    tuner = package("tuner", "tuner", {"monitoring": {"provides": [{"name": "tuning/score"}]}})
    unrelated = package("unrelated", "model", {"monitoring": {"provides": [{"name": "unrelated/"}]}})
    standalone = package("standalone", "monitor", {"requires": ["scalars.v1"],
        "monitoring": {"provides": [{"name": "standalone/score"}]}})
    catalog.packages = [runner, model, tuner, unrelated, standalone]
    script = str(runner.root / "component.py")
    profiles = {("agent", "runner"): {"entrypoint": script}, ("model", "model_profile"): {"name": "model"}}
    read = lambda kind, selector: profiles.get((kind, selector), {})
    cfg = {"methods": {"one": {"agent": "runner", "model": "model_profile"},
                       "off": {"enabled": False, "agent": "runner", "model": "unrelated"}},
           "tuning": {"component": "tuner"}}
    assert {spec.name for spec in available_monitors(cfg, catalog=catalog, read_config=read)} == {
        "train/loss", "blend/", "tuning/score"}
    cfg["monitors"] = {"optional": {"component": "standalone"}}
    assert "standalone/score" in {spec.name for spec in available_monitors(cfg, catalog=catalog, read_config=read)}
    cfg["monitors"]["broken"] = {"component": "missing-provider"}
    errors = []
    assert {spec.name for spec in available_monitors(cfg, catalog=catalog, read_config=read, errors=errors)} == {
        "train/loss", "blend/", "tuning/score", "standalone/score"}
    assert len(errors) == 1 and "broken" in errors[0]
    with pytest.raises(ValueError, match="missing-provider"):
        available_monitors(cfg, catalog=catalog, read_config=read)
    cfg["monitors"].pop("broken")
    errors = []
    available_monitors(cfg, catalog=catalog, read_config=read, errors=errors)
    assert errors == []
    cfg.pop("monitors")
    # Custom script wins over the selected algorithm profile; no hooks declared.
    cfg["methods"]["one"]["entrypoint"] = str(tmp_path / "custom.py")
    assert available_monitors(cfg, catalog=catalog, read_config=read) == []


@pytest.mark.parametrize("metadata", [{"provides": [{"name": "step"}]},
    {"provides": [{"name": "x/", "requires": "observations"}]}, {"supports": "scalar"},
    {"provides": [{"name": "x/"}, {"name": "x/"}]}])
def test_invalid_manifest_contracts_are_rejected(metadata):
    with pytest.raises(ValueError):
        validate_monitoring_metadata(metadata)


def test_lightning_bridge_applies_live_settings_without_filtering_csv(tmp_path):
    from omegaconf import OmegaConf
    from src.app.core.lightning_builder import setup_loggers
    from src.app.telemetry.lightning import finish_telemetry
    from types import SimpleNamespace

    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "agent": {"name": "m"}, **config("val/loss")})
    loggers = setup_loggers(cfg, str(tmp_path))
    for logger in loggers:
        logger.log_metrics({"val/loss": 1., "train/loss": 2.}, step=0)
        logger.save()
    finish_telemetry(SimpleNamespace(loggers=loggers))
    version = next((tmp_path / "results/logs/g/e/m").glob("version_*"))
    assert "val/loss" not in next(rec for rec in records(version) if rec["type"] == "metrics")["data"]
    assert "val/loss" in (version / "metrics.csv").read_text()


def test_config_catalog_enumerates_experts_without_importing_implementation(tmp_path):
    from src.app.core.provider_catalog import Capability, Package, ProviderCatalog
    from src.app.workspace import WorkspacePaths

    root = tmp_path / "provider"
    root.mkdir()
    (root / "__init__.py").write_text("raise AssertionError('imported package initializer')")
    (root / "runner.py").write_text("raise AssertionError('imported runner implementation')")
    (root / "catalog.py").write_text('''
def describe(config, context):
    return [{"name": f"custom/expert_{i}/weight", "group": "Experts"}
            for i in range(config["experts"])]
''')
    catalog = ProviderCatalog.__new__(ProviderCatalog)
    catalog.paths = WorkspacePaths(tmp_path)
    catalog.overrides = {}
    metadata = {"monitoring": {"supports": ["scalars.v1"], "catalog": "catalog.py:describe"}}
    catalog.packages = [Package("custom", root, "user", "1.0.0", {},
        [Capability("custom", "algorithm", "runner.py:main", metadata=metadata)])]
    cfg = {"agent": {"name": "launch_name", "algorithm": "custom", "experts": 3,
                      "entrypoint": str(root / "runner.py")}}
    specs = available_monitors(cfg, catalog=catalog, read_config=lambda *_: {})
    assert {spec.name for spec in specs} == {f"custom/expert_{i}/weight" for i in range(3)}
    assert all(spec.group == "Experts" for spec in specs)


def test_all_individual_diagnostics_removed_skip_optional_collection(tmp_path):
    from src.app.telemetry.monitors import system_monitors

    cfg = {"agent": {"name": "sb3-dev/ppo", "algorithm": "sb3-dev/ppo",
                      "entrypoint": "package://sb3-dev/sb3_runner.py"},
           "model": {"name": "blendrl-dev", "neural": {"name": "dnn"}, "symbolic": {"name": "nsfr"}}}
    specs = available_monitors(cfg)
    removed = [spec.name for spec in specs if spec.name.startswith("blend/train/")]
    assert "blend/train/logic_0/weight_mean" in removed
    assert all(not name.endswith("/") for name in removed)
    cfg.update(config(*removed))
    with Run(tmp_path / "model", ["stream"], config=cfg, system_interval=0) as run:
        collect = Mock(return_value={"blend/train/entropy": 2})
        assert not run.monitor("blend/train/", collect)
        collect.assert_not_called()
        assert run.wants_monitor("blend/eval/")
    disabled_system = [spec.name for spec in system_monitors()]
    with Run(tmp_path / "system", ["stream"], config=config(*disabled_system)) as run:
        assert run.sampler is None
