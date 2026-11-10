"""Visual contracts, lazy artifacts, physical topology and live inspection."""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from src.app.core.provider_catalog import get_catalog
from src.app.monitoring.api import Context, ModelAccess, ProbeSet, load_snapshot, write_snapshot
from src.app.monitoring.runtime import MonitorManager
from src.app.monitoring.topology import NetworkStructure
from src.app.telemetry.monitors import available_monitors, validate_monitoring_metadata
from src.app.telemetry.run import Run
from src.app.telemetry.sinks import MemorySink


class SmallNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
        with torch.no_grad():
            self.network[0].weight.copy_(torch.eye(2))
            self.network[0].bias.zero_()
            self.network[2].weight.copy_(torch.tensor([[100., -2.], [-3., 4.]]))
            self.network[2].bias.copy_(torch.tensor([.5, -.25]))

    def get_q_values(self, obs):
        return self.network(obs)


def ctx():
    bank = ProbeSet(np.array([[-1, 2], [3, 1], [0, 1], [2, -1]], np.float32),
                    np.array([11, 22, 33, 44]), {"sampling": "fixed"})
    return Context("net", {"target": "network"}, ModelAccess(SmallNetwork().eval(), "get_q_values", "q_values"),
                   bank, 100, 10, 0, lineage="same-initialization")


def save(directory, context, result):
    from src.app.telemetry.monitors import _catalog_outputs
    package, cap = get_catalog().resolve("monitor", "network-structure")
    declaration = _catalog_outputs(package, cap.metadata["monitoring"]["catalog"], {"instance": "net"}, {})[0]
    result.visuals = {"network": {**declaration, "data": result.visuals["network"]}}
    return write_snapshot(directory, context, result, model_hash="model", provider={}, elapsed=0.)


def test_known_network_signs_biases_and_zero_contribution_despite_large_weight():
    context = ctx()
    result = NetworkStructure().collect(context)
    graph = result.visuals["network"]["graph"]
    last = graph["connections"][-1]
    assert last["orientation"] == "destination,source"
    weight = result.arrays[last["weights"]][0, 0]
    source = result.arrays[last["source_values"]][0, 0]
    assert weight == 100 and source == 0 and source * weight == 0
    assert result.arrays[last["weights"]][0, 1] == -2
    assert result.arrays[graph["layers"][-1]["bias"]].tolist() == [.5, -.25]
    assert graph["connections"][1]["kind"] == "operation"
    assert len(last["edges"]) == 4
    for i, connection in enumerate(graph["connections"]):
        if connection["kind"] == "linear":
            expected = result.arrays[connection["source_values"]] @ result.arrays[connection["weights"]].T
            expected += result.arrays[graph["layers"][i+1]["bias"]]
            np.testing.assert_allclose(expected, result.arrays[graph["layers"][i+1]["activation"]])
    assert all(not m._forward_hooks for m in context.access.model.modules())


def test_budget_coverage_topology_change_and_unsupported_architecture():
    context = ctx(); monitor = NetworkStructure(max_layers=1, max_units=1, max_edges=1)
    result = monitor.collect(context)
    graph = result.visuals["network"]["graph"]
    assert graph["coverage"]["edges_recorded"] == 1
    assert graph["coverage"]["edges_total"] == 10
    assert graph["coverage"]["layers_total"] == 3
    assert graph["layers"][1]["units"] == 1 and graph["layers"][1]["total_units"] == 2
    context.access.model.network[1] = nn.Tanh()
    with pytest.raises(ValueError, match="topology changed"):
        monitor.collect(context)
    context.access.model.network[1] = nn.Conv1d(2, 2, 1)
    with pytest.raises(ValueError, match="architecture adapter"):
        monitor.collect(context)
    context.spec["target"] = ""
    with pytest.raises(ValueError, match="Sequential target"):
        NetworkStructure().collect(context)


def test_discovery_rich_outputs_for_model_owned_and_standalone_instances():
    cfg = {"agent": "sb3-dev/dqn", "model": "basics/dnn", "monitors": {
        "network": {"component": "network-structure", "target": "network"},
        "activation": {"component": "activation-evolution", "target": "network"}}}
    outputs = available_monitors(cfg)
    assert any(s.kind == "matrix" and s.instance == "activation" for s in outputs)
    assert any(s.kind == "network_graph" and s.instance == "network" for s in outputs)
    assert any(s.kind == "scalar_series" for s in outputs)
    class Owned(SmallNetwork):
        def get_monitor_specs(self):
            return {"owned": {"component": "network-structure", "target": "network",
                              "probes": {"count": 4, "steps": 4}}}
    # Runtime and declarative defaults use exactly the same provider path.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        with Run(Path(directory), [], config={}, system_interval=0) as run:
            manager = MonitorManager({}, run, Owned(), "dqn", lambda: gym.make("MountainCar-v0"))
            manager.collect(10, 1, final=True)
            assert manager.entries[0]["outputs"] == {"network"}
            manager.close()


def test_visual_declaration_rejects_unknown_kind_and_missing_semantics():
    with pytest.raises(ValueError, match="kind"):
        validate_monitoring_metadata({"provides": [{"name": "x", "kind": "unknown"}]})
    with pytest.raises(ValueError, match="visual_schema"):
        validate_monitoring_metadata({"provides": [{"name": "x", "kind": "matrix"}]})
    validate_monitoring_metadata({"provides": [{"name": "legacy/scalar"}]})


def test_visual_only_selection_emits_references_without_tensor_payloads(tmp_path):
    cfg = {"telemetry": {"metrics": ["interp/net/network"]}, "monitors": {
        "net": {"component": "network-structure", "target": "network", "probes": {"count": 4, "steps": 4}}}}
    sink = MemorySink()
    with Run(tmp_path, [sink], config=cfg, system_interval=0) as run:
        manager = MonitorManager(cfg, run, SmallNetwork(), "dqn", lambda: gym.make("MountainCar-v0"))
        manager.collect(100, 10, final=True); manager.close()
    path = next((tmp_path / "monitors/net").glob("snapshot_*.json"))
    meta, arrays = load_snapshot(path)
    assert meta["visuals"]["network"]["data"]["graph"]["schema"] == "theta.network-graph/v1"
    assert arrays["input"].shape == (4, 2)
    records = sink.records
    artifact = next(r for r in records if r.get("kind") == "monitor_snapshot")
    assert artifact["data"]["visuals"][0]["kind"] == "network_graph"
    assert "data" not in artifact["data"]["visuals"][0]
    assert not any("graph" in (r.get("data") or {}) for r in records if r["type"] == "metrics")


def test_hidden_only_output_allocates_no_probes(tmp_path):
    cfg = {"agent": "sb3-dev/dqn", "model": "basics/dnn", "telemetry": {"monitors": {"disabled": ["interp/net/network"]}}, "monitors": {
        "net": {"component": "network-structure", "target": "network"}}}
    with Run(tmp_path, [], config=cfg, system_interval=0) as run:
        manager = MonitorManager(cfg, run, SmallNetwork(), "dqn", lambda: pytest.fail("Hidden output built probes"))
        assert not manager.entries or all(e["disabled"] for e in manager.entries)
        manager.close()
    assert not (tmp_path / "monitors").exists()


def reference(path, updates=10):
    declaration = json.loads(path.read_text())["visuals"]["network"]
    return {"run": "DQN", "instance": "net", "path": str(path), "updates": updates,
            "kind": "monitor_snapshot", "data": {"instance": "net", "optimizer_updates": updates,
             "transitions": updates*10, "visuals": [{k:v for k,v in declaration.items() if k != "data"}]}}


def test_native_view_follows_new_steps_preserves_history_and_checks_lineage(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from frontend.live_run import LiveRun
    from frontend.monitor.visual_view import VisualsView
    app = QApplication.instance() or QApplication([])
    context = ctx(); monitor = NetworkStructure()
    first = save(tmp_path, context, monitor.collect(context))
    live = LiveRun(); live.add([{"type": "artifact", **reference(first)}])
    view = VisualsView(); view.resize(1400, 800); view.show()
    view.show_lives({"DQN": live}); app.processEvents()
    assert view.step.count() == 1 and "optimizer step 10" in view.caption.text()
    assert view.canvas.scene().items()
    context.snapshot_index = 1; context.updates = 20
    second = save(tmp_path, context, monitor.collect(context))
    live.add([{"type": "artifact", **reference(second, 20)}]); view.show_lives({"DQN": live})
    assert view.step.currentData() == str(second)
    view.follow.setChecked(False); view.step.setCurrentIndex(0)
    view.state.setCurrentIndex(1); view.mode.setCurrentText("Local contributions")
    context.snapshot_index = 2; context.updates = 30
    third = save(tmp_path, context, monitor.collect(context))
    live.add([{"type": "artifact", **reference(third, 30)}]); view.show_lives({"DQN": live})
    assert view.step.currentData() == str(first) and view.state.currentData() == 0
    view.compare.setCurrentIndex(2)
    assert "unavailable" not in view.caption.text().lower()
    meta = json.loads(second.read_text()); meta["lineage"] = "different-initialization"; second.write_text(json.dumps(meta))
    view._draw()
    assert "lineage" in view.caption.text()
    # Completed runs reopen through the same references.
    reopened = VisualsView(); reopened.show_lives({"DQN": live})
    assert reopened.step.count() == 3
    view.close(); reopened.close()


def test_incomplete_snapshot_and_array_traversal_are_unavailable(tmp_path):
    from src.app.telemetry.visuals import completed_metadata
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"schema": "theta.monitor-snapshot/v1", "arrays": "missing.npz"}))
    with pytest.raises(ValueError, match="completed"):
        completed_metadata(path)
    path.write_text(json.dumps({"schema": "theta.monitor-snapshot/v1", "arrays": "../elsewhere.npz"}))
    with pytest.raises(ValueError, match="beside"):
        completed_metadata(path)


def test_layernorm_is_a_coupled_operation_with_saved_affine_parameters():
    context = ctx()
    context.access.model.network = nn.Sequential(context.access.model.network[0], nn.LayerNorm(2), nn.ReLU(), context.access.model.network[2])
    result = NetworkStructure().collect(context)
    graph = result.visuals['network']['graph']
    norm = graph['connections'][1]
    assert norm['kind'] == 'normalization' and norm['edges'] == []
    assert 'full layer' in norm['dependency']
    np.testing.assert_array_equal(result.arrays[graph['layers'][2]['scale']], [1, 1])
    np.testing.assert_array_equal(result.arrays[graph['layers'][2]['bias']], [0, 0])
    assert result.arrays['source_2'][0, 0] < 0
    assert result.arrays['activation_2'][0, 0] == 0


def test_scalar_only_selection_omits_heatmap_arrays_and_keeps_scalar(tmp_path):
    cfg = {'telemetry': {'metrics': ['interp/features/activation_norm']}, 'monitors': {
        'features': {'component': 'activation-evolution', 'target': 'network', 'probes': {'count': 4, 'steps': 4}}}}
    with Run(tmp_path, [], config=cfg, system_interval=0) as run:
        manager = MonitorManager(cfg, run, SmallNetwork(), 'dqn', lambda: gym.make('MountainCar-v0'))
        manager.collect(100, 10, final=True); manager.close()
    meta, arrays = load_snapshot(next((tmp_path/'monitors/features').glob('snapshot_*.json')))
    assert set(meta['metrics']) == {'activation_norm'}
    assert not meta['visuals'] and not arrays


def test_topology_identity_survives_weight_changes():
    context = ctx(); monitor = NetworkStructure()
    first = monitor.collect(context)
    with torch.no_grad():
        context.access.model.network[2].weight.add_(1)
    second = monitor.collect(context)
    a, b = first.visuals['network']['graph'], second.visuals['network']['graph']
    assert a['topology_sha256'] == b['topology_sha256']
    assert [l['id'] for l in a['layers']] == [l['id'] for l in b['layers']]
    assert not np.array_equal(first.arrays['weights_2'], second.arrays['weights_2'])


def test_network_provider_exports_and_installs_with_visual_catalog(tmp_path):
    from src.app.services.component_exporter import ComponentExporter, ComponentExportOptions
    from src.app.components.archive import extract, validate_native_package
    from src.app.components.installer import HubInstaller
    from src.app.components.models import HubComponent, ReleaseInfo
    from src.app.core.provider_catalog import file_hash, ProviderCatalog
    from src.app.workspace import WorkspacePaths
    result = ComponentExporter().export('network-structure', ComponentExportOptions(
        output_path=tmp_path/'network-structure.zip', version='0.1.0', generate_git_scaffold=False))
    assert result.success, result.error
    unpacked = extract(result.export_path, tmp_path/'extracted')
    manifest = json.loads((unpacked/'.theta_component.json').read_text())
    validate_native_package(manifest, unpacked)
    component = HubComponent.from_dict(manifest)
    component.releases = {component.version: ReleaseInfo(component.version, result.export_path.as_uri(), file_hash(result.export_path))}
    workspace = tmp_path/'workspace'
    HubInstaller(workspace, workspace/'.thetaide').install(component, install_dependencies=False)
    selected = ProviderCatalog(WorkspacePaths(workspace)).resolve('monitor', 'network-structure')
    assert selected and selected[0].root.is_relative_to(workspace)
    from src.app.telemetry.monitors import _catalog_outputs
    outputs = _catalog_outputs(selected[0], selected[1].metadata['monitoring']['catalog'], {'instance': 'installed'}, {})
    assert outputs[0]['kind'] == 'network_graph' and outputs[0]['name'] == 'interp/installed/network'


def test_legacy_scalar_namespace_catalog_keeps_dynamic_metrics(tmp_path, monkeypatch):
    from src.app.monitoring.api import Result
    catalog = get_catalog()
    _, capability = catalog.resolve('monitor', 'activation-evolution')
    monkeypatch.setitem(capability.metadata, 'monitoring', {'provides': [{'name': 'interp/legacy/'}]})
    class LegacyProvider:
        def collect(self, context):
            return Result(metrics={'dynamic': 2., 'other': 3.})
    monkeypatch.setattr('src.app.core.provider_loader.load_capability', lambda *args, **kwargs: LegacyProvider)
    cfg = {'telemetry': {'metrics': ['interp/legacy/dynamic']}, 'monitors': {
        'legacy': {'component': 'activation-evolution', 'target': 'network', 'probes': {'count': 4, 'steps': 4}}}}
    with Run(tmp_path, [], config=cfg, system_interval=0) as run:
        manager = MonitorManager(cfg, run, SmallNetwork(), 'dqn', lambda: gym.make('MountainCar-v0'), catalog=catalog)
        manager.collect(100, 10, final=True); manager.close()
    meta, _ = load_snapshot(next((tmp_path/'monitors/legacy').glob('snapshot_*.json')))
    assert meta['metrics'] == {'dynamic': 2.}
