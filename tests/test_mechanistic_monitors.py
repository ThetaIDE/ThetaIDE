"""Causal correctness, state isolation, lifecycle and native monitor distribution."""
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from src.app.core.provider_catalog import get_catalog
from src.app.core.provider_loader import load_module
from src.app.monitoring.api import Context, ModelAccess, ProbeSet, load_snapshot, preserve_rng, state_digest
from src.app.monitoring.config import instances, validate_context
from src.app.monitoring.measurements import linear_cka
from src.app.monitoring.runtime import MonitorManager
from src.app.monitoring.sb3 import PostUpdateHook
from src.app.telemetry.run import Run
from src.app.telemetry.sinks import MemorySink


class KnownCircuit(nn.Module):
    """Only unit 0 changes the action-0 versus action-1 score."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, 4, bias=False), nn.ReLU())
        with torch.no_grad():
            self.network[0].weight.copy_(torch.eye(4))

    def get_q_values(self, obs):
        features = self.network(obs)
        self.cached = features * 2  # exercise deepcopy of non-leaf caches
        return torch.stack((features[:, 0], torch.zeros_like(features[:, 0])), 1)

    def get_action_probs(self, obs):
        return self.get_q_values(obs).softmax(-1)


def probes():
    rng = np.random.default_rng(15)
    return ProbeSet(rng.normal(size=(32, 4)).astype(np.float32), np.arange(32), {"source": "known-circuit", "sampling": "fixed"})


def context(model=None):
    model = model or KnownCircuit()
    return Context("test", {"target": "network", "probes": {"seed": 12}},
                   ModelAccess(model, "get_q_values", "q_values"), probes(), 10, 2, 0)


def provider(package):
    return get_catalog().load("monitor", package)


def test_known_circuit_finds_causal_unit_and_patching_recovers_behavior():
    monitor = provider("causal-circuit")(max_units=4, top_k=1, random_controls=4)
    ctx = context()
    result = monitor.collect(ctx)
    selected = [node for node in result.details["graph"]["nodes"] if node.get("selected")]
    assert [(node["module"], node["unit"]) for node in selected] == [("1", 0)]
    assert result.metrics["heldout_ablation_effect"] > 0
    assert result.metrics["complement_ablation_effect"] == pytest.approx(0)
    assert result.metrics["heldout_patch_recovery"] == pytest.approx(1)
    assert result.arrays["donor_indices"][:16].max() < 16
    assert result.arrays["donor_indices"][16:].min() >= 16
    assert all(not module._forward_hooks for module in ctx.access.model.modules())


def test_cka_is_rotation_invariant_and_degenerate_reference_is_explicit():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(32, 8))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert linear_cka(x, x @ rotation) == pytest.approx(1)
    assert linear_cka(np.zeros_like(x), x) is None


def test_qualified_algorithm_identity_preserves_q_value_semantics():
    from src.app.monitoring.api import model_access
    access = model_access(KnownCircuit(), "SB3-DEV/DQN")
    assert access.score_kind == "q_values"
    assert access.score_method == "get_q_values"


def test_default_adapter_requires_explicit_handling_of_existing_architecture_hooks():
    model = KnownCircuit()
    handle = model.network[0].register_forward_pre_hook(lambda *_: None)
    try:
        with pytest.raises(ValueError, match="architecture-specific"):
            context(model).access.fork()
        assert len(model.network[0]._forward_pre_hooks) == 1
    finally:
        handle.remove()


def test_architecture_owned_monitor_rejects_unrelated_model_before_loading_code():
    catalog = get_catalog()
    cfg = {"agent": "sb3-dev/ppo", "model": "basics/dnn", "monitors": {
        "wrong": {"component": "blendrl-dev/reliance"}}}
    with pytest.raises(ValueError, match="blendrl.expert_interventions"):
        validate_context(cfg, catalog=catalog)


def test_activation_state_roundtrip_and_reference_change():
    monitor_class = provider("activation-evolution")
    monitor = monitor_class(max_features=4)
    ctx = context()
    first = monitor.collect(ctx)
    assert "cka_initial" not in first.metrics
    saved = copy.deepcopy(monitor.state_dict())
    restored = monitor_class(max_features=4)
    restored.load_state_dict(saved)
    second = restored.collect(ctx)
    assert second.metrics["cka_initial"] == pytest.approx(1)
    assert second.metrics["cka_previous"] == pytest.approx(1)
    assert first.details["layers"][0]["module"] == "1"


def monitor_cfg(component="activation-evolution", **overrides):
    return {"monitors": {"diagnostic": {"component": component, "target": "network",
            "interval_updates": 2, "max_snapshots": 2, "probes": {"count": 16, "steps": 32, "seed": 19}, **overrides}}}


def test_runtime_isolates_rng_modes_buffers_gradients_and_nonleaf_caches(tmp_path):
    model = KnownCircuit()
    model.get_q_values(torch.ones(4, 4)).sum().backward()
    before = state_digest(model)
    gradients = [p.grad.clone() for p in model.parameters()]
    random.seed(71); np.random.seed(71); torch.manual_seed(71)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with Run(tmp_path, [MemorySink()], config=monitor_cfg(), system_interval=0) as run:
        manager = MonitorManager(monitor_cfg(), run, model, "dqn", lambda: gym.make("CartPole-v1"))
        manager.collect(10, 1)
        manager.collect(20, 2)  # not due
        manager.collect(30, 3, final=True)
        manager.collect(50, 5)  # cap reached
        state = manager.state_dict()
        manager.load_state_dict(state)
        manager.close(); manager.close()
    assert model.training
    assert state_digest(model) == before
    for parameter, gradient in zip(model.parameters(), gradients):
        assert torch.equal(parameter.grad, gradient)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    snapshots = sorted((tmp_path / "monitors/diagnostic").glob("snapshot_*.json"))
    assert len(snapshots) == 2
    metadata, arrays = load_snapshot(snapshots[-1])
    assert metadata["optimizer_updates"] == 3
    assert metadata["model_state_sha256"] == before
    assert metadata["probe_set_sha256"] == state["diagnostic"]["probe_set"]
    assert arrays["layer_0"].shape == (16, 4)
    assert (snapshots[-1].parent / "index.html").is_file()


@pytest.mark.parametrize("cfg", [{}, monitor_cfg(enabled=False), {**monitor_cfg(), "telemetry": {"monitors": {"disabled": ["interp/"]}}}])
def test_disabled_monitors_allocate_no_probes_or_provider(tmp_path, cfg):
    with Run(tmp_path, [], config=cfg, system_interval=0) as run:
        manager = MonitorManager(cfg, run, KnownCircuit(), "dqn", lambda: pytest.fail("disabled monitor built probes"))
        assert manager.entries == []
        manager.close()
    assert not (tmp_path / "monitors").exists()


def test_whole_layer_patching_and_state_artifact_roundtrip(tmp_path):
    result = provider("causal-circuit")(granularity="layer", top_k=1).collect(context())
    assert result.metrics["heldout_patch_recovery"] == pytest.approx(1)
    assert result.details["granularity"] == "layer"
    assert result.details["graph"]["nodes"][0]["unit"] is None
    from src.app.monitoring.api import save_monitor_state, load_monitor_state
    state = {"one": {"reference": probes().observations, "count": 3}}
    save_monitor_state(tmp_path / "state.npz", state)
    loaded = load_monitor_state(tmp_path / "state.npz")
    np.testing.assert_array_equal(loaded["one"]["reference"], state["one"]["reference"])
    assert loaded["one"]["count"] == 3


def test_known_sequential_path_has_downstream_mediation_and_rescue():
    model = KnownCircuit()
    model.network.extend([nn.Linear(4, 4, bias=False), nn.ReLU()])
    with torch.no_grad():
        model.network[2].weight.copy_(torch.eye(4))
    result = provider("causal-circuit")(max_units=8, top_k=2, max_edges=2).collect(context(model))
    edges = [edge for edge in result.details["graph"]["edges"] if edge["kind"] == "measured_path_mediation"]
    assert len(edges) == 1
    assert edges[0]["source"] == "1:0"
    assert edges[0]["target"] == "3:0"
    assert edges[0]["effect"] > 0
    assert edges[0]["rescue_effect"] == pytest.approx(edges[0]["effect"])


def test_failure_policy_records_error_and_disables_collector(tmp_path):
    cfg = monitor_cfg(target="missing", failure_policy="disable")
    memory = MemorySink()
    with Run(tmp_path, [memory], config=cfg, system_interval=0) as run:
        manager = MonitorManager(cfg, run, KnownCircuit(), "dqn", lambda: gym.make("CartPole-v1"))
        manager.collect(10, 1); manager.close()
    assert any(row.get("kind") == "monitor_error" for row in memory.records)
    assert not list(tmp_path.rglob("snapshot_*.json"))


def test_model_owned_specs_and_explicit_overrides_share_lifecycle(tmp_path):
    model = KnownCircuit()
    model.get_monitor_specs = lambda: {"owned": {"component": "activation-evolution", "target": "network",
        "probes": {"count": 8, "steps": 8}, "max_snapshots": 1}}
    cfg = {"monitors": {"owned": {"enabled": False}, "external": {"component": "activation-evolution", "target": "network",
           "probes": {"count": 8, "steps": 8}, "max_snapshots": 1}}}
    with Run(tmp_path, [], config=cfg, system_interval=0) as run:
        manager = MonitorManager(cfg, run, model, "dqn", lambda: gym.make("CartPole-v1"))
        assert [entry["identity"] for entry in manager.entries] == ["external"]
        manager.collect(10, 1); manager.close()


def test_post_update_boundary_counts_optimizer_steps_and_restores_wrapper():
    model = KnownCircuit()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    class Algorithm:
        policy = SimpleNamespace(optimizer=optimizer)
        num_timesteps = 8
        def train(self):
            optimizer.zero_grad()
            model.get_q_values(torch.ones(4, 4)).sum().backward()
            optimizer.step(); optimizer.step()
    calls = []
    algorithm = Algorithm()
    hook = PostUpdateHook(algorithm, SimpleNamespace(collect=lambda *args: calls.append(args)))
    algorithm.train()
    assert calls == [(8, 2)]
    hook.close();hook.close()
    assert "train" not in algorithm.__dict__
    assert not optimizer._optimizer_step_post_hooks


def test_preflight_rejects_unsupported_runner_and_invalid_params():
    catalog = get_catalog()
    cfg = {**monitor_cfg(), "agent": {"name": "sb3-dev/sac", "entrypoint": "package://sb3-dev/sb3_runner.py"}}
    with pytest.raises(ValueError, match="lacks required hooks"):
        validate_context(cfg, catalog=catalog)
    cfg["agent"]["name"] = "sb3-dev/dqn"
    cfg["monitors"]["diagnostic"]["params"] = {"max_features": -2}
    with pytest.raises(ValueError, match="Monitor diagnostic"):
        validate_context(cfg, catalog=catalog)


def test_probe_artifact_roundtrip_reopens_identity(tmp_path):
    bank = probes()
    bank.save(tmp_path / "probes.npz")
    loaded = ProbeSet.load(tmp_path / "probes.npz")
    assert loaded.identity == bank.identity


def test_reference_experiment_discovers_both_monitor_components():
    import yaml
    from src.app.telemetry.monitors import available_monitors
    cfg = yaml.safe_load((Path(__file__).parent / "fixtures/workspace_config/experiment/newMonitors/onlineRL.yaml").read_text())
    names = {spec.name for spec in available_monitors(cfg)}
    assert "interp/activations/cka_initial" in names
    assert "interp/circuit/heldout_mediation_effect" in names


def test_monitor_native_export_and_install(tmp_path):
    from src.app.services.component_exporter import ComponentExporter, ComponentExportOptions
    from src.app.components.archive import extract, validate_native_package
    result = ComponentExporter().export("activation-evolution", ComponentExportOptions(
        output_path=tmp_path / "activation-evolution-0.1.0.zip", version="0.1.0", generate_git_scaffold=False))
    assert result.success, result.error
    stage = extract(result.export_path, tmp_path / "extracted")
    manifest = json.loads((stage / ".theta_component.json").read_text())
    validate_native_package(manifest, stage)
    assert manifest["kind"] == "monitor"
    from src.app.components.installer import HubInstaller
    from src.app.components.models import HubComponent, ReleaseInfo
    from src.app.core.provider_catalog import file_hash, ProviderCatalog
    from src.app.workspace import WorkspacePaths
    component = HubComponent.from_dict(manifest)
    component.releases = {component.version: ReleaseInfo(component.version, result.export_path.as_uri(), file_hash(result.export_path))}
    isolated = tmp_path / "workspace"
    installer = HubInstaller(isolated, isolated / ".thetaide")
    installer.install(component, install_dependencies=False)
    installed = ProviderCatalog(WorkspacePaths(isolated)).resolve("monitor", "activation-evolution")
    assert installed is not None
    assert installed[0].root.is_relative_to(isolated)


def test_monitor_scaffold_has_native_capability_and_exact_output_catalog(tmp_path):
    from src.app.services.plugin_scaffolder import PluginScaffolder, ScaffoldOptions
    from src.app.components.archive import validate_native_package
    result = PluginScaffolder(tmp_path).scaffold(ScaffoldOptions(name="custom_diagnostic", kind="monitor"))
    assert result.success, result.error
    meta = json.loads((result.target_dir / ".theta_component.json").read_text())
    validate_native_package(meta, result.target_dir)
    import runpy
    outputs = runpy.run_path(str(result.target_dir / "catalog.py"))["describe"]({"instance": "owned"}, {})
    assert outputs[0]["name"] == "interp/owned/samples"


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_real_sb3_training_is_identical_with_and_without_diagnostics(tmp_path, algorithm):
    module = load_module(get_catalog().package("sb3-dev"), "theta_sb3_policy.py")
    algorithm_class = module.ThetaPPO if algorithm == "ppo" else module.ThetaDQN
    policy = module.ThetaActorCriticPolicy if algorithm == "ppo" else module.ThetaDQNPolicy
    from src.app.monitoring.sb3 import monitored_learn
    kwargs = {"seed": 13, "device": "cpu", "policy_kwargs": {
        "theta_config": {"model": {"hidden_sizes": [8], "dueling": False}},
        "theta_model": "dnn", "env_name": "cartpole"}}
    kwargs.update({"n_steps": 8, "batch_size": 4, "n_epochs": 1} if algorithm == "ppo" else
                  {"learning_starts": 0, "buffer_size": 32, "batch_size": 4, "train_freq": 1, "gradient_steps": 1})
    results = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for enabled in (False, True):
            env = gym.make("CartPole-v1")
            try:
                agent = algorithm_class(policy, env, **kwargs)
                selected = (agent.policy.theta_model_module if algorithm == "ppo" else agent.policy.q_net.theta_model_module)
                cfg = monitor_cfg(enabled=enabled)
                with Run(tmp_path / str(enabled), [], config=cfg, system_interval=0) as run:
                    manager = MonitorManager(cfg, run, selected, algorithm, lambda: gym.make("CartPole-v1"))
                    monitored_learn(agent, manager, total_timesteps=16)
                    assert manager.closed
                    assert "train" not in agent.__dict__
                results.append(copy.deepcopy(agent.policy.state_dict()))
            finally:
                env.close()
    finally:
        torch.set_num_threads(previous_threads)
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        assert torch.equal(results[0][key], results[1][key]), key
    assert list((tmp_path / "True/monitors").rglob("snapshot_*.json"))
