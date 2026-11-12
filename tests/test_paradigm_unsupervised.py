"""Tests for the unsupervised learning paradigm.

Covers the three layers a paradigm spans: its YAML declaration, the components
the loader resolves from it, and the pipeline branches that used to be keyed on
paradigm names. The behaviour-preservation tests matter most — the capability
helpers replaced hardcoded tuples, so they must answer exactly as those tuples
did for the paradigms that already shipped.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import entry_points

import pytest
import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

from src.app.core.paradigm_loader import get_component, load_paradigm_definition
from src.app.pipeline.commands import build_method_overrides
from src.app.pipeline.validation import (
    list_paradigms,
    load_paradigm,
    paradigm_uses_agents,
    paradigm_uses_static_dataset,
)

# Plugin-specific checks run only when the component exports its training module.
from src.app.core.provider_catalog import get_catalog
AUTOENCODER_INSTALLED = get_catalog().resolve('component','autoencoder_training') is not None
needs_autoencoder = pytest.mark.skipif(
    not AUTOENCODER_INSTALLED,
    reason="the autoencoder model component is not installed",
)

PARADIGM = "unsupervised"


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_paradigm_is_registered():
    assert PARADIGM in list_paradigms()


def test_declares_no_agents():
    """An empty allow-list is what makes methods model-only, as in supervised."""
    assert load_paradigm(PARADIGM)["allowed_agents"] == []


def test_forbids_the_rl_agents():
    """An empty allow-list is what disallows RL agent harnesses in unsupervised learning."""
    defn = load_paradigm(PARADIGM)
    assert defn.get("allowed_agents") == []


def test_requires_methods():
    requires = load_paradigm(PARADIGM)["constraints"]["requires"]
    assert requires["methods"] == "non_empty"


def test_forbids_simulator_rollouts():
    assert load_paradigm(PARADIGM)["constraints"]["forbids"]["eval_episodes"] == "non_zero"


def test_does_not_opt_into_eval_episodes():
    """Simulator rollouts are opt-in; a paradigm reading a static dataset must not set them."""
    defn = load_paradigm(PARADIGM)
    assert defn.get("allows_eval_episodes", False) is False


def test_declares_only_registered_callbacks():
    """default_callbacks resolves through get_component, so unregistered names are a trap."""
    for name in load_paradigm(PARADIGM).get("default_callbacks") or []:
        assert get_component(name) is not None, f"{name} is declared but not registered"


# ---------------------------------------------------------------------------
# Component resolution
# ---------------------------------------------------------------------------


def test_loader_resolves_all_three_components():
    definition = load_paradigm_definition(PARADIGM)
    assert definition.runner_cls.__name__ == "UnsupervisedRunner"
    assert definition.data_module_cls.__name__ == "UnsupervisedDataModule"
    assert definition.eval_protocol_cls.__name__ == "ReconstructionEvalProtocol"


def test_runner_reuses_the_supervised_training_loop():
    from src.app.core.paradigm_impls.base.supervised import SupervisedRunner

    assert issubclass(load_paradigm_definition(PARADIGM).runner_cls, SupervisedRunner)


def test_data_module_reuses_the_supervised_loaders():
    from src.app.core.paradigm_impls.base.supervised import SupervisedDataModule

    assert issubclass(load_paradigm_definition(PARADIGM).data_module_cls, SupervisedDataModule)


# ---------------------------------------------------------------------------
# Capability helpers — these replaced hardcoded paradigm-name tuples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "paradigm,expected",
    [("online_rl", True), ("offline_rl", True), ("supervised", False), ("unsupervised", False)],
)
def test_uses_agents_matches_the_old_supervised_check(paradigm, expected):
    """Was `paradigm == "supervised"`, inverted."""
    assert paradigm_uses_agents(paradigm) is expected


@pytest.mark.parametrize(
    "paradigm,expected",
    [("online_rl", False), ("offline_rl", True), ("supervised", True), ("unsupervised", True)],
)
def test_static_dataset_matches_the_old_tuple_check(paradigm, expected):
    """Was `paradigm in ("offline_rl", "supervised")`."""
    assert paradigm_uses_static_dataset(paradigm) is expected


def test_unknown_paradigm_keeps_the_rl_assumption():
    """An unknown name must not silently become agent-less and skip agent checks."""
    assert paradigm_uses_agents("not_a_paradigm") is True
    assert paradigm_uses_static_dataset("not_a_paradigm") is False


def test_missing_allow_list_is_not_the_same_as_an_empty_one():
    """validation treats absent as unrestricted and [] as forbidding agents."""
    assert paradigm_uses_agents(None) is True


# ---------------------------------------------------------------------------
# Override construction
# ---------------------------------------------------------------------------


class _Cfg(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


def _overrides(paradigm, spec, dataset_path="/data/x.npz"):
    cfg = _Cfg(paradigm=paradigm, experiment_name="demo")
    return build_method_overrides("m1", spec, dataset_path=dataset_path, extra_args=None, cfg=cfg)


def test_model_only_method_does_not_require_an_agent():
    """The RL path raises on a missing agent; the agent-less path must not."""
    overrides = _overrides(PARADIGM, {"model": "autoencoder"})
    assert "paradigm=unsupervised" in overrides
    assert "model=autoencoder" in overrides
    assert not any(o.startswith("agent=") for o in overrides)


def test_dataset_path_is_passed_through():
    assert "++dataset_path=/data/x.npz" in _overrides(PARADIGM, {"model": "autoencoder"})


def test_hyperparameters_route_to_the_model_not_the_agent():
    overrides = _overrides(PARADIGM, {"model": "autoencoder", "latent_dim": 8})
    assert "++model.latent_dim=8" in overrides
    assert "++agent.latent_dim=8" not in overrides


def test_overrides_match_supervised_for_the_same_method():
    """Both paradigms are agent-less, so only the paradigm= token should differ."""
    spec = {"model": "autoencoder", "latent_dim": 8}
    unsup = [o for o in _overrides(PARADIGM, spec) if not o.startswith("paradigm=")]
    sup = [o for o in _overrides("supervised", spec) if not o.startswith("paradigm=")]
    assert unsup == sup


# ---------------------------------------------------------------------------
# Autoencoder architectures
# ---------------------------------------------------------------------------


@needs_autoencoder
def test_architectures_are_registered():
    from src.app.core.model_registry import MODEL_REGISTRY, auto_discover_models

    auto_discover_models()
    assert get_catalog().resolve("model","autoencoder") is not None
    assert get_catalog().resolve("model","dense_autoencoder") is not None


@needs_autoencoder
def test_sequence_autoencoder_reconstructs_its_input_shape():
    from src.app.core.model_registry import get_model_class
    SequenceAutoencoder = get_model_class("autoencoder")

    model = SequenceAutoencoder(input_dim=7, hidden_dim=16, latent_dim=8)
    x = torch.randn(4, 12, 7)
    recon, latent = model(x, lengths=torch.full((4,), 12))
    assert recon.shape == x.shape
    assert latent.shape == (4, 8)


@needs_autoencoder
def test_sequence_encoder_reads_the_last_real_step_not_the_padding():
    """A padded tail must not change the representation of a shorter sequence."""
    from src.app.core.model_registry import get_model_class
    SequenceAutoencoder = get_model_class("autoencoder")

    torch.manual_seed(0)
    model = SequenceAutoencoder(input_dim=3, hidden_dim=8, latent_dim=4).eval()
    x = torch.randn(1, 10, 3)
    padded = x.clone()
    padded[:, 6:, :] = 99.0  # garbage beyond the real length
    with torch.no_grad():
        a = model.encode(x[:, :6, :], lengths=torch.tensor([6]))
        b = model.encode(padded, lengths=torch.tensor([6]))
    assert torch.allclose(a, b, atol=1e-5)


@needs_autoencoder
def test_dense_autoencoder_round_trips_flat_features():
    from src.app.core.model_registry import get_model_class
    DenseAutoencoder = get_model_class("dense_autoencoder")

    model = DenseAutoencoder(input_dim=9, hidden_dim=8, latent_dim=4)
    x = torch.randn(5, 9)
    recon, latent = model(x)
    assert recon.shape == x.shape
    assert latent.shape == (5, 4)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------


def _module(architecture="autoencoder", input_dim=7, **kwargs):
    cls = get_component("autoencoder_training")
    return cls(architecture_name=architecture, input_dim=input_dim, lr=1e-3, **kwargs)


def _labeled_loader(n=40, length=12, dim=7, pad_from=None):
    x = torch.randn(n, length, dim)
    y = torch.randint(0, 2, (n,)).float()
    lengths = torch.full((n,), pad_from or length)
    mask = torch.zeros(n, length, dtype=torch.bool)
    if pad_from is not None:
        mask[:, pad_from:] = True
    return DataLoader(TensorDataset(x, y, lengths, mask), batch_size=8)


@needs_autoencoder
def test_module_is_resolvable_as_a_component():
    """train.py finds it through model.lightning_module, so it must be registered."""
    assert get_component("autoencoder_training") is not None


@needs_autoencoder
def test_unknown_architecture_is_rejected_with_the_registered_names():
    with pytest.raises(ValueError, match="Unknown unsupervised architecture"):
        _module(architecture="not_an_architecture")


@needs_autoencoder
def test_training_step_needs_no_labels():
    """The batch carries y because the loaders are shared; the loss must ignore it."""
    module = _module()
    loss = module.training_step(next(iter(_labeled_loader())), 0)
    assert loss.requires_grad
    assert float(loss.detach()) > 0


@needs_autoencoder
def test_padding_does_not_contribute_to_the_loss():
    """Averaging over padded steps would reward predicting zeros."""
    torch.manual_seed(0)
    module = _module(input_dim=4).eval()
    x = torch.randn(2, 10, 4)
    short = (x[:, :5, :], torch.zeros(2), torch.full((2,), 5), torch.zeros(2, 5, dtype=torch.bool))
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 5:] = True
    padded = (x, torch.zeros(2), torch.full((2,), 5), mask)
    with torch.no_grad():
        assert torch.allclose(module._step(short, "val"), module._step(padded, "val"), atol=1e-5)


@needs_autoencoder
def test_module_logs_the_metric_build_trainer_monitors():
    """build_trainer defaults to monitoring val/loss with mode min."""
    module = _module()
    logged = []
    module.log = lambda name, *a, **k: logged.append(name)
    module.validation_step(next(iter(_labeled_loader())), 0)
    assert "val/loss" in logged


@needs_autoencoder
def test_declared_monitor_metric_is_one_the_module_logs():
    monitor = load_paradigm(PARADIGM).get("monitor_metric")
    module = _module()
    logged = []
    module.log = lambda name, *a, **k: logged.append(name)
    module.validation_step(next(iter(_labeled_loader())), 0)
    assert monitor in logged


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------


def _protocol():
    return get_component("ReconstructionEvalProtocol")()


@needs_autoencoder
def test_evaluate_reports_reconstruction_error():
    metrics = _protocol().evaluate(_module(), _labeled_loader())
    assert metrics["recon_mse"] > 0
    assert metrics["recon_mae"] > 0


@needs_autoencoder
def test_evaluate_reports_latent_clustering_quality():
    """With no labels, internal cluster scores stand in for accuracy."""
    metrics = _protocol().evaluate(_module(), _labeled_loader(), n_clusters=3)
    assert -1.0 <= metrics["silhouette"] <= 1.0
    assert metrics["davies_bouldin"] >= 0
    assert metrics["n_clusters"] == 3


@needs_autoencoder
def test_clustering_metrics_are_omitted_rather_than_faked_when_too_small():
    """Two samples cannot support three clusters, and a made-up score is worse than none."""
    metrics = _protocol().evaluate(_module(), _labeled_loader(n=2), n_clusters=3)
    assert "recon_mse" in metrics
    assert "silhouette" not in metrics


@needs_autoencoder
def test_evaluate_accepts_unlabeled_batches():
    loader = DataLoader(TensorDataset(torch.randn(30, 9)), batch_size=10)
    metrics = _protocol().evaluate(_module(architecture="dense_autoencoder", input_dim=9), loader)
    assert metrics["recon_mse"] > 0


@needs_autoencoder
def test_evaluate_on_an_empty_loader_returns_zeroed_metrics():
    empty = DataLoader(TensorDataset(torch.zeros(0, 12, 7)), batch_size=4)
    assert _protocol().evaluate(_module(), empty) == {"recon_mse": 0.0, "recon_mae": 0.0}


# ---------------------------------------------------------------------------
# End-to-end training loop
# ---------------------------------------------------------------------------


@needs_autoencoder
def test_the_runner_actually_trains_the_autoencoder():
    """Drives the real runner through a real Lightning Trainer.

    The MIMIC dataset this paradigm targets is not in the repo, so the loop is
    exercised on synthetic sequences with a recoverable structure: each sequence
    is one vector repeated with noise, which a working autoencoder compresses
    and so reconstructs better after training than before.
    """
    torch.manual_seed(0)
    definition = load_paradigm_definition(PARADIGM)
    module = _module(input_dim=6, hidden_dim=24, latent_dim=6)

    n, length, dim = 128, 10, 6
    base = torch.randn(n, 1, dim).repeat(1, length, 1)
    x = base + 0.05 * torch.randn(n, length, dim)
    loader = DataLoader(
        TensorDataset(x, torch.zeros(n), torch.full((n,), length), torch.zeros(n, length, dtype=torch.bool)),
        batch_size=16,
    )

    protocol = definition.eval_protocol_cls()
    before = protocol.evaluate(module, loader)["recon_mse"]
    L.Trainer(max_epochs=5, accelerator='cpu', devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False).fit(module, train_dataloaders=loader, val_dataloaders=loader)
    after = protocol.evaluate(module, loader)["recon_mse"]

    assert after < before, f"reconstruction did not improve: {before:.4f} -> {after:.4f}"


@needs_autoencoder
def test_evaluating_does_not_leave_the_model_in_eval_mode():
    """Scoring a checkpoint and then resuming training must not lose dropout."""
    module = _module()
    module.train()
    _protocol().evaluate(module, _labeled_loader())
    assert module.training is True


@needs_autoencoder
def test_evaluating_restores_eval_mode_when_that_is_where_it_started():
    module = _module()
    module.eval()
    _protocol().evaluate(module, _labeled_loader())
    assert module.training is False


# Native package/config/training resolution is exercised in the maintained
# theta-plugins/tests/test_native_integration.py suite.


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


def _breast_cancer_split(seed=0):
    """Real clinical tabular data: 569 biopsies, 30 features, benign/malignant.

    The MIMIC dataset this paradigm targets needs PhysioNet credentials and is
    not in the repo, so this stands in as real data that ships with sklearn.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    x_train, x_test, _, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=seed, stratify=data.target
    )
    scaler = StandardScaler().fit(x_train)
    return (
        scaler.transform(x_train).astype("float32"),
        scaler.transform(x_test).astype("float32"),
        y_test,
    )


@needs_autoencoder
def test_autoencoder_learns_real_clinical_data():
    """Trains on real features with the labels withheld, then uses them only to
    score whether the latent space recovered the real benign/malignant split.

    Asserts the two properties that held across every seed tried. Notably it
    does not assert the latent beats k-means on the raw features: that was true
    for only two seeds in five, so it is not a property of the model.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    torch.manual_seed(0)
    x_train, x_test, y_test = _breast_cancer_split()
    definition = load_paradigm_definition(PARADIGM)
    module = _module(architecture="dense_autoencoder", input_dim=x_train.shape[1], hidden_dim=64, latent_dim=8)

    train_loader = DataLoader(TensorDataset(torch.tensor(x_train)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test)), batch_size=32)

    protocol = definition.eval_protocol_cls()
    before = protocol.evaluate(module, test_loader, n_clusters=2)["recon_mse"]
    L.Trainer(max_epochs=30, accelerator='cpu', devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False).fit(module, train_dataloaders=train_loader, val_dataloaders=test_loader)
    after = protocol.evaluate(module, test_loader, n_clusters=2)["recon_mse"]

    assert after < before / 2, f"reconstruction barely improved: {before:.3f} -> {after:.3f}"

    with torch.no_grad():
        latent = module.encode(torch.tensor(x_test)).numpy()
    clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(latent)
    score = adjusted_rand_score(y_test, clusters)
    # Chance is ~0. The five seeds tried landed between 0.56 and 0.86.
    assert score > 0.3, f"latent space carries no real class structure (ARI={score:.3f})"


@needs_autoencoder
def test_reconstruction_quality_and_cluster_quality_are_not_the_same_thing():
    """A wider latent reconstructs better while clustering worse, which is why
    monitor_metric: val/loss selects for reconstruction rather than for a
    representation a downstream task can use. Worth knowing before tuning on it.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    x_train, x_test, y_test = _breast_cancer_split()
    definition = load_paradigm_definition(PARADIGM)
    results = {}
    for latent_dim in (8, 48):
        torch.manual_seed(0)
        module = _module(
            architecture="dense_autoencoder", input_dim=x_train.shape[1], hidden_dim=64, latent_dim=latent_dim
        )
        train_loader = DataLoader(TensorDataset(torch.tensor(x_train)), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(x_test)), batch_size=32)
        L.Trainer(max_epochs=30, accelerator='cpu', devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False).fit(module, train_dataloaders=train_loader, val_dataloaders=test_loader)
        mse = definition.eval_protocol_cls().evaluate(module, test_loader, n_clusters=2)["recon_mse"]
        with torch.no_grad():
            latent = module.encode(torch.tensor(x_test)).numpy()
        clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(latent)
        results[latent_dim] = (mse, adjusted_rand_score(y_test, clusters))

    wide_mse, wide_ari = results[48]
    narrow_mse, narrow_ari = results[8]
    assert wide_mse < narrow_mse, "expected the wider latent to reconstruct better"
    assert wide_ari <= narrow_ari, "expected the wider latent not to cluster better"
