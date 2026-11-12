"""Tests for supervised training on datasets other than MIMIC.

Supervised previously only worked for the early_prediction setup: train.py fell
back to EPSepsisLightningModule, which accepts only lstm/transformer and expects
MIMIC-shaped sequence batches, and SupervisedDataModule had no loader for
anything else. A supervised group pointing at any other dataset validated and
then failed at training time.

These cover the three pieces that close that: a classifier architecture, a
labelled data module, and a LightningModule train.py can resolve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from importlib.metadata import entry_points

import pytest
import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

from src.app.core.paradigm_loader import get_component, load_paradigm_definition

MLP_INSTALLED = any(
    point.name == "mlp_classifier_training"
    for point in entry_points().select(group="theta.components")
)
needs_mlp = pytest.mark.skipif(not MLP_INSTALLED, reason="the MLP model component is not installed")

CONFIG = Path("in/config")


class _Cfg(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


def _write_npz(path: Path, n=120, dim=8, separable=True):
    """A labelled dataset with recoverable structure, so training can be judged."""
    rng = np.random.default_rng(0)
    labels = (np.arange(n) % 2).astype(np.int64)
    features = rng.normal(size=(n, dim)).astype(np.float32)
    if separable:
        features[labels == 1] += 1.5
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, labels=labels)
    return path


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


@needs_mlp
def test_the_rl_dnn_is_not_a_classifier():
    """`dnn` is an action/value model, so supervised uses a classifier instead."""
    from src.app.core.model_registry import MODEL_REGISTRY, auto_discover_models

    auto_discover_models()
    dnn = MODEL_REGISTRY["dnn"]
    assert hasattr(dnn, "get_action_and_value")
    assert hasattr(dnn, "get_q_values")


@needs_mlp
def test_the_classifier_is_registered():
    from src.app.core.model_registry import MODEL_REGISTRY, auto_discover_models

    auto_discover_models()
    assert "mlp_classifier" in MODEL_REGISTRY


@needs_mlp
def test_a_binary_classifier_emits_one_logit_per_sample():
    """ClassificationEvalProtocol takes the sigmoid of a 1-D output."""
    from src.usr.models.mlp.mlp_model import MLPClassifier

    assert MLPClassifier(input_dim=8)(torch.randn(5, 8)).shape == (5,)


@needs_mlp
def test_a_multiclass_classifier_emits_one_logit_per_class():
    from src.usr.models.mlp.mlp_model import MLPClassifier

    assert MLPClassifier(input_dim=8, n_classes=4)(torch.randn(5, 8)).shape == (5, 4)


# ---------------------------------------------------------------------------
# Labelled data module
# ---------------------------------------------------------------------------


def _supervised_dm(path, **extra):
    return get_component("SupervisedNpzDataModule")(_Cfg(dataset_path=str(path), **extra))


def test_the_labelled_module_is_registered():
    assert get_component("SupervisedNpzDataModule") is not None


def test_labels_are_in_the_batch():
    """The unsupervised module deliberately drops them; this one must not."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dm = _supervised_dm(_write_npz(Path(tmp) / "d.npz"))
        batch = next(iter(dm.train_dataloader()))
        assert len(batch) == 2
        x, y = batch
        assert x.shape[0] == y.shape[0]


def test_the_unsupervised_module_still_withholds_labels():
    """Both read the same file; only one may see the answers."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_npz(Path(tmp) / "d.npz")
        unsup = get_component("TabularNpzDataModule")(_Cfg(dataset_path=str(path)))
        assert len(next(iter(unsup.train_dataloader()))) == 1


def test_train_and_validation_labels_line_up_with_their_rows():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dm = _supervised_dm(_write_npz(Path(tmp) / "d.npz", n=100), val_fraction=0.3, batch_size=1000)
        _, y = next(iter(dm.val_dataloader()))
        assert len(y) == len(dm.val_indices)
        assert np.array_equal(y.numpy().astype(np.int64), dm.labels[dm.val_indices])


def test_a_dataset_without_labels_is_refused_with_a_usable_message():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.npz"
        np.savez_compressed(path, features=np.zeros((10, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="prepare_tabular.py"):
            _supervised_dm(path)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------


def _module(**kwargs):
    kwargs.setdefault("architecture_name", "mlp_classifier")
    kwargs.setdefault("input_dim", 8)
    kwargs.setdefault("lr", 1e-3)
    return get_component("mlp_classifier_training")(**kwargs)


@needs_mlp
def test_the_module_is_resolvable_as_a_component():
    """train.py finds it through model.lightning_module."""
    assert get_component("mlp_classifier_training") is not None


@needs_mlp
def test_an_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unknown supervised architecture"):
        _module(architecture_name="not_an_architecture")


@needs_mlp
def test_it_logs_the_metric_build_trainer_monitors():
    module = _module()
    logged = []
    module.log = lambda name, *a, **k: logged.append(name)
    loader = DataLoader(TensorDataset(torch.randn(8, 8), torch.randint(0, 2, (8,)).float()), batch_size=4)
    module.validation_step(next(iter(loader)), 0)
    assert "val/loss" in logged


@needs_mlp
def test_a_binary_task_uses_a_binary_loss():
    assert type(_module().loss_fn).__name__ == "BCEWithLogitsLoss"


@needs_mlp
def test_a_multiclass_task_uses_cross_entropy():
    assert type(_module(n_classes=3).loss_fn).__name__ == "CrossEntropyLoss"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@needs_mlp
def test_the_supervised_runner_trains_a_classifier():
    """Drives the shipped supervised runner and eval protocol, unchanged."""
    import tempfile

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        dm = _supervised_dm(_write_npz(Path(tmp) / "d.npz", n=200, dim=8), batch_size=32)
        module = _module(input_dim=dm.input_dim, hidden_sizes=[32])
        definition = load_paradigm_definition("supervised")
        protocol = definition.eval_protocol_cls()

        before = protocol.evaluate(module, dm.val_dataloader())["auc"]
        L.Trainer(max_epochs=25, accelerator='cpu', devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False).fit(module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
        after = protocol.evaluate(module, dm.val_dataloader())["auc"]

        assert after > before
        assert after > 0.9, f"a separable dataset should be learnable (auc={after:.3f})"


def test_a_supervised_group_is_configured_for_this_path():
    """Whatever supervised groups are present must be runnable by this path.

    No experiment group ships in the repo any more: they are Hub components,
    and CI sees only the fixture workspace. So this checks every supervised
    group it can find rather than naming one, and says so when there are none
    instead of asserting against a group that is not there.

    It no longer requires `data_module`. The supervised paradigm declares
    `default_data_module: SupervisedDataModule`, which carries
    `npz_delegate = "SupervisedNpzDataModule"`, so a group naming the loader
    itself is redundant: the bug that assertion guarded is fixed structurally.
    """
    import yaml

    bases = sorted((CONFIG / "experiment").glob("*/_base.yaml"))
    supervised = []
    for path in bases:
        raw = yaml.safe_load(path.read_text()) or {}
        if raw.get("paradigm") == "supervised":
            supervised.append((path, raw))

    if not supervised:
        pytest.skip("no supervised group present: they are Hub components now")

    for path, base in supervised:
        where = path.parent.name
        assert "intervals_count" not in base, f"{where} uses the key that was renamed to epochs"
        if "eval_episodes" in base:
            assert base["eval_episodes"] == 0, f"{where}: supervised has no simulator to roll out in"
        for name, spec in (base.get("methods") or {}).items():
            assert "agent" not in spec, f"{where}.{name}: supervised methods name a model, not an agent"


def test_the_model_config_names_a_resolvable_lightning_module():
    """Without this key supervised silently falls back to the MIMIC module."""
    import yaml

    cfg_candidates = list((CONFIG / "model").rglob("mlp_classifier.yaml"))
    if not cfg_candidates or not MLP_INSTALLED:
        pytest.skip("mlp_classifier.yaml not in config tree")
    model_cfg = yaml.safe_load(cfg_candidates[0].read_text())
    assert get_component(model_cfg.get("lightning_module")) is not None
    assert model_cfg.get("architecture") == "mlp_classifier" or model_cfg.get("type") == "mlp_classifier"


@needs_mlp
def test_scoring_does_not_leave_the_model_in_eval_mode():
    """The classifier has dropout, so evaluating a checkpoint and then resuming
    training would quietly train without it."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dm = _supervised_dm(_write_npz(Path(tmp) / "d.npz"))
        module = _module(input_dim=dm.input_dim)
        module.train()
        load_paradigm_definition("supervised").eval_protocol_cls().evaluate(module, dm.val_dataloader())
        assert module.training is True


@needs_mlp
def test_scoring_leaves_eval_mode_alone_when_it_started_there():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dm = _supervised_dm(_write_npz(Path(tmp) / "d.npz"))
        module = _module(input_dim=dm.input_dim)
        module.eval()
        load_paradigm_definition("supervised").eval_protocol_cls().evaluate(module, dm.val_dataloader())
        assert module.training is False


# ---------------------------------------------------------------------------
# Resolving a loader without a data_module key
# ---------------------------------------------------------------------------


def _env_cfg(tmp_path, name="breast_cancer"):
    """A config carrying an env but no `data_module`, over a dataset written here.

    in/datasets/ is gitignored - its contents are produced by
    in/datasets/prepare_tabular.py - so a test that reads from it passes only on
    a machine where someone has run that script, and fails in CI.
    """
    return _Cfg(
        dataset_path=str(_write_npz(tmp_path / "d.npz")),
        env=_Cfg(name=name, dataset_name=f"{name}.npz"),
        batch_size=32,
    )


def test_supervised_finds_a_labelled_loader_with_no_data_module_key(tmp_path):
    """`data_module` cannot be set from the configuration panel, so requiring it
    would mean supervised still only worked for MIMIC as far as the UI goes."""
    dm = get_component("SupervisedDataModule")(_env_cfg(tmp_path))
    assert type(dm._delegate).__name__ == "SupervisedNpzDataModule"
    assert len(next(iter(dm.train_dataloader()))) == 2


def test_unsupervised_resolves_the_unlabelled_loader_instead(tmp_path):
    """Same dataset, same absent key, different paradigm: labels must not leak."""
    dm = get_component("UnsupervisedDataModule")(_env_cfg(tmp_path))
    assert type(dm._delegate).__name__ == "TabularNpzDataModule"
    assert len(next(iter(dm.train_dataloader()))) == 1


def test_the_two_paradigms_declare_different_delegates():
    supervised = get_component("SupervisedDataModule")
    unsupervised = get_component("UnsupervisedDataModule")
    assert supervised.npz_delegate != unsupervised.npz_delegate


def test_the_environment_does_not_choose_the_loader():
    """An env is shared across paradigms, and train.py prefers the env's choice
    over the paradigm's, so naming a loader there picked the wrong one."""
    import yaml

    for name in ("breast_cancer", "digits"):
        env_file = CONFIG / "env" / f"{name}.yaml"
        if not env_file.is_file():
            continue
        env = yaml.safe_load(env_file.read_text())
        assert "data_module" not in env, f"{name}.yaml should leave the loader to the paradigm"


def test_paradigms_do_not_choose_an_installed_model_for_the_user():
    from frontend.config_model import ConfigTree, default_method_model

    tree = ConfigTree.discover()
    for paradigm in ("supervised", "unsupervised"):
        assert "default_model" not in tree.paradigms[paradigm].raw
        assert default_method_model(None, tree.models, paradigm=tree.paradigms[paradigm]) is None


def test_an_existing_group_supplies_its_own_model_default():
    """A group may declare its own architecture without a paradigm default.

    Built here rather than read from a shipped group: early_prediction used to
    provide one, and experiment groups are optional components, so this test
    defines its own group rather than depending on an installed example.
    """
    from frontend.config_model import ConfigTree, ExperimentGroup, default_method_model

    tree = ConfigTree.discover()
    settled = ExperimentGroup(
        name="already_chosen",
        paradigm="supervised",
        has_base=True,
        base_methods={"early": {"model": "lstm"}},
    )
    chosen = default_method_model(settled, tree.models, paradigm=tree.paradigms["supervised"])
    assert chosen == "lstm", "a group that has settled on an architecture should not be overridden"


def test_rl_paradigms_do_not_choose_a_model_plugin_by_default():
    from frontend.config_model import ConfigTree, default_method_model

    tree = ConfigTree.discover()
    for paradigm in ("online_rl", "offline_rl"):
        assert "default_model" not in tree.paradigms[paradigm].raw
        assert default_method_model(None, tree.models, paradigm=tree.paradigms[paradigm]) is None


def test_a_named_dataset_that_is_missing_is_reported(tmp_path, monkeypatch, caplog):
    """Missing data used to be logged at debug level, so a run only failed later with
    "train_loader has not been set" and no word about the dataset."""
    import logging

    monkeypatch.chdir(tmp_path)  # no in/datasets here
    with caplog.at_level(logging.WARNING, logger="src.app.core.paradigm_impls.base.supervised"):
        # Named but never written: no dataset_path, so it is looked up under in/datasets/.
        name = "breast_cancer"
        named = _Cfg(env=_Cfg(name=name, dataset_name=f"{name}.npz"), batch_size=32)
        dm = get_component("SupervisedDataModule")(named)
    assert dm._delegate is None
    assert "No dataset for 'breast_cancer' under in/datasets/" in caplog.text


def test_a_config_naming_no_dataset_stays_quiet(caplog):
    """Loaders may be supplied directly instead (cross-validation folds), so no warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="src.app.core.paradigm_impls.base.supervised"):
        get_component("SupervisedDataModule")(_Cfg(batch_size=32))
    assert caplog.text == ""
