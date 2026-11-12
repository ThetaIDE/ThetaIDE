"""Tests for the offline dataset path of the unsupervised paradigm.

MIMIC is unavailable to this project, so unsupervised experiments train on
datasets in/datasets/prepare_tabular.py materialises from scikit-learn. That
directory is gitignored, so nothing here reads it: the data module tests write
their own .npz, and the prepare tests build into a temporary directory.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import pytest
import torch
import lightning as L

from src.app.core.paradigm_loader import get_component

AUTOENCODER_INSTALLED = any(
    point.name == "autoencoder_training"
    for point in entry_points().select(group="theta.components")
)
needs_autoencoder = pytest.mark.skipif(
    not AUTOENCODER_INSTALLED,
    reason="the autoencoder model component is not installed",
)

PREPARE = Path("in/datasets/prepare_tabular.py")


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_tabular", PREPARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_npz(path: Path, n=40, dim=6, sequences=False, labels=True):
    arrays = {"features": np.random.default_rng(0).normal(size=(n, dim)).astype(np.float32)}
    if sequences:
        arrays["sequences"] = arrays["features"].reshape(n, 2, dim // 2)
    if labels:
        arrays["labels"] = np.arange(n) % 2
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


class _Cfg(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


def _cfg(path, **extra):
    return _Cfg(dataset_path=str(path), **extra)


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------


def test_is_registered_as_a_component():
    assert get_component("TabularNpzDataModule") is not None


def test_is_a_lightning_data_module():
    """Trainer.fit calls is_overridden() on whatever it is handed, which raises
    'Expected a parent' for a plain object."""
    import lightning as L

    assert issubclass(get_component("TabularNpzDataModule"), L.LightningDataModule)


def test_loads_features_and_reports_input_dim(tmp_path):
    dm = get_component("TabularNpzDataModule")(_cfg(_write_npz(tmp_path / "d.npz", dim=6)))
    assert dm.input_dim == 6


def test_splits_into_train_and_validation(tmp_path):
    dm = get_component("TabularNpzDataModule")(_cfg(_write_npz(tmp_path / "d.npz", n=40), val_fraction=0.25))
    assert len(dm.train_indices) == 30
    assert len(dm.val_indices) == 10


def test_the_split_is_a_partition(tmp_path):
    dm = get_component("TabularNpzDataModule")(_cfg(_write_npz(tmp_path / "d.npz", n=40)))
    assert not set(dm.train_indices) & set(dm.val_indices)
    assert len(set(dm.train_indices) | set(dm.val_indices)) == 40


def test_the_split_is_reproducible_for_a_seed(tmp_path):
    path = _write_npz(tmp_path / "d.npz", n=40)
    first = get_component("TabularNpzDataModule")(_cfg(path, seed=7)).val_indices
    second = get_component("TabularNpzDataModule")(_cfg(path, seed=7)).val_indices
    assert list(first) == list(second)


def test_no_label_reaches_the_training_batch(tmp_path):
    """A label in the batch would make the run supervised by accident."""
    dm = get_component("TabularNpzDataModule")(_cfg(_write_npz(tmp_path / "d.npz", labels=True)))
    batch = next(iter(dm.train_dataloader()))
    assert len(batch) == 1, "flat batches carry inputs only"


def test_labels_are_still_available_for_scoring(tmp_path):
    dm = get_component("TabularNpzDataModule")(_cfg(_write_npz(tmp_path / "d.npz", n=40)))
    held_out = dm.held_out_labels()
    assert held_out is not None
    assert len(held_out) == len(dm.val_indices)


def test_a_dataset_without_labels_still_loads(tmp_path):
    dm = get_component("TabularNpzDataModule")(_cfg(_write_npz(tmp_path / "d.npz", labels=False)))
    assert dm.held_out_labels() is None
    assert dm.input_dim == 6


def test_sequence_form_yields_the_four_part_batch(tmp_path):
    """The sequence models expect (x, y, lengths, padding_mask); y stays zeros."""
    path = _write_npz(tmp_path / "d.npz", dim=6, sequences=True)
    dm = get_component("TabularNpzDataModule")(_cfg(path, use_sequences=True))
    batch = next(iter(dm.train_dataloader()))
    assert len(batch) == 4
    x, y, lengths, mask = batch
    assert x.ndim == 3
    assert torch.count_nonzero(y) == 0
    assert torch.count_nonzero(mask) == 0
    assert dm.input_dim == 3, "input_dim is the per-step width for sequences"


def test_a_missing_dataset_says_how_to_build_it(tmp_path):
    cfg = _Cfg(env=_Cfg(dataset_name="not_a_dataset.npz"))
    with pytest.raises(FileNotFoundError, match="prepare_tabular.py"):
        get_component("TabularNpzDataModule")(cfg)


@needs_autoencoder
def test_the_module_feeds_the_autoencoder(tmp_path):
    """End of the chain: a file on disk trains the paradigm's own model."""
    from src.app.core.paradigm_loader import load_paradigm_definition

    dm = get_component("TabularNpzDataModule")(_cfg(_write_npz(tmp_path / "d.npz", n=64, dim=6)))
    module = get_component("autoencoder_training")(
        architecture_name="dense_autoencoder", input_dim=dm.input_dim, lr=1e-3, hidden_dim=16, latent_dim=4
    )
    definition = load_paradigm_definition("unsupervised")
    protocol = definition.eval_protocol_cls()
    before = protocol.evaluate(module, dm.val_dataloader())["recon_mse"]
    L.Trainer(max_epochs=5, accelerator='cpu', devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False).fit(module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    assert protocol.evaluate(module, dm.val_dataloader())["recon_mse"] < before


# ---------------------------------------------------------------------------
# The prepare script
# ---------------------------------------------------------------------------


def test_prepare_script_exists_since_the_datasets_are_gitignored():
    assert PREPARE.is_file(), "in/datasets/ is gitignored, so the script is what ships"


def test_prepare_builds_a_loadable_dataset(tmp_path, monkeypatch):
    module = _load_prepare_module()
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path)
    path = module.build("breast_cancer")
    with np.load(path) as handle:
        assert handle["features"].shape == (569, 30)
        assert handle["labels"].shape == (569,)


def test_prepare_standardises_features(tmp_path, monkeypatch):
    """Unstandardised, the loss is dominated by whichever feature has the
    largest units rather than by anything structural."""
    module = _load_prepare_module()
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path)
    with np.load(module.build("breast_cancer")) as handle:
        features = handle["features"]
    assert np.allclose(features.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(features.std(axis=0), 1, atol=1e-4)


def test_prepare_emits_a_sequence_form_where_the_shape_allows(tmp_path, monkeypatch):
    module = _load_prepare_module()
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path)
    with np.load(module.build("digits")) as handle:
        assert handle["sequences"].shape == (1797, 8, 8)


def test_prepare_rejects_an_unknown_dataset(tmp_path, monkeypatch):
    module = _load_prepare_module()
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path)
    assert module.main(["prepare_tabular.py", "not_a_dataset"]) == 1


# ---------------------------------------------------------------------------
# Training schedule resolution
# ---------------------------------------------------------------------------


def test_agent_setting_wins_when_the_agent_declares_it():
    """cql and iql declare epochs: 50; that must keep winning."""
    from src.app.core.lightning_builder import _agent_or_global

    cfg = _Cfg(epochs=3, agent=_Cfg(epochs=50))
    assert _agent_or_global(cfg, "epochs", 1) == 50


def test_top_level_setting_is_used_when_the_agent_omits_it():
    """An agent-less paradigm still composes the default agent group (ppo), which
    declares no schedule, so reading only the agent pinned every run to 1 epoch."""
    from src.app.core.lightning_builder import _agent_or_global

    cfg = _Cfg(epochs=20, agent=_Cfg(lr=1e-3))
    assert _agent_or_global(cfg, "epochs", 1) == 20


def test_the_default_applies_when_nobody_declares_it():
    from src.app.core.lightning_builder import _agent_or_global

    assert _agent_or_global(_Cfg(agent=_Cfg()), "epochs", 1) == 1


def test_a_config_without_an_agent_section_still_resolves():
    from src.app.core.lightning_builder import _agent_or_global

    assert _agent_or_global(_Cfg(epochs=7), "epochs", 1) == 7


def test_no_test_here_depends_on_a_generated_dataset():
    """in/datasets/ is gitignored, so anything read from it exists only where
    someone has run prepare_tabular.py. Two tests once resolved breast_cancer.npz
    from there and passed locally while failing in CI; this keeps that from
    coming back unnoticed.
    """
    import re

    offenders = []
    for path in sorted(Path("tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8")  # the default on Windows is cp1252, which some test files are not
        for number, line in enumerate(text.splitlines(), start=1):
            if "in/datasets" not in line or line.lstrip().startswith("#"):
                continue
            # A prepare-script reference or a skip-guarded lookup is fine; reading a
            # dataset file as a fixture is not.
            if re.search(r"in/datasets/\w+/\w+\.npz|dataset_name=\"?\w+\.npz", line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "tests must build their own dataset:\n" + "\n".join(offenders)
