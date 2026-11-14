"""Data module for the .npz datasets in/datasets/prepare_tabular.py writes.

The unsupervised paradigm needs an offline dataset, and clinical trajectory data is not available
to this project, so these are real datasets bundled with scikit-learn. The
loader is deliberately generic: anything stored as `features` (n, d) or
`sequences` (n, steps, per_step) works, so swapping in another offline dataset
means writing an .npz, not writing code.

Labels are loaded when present but never put in the training batch. They exist
so an evaluation can ask whether a learned latent space lines up with real
classes; letting them reach the model would make the run supervised.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.app.core.interfaces import BaseDataModule
from src.app.core.paradigm_loader import register_component

log = logging.getLogger(__name__)


@register_component("TabularNpzDataModule")
class TabularNpzDataModule(L.LightningDataModule, BaseDataModule):
    """Train/val split over an .npz of unlabeled feature vectors or sequences.

    LightningDataModule is a real base here, not decoration: Trainer.fit calls
    is_overridden() and the datamodule hooks on whatever it is handed, and both
    fail on a plain object.
    """

    def __init__(self, cfg: Any = None, *, paths=None):
        super().__init__()
        self.cfg = cfg
        from src.app.workspace import WorkspacePaths
        self.paths = paths or WorkspacePaths.for_process()
        self.input_dim = 0
        self.labels: np.ndarray | None = None
        self._train: DataLoader | None = None
        self._val: DataLoader | None = None
        if cfg is not None:
            self.setup(cfg=cfg)

    # -- config helpers ---------------------------------------------------

    @staticmethod
    def _get(cfg: Any, key: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if hasattr(cfg, "get"):
            value = cfg.get(key, default)
            return default if value is None else value
        return getattr(cfg, key, default)

    def _resolve_path(self, cfg: Any) -> Path:
        """The dataset file, from an explicit override or the env's dataset_name."""
        explicit = self._get(cfg, "dataset_path")
        if explicit:
            from src.app.pipeline.datasets import resolve_dataset_for_method
            return resolve_dataset_for_method("tabular", {}, cfg, paths=self.paths)

        env = self._get(cfg, "env")
        name = self._get(env, "dataset_name") or self._get(env, "name")
        if not name:
            raise FileNotFoundError("TabularNpzDataModule needs dataset_path, or dataset_name on the env config.")
        stem = str(name).replace(".npz", "")
        for candidate in (
            self.paths.datasets / stem / f"{stem}.npz",
            self.paths.datasets / f"{stem}.npz",
            self.paths.datasets / stem,
        ):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No dataset for '{stem}' under {self.paths.datasets}. Set dataset_path to an existing input, "
            f"or generate an example with {self.paths.application / 'in/datasets/prepare_tabular.py'}."
        )

    # -- BaseDataModule ---------------------------------------------------

    def setup(self, stage: str | None = None, cfg: Any = None) -> None:
        if cfg is not None:
            self.cfg = cfg
        elif stage is not None and not isinstance(stage, str):
            self.cfg = stage
        if self._train is not None:
            return

        path = self._resolve_path(self.cfg)
        with np.load(path) as handle:
            keys = list(handle.keys())
            prefer_sequences = bool(self._get(self.cfg, "use_sequences", False))
            key = "sequences" if (prefer_sequences and "sequences" in keys) else "features"
            if key not in keys:
                raise KeyError(f"{path} has no '{key}' array (found: {keys})")
            data = np.asarray(handle[key], dtype=np.float32)
            self.labels = np.asarray(handle["labels"]) if "labels" in keys else None

        # Width the model is built against: per-step for sequences, else total.
        self.input_dim = int(data.shape[-1])

        val_fraction = float(self._get(self.cfg, "val_fraction", 0.3) or 0.3)
        seed = int(self._get(self.cfg, "seed", 0) or 0)
        batch_size = int(self._get(self.cfg, "batch_size", 32) or 32)

        order = np.random.default_rng(seed).permutation(len(data))
        split = max(1, int(len(data) * (1.0 - val_fraction)))
        train_idx, val_idx = order[:split], order[split:]
        self.train_indices, self.val_indices = train_idx, val_idx

        self._train = self._loader(data[train_idx], batch_size, shuffle=True)
        self._val = self._loader(data[val_idx], batch_size, shuffle=False)
        log.info(
            "TabularNpzDataModule: %s %s -> train %d / val %d, input_dim=%d",
            path,
            data.shape,
            len(train_idx),
            len(val_idx),
            self.input_dim,
        )

    def _loader(self, array: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if tensor.ndim == 3:
            # Match the (x, y, lengths, padding_mask) shape the sequence models
            # expect. y is zeros: no label reaches the model.
            n, steps = tensor.shape[0], tensor.shape[1]
            dataset = TensorDataset(
                tensor,
                torch.zeros(n),
                torch.full((n,), steps, dtype=torch.long),
                torch.zeros(n, steps, dtype=torch.bool),
            )
        else:
            dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_dataloader(self) -> DataLoader:
        if self._train is None:
            self.setup()
        assert self._train is not None
        return self._train

    def val_dataloader(self) -> DataLoader | None:
        if self._val is None:
            self.setup()
        return self._val

    def held_out_labels(self) -> np.ndarray | None:
        """Validation-split labels, for scoring a latent space after training."""
        if self.labels is None or not hasattr(self, "val_indices"):
            return None
        return self.labels[self.val_indices]


@register_component("SupervisedNpzDataModule")
class SupervisedNpzDataModule(TabularNpzDataModule):
    """The same .npz datasets, with the labels in the batch.

    Kept separate from TabularNpzDataModule rather than switched by a flag: the
    unsupervised paradigm must not be one config key away from quietly training
    against labels, and an experiment names the module it wants anyway.
    """

    def _loader(self, array: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        if self.labels is None:
            raise ValueError(
                "SupervisedNpzDataModule needs a 'labels' array in the dataset. "
                "in/datasets/prepare_tabular.py writes one."
            )
        split = self.train_indices if shuffle else self.val_indices
        x = torch.from_numpy(np.ascontiguousarray(array))
        y = torch.from_numpy(np.ascontiguousarray(self.labels[split])).float()
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)
