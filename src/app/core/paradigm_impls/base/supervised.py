"""Generic supervised learning paradigm components.

Implements domain-independent interfaces for supervised learning:
- SupervisedDataModule: Manages PyTorch DataLoader instances for train/val/test.
- ClassificationEvalProtocol: Domain-agnostic evaluation computing AUC-ROC, AUPRC, F1, and loss.
- SupervisedRunner: Orchestrates model training using standard PyTorch Lightning Trainer.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from src.app.core.interfaces import BaseDataModule, BaseEvalProtocol, BaseParadigmRunner
from src.app.core.paradigm_loader import register_component

log = logging.getLogger(__name__)


def _names_a_dataset(cfg: Any) -> bool:
    """Whether a config points at a dataset file (dataset_path, or the env's dataset_name)."""

    def get(obj, key):
        return obj.get(key) if hasattr(obj, "get") else getattr(obj, key, None)

    env = get(cfg, "env")
    return bool(get(cfg, "dataset_path") or (env is not None and get(env, "dataset_name")))


# ---------------------------------------------------------------------------
# 1. Generic Supervised Data Module
# ---------------------------------------------------------------------------


@register_component("SupervisedDataModule", "CrossValidationDataModule")
class SupervisedDataModule(L.LightningDataModule, BaseDataModule):
    """Domain-independent data module wrapping train, val, and test DataLoaders."""

    #: Loader used for a plain .npz dataset. Supervised needs the labels in the
    #: batch; paradigms that must not see them override this.
    npz_delegate = "SupervisedNpzDataModule"

    def __init__(
        self,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        cfg: Any = None,
    ):
        super().__init__()
        if train_loader is not None and not isinstance(train_loader, DataLoader) and cfg is None:
            cfg = train_loader
            train_loader = None

        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader
        self._cfg = None
        self._delegate = None
        self.input_dim = 64
        if cfg is not None:
            self.setup(cfg=cfg)

    def setup(self, stage: str | None = None, cfg: Any = None) -> None:
        """Initialize data module from config if loaders were not passed at init."""
        if cfg is not None and not isinstance(cfg, str):
            self._cfg = cfg
        elif stage is not None and not isinstance(stage, str) and cfg is None:
            self._cfg = stage
        if self._train_loader is None and self._cfg is not None:
            dm_name = (
                self._cfg.get("data_module") if hasattr(self._cfg, "get") else getattr(self._cfg, "data_module", None)
            )
            if hasattr(dm_name, "get"):
                dm_name = dm_name.get("name") or dm_name.get("class") or dm_name.get("type")
            # Excluding this class's own name too: a config naming the
            # subclass in use would otherwise delegate to itself forever.
            if dm_name and dm_name not in (type(self).__name__, "SupervisedDataModule"):
                from src.app.core.paradigm_loader import get_component

                try:
                    cls = get_component(dm_name)
                except KeyError:
                    cls = None
                if cls is not None:
                    self._delegate = cls(self._cfg)
                    self.input_dim = getattr(self._delegate, "input_dim", 64)
                    return

            # Generic fallback for an offline dataset: a labelled .npz. Resolve it
            # here rather than requiring a data_module key that the config panel
            # cannot set. Staying undelegated is fine when loaders were supplied
            # directly instead.
            try:
                from src.app.core.paradigm_loader import get_component

                delegate = get_component(self.npz_delegate)(self._cfg)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                # A dataset the config names but that cannot be loaded is worth saying out loud:
                # the run would otherwise fail later with only "train_loader has not been set".
                level = logging.WARNING if _names_a_dataset(self._cfg) else logging.DEBUG
                log.log(level, "No .npz dataset for this config: %s", exc)
            else:
                self._delegate = delegate
                self.input_dim = getattr(delegate, "input_dim", 64)

    def train_dataloader(self) -> DataLoader:
        if self._delegate is not None:
            return self._delegate.train_dataloader()
        if self._train_loader is None:
            raise RuntimeError("train_loader has not been set or loaded.")
        return self._train_loader

    def val_dataloader(self) -> DataLoader | None:
        if self._delegate is not None:
            return self._delegate.val_dataloader()
        return self._val_loader

    def test_dataloader(self) -> DataLoader | None:
        if self._delegate is not None:
            return self._delegate.test_dataloader()
        return self._test_loader

    def set_loaders(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
    ) -> None:
        """Set or update active loaders (e.g. for cross-validation fold iterations)."""
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader
        self._delegate = None


# ---------------------------------------------------------------------------
# 2. Generic Classification Evaluation Protocol
# ---------------------------------------------------------------------------


@register_component("ClassificationEvalProtocol")
class ClassificationEvalProtocol(BaseEvalProtocol):
    """Domain-independent binary classification evaluation protocol.

    Computes:
    - AUC-ROC
    - AUPRC (Area Under Precision-Recall Curve)
    - F1-opt (optimal threshold determined on validation)
    - F1-max (maximum achievable F1 on test curve)
    - F1-0.5 (standard 0.5 decision threshold)
    """

    def evaluate(
        self,
        model: nn.Module,
        data_source: Any,
        cfg: Any = None,
        device: torch.device | str | None = None,
        opt_thresh: float | None = None,
    ) -> dict[str, float]:
        """Evaluate a classification model against a DataLoader.

        Args:
            model: PyTorch Module or LightningModule.
            data_source: PyTorch DataLoader yielding batches.
            cfg: Optional experiment config.
            device: Target torch device.
            opt_thresh: Optional precomputed optimal decision threshold.

        Returns:
            Dictionary with metrics: auc, auprc, f1_opt, f1_max, f1_05, bce_loss.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        # Scoring must not leave the model in eval mode: a caller that evaluates a
        # checkpoint and then keeps training would lose dropout without saying so.
        # Restored before the early returns below, which no longer touch the model.
        was_training = model.training
        model.eval()
        model.to(device)

        all_probs: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        try:
            with torch.no_grad():
                for batch in data_source:
                    x, y, kwargs = self._unpack_batch(batch, device)
                    logits = self._forward_model(model, x, kwargs)

                    if logits.ndim > 1 and logits.shape[-1] > 1:
                        probs = torch.softmax(logits, dim=-1)[:, 1]
                    else:
                        probs = torch.sigmoid(logits).view(-1)

                    all_probs.append(probs.detach().cpu().numpy())
                    all_targets.append(y.detach().cpu().numpy().reshape(-1))
        finally:
            model.train(was_training)

        if not all_targets or len(all_targets[0]) == 0:
            return {"auc": 0.0, "auprc": 0.0, "f1_opt": 0.0, "f1_max": 0.0, "f1_05": 0.0}

        y_true = np.concatenate(all_targets).astype(np.int32)
        y_prob = np.concatenate(all_probs).astype(np.float32)

        # Guard against single-class evaluation splits
        if len(np.unique(y_true)) < 2:
            log.warning("Evaluation batch contains only one class. Returning neutral metrics.")
            return {"auc": 0.5, "auprc": 0.0, "f1_opt": 0.0, "f1_max": 0.0, "f1_05": 0.0}

        auc_roc = float(roc_auc_score(y_true, y_prob))
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        auprc_val = float(auc(recalls, precisions))

        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        f1_max_val = float(np.max(f1_scores))

        if opt_thresh is None:
            best_idx = int(np.argmax(f1_scores))
            opt_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

        preds_opt = (y_prob >= opt_thresh).astype(np.int32)
        f1_opt_val = float(f1_score(y_true, preds_opt, zero_division=0))

        preds_05 = (y_prob >= 0.5).astype(np.int32)
        f1_05_val = float(f1_score(y_true, preds_05, zero_division=0))

        return {
            "auc": auc_roc,
            "auprc": auprc_val,
            "f1_opt": f1_opt_val,
            "f1_max": f1_max_val,
            "f1_05": f1_05_val,
            "opt_thresh": float(opt_thresh),
        }

    def _unpack_batch(self, batch: Any, device: torch.device):
        """Unpack standard and sequence batches generically."""
        kwargs: dict[str, Any] = {}
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 4:
                # Sequence format: (x, y, lengths, padding_mask, ...)
                x, y, lengths, padding_mask = batch[:4]
                kwargs["lengths"] = lengths.to(device) if lengths is not None else None
                kwargs["padding_mask"] = padding_mask.to(device) if padding_mask is not None else None
            elif len(batch) >= 2:
                x, y = batch[:2]
            else:
                x = batch[0]
                y = torch.zeros(len(x))
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        return x.to(device), y.to(device), kwargs

    def _forward_model(self, model: nn.Module, x: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
        """Call model forward pass passing matching keyword arguments."""
        sig = inspect.signature(model.forward)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
        return model(x, **filtered_kwargs)


# ---------------------------------------------------------------------------
# 3. Generic Supervised Runner
# ---------------------------------------------------------------------------


@register_component("SupervisedRunner")
class SupervisedRunner(BaseParadigmRunner):
    """Domain-independent runner for supervised learning tasks.

    Provides a standard training loop utilizing PyTorch Lightning Trainer,
    handling optimizer stepping, epoch scheduling, and metric logging.
    """

    def run(self, cfg, context):
        from src.app.core.execution import execute_backend
        return execute_backend(cfg, context)
