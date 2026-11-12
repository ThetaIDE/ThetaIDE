"""Generic unsupervised learning paradigm components.

Implements domain-independent interfaces for representation learning on
unlabeled data:
- UnsupervisedDataModule: reuses the supervised loaders and discards the labels.
- ReconstructionEvalProtocol: label-free evaluation (reconstruction error plus
  internal clustering quality on the learned latent space).
- UnsupervisedRunner: the standard PyTorch Lightning loop.

The distinction from `supervised` is the training signal, not the data: both
read a static dataset, but nothing here consults `y`, so the same trajectory-style
loaders serve an autoencoder without a labeled target.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.app.core.interfaces import BaseEvalProtocol
from src.app.core.paradigm_impls.base.supervised import SupervisedDataModule, SupervisedRunner
from src.app.core.paradigm_loader import register_component

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Unsupervised Data Module
# ---------------------------------------------------------------------------


@register_component("UnsupervisedDataModule")
class UnsupervisedDataModule(SupervisedDataModule):
    """Loads the same static datasets as `supervised`, without using the labels.

    Subclassing rather than reimplementing keeps one copy of delegation to
    domain data modules discovered through the component registry.
    Batches still carry `y` because the loaders are shared; the unsupervised
    model and eval protocol simply never read it.
    """

    #: Batches are (x, y, ...); nothing downstream of here consults index 1.
    label_index = 1

    #: The unlabelled loader: letting labels into the batch here would make the
    #: run supervised by accident.
    npz_delegate = "TabularNpzDataModule"


# ---------------------------------------------------------------------------
# 2. Label-free Evaluation Protocol
# ---------------------------------------------------------------------------


@register_component("ReconstructionEvalProtocol")
class ReconstructionEvalProtocol(BaseEvalProtocol):
    """Evaluates a representation model without any labels.

    Reports:
    - recon_mse / recon_mae: how well the model reconstructs its own input.
    - silhouette / davies_bouldin / calinski_harabasz: internal clustering
      quality of the latent space under k-means, which is the usual stand-in
      for accuracy when no ground truth exists.

    Clustering metrics are omitted rather than faked when the latent space has
    too few samples to partition.
    """

    def evaluate(
        self,
        model: Any,
        data_source: Any,
        cfg: Any = None,
        device: torch.device | str | None = None,
        n_clusters: int = 2,
    ) -> dict[str, float]:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        # Evaluating must not leave the model in eval mode: a caller that scores
        # a checkpoint and then keeps training would silently lose dropout.
        was_training = model.training
        model.eval()
        model.to(device)

        sq_err_sum, abs_err_sum, element_count = 0.0, 0.0, 0
        latents: list[np.ndarray] = []

        try:
            with torch.no_grad():
                for batch in data_source:
                    x, kwargs = self._unpack_batch(batch, device)
                    recon, latent = self._forward_model(model, x, kwargs)

                    diff = (recon - x).reshape(-1)
                    sq_err_sum += float(torch.sum(diff**2).item())
                    abs_err_sum += float(torch.sum(torch.abs(diff)).item())
                    element_count += diff.numel()

                    if latent is not None:
                        latents.append(latent.reshape(latent.shape[0], -1).detach().cpu().numpy())
        finally:
            model.train(was_training)

        if element_count == 0:
            return {"recon_mse": 0.0, "recon_mae": 0.0}

        metrics = {
            "recon_mse": sq_err_sum / element_count,
            "recon_mae": abs_err_sum / element_count,
        }
        metrics.update(self._cluster_metrics(latents, n_clusters))
        return metrics

    def _cluster_metrics(self, latents: list[np.ndarray], n_clusters: int) -> dict[str, float]:
        """Internal clustering scores for the latent space, or {} if not computable."""
        if not latents:
            return {}
        z = np.concatenate(latents).astype(np.float64)
        # k-means needs more samples than clusters, and the scores need at least
        # two populated clusters to mean anything.
        if z.shape[0] <= n_clusters or n_clusters < 2:
            log.warning("Latent space has %d samples; skipping clustering metrics.", z.shape[0])
            return {}
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

            labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(z)
            if len(np.unique(labels)) < 2:
                return {}
            return {
                "silhouette": float(silhouette_score(z, labels)),
                "davies_bouldin": float(davies_bouldin_score(z, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(z, labels)),
                "n_clusters": float(n_clusters),
            }
        except Exception as exc:  # pragma: no cover - sklearn edge cases
            log.warning("Could not compute clustering metrics: %s", exc)
            return {}

    def _unpack_batch(self, batch: Any, device: torch.device):
        """Take only the inputs. A labeled batch is accepted and its `y` dropped."""
        kwargs: dict[str, Any] = {}
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 4:
                # Sequence format: (x, y, lengths, padding_mask, ...)
                x, _, lengths, padding_mask = batch[:4]
                kwargs["lengths"] = lengths.to(device) if lengths is not None else None
                kwargs["padding_mask"] = padding_mask.to(device) if padding_mask is not None else None
            else:
                x = batch[0]
        elif isinstance(batch, torch.Tensor):
            x = batch
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        return x.to(device), kwargs

    def _forward_model(self, model: Any, x: torch.Tensor, kwargs: dict[str, Any]):
        """Return (reconstruction, latent). Latent is None if the model exposes none."""
        import inspect

        encode = getattr(model, "encode", None)
        latent = None
        if callable(encode):
            sig = inspect.signature(encode)
            latent = encode(x, **{k: v for k, v in kwargs.items() if k in sig.parameters and v is not None})

        sig = inspect.signature(model.forward)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
        out = model(x, **filtered)
        if isinstance(out, (tuple, list)):
            recon = out[0]
            if latent is None and len(out) > 1:
                latent = out[1]
        else:
            recon = out
        return recon, latent


# ---------------------------------------------------------------------------
# 3. Unsupervised Runner
# ---------------------------------------------------------------------------


@register_component("UnsupervisedRunner")
class UnsupervisedRunner(SupervisedRunner):
    """Runs the standard Lightning training loop for representation learning.

    The loop is identical to the supervised one - the difference lives in the
    model's own loss, which reconstructs its input instead of predicting a
    label - so this subclasses rather than duplicates it.
    """

    def run(self, cfg, context):
        from src.app.core.execution import execute_backend
        return execute_backend(cfg, context)
