"""Independent, CPU-friendly reconstruction of the MT-TICC + EM-EDM stages.

This does not import or silently replace the incomplete original project.
See README.md for choices that differ from the supplied research scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn import functional as F

LOG = logging.getLogger(__name__)


def toeplitz_projection(a: np.ndarray, features: int) -> np.ndarray:
    """Orthogonal projection onto symmetric block-Toeplitz matrices."""
    a = (a + a.T) / 2
    p = features
    if len(a) % p or a.shape != (len(a),len(a)):
        raise ValueError('Precision dimensions must be a multiple of feature count')
    if len(a) != 2*p:
        blocks=len(a)//p
        result=np.empty_like(a)
        for lag in range(blocks):
            shared=sum(a[i*p:(i+1)*p,(i+lag)*p:(i+lag+1)*p]
                       for i in range(blocks-lag))/(blocks-lag)
            for i in range(blocks-lag):
                result[i*p:(i+1)*p,(i+lag)*p:(i+lag+1)*p]=shared
                result[(i+lag)*p:(i+lag+1)*p,i*p:(i+1)*p]=shared.T
        return result
    diagonal = (a[:p, :p] + a[p:, p:]) / 2
    cross = a[:p, p:]
    return np.block([[diagonal, cross], [cross.T, diagonal]])


def sparse_toeplitz_precision(cov: np.ndarray, features: int, penalty: float,
                             iterations: int = 500) -> tuple[np.ndarray, dict]:
    """ADMM graphical lasso with a block-Toeplitz precision constraint.

    Objective: -logdet(theta) + trace(cov @ theta) + penalty*||theta||_1.
    Diagonal entries are unpenalized. A covariance ridge is applied by the caller.
    """
    n = len(cov)
    z, u = np.eye(n), np.zeros_like(cov)
    rho = 1.0
    converged = False
    for iteration in range(iterations):
        values, vectors = np.linalg.eigh(rho * (z - u) - cov)
        theta = (vectors * ((values + np.sqrt(values**2 + 4 * rho)) / (2 * rho))) @ vectors.T
        projected = toeplitz_projection(theta + u, features)
        previous = z
        z = np.sign(projected) * np.maximum(np.abs(projected) - penalty / rho, 0)
        np.fill_diagonal(z, np.diag(projected))
        u += theta - z
        primal = float(np.linalg.norm(theta - z))
        dual = float(rho * np.linalg.norm(z - previous))
        if max(primal, dual) < 1e-4 * n:
            converged = True
            break
        if (iteration + 1) % 10 == 0:
            # Residual balancing is important for high-precision, low-variance
            # clusters; a fixed rho otherwise stalls one side of the constraint.
            if primal > 10 * dual and rho < 1e4:
                rho *= 2
                u /= 2
            elif dual > 10 * primal and rho > 1e-4:
                rho /= 2
                u *= 2
    # A diagonal shift preserves both symmetry and the Toeplitz constraint.
    min_eigenvalue = float(np.linalg.eigvalsh(z).min())
    shift = max(0.0, 1e-5 - min_eigenvalue)
    z = z + shift * np.eye(n)
    return z, dict(iterations=iteration + 1, converged=converged,
                   primal_residual=primal, dual_residual=dual, pd_shift=shift)


def viterbi(cost: np.ndarray, switch_penalty: float) -> np.ndarray:
    """Minimum-cost stage sequence, with a fixed penalty for each switch.

    Call separately for every patient. Time is measured in row-index units here;
    the NPZ has no timestamps, so an irregular-time decay is not inferred.
    """
    if not len(cost):
        return np.empty(0, dtype=np.int64)
    k = cost.shape[1]
    previous = cost[0].copy()
    back = np.zeros((len(cost), k), dtype=np.int64)
    for t in range(1, len(cost)):
        best = int(previous.argmin())
        stay = previous
        change = previous[best] + switch_penalty
        change_is_better = change < stay
        back[t] = np.where(change_is_better, best, np.arange(k))
        previous = cost[t] + np.minimum(stay, change)
    labels = np.empty(len(cost), dtype=np.int64)
    labels[-1] = previous.argmin()
    for t in range(len(cost) - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    return labels


def stack_windows(episodes: list[np.ndarray], window_rows: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Causal [previous, current] windows; repeat first observation at an edge."""
    if type(window_rows) is not int or window_rows < 1:
        raise ValueError('window_rows must be a positive integer')
    offsets = np.r_[0, np.cumsum([len(x) for x in episodes])]
    if window_rows != 2:
        windows=[np.concatenate([x[np.maximum(np.arange(len(x))-lag,0)]
                 for lag in range(window_rows-1,-1,-1)],axis=1) for x in episodes]
        return np.concatenate(windows),offsets
    windows = [np.concatenate([np.vstack([x[:1], x[:-1]]), x], axis=1) for x in episodes]
    return np.concatenate(windows), offsets


class MultiSeriesTICC:
    """Fixed-row-spacing, multi-series TICC with two-observation windows."""

    def __init__(self, clusters=5, penalty=0.01, switch_penalty=4.0,
                 iterations=12, seed=42):
        self.clusters, self.penalty = clusters, penalty
        self.switch_penalty, self.iterations, self.seed = switch_penalty, iterations, seed
        self.history: list[dict] = []
        self.observer = None

    def costs(self, windows):
        cost = []
        for mean, precision in zip(self.means, self.precisions):
            delta = windows - mean
            # Cholesky avoids constructing one quadratic-form matrix per row.
            transformed = delta @ np.linalg.cholesky(precision)
            cost.append(0.5 * ((transformed**2).sum(axis=1) - np.linalg.slogdet(precision)[1]))
        return np.column_stack(cost)

    def assign(self, windows, offsets):
        costs = self.costs(windows)
        return np.concatenate([viterbi(costs[a:b], self.switch_penalty)
                               for a, b in zip(offsets[:-1], offsets[1:])])

    def fit(self, episodes):
        windows, offsets = stack_windows(episodes)
        p = windows.shape[1] // 2
        if len(windows) < self.clusters * 3:
            raise ValueError("Too few training observations for the requested stage count")
        labels = KMeans(self.clusters, n_init=5, random_state=self.seed).fit_predict(windows)
        for iteration in range(self.iterations):
            means, precisions, diagnostics = [], [], []
            for cluster in np.unique(labels):
                points = windows[labels == cluster]
                mean = points.mean(axis=0)
                delta = points - mean
                cov = delta.T @ delta / max(1, len(points)) + 0.05 * np.eye(2 * p)
                precision, diag = sparse_toeplitz_precision(cov, p, self.penalty)
                means.append(mean)
                precisions.append(precision)
                diagnostics.append(diag)
            self.means, self.precisions = np.array(means), np.array(precisions)
            new_labels = self.assign(windows, offsets)
            # If a stage is unused, the next M-step removes it and reindexes it.
            change = float((new_labels != labels).mean())
            self.history.append(dict(iteration=iteration, changed_fraction=change,
                                     counts=np.bincount(new_labels, minlength=len(means)).tolist(),
                                     precision_solver=diagnostics))
            LOG.info("TICC %d: stage counts=%s; changed=%.3f", iteration + 1,
                     self.history[-1]['counts'], change)
            if self.observer is not None:
                self.observer('ticc', dict(self.history[-1]))
            labels = new_labels
            if change < 0.001:
                break
        return self

    def predict(self, episodes):
        windows, offsets = stack_windows(episodes)
        labels = self.assign(windows, offsets)
        return [labels[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


@dataclass
class Segment:
    patient: int
    start: int  # Global flattened row index, inclusive.
    stop: int   # Exclusive.
    stage: int
    split: str


def make_segments(stage_sequences, offsets, splits):
    result = []
    for patient, (labels, split) in enumerate(zip(stage_sequences, splits)):
        bounds = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            result.append(Segment(patient, int(offsets[patient] + a),
                                  int(offsets[patient] + b), int(labels[a]), str(split)))
    return result


def fit_edm(model, observations, actions, updates=250, sgld_steps=10,
            density_weight=0.1, buffer=None, observer=None):
    """Cross-entropy + energy distribution matching, with persistent SGLD.

    All states are train-fitted and scaled to [-1,1]. The bounded state space,
    energy regularizer, uniform row sampling, and warm starts are demo choices.
    """
    x = torch.as_tensor(observations, dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.long)
    if not len(x):
        raise ValueError("Cannot fit a policy without valid training rows")
    if buffer is None:
        buffer = torch.rand(2048, x.shape[1]) * 2 - 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    losses = []
    for update in range(updates):
        ix = torch.randint(len(x), (128,))
        bx, by = x[ix], y[ix]
        bi = torch.randperm(len(buffer))[:128]
        negative = buffer[bi].clone()
        reset = torch.rand(len(negative)) < 0.05
        negative[reset] = torch.rand(int(reset.sum()), x.shape[1]) * 2 - 1
        for _ in range(sgld_steps):
            negative = negative.detach().requires_grad_(True)
            energy = -torch.logsumexp(model(negative), dim=1)
            gradient = torch.autograd.grad(energy.sum(), negative)[0]
            negative = (negative - 0.01 * gradient + 0.01 * torch.randn_like(negative)).clamp(-1, 1)
        negative = negative.detach()
        buffer[bi] = negative
        logits = model(bx)
        positive_energy = -torch.logsumexp(logits, dim=1)
        negative_energy = -torch.logsumexp(model(negative), dim=1)
        ce = F.cross_entropy(logits, by)
        density = positive_energy.mean() - negative_energy.mean()
        loss = ce + density_weight * density + 0.001 * (positive_energy.square().mean()
                                                       + negative_energy.square().mean())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite EDM loss")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if update in (0, updates - 1):
            losses.append(dict(update=update, loss=float(loss.detach()),
                               cross_entropy=float(ce.detach()), density=float(density.detach())))
        if observer is not None and (update % max(1, updates // 10) == 0 or update == updates - 1):
            observer('edm', dict(update=update + 1, loss=float(loss.detach()),
                cross_entropy=float(ce.detach()), density=float(density.detach()),
                sgld_gradient_norm=float(gradient.detach().norm(dim=1).mean()),
                negative_energy=float(negative_energy.detach().mean())))
    model.eval()
    return buffer.detach(), losses


def segment_log_scores(models, observations, actions, valid, segments):
    """Sum conditional log probabilities within each segment (no underflow)."""
    x = torch.as_tensor(observations, dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.long)
    scores = np.empty((len(segments), len(models)), dtype=np.float64)
    for k, model in enumerate(models):
        with torch.no_grad():
            rows = np.concatenate([F.log_softmax(model(batch), dim=1).numpy()
                                   for batch in x.split(8192)])
        observed = rows[np.arange(len(y)), y.numpy()] * valid
        prefix = np.r_[0.0, np.cumsum(observed, dtype=np.float64)]
        scores[:, k] = [prefix[s.stop] - prefix[s.start] for s in segments]
    return scores


def responsibilities(scores, priors):
    joint = scores + np.log(np.maximum(priors, 1e-15))
    normalizer = logsumexp(joint, axis=1)
    return np.exp(joint - normalizer[:, None]), normalizer


class EMEDM:
    """Historical demo mixture; not equivalent to original EM_EDM_utils.py."""

    def __init__(self, groups=3, iterations=6, updates=250, sgld_steps=10,
                 density_weight=0.1, seed=42, min_group_segments=5, *, policy_factory):
        if not callable(policy_factory):
            raise ValueError("EMEDM requires a policy factory supplied by the selected model")
        self.policy_factory = policy_factory
        self.groups, self.iterations, self.updates = groups, iterations, updates
        self.sgld_steps, self.density_weight, self.seed = sgld_steps, density_weight, seed
        self.min_group_segments = min_group_segments
        self.history = []
        self.observer = None

    def fit(self, observations, actions, valid, segments):
        torch.manual_seed(self.seed)
        usable = [s for s in segments if s.split == 'train' and valid[s.start:s.stop].any()]
        if len(usable) < self.groups:
            raise ValueError("Too few training segments for the requested policy count")
        # Treatment rate is used only to initialize membership, never as policy input.
        summaries = []
        for s in usable:
            keep = valid[s.start:s.stop]
            summaries.append(np.r_[0.15 * observations[s.start:s.stop][keep].mean(axis=0),
                                   4 * actions[s.start:s.stop][keep].mean()])
        labels = KMeans(self.groups, n_init=10, random_state=self.seed).fit_predict(summaries)
        self.models = [self.policy_factory(observations.shape[1]) for _ in range(self.groups)]
        buffers = [None] * self.groups
        self.component_ids = list(range(self.groups))
        self.optimizer_updates = 0
        for iteration in range(self.iterations):
            counts = np.bincount(labels, minlength=len(self.models))
            active = np.flatnonzero(counts >= self.min_group_segments)
            if not len(active):
                active = np.array([int(counts.argmax())])
            # Reassign observations from removed components using the retained
            # policies, then renumber. No segment or patient is discarded.
            if len(active) < len(self.models):
                LOG.info('Removing policy groups with fewer than %d training segments; retaining %s',
                         self.min_group_segments, active.tolist())
                scores = segment_log_scores([self.models[k] for k in active],
                                            observations, actions, valid, usable)
                removed = ~np.isin(labels, active)
                labels[removed] = active[scores[removed].argmax(axis=1)]
            self.component_ids = [self.component_ids[i] for i in active]
            self.models = [self.models[i] for i in active]
            buffers = [buffers[i] for i in active]
            labels = np.searchsorted(active, labels)
            self.priors = np.bincount(labels, minlength=len(active)) / len(labels)
            losses = []
            for k, model in enumerate(self.models):
                rows = np.concatenate([np.arange(s.start, s.stop)[valid[s.start:s.stop]]
                                       for i, s in enumerate(usable) if labels[i] == k])
                def observe(event, payload, policy=self.component_ids[k], base=self.optimizer_updates):
                    if self.observer is not None:
                        self.observer(event, {**payload, 'round': iteration + 1, 'policy': policy, 'optimizer_updates': base + payload['update']})
                buffers[k], loss = fit_edm(model, observations[rows], actions[rows],
                                          self.updates, self.sgld_steps, self.density_weight, buffers[k],
                                          observer=observe if self.observer is not None else None)
                self.optimizer_updates += self.updates
                losses.append(loss)
            scores = segment_log_scores(self.models, observations, actions, valid, usable)
            probs, evidence = responsibilities(scores, self.priors)
            new_labels = probs.argmax(axis=1)
            change = float((new_labels != labels).mean())
            history = dict(iteration=iteration, changed_fraction=change,
                           fitted_counts=np.bincount(labels, minlength=len(active)).tolist(),
                           assigned_counts=np.bincount(new_labels, minlength=len(active)).tolist(),
                           conditional_log_likelihood=float(evidence.sum()), training_losses=losses)
            history['component_ids'] = list(self.component_ids)
            history['priors'] = self.priors.tolist()
            history['optimizer_updates'] = self.optimizer_updates
            self.history.append(history)
            if self.observer is not None:
                self.observer('em', {**history, 'learner': self, 'probabilities': probs})
            LOG.info("EM-EDM %d: policy counts=%s; changed=%.3f; logL=%.1f",
                     iteration + 1, history['assigned_counts'], change, evidence.sum())
            labels = new_labels
            next_counts = np.bincount(new_labels, minlength=len(active))
            needs_pruning = len(active) > 1 and (next_counts < self.min_group_segments).any()
            if change < 0.005 and not needs_pruning:
                break
        return self

    def predict(self, observations, actions, valid, segments):
        scores = segment_log_scores(self.models, observations, actions, valid, segments)
        return responsibilities(scores, self.priors)
