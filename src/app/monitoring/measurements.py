"""Numerical primitives shared by diagnostic providers, independent of runners."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn


def linear_cka(first, second):
    x, y = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("CKA requires aligned [samples, features] arrays")
    x, y = x - x.mean(0), y - y.mean(0)
    numerator = np.square(x.T @ y).sum()
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(numerator / denominator) if denominator > 1e-15 else None


def effective_rank(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean(0)
    spectrum = np.linalg.svd(values, compute_uv=False)
    if spectrum.sum() <= 1e-15:
        return 0.0
    probabilities = spectrum / spectrum.sum()
    return float(np.exp(-(probabilities * np.log(np.maximum(probabilities, 1e-30))).sum()))


def layers_for(context, names=None, *, max_layers=16):
    target = context.access.target(context.spec["target"])
    available = dict(target.named_modules())
    if names:
        missing = set(names) - set(available)
        if missing:
            raise ValueError(f"Unknown target layers: {sorted(missing)}")
        selected = [(name, available[name]) for name in names]
    else:
        kinds = (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU, nn.SiLU, nn.LeakyReLU)
        selected = [(name, module) for name, module in available.items() if isinstance(module, kinds)]
        if not selected:
            selected = [(name, module) for name, module in available.items() if isinstance(module, nn.Linear)]
    if not selected:
        raise ValueError("Target has no supported activation/linear modules; supply an architecture adapter")
    if len(selected) > max_layers:
        raise ValueError(f"Target has {len(selected)} layers; select at most {max_layers} explicitly")
    return selected


def capture(context, layers, observations, *, max_features=128, require_full=False, coverage=None):
    values, handles = {}, []
    def hook(name):
        def save(module, inputs, output):
            if name in values:
                raise ValueError(f"Layer {name} executes repeatedly; v1 requires explicit single-execution targets")
            if not torch.is_tensor(output) or output.ndim != 2 or len(output) != len(observations):
                raise ValueError(f"Layer {name} must emit [samples, features]; recurrent/spatial targets need an adapter")
            if require_full and output.shape[1] > max_features:
                raise ValueError(f"Whole-layer intervention exceeds the {max_features}-feature budget")
            if coverage is not None:
                coverage[name] = int(output.shape[1])
            values[name] = output[:, :max_features].detach().clone()
        return save
    try:
        handles = [module.register_forward_hook(hook(name)) for name, module in layers]
        scores = context.access.scores(observations).detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    if set(values) != {name for name, _ in layers}:
        raise ValueError("Some selected layers were not executed by the model's scoring path")
    return scores, values


@contextmanager
def replace_units(layers, replacements):
    """Replace selected output units on an isolated model; always remove hooks."""
    handles = []
    try:
        for name, module in layers:
            if name not in replacements:
                continue
            units, values = replacements[name]
            def hook(module, inputs, output, units=units, values=values):
                changed = output.clone()
                changed[:, units] = values.to(output.device, output.dtype)
                return changed
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def action_difference(scores, action=0, reference=1):
    if action == reference or not 0 <= action < scores.shape[1] or not 0 <= reference < scores.shape[1]:
        raise ValueError("action and reference_action must be distinct valid action indices")
    return scores[:, action] - scores[:, reference]


def validate_params(params, allowed, limits):
    if set(params) - set(allowed):
        raise ValueError(f"Unknown parameters: {sorted(set(params) - set(allowed))}")
    for key, (low, high) in limits.items():
        if key in params and (type(params[key]) is not int or not low <= params[key] <= high):
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    if "layers" in params and (not isinstance(params["layers"], list) or not params["layers"]
                              or not all(isinstance(v, str) for v in params["layers"]) or len(set(params["layers"])) != len(params["layers"])):
        raise ValueError("layers must be a nonempty list of unique module paths")
