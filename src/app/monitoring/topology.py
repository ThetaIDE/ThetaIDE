"""Bounded, explicit sequential topology adapter on isolated scoring forwards."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn

from .api import Result
from .measurements import validate_params


POINTWISE = (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU, nn.SiLU, nn.LeakyReLU,
             nn.ELU, nn.Identity, nn.Dropout)


def sequential_modules(access, target):
    """Override ModelAccess.topology_modules for architecture-owned adapters.

    The default supports an explicitly selected Sequential, not arbitrary graphs
    inferred from module registration order. No edges are guessed across branches.
    """
    module = access.target(target)
    if not isinstance(module, nn.Sequential):
        raise ValueError("Network viewer requires a Sequential target or an architecture-owned topology_modules adapter")
    result = []

    def visit(parent, prefix):
        for name, child in parent.named_children():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Sequential):
                visit(child, path)
            elif isinstance(child, (nn.Linear, nn.LayerNorm, *POINTWISE)):
                result.append((path, child))
            else:
                raise ValueError(f"Unsupported topology operation {path}: {type(child).__name__}; supply an architecture adapter")
    visit(module, "")
    if not result or not any(isinstance(m, nn.Linear) for _, m in result):
        raise ValueError("Network topology needs at least one Linear operation")
    return result


class NetworkStructure:
    @staticmethod
    def validate(params):
        validate_params(params, {"max_layers", "max_units", "max_edges"},
                        {"max_layers": (1, 32), "max_units": (1, 256), "max_edges": (1, 65536)})

    def __init__(self, max_layers=16, max_units=64, max_edges=4096):
        self.validate(dict(max_layers=max_layers, max_units=max_units, max_edges=max_edges))
        self.max_layers, self.max_units, self.max_edges = max_layers, max_units, max_edges
        self.fingerprint = None

    def setup(self, context):
        context.access.topology_modules(context.spec["target"])

    def collect(self, context):
        if not context.wants_output("network"):
            return Result()
        modules = context.access.topology_modules(context.spec["target"])
        # Fingerprint every operation, including omitted layers. Parameter values
        # deliberately do not participate: changing weights preserves unit IDs.
        definition = [{"module": name, "operation": repr(module),
                       "parameters": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in module.named_parameters()}}
                      for name, module in modules]
        fingerprint = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        if self.fingerprint is not None and self.fingerprint != fingerprint:
            raise ValueError("Network topology changed; start a new monitor reference")
        selected = modules[:self.max_layers]
        captured, handles = {}, []

        def hook(name):
            def save(module, inputs, output):
                if name in captured:
                    raise ValueError("Repeated topology module execution requires an architecture adapter")
                source = inputs[0]
                if not torch.is_tensor(output) or source.ndim != 2 or output.ndim != 2:
                    raise ValueError("Network topology requires flat [probe, unit] tensors")
                captured[name] = (source.detach().clone(), output.detach().clone())
            return save
        try:
            handles = [module.register_forward_hook(hook(name)) for name, module in selected]
            context.access.scores(context.observations)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != {name for name, _ in selected}:
            raise ValueError("Topology modules must execute on the selected scoring path")

        arrays = {"sample_ids": context.probes.sample_ids}
        layers, connections = [], []
        prefix = context.spec["target"]
        previous, edge_count, total_edges = None, 0, 0

        def values(tensor):
            data = tensor.detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(data).all():
                raise ValueError("Network snapshots require finite parameters and activations")
            return data

        for i, (name, module) in enumerate(selected):
            source, output = captured[name]
            if previous is not None and (previous.shape != source.shape or not torch.equal(previous, source)):
                raise ValueError("Operations between topology modules require an explicit architecture adapter")
            path = f"{prefix}.{name}" if prefix else name
            if i == 0:
                arrays["input"] = values(source[:, :self.max_units])
                layers.append({"id": f"{prefix}:input", "label": "Effective target input", "operation": "input",
                               "units": min(source.shape[1], self.max_units), "total_units": source.shape[1],
                               "activation": "input"})
            arrays[f"activation_{i}"] = values(output[:, :self.max_units])
            arrays[f"source_{i}"] = values(source[:, :self.max_units])
            layer = {"id": path, "label": path, "operation": repr(module),
                     "units": min(output.shape[1], self.max_units), "total_units": output.shape[1],
                     "activation": f"activation_{i}"}
            connection = {"source": layers[-1]["id"], "target": path,
                          "kind": "linear" if isinstance(module, nn.Linear) else "operation",
                          "source_values": f"source_{i}", "edges": []}
            if isinstance(module, nn.Linear):
                weights = values(module.weight[:self.max_units, :self.max_units])
                arrays[f"weights_{i}"] = weights
                arrays[f"bias_{i}"] = values(module.bias[:self.max_units]) if module.bias is not None else np.zeros(layer["units"], np.float32)
                layer["bias"] = f"bias_{i}"
                layer["has_bias"] = module.bias is not None
                connection["weights"] = f"weights_{i}"
                connection["orientation"] = "destination,source"
                total_edges += module.in_features * module.out_features
                candidates = ((src, dst) for dst in range(weights.shape[0]) for src in range(weights.shape[1]))
            elif isinstance(module, nn.LayerNorm):
                if len(module.normalized_shape) != 1 or source.shape != output.shape:
                    raise ValueError("Network LayerNorm requires one flat feature axis")
                connection["kind"] = "normalization"
                connection["dependency"] = "Coupled operation over the full layer, including omitted units; not per-unit synapses"
                if module.weight is not None:
                    arrays[f"scale_{i}"] = values(module.weight[:self.max_units])
                    layer["scale"] = f"scale_{i}"
                if module.bias is not None:
                    arrays[f"norm_bias_{i}"] = values(module.bias[:self.max_units])
                    layer["bias"] = f"norm_bias_{i}"
                candidates = ()
            else:
                if source.shape != output.shape:
                    raise ValueError("Pointwise topology operations must preserve unit identities")
                total_edges += output.shape[1]
                candidates = ((unit, unit) for unit in range(layer["units"]))
            for src, dst in candidates:
                if edge_count >= self.max_edges:
                    break
                connection["edges"].append([src, dst])
                edge_count += 1
            layers.append(layer)
            connections.append(connection)
            previous = output
        # Count omitted layers' known edges without allocating their tensors.
        width = previous.shape[1]
        for _, module in modules[len(selected):]:
            if isinstance(module, nn.Linear):
                total_edges += module.in_features * module.out_features
                width = module.out_features
            elif not isinstance(module, nn.LayerNorm):
                total_edges += width
        for row in layers:
            row["unit_ids"] = [f"{row['id']}:{unit}" for unit in range(row["units"])]
        graph = {"schema": "theta.network-graph/v1", "array_precision": "float32", "topology_sha256": fingerprint,
                 "layers": layers, "connections": connections, "sample_ids": "sample_ids",
                 "preprocessing": {"source": "Actual scoring-path input to selected target",
                                   "probe_metadata": context.probes.metadata,
                                   "scope": "Selected target only; operations outside it are not represented"},
                 "coverage": {"layers_recorded": len(selected), "layers_total": len(modules),
                              "edges_recorded": edge_count, "edges_total": total_edges,
                              "selection": "First layers and first units; destination-major unit edges up to budget. LayerNorm shown as a coupled operation group"},
                 "semantics": "Linear edges are physical parameters; operation links are separate. Local contribution is source activation × weight to preactivation; bias is separate. Neither establishes causal importance."}
        self.fingerprint = fingerprint
        return Result(arrays=arrays, visuals={"network": {"graph": graph}})

    def state_dict(self):
        return {"fingerprint": self.fingerprint}

    def load_state_dict(self, state):
        self.fingerprint = state["fingerprint"]
