"""Fixed-probe activation evolution for compatible feedforward PyTorch models."""
import numpy as np
from torch import nn

from src.app.monitoring.api import Result
from src.app.monitoring.measurements import capture, effective_rank, layers_for, linear_cka, validate_params


class ActivationEvolution:
    @staticmethod
    def validate(params):
        validate_params(params, {"layers", "max_features", "max_layers"},
                        {"max_features": (1, 512), "max_layers": (1, 32)})

    def __init__(self, layers=None, max_features=128, max_layers=16):
        self.validate({"max_features": max_features, "max_layers": max_layers, **({"layers": layers} if layers is not None else {})})
        self.layers, self.max_features, self.max_layers = layers, max_features, max_layers
        self.initial, self.previous = {}, {}

    def setup(self, context):
        layers = layers_for(context, self.layers, max_layers=self.max_layers)
        capture(context, layers, context.observations, max_features=self.max_features)

    def collect(self, context):
        layers = layers_for(context, self.layers, max_layers=self.max_layers)
        coverage = {}
        _, activations = capture(context, layers, context.observations, max_features=self.max_features, coverage=coverage)
        if self.initial and set(activations) != set(self.initial):
            raise ValueError("Layer topology changed; start a new monitor reference")
        arrays, statistics, current = {}, [], {}
        for index, (name, module) in enumerate(layers):
            values = activations[name].cpu().numpy().astype(np.float32)
            current[name] = values
            if name in self.initial and values.shape != self.initial[name].shape:
                raise ValueError("Layer dimensions changed; start a new monitor reference")
            stats = {"module": name, "type": type(module).__name__, "features_recorded": values.shape[1],
                     "features_total": coverage[name],
                     "activation_norm": float(np.linalg.norm(values, axis=1).mean()),
                     "effective_rank": effective_rank(values),
                     "inactive_fraction": float(np.all(np.abs(values) < 1e-8, axis=0).mean())}
            if isinstance(module, nn.Tanh):
                stats["saturation_fraction"] = float((np.abs(values) >= .99).mean())
            elif isinstance(module, nn.Sigmoid):
                stats["saturation_fraction"] = float(((values <= .01) | (values >= .99)).mean())
            for key, reference in (("cka_initial", self.initial), ("cka_previous", self.previous)):
                if name in reference:
                    score = linear_cka(values, reference[name])
                    if score is not None:
                        stats[key] = score
            statistics.append(stats)
            if context.wants_output("heatmap"):
                arrays[f"layer_{index}"] = values
        metrics = {key: float(np.mean([row[key] for row in statistics if key in row]))
                   for key in ("activation_norm", "effective_rank", "inactive_fraction", "saturation_fraction", "cka_initial", "cka_previous")
                   if any(key in row for row in statistics)}
        self.initial = self.initial or {key: value.copy() for key, value in current.items()}
        self.previous = current
        visuals = {}
        if context.wants_output("heatmap"):
            arrays["sample_ids"] = context.probes.sample_ids
            visuals["heatmap"] = {"schema": "theta.activation-matrix/v1", "layers": [{"id": row["module"], "array": f"layer_{i}",
                "feature_ids": list(range(row["features_recorded"])), "coverage": f"{row['features_recorded']}/{row['features_total']} features, first indices"}
                for i, row in enumerate(statistics)], "sample_ids": "sample_ids"}
        return Result(metrics, arrays, {"layers": statistics, "axes": ["probe_sample", "feature"],
                      "feature_selection": f"first {self.max_features} features per layer", "aggregation": "mean over applicable layers",
                      "interpretation": "Representation measurements; no causal circuit claim"}, visuals)

    def state_dict(self):
        return {"initial": self.initial, "previous": self.previous}

    def load_state_dict(self, state):
        self.initial, self.previous = state["initial"], state["previous"]
