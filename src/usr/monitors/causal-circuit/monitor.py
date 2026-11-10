"""Bounded neuron ablation, donor patching and held-out candidate-circuit tests."""
import numpy as np
import torch
from torch import nn

from src.app.monitoring.api import Result
from src.app.monitoring.measurements import action_difference, capture, layers_for, replace_units, validate_params


class CausalCircuit:
    @staticmethod
    def validate(params):
        validate_params(params, {"layers", "max_units", "top_k", "random_controls", "action", "reference_action", "baseline", "granularity", "max_edges"},
                        {"max_units": (2, 64), "top_k": (1, 16), "random_controls": (1, 16), "action": (0, 10000), "reference_action": (0, 10000), "max_edges": (0, 32)})
        if params.get("baseline", "mean") not in ("mean", "zero"):
            raise ValueError("baseline must be mean or zero")
        if params.get("granularity", "unit") not in ("unit", "layer"):
            raise ValueError("granularity must be unit or layer")
        if params.get("top_k", 4) > params.get("max_units", 16):
            raise ValueError("top_k cannot exceed max_units")

    def __init__(self, layers=None, max_units=16, top_k=4, random_controls=4, action=0, reference_action=1, baseline="mean", granularity="unit", max_edges=8):
        self.validate({"max_units": max_units, "top_k": top_k, "random_controls": random_controls,
                       "action": action, "reference_action": reference_action, "baseline": baseline, "granularity": granularity, "max_edges": max_edges,
                       **({"layers": layers} if layers is not None else {})})
        self.layers, self.max_units, self.top_k = layers, max_units, top_k
        self.random_controls, self.action, self.reference_action, self.baseline = random_controls, action, reference_action, baseline
        self.granularity = granularity
        self.max_edges = max_edges

    def setup(self, context):
        if len(context.probes.observations) < 8:
            raise ValueError("Circuit analysis requires at least eight probe samples")
        layers = layers_for(context, self.layers)
        scores, _ = capture(context, layers, context.observations,
                            max_features=512 if self.granularity == "layer" else self.max_units,
                            require_full=self.granularity == "layer")
        action_difference(scores, self.action, self.reference_action)

    @torch.no_grad()
    def collect(self, context):
        layers = layers_for(context, self.layers)
        obs = context.observations
        scores, clean = capture(context, layers, obs, max_features=512 if self.granularity == "layer" else self.max_units,
                                require_full=self.granularity == "layer")
        difference = lambda values: action_difference(values, self.action, self.reference_action)
        baseline_scores = difference(scores)
        # Deterministic candidate coverage across layers; never silently treats
        # a sampled subset as the entire network's circuit.
        candidates = [(name, unit) for unit in range(self.max_units) for name, _ in layers
                      if unit < clean[name].shape[1]][:self.max_units]
        if self.granularity == "layer":
            candidates = [(name, None) for name, _ in layers][:self.max_units]
        if not candidates:
            raise ValueError("Circuit analysis requires candidate units or layers")
        split = len(obs) // 2
        if split < 4:
            raise ValueError("Circuit analysis requires at least eight probe samples")
        heldout = slice(split, None)
        replacements = {name: values[:split].mean(0, keepdim=True).expand(len(obs), -1) if self.baseline == "mean" else torch.zeros_like(values)
                        for name, values in clean.items()}

        def interventions(units, source):
            grouped = {}
            for name, unit in units:
                grouped.setdefault(name, []).extend(range(clean[name].shape[1]) if unit is None else [unit])
            return {name: (indices, source[name][:, indices]) for name, indices in grouped.items()}

        def evaluate(inputs, units, source):
            with replace_units(layers, interventions(units, source)):
                return context.access.scores(inputs).detach().clone()

        ablated = torch.stack([difference(evaluate(obs, [candidate], replacements)) for candidate in candidates], 1)
        effects = baseline_scores[:, None] - ablated
        discovery = effects[:split].abs().mean(0).cpu().numpy()
        ranking = np.argsort(-discovery, kind="stable")
        selected_indices = ranking[:min(self.top_k, len(candidates))]
        selected = [candidates[int(index)] for index in selected_indices]
        complement = [item for item in candidates if item not in selected]
        ablation = difference(evaluate(obs, selected, replacements))
        complement_ablation = difference(evaluate(obs, complement, replacements))
        # Donors stay within each split, so reference data cannot leak across
        # discovery/validation. A cyclic shift never pairs a sample with itself.
        donors = torch.cat((torch.roll(torch.arange(split, device=obs.device), 1),
                            torch.roll(torch.arange(split, len(obs), device=obs.device), 1)))
        donor_obs = obs[donors]
        corrupt = difference(context.access.scores(donor_obs))
        patched = difference(evaluate(donor_obs, selected, clean))
        gap = baseline_scores - corrupt
        valid = gap[heldout].abs() > 1e-4
        recovery = ((patched[heldout] - corrupt[heldout])[valid] / gap[heldout][valid]).cpu().numpy()
        rng = np.random.default_rng(context.spec["probes"]["seed"])
        controls = []
        for _ in range(self.random_controls):
            indices = rng.choice(len(candidates), len(selected), replace=False)
            control = difference(evaluate(obs, [candidates[int(i)] for i in indices], replacements))
            controls.append(float((baseline_scores[heldout] - control[heldout]).abs().mean()))
        raw_effect = (baseline_scores[heldout] - ablation[heldout]).abs().cpu().numpy()
        metrics = {"heldout_ablation_effect": float(raw_effect.mean()),
                   "heldout_ablation_se": float(raw_effect.std(ddof=1) / np.sqrt(len(raw_effect))),
                   "random_ablation_effect": float(np.mean(controls)),
                   "complement_ablation_effect": float((baseline_scores[heldout] - complement_ablation[heldout]).abs().mean()),
                   "patch_recovery_valid_samples": float(valid.sum()), "candidate_units": float(len(candidates)),
                   "selected_units": float(len(selected))}
        if len(recovery):
            metrics["heldout_patch_recovery"] = float(recovery.mean())
        mediation_edges = []
        if isinstance(context.access.target(context.spec["target"]), nn.Sequential):
            order = {name: index for index, (name, _) in enumerate(layers)}
            pairs = [(source, target) for source in selected for target in selected if order[source[0]] < order[target[0]]]
            for source, target in pairs[:self.max_edges]:
                with replace_units(layers, interventions([source], replacements)):
                    _, changed_values = capture(context, layers, obs,
                        max_features=512 if self.granularity == "layer" else self.max_units,
                        require_full=self.granularity == "layer")
                mediated = difference(evaluate(obs, [target], changed_values))
                with replace_units(layers, interventions([source], replacements)):
                    rescued = difference(evaluate(obs, [target], clean))
                source_index = candidates.index(source)
                source_ablated = baseline_scores - effects[:, source_index]
                mediation_edges.append({"source": f"{source[0]}:{source[1]}", "target": f"{target[0]}:{target[1]}",
                    "kind": "measured_path_mediation", "effect": float((baseline_scores[heldout] - mediated[heldout]).abs().mean()),
                    "rescue_effect": float((rescued[heldout] - source_ablated[heldout]).abs().mean())})
        metrics["tested_paths"] = float(len(mediation_edges))
        if mediation_edges:
            metrics["heldout_mediation_effect"] = float(np.mean([edge["effect"] for edge in mediation_edges]))
        nodes = [{"id": f"{name}:{unit}", "module": name, "unit": unit,
                  "selected": (name, unit) in selected, "discovery_effect": float(discovery[index]),
                  "heldout_effect": float(effects[heldout, index].abs().mean())}
                 for index, (name, unit) in enumerate(candidates)]
        graph = {"schema": "theta.intervention-graph/v1", "nodes": nodes + [{"id": "behavior", "label": f"action {self.action} minus {self.reference_action}"}],
                 "edges": [{"source": node["id"], "target": "behavior", "kind": "measured_total_intervention_effect", "effect": node["heldout_effect"]} for node in nodes] + mediation_edges,
                 "interpretation": "Behavior edges measure total effects. Path edges test downstream mediation/rescue in sequential targets and may include intermediate layers; they are not direct synaptic connections. Selected units are a candidate set; complement tests cover sampled candidates only."}
        return Result(metrics, {"baseline": baseline_scores.cpu().numpy(), "ablated": ablation.cpu().numpy(),
                      "complement_ablated": complement_ablation.cpu().numpy(), "donor_scores": corrupt.cpu().numpy(),
                      "patched": patched.cpu().numpy(), "donor_indices": donors.cpu().numpy(), "unit_effects": effects.cpu().numpy(),
                      "random_control_effects": np.asarray(controls)},
                      {"graph": graph, "discovery_samples": split, "heldout_samples": len(obs)-split,
                       "baseline": self.baseline, "patch": "clean activations into real donor observations",
                       "granularity": self.granularity, "complement_scope": "unselected sampled candidates; not all network units",
                       "forward_passes": 2 + len(candidates) + 3 + self.random_controls + 3 * len(mediation_edges),
                       "mediation_scope": "Selected pairs in sequential targets only; donor patching and ranking remain separate tests",
                       "axes": {"unit_effects": ["probe_sample", "candidate_unit"]}},
                      {"interventions": {"graph": graph}} if context.wants_output("interventions") else {})
