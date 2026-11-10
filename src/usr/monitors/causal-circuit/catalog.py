def describe(config, context):
    identity = config["instance"]
    outputs = [{"name": f"interp/{identity}/{key}", "label": key.replace("_", " "),
             "group": f"Causal interventions · {identity}", "requires": ["probe_forward.v1", "module_interventions.v1", "post_update.v1"]}
            for key in ("heldout_ablation_effect", "heldout_ablation_se", "random_ablation_effect", "complement_ablation_effect",
                        "patch_recovery_valid_samples", "candidate_units", "selected_units", "heldout_patch_recovery", "tested_paths", "heldout_mediation_effect")]

    outputs.append({"name": f"interp/{identity}/interventions", "label": "Intervention evidence", "group": f"Visuals · {identity}",
        "requires": ['probe_forward.v1', 'post_update.v1', 'module_interventions.v1'], "kind": "intervention_graph", "visual_schema": "theta.visual-output/v1",
        "artifact_schema": "theta.intervention-graph/v1", "axes": ['candidate', 'behavior'],
        "value_semantics": "Measured held-out action-score effects; indirect paths are not synapses", "retention": "snapshots"})
    return outputs
