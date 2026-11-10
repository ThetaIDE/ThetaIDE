def describe(config, context):
    identity = config["instance"]
    outputs = [{"name": f"interp/{identity}/{key}", "label": key.replace("_", " "),
             "group": f"Activation evolution · {identity}", "requires": ["probe_forward.v1", "post_update.v1"]}
            for key in ("activation_norm", "effective_rank", "inactive_fraction", "saturation_fraction", "cka_initial", "cka_previous")]

    outputs.append({"name": f"interp/{identity}/heatmap", "label": "Activation heatmaps", "group": f"Visuals · {identity}",
        "requires": ['probe_forward.v1', 'post_update.v1'], "kind": "matrix", "visual_schema": "theta.visual-output/v1",
        "artifact_schema": "theta.activation-matrix/v1", "axes": ['probe_sample', 'feature'],
        "value_semantics": "Activation at the selected module output; model units", "retention": "snapshots"})
    return outputs
