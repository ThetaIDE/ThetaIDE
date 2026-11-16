def describe(config, context):
    identity = config["instance"]
    return [{"name": f"interp/{identity}/network", "label": "Layered network",
        "group": f"Network · {identity}", "requires": ["probe_forward.v1", "post_update.v1"],
        "kind": "network_graph", "visual_schema": "theta.visual-output/v1",
        "artifact_schema": "theta.network-graph/v1", "axes": ["module", "unit", "probe_sample"],
        "value_semantics": "Weights, probe activations and linear local contributions; no causal claim",
        "retention": "snapshots"}]
