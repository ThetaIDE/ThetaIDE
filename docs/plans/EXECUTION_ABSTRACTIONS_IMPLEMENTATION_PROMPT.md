# Implementation prompt: execution contracts, then THEMES/AL

Implement `docs/plans/EXECUTION_ABSTRACTIONS_AND_THEMES_PLAN.md` in this repository. Read the complete revised plan, applicable AGENTS.md guidance, and relevant runtime/provider contracts before changing code. Treat this as implementation work, not another planning-only task.

First fix Theta's execution abstractions (phases A–D), then extract and integrate the TICC + EM-EDM reconstruction and deliver the initial AL experiments (phases E–G). Full inverse-reward/reward-regulated THEMES (H) is a later milestone; document its boundary accurately. General workflows are explicitly deferred: do not build a graph engine, workflow editor, generalized artifact routing, or meta-pipeline framework.

Execution requirements:

- Route every supported normal method/trial/replicate through one Theta-owned worker bootstrap that actually resolves and invokes the registered paradigm runner exactly once.
- Give the runner one resolved method execution. Keep experiment preparation, method expansion, site/queue/resource scheduling and plotting in the coordinator. Do not recurse into experiment dispatch from a runner.
- Define small versioned runner/context/result contracts; integrate Lightning and editable first-party SB3/d3rlpy adapters. Preserve supported learning behavior, backend-native checkpoints, alternate Python environments, telemetry, cancellation and isolated output ownership.
- Remove old execution-selector aliases and implicit fallback. Reject execution-selection `entrypoint`, method `runner`, `script`, and legacy `engine` at their relevant effective-config locations. No generic script adapter, old-precedence normalization or compatibility-only wrappers. Preserve provider manifest `entry_point`, which is implementation import metadata.
- Fail clearly on unresolved runners, incompatible selected providers, conflicting old selectors, and unsupported resume. Prove actual launcher dispatch, not just direct class construction. Identify evaluator invocation sites and test that each declared metric is computed once.

Local development policy:

- Start with the existing editable `src/usr/algorithms/sb3-dev` component. Inspect active source/profiles/manifests before editing; preserve unrelated work. Use or create an explicit editable d3rlpy dev component and local THEMES package with correct identities, schemas, capabilities and provenance.
- Update active integration recipes to select dev IDs explicitly. Never shadow production IDs or silently redirect production selectors. Do not assume algorithm copying is supported by the model deep-copy tool.
- Do not edit protected installed Hub payloads or retained package snapshots. Do not build, upload, publish or redeploy production Hub packages during this implementation. Maintain a separate production-source migration/release handoff listing changed files and outstanding release steps.
- Breaking old Hub component execution contracts is explicitly accepted. Add minimal execution-compatibility validation; reject incompatible selected packages with ID/version and concrete migration guidance, while allowing unrelated runs/catalog browsing. Do not keep bad runtime behavior to make old Hub packages pass.
- Provide a preview/apply migration tool for explicitly selected editable first-party configs. Unknown external scripts and ambiguous declarations require manual resolution. Historical configs remain inspectable without a promise of execution compatibility.

THEMES/AL requirements:

- Locate THEMES from the actual workspace layout; it was inspected at `/Users/cameronegbert/Research/THEMES`. Use its `demo/core.py`, `demo/run_mimic.py` and meaningful verification/tests as the initial reconstruction, not the incomplete original scripts as production imports.
- Extract standalone numerical code shared by the CLI and Theta adapters. Expose independently reusable loading/preprocessing, TICC, segmentation, policy networks, EDM, EM-EDM, evaluation and plotting.
- Add the apprenticeship contract and a dedicated THEMES runner with an explicit phase schedule behind the common bootstrap. Do not add a THEMES-specific alternate Theta training entrypoint.
- Deliver the AL synthetic smoke, configured-local-data MIMIC demo, and independent TICC/EDM/EM-EDM/behavior-cloning recipes. Keep source lineage, train-only fitting, fixed splits, causal feature lagging, retrospective membership semantics, pruning identities and reloadable artifacts correct. Compare held-out mixture action likelihood with honest baselines; do not require performance gains to prove wiring.
- Label this first scientific milestone TICC + EM-EDM reconstruction. Do not claim full reward-feedback THEMES, irregular-time MT-TICC, expert behavior, or clinical benefit.

Inspect the current tree and preserve unrelated changes. Implement changes in phase order, run targeted and launcher-level checks at each gate, and continue through the authorized local implementation without requesting routine design confirmation. Request permission only for actual sandbox restrictions or genuinely destructive actions. Do not publish or run broad production migrations.

Maintain an implementation status document with completed gates, exact validation evidence, remaining work and the production package migration matrix. Include real local smoke results and clearly distinguish locally checked Slurm composition from remote execution. Finish by reporting changed behavior, tests, limitations and the deferred H/release/workflow work. Do not stop at updating documents when implementation remains possible.

Keep progress updates and the final report concise and outcome-focused: changes made, validation results, blockers and remaining work. Do not include internal deliberation or step-by-step reasoning.
