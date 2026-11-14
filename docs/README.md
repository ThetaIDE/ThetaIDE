# Documentation

Start with [installation](guides/INSTALL.md), [component development](guides/COMPONENT_DEVELOPMENT.md), or the [contributor guide](../CONTRIBUTING.md). The IDE also has built-in help under **F1 → Documentation**.

Documentation is organized by purpose and shared by human contributors and coding agents. Current guidance lives in `guides/`, `reference/`, and `development/`; proposals and historical notes are kept separately.

## Guides

- [Install and launch ThetaIDE](guides/INSTALL.md) — setup, local state, and the developer reload button.
- [Develop components from the GUI or terminal](guides/COMPONENT_DEVELOPMENT.md)
- [Create editable objectives](guides/CUSTOM_OBJECTIVES.md)
- [Use mechanistic monitors](guides/MECHANISTIC_MONITORS.md)
- [Run THEMES/AL experiments](guides/THEMES_AL.md)
- [Configure local training resources](guides/TRAINING_RESOURCES.md)
- [Run and sync cluster jobs](guides/CLUSTER_JOBS.md)

## Reference

- [Theta SDK](reference/SDK.md) — the supported `src.app.api` import surface for components and app plugins, the `PluginContext` host facade, SDK version checks and boundary enforcement.
- [Layer ownership and configuration provenance](reference/LAYER_BOUNDARIES.md) — active architecture contract, source-aware resolution, compatibility exceptions and inventory command.
- [Method execution contract v1](reference/EXECUTION_CONTRACT.md) — provider features, preflight hooks, resource hints and tuner objective/pruning compatibility.
- [Extension contributions and readiness v1](reference/EXTENSION_CONTRIBUTIONS.md) — metadata-only discovery, enablement, lifecycle cleanup and tuner controls/monitor availability.
- [Scheduler component contract v1](reference/SCHEDULER_CONTRACT.md) — explicit `site.scheduler` selection, runtime methods, receipt ownership, compatibility adapters and installing the optional Slurm component.
- [Portable tuner execution contract v1](reference/TUNER_EXECUTION_CONTRACT.md) — explicit provider selection, launch bindings, state snapshots, continuation, dependency pins, headless summaries and old-receipt readers.
- [Execution refactor implementation status](development/EXECUTION_ABSTRACTIONS_STATUS.md)
- [Saved result layout](reference/RESULT_STORAGE.md) — experiment folders, unique runs, and legacy compatibility.
- [Pipeline and component service API](reference/API.md)
- [Provider packages, storage, and ownership](reference/PLUGIN_STORAGE.md) — including declared cross-package exports.
- [Data loader ownership and selection](reference/DATA_LOADER_CONTRACT.md)
- [Experiment monitor contract](reference/MONITOR_CONTRACT.md) — declarations, hover descriptions, nested chart sections, and stream selection.
- [Configuration parameter form conventions](reference/CONFIG_PARAMETER_FORMS.md)
- [Third-party libraries and licenses](reference/THIRD_PARTY_LIBRARIES.md)

## Development

- [Layer boundary implementation status](development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md) — per-phase progress outline, implemented changes, acceptance evidence and outstanding gates.
- [Phase 4 handoff prompt](development/LAYER_BOUNDARY_PHASE4_HANDOFF.md) — prompt for a new agent session continuing the layer boundary work at phase 4.
- [Config panel implementation and extension guide](development/CONFIG_PANEL_IMPLEMENTATION_STATUS.md) — shared document, rows, regions, section adapters, verification and remaining integration gates.
- [Measure startup and understand deferred services](development/STARTUP_PERFORMANCE.md)
- [Author component plugins](development/COMPONENT_PLUGIN_GUIDE.md)
- [Run native provider integration checks](development/PLUGIN_NATIVE_INTEGRATION.md)
- [Release components to Theta Hub](development/THETA_HUB_RELEASE_WORKFLOW.md)
- [Local component development status](development/LOCAL_COMPONENT_DEVELOPMENT_STATUS.md)
- [Local variability implementation status](development/VARIABILITY_IMPLEMENTATION_STATUS.md)
- [NCShare CPU Slurm implementation handoff](development/SLURM_NCSHARE_IMPLEMENTATION_HANDOFF.md)
- [NCShare CPU implementation status and live validation commands](development/SLURM_NCSHARE_IMPLEMENTATION_STATUS.md)

## Plans

These documents describe proposals, intended behavior, or remaining work. Consult the current guides and implementation status above for supported behavior.

- [Phase 4 plan: retire `in/`](plans/LAYER_BOUNDARY_PHASE4_PLAN.md) — proposed; package format v3, layout v2 cutover, migration command with backup, Hub republish preparation.
- [IDE, plugin, and user layer implementation plan](plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md) — active (phases 0–3 locally complete, phase 4 next); contract boundaries, recoverable `in/` migration, provider-neutral execution, and eventual `src/ide` consolidation.

- [Config panel refactor plan](plans/CONFIG_PANEL_REFACTOR_PLAN.md) — shared parameter rows, explicit layout regions, typed document edits, and staged section migration.

- [Execution abstractions, THEMES components, and the AL group](plans/EXECUTION_ABSTRACTIONS_AND_THEMES_PLAN.md) — strict execution-contract migration using editable first-party dev components, then THEMES/AL; production Hub release and general workflows deferred. [Implementation prompt](plans/EXECUTION_ABSTRACTIONS_IMPLEMENTATION_PROMPT.md).
- [Slurm hardening and NCShare CPU rollout](plans/SLURM_NCSHARE_HARDENING_PLAN.md) — storage/publication policy, parallelism and continuation boundaries, implementation sequence, and CPU acceptance gate.
- [Project organization and workspace separation proposal](plans/PROJECT_ORGANIZATION_PLAN.md) — deferred architecture proposal.
- [Hyperparameter sweep design](plans/HYPERPARAMETER_SWEEPS_DESIGN.md) — design draft.
- [Training and evaluation variability design](plans/TRAINING_EVALUATION_VARIABILITY_DESIGN.md) — proposed design; see the [implementation status](development/VARIABILITY_IMPLEMENTATION_STATUS.md) for current behavior.
- [Mechanistic monitoring roadmap](plans/MECHANISTIC_MONITORS_ROADMAP.md) — remaining work after the initial foundation.

## Reports

- [Installed layer-boundary distribution evidence](reports/LAYER_BOUNDARY_DISTRIBUTION_20261113.json) — artifact hashes, bundled inventory, dependency versions, external-workspace/worker agreement and relocated checkpoint reload.

- [IDE, plugin, and user layer boundary audit](reports/LAYER_BOUNDARY_AUDIT_20261113.md) — ownership gaps in `in/`, execution, desktop extensions, package APIs, distribution, and remote snapshots.

- [Configuration panel architecture audit](reports/CONFIG_PANEL_AUDIT_20261113.md) — shared-control adoption gaps, column/nesting fragility, correctness findings, and verification baseline.

- [Desktop startup regression and post-launch crash](reports/STARTUP_REGRESSION_20261114.md) — site-sync polling and per-call config resolvers on the layer-boundary branch.
- [Desktop startup investigation](reports/STARTUP_PERFORMANCE_20261112.md)
- [Monitor responsiveness investigation](reports/MONITOR_PERFORMANCE_20261112.md)
- [Frontend performance diagnosis](reports/FRONTEND_PERFORMANCE_DIAGNOSIS_20261111.md)
- [Slurm integration and NCShare operational review](reports/SLURM_NCSHARE_REVIEW_20261110.md)
- [CartPole final sweep and objective review](reports/CARTPOLE_FINAL_SWEEP_20261109.md)
- [Sweep dashboard comparison](reports/OPTUNA_DASHBOARD_GAPS.md)
- [Results browser requirements report (PDF)](reports/results_browser_must_haves_report.pdf)

## Archive

Historical context for earlier ThetaIDE, NeSyRL, and BlendRL work; commands and paths in these documents may refer to former projects.

- [Configuration integration and training monitor change log](archive/CHANGES_M0_config_integration.md)
- [Former BlendRL workflow guide](archive/WORKFLOW_GUIDE.md)
- [Multi-model training pipeline handoff](archive/handoff_multimodel_training_pipeline.md)
- [Single-model deployment and verification handoff](archive/handoff_single_model_verification.md)

## Assets and placement

The [demo video](assets/demo%20video.mp4) and documentation media live in `assets/`. Reports and attachments live in `reports/`, keeping the top level of `docs/` limited to this index.

Add documentation to the matching folder and link it from this index. Put design proposals and roadmaps in `plans/`, and move obsolete guidance to `archive/` with a historical-context note. Write shared documentation once; keep agent-specific instructions in the repository's `AGENTS.md` entry point. The project root retains the README, contributor guide, code of conduct, license, citation, and required tool/configuration entry points.

The native visual monitoring foundation (roadmap sections 1–3) is described in
[Mechanistic monitoring](guides/MECHANISTIC_MONITORS.md#native-diagnostics-and-network-inspection)
and the [rich output contract](reference/MONITOR_CONTRACT.md#rich-output-declarations-v1).

- [THEMES source audit and corrective implementation](guides/THEMES_SOURCE_AUDIT.md)
