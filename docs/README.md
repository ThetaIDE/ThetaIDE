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
- [Create figures and write discussions](guides/DISCUSSION.md)

## Reference

- [Theta SDK](reference/SDK.md) — SDK 3.4: the supported `src.app.api` import surface for components and app plugins, the `PluginContext` host facade, SDK version checks and boundary enforcement.
- [Discussion figure contract](reference/FIGURE_CONTRACT.md) — Vega-Lite plot components, live parameters, regeneration and storage.
- [Layer ownership and configuration provenance](reference/LAYER_BOUNDARIES.md) — active architecture contract, source-aware resolution, the exception register and inventory command.
- [Method execution contract v1](reference/EXECUTION_CONTRACT.md) — provider features, resource hints and tuner objective/pruning compatibility.
- [Compatibility requirements](reference/COMPATIBILITY_REQUIREMENTS.md) — capability `requires`/`provides`/`preflight`, component references, workspace resources and run logger result categories; how validation, forms and the run record use the findings.
- [Extension contributions and readiness v1](reference/EXTENSION_CONTRIBUTIONS.md) — metadata-only discovery, enablement, lifecycle cleanup and tuner controls/monitor availability.
- [Scheduler component contract v1](reference/SCHEDULER_CONTRACT.md) — explicit `site.scheduler` selection, runtime methods, receipt ownership and installing the optional Slurm component.
- [Portable tuner execution contract v1](reference/TUNER_EXECUTION_CONTRACT.md) — explicit provider selection, launch bindings, state snapshots, continuation, dependency pins, headless summaries and receipts.
- [What a launch fixes at start](reference/LAUNCHES.md) — the config snapshot every trial composes from, seeds drawn once per launch, and reproducing a run.
- [Saved result layout](reference/RESULT_STORAGE.md) — experiment folders, unique runs, and legacy compatibility.
- [Pipeline and component service API](reference/API.md)
- [Provider packages, storage, and ownership](reference/PLUGIN_STORAGE.md) — package format v3, profile resolution and declared cross-package exports.
- [Data loader ownership and selection](reference/DATA_LOADER_CONTRACT.md)
- [Experiment monitor contract](reference/MONITOR_CONTRACT.md) — declarations, hover descriptions, nested chart sections, and stream selection.
- [Configuration parameter form conventions](reference/CONFIG_PARAMETER_FORMS.md)
- [Third-party libraries and licenses](reference/THIRD_PARTY_LIBRARIES.md)

## Development

- [Layer boundary implementation status](development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md) — current state only: per-phase progress (0–7 implemented), what still has more than one home, repositories and test conventions.
- [Layer boundary handoff prompt](development/LAYER_BOUNDARY_HANDOFF.md) — prompt for a new agent session: publication, the remaining one-home settings, constraints and testing.
- [Config panel implementation and extension guide](development/CONFIG_PANEL_IMPLEMENTATION_STATUS.md) — shared document, rows, regions, section adapters, verification and remaining integration gates.
- [Measure startup and understand deferred services](development/STARTUP_PERFORMANCE.md)
- [Author component plugins](development/COMPONENT_PLUGIN_GUIDE.md)
- [Run native provider integration checks](development/PLUGIN_NATIVE_INTEGRATION.md)
- [Release components to Theta Hub](development/THETA_HUB_RELEASE_WORKFLOW.md)
- [Local variability implementation status](development/VARIABILITY_IMPLEMENTATION_STATUS.md)
- [NCShare CPU implementation status and live validation commands](development/SLURM_NCSHARE_IMPLEMENTATION_STATUS.md)

## Plans

- [Discussion panel plan](plans/DISCUSSION_PANEL_PLAN.md) — implementation and verification status for per-experiment figures and discussion text. [Implementation prompt](plans/DISCUSSION_PANEL_IMPLEMENTATION_PROMPT.md).
- [Discussion figures enhancement plan](plans/DISCUSSION_FIGURES_ENHANCEMENT_PLAN.md) — Phases 1–4 implemented (pickers, figure management, grouped parameters, eight figure types, interaction); export, reports and cross-target figures remain.
- [BlendRL Bets plan](plans/BLENDRL_BETS_PLAN.md) — phases 0–3 implemented (component `blendrl_bets`, sb3-suite 2.2.0 protocols); confidence-aware blending (free signals, per-expert value bets, GPI arbitration) and module dropout as one `blendrl_bets` component, the SB3 adapter protocols it needs, and the CartPole ablation matrix.
- [UI refinements during the layer refactor](plans/UI_REFINEMENTS_20261115.md) — config forms, Monitor/Results consolidation, retired panels and terminal shell controls; Vega deferred.

These documents describe proposals, intended behavior, or remaining work. Consult the current guides and implementation status above for supported behavior.

- [Phase 5 plan: results in `src/usr/results` and explicit bundles](plans/LAYER_BOUNDARY_PHASE5_PLAN.md) — bundle manifest `theta.bundle/v1`, application/workspace resolution, bundle inclusions and exclusions, units 5a–5e.
- [Phase 4 plan: retire `in/`](plans/LAYER_BOUNDARY_PHASE4_PLAN.md) — units 4a–4g implemented; package format v3 only, `src/usr` cutover, retirement of every compatibility adapter, complete move of the user's workspace, Hub republish preparation.
- [Compatibility requirements plan](plans/COMPATIBILITY_REQUIREMENTS_PLAN.md) — done: components declare compatibility (`requires` at `block` or `warn` level, `provides`, `preflight`), paradigms describe only themselves, and core, the SDK and the desktop do nothing extra for any component.
- [IDE, plugin, and user layer implementation plan](plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md) — active; confirmed decisions (no legacy support, generic rejection, results in `src/usr/results`), phase 4 close-out, remote bundles, `src/ide` consolidation and enforcement.

- [Config panel refactor plan](plans/CONFIG_PANEL_REFACTOR_PLAN.md) — shared parameter rows, explicit layout regions, typed document edits, and staged section migration.

- [Execution abstractions, THEMES components, and the AL group](plans/EXECUTION_ABSTRACTIONS_AND_THEMES_PLAN.md) — strict execution-contract migration using editable first-party dev components, then THEMES/AL; production Hub release and general workflows deferred. [Implementation prompt](plans/EXECUTION_ABSTRACTIONS_IMPLEMENTATION_PROMPT.md).
- [Slurm hardening and NCShare CPU rollout](plans/SLURM_NCSHARE_HARDENING_PLAN.md) — storage/publication policy, parallelism and continuation boundaries, implementation sequence, and CPU acceptance gate.
- [Hyperparameter sweep design](plans/HYPERPARAMETER_SWEEPS_DESIGN.md) — design draft.
- [Training and evaluation variability design](plans/TRAINING_EVALUATION_VARIABILITY_DESIGN.md) — proposed design; see the [implementation status](development/VARIABILITY_IMPLEMENTATION_STATUS.md) for current behavior.
- [Mechanistic monitoring roadmap](plans/MECHANISTIC_MONITORS_ROADMAP.md) — remaining work after the initial foundation.

## Reports

- [BlendRL Bets CartPole ablations](reports/BLENDRL_BETS_CARTPOLE_ABLATIONS_20261118.md) — 12 methods × 5 seeds: ceiling at initialization, no method distinguishable from BlendRL, gate trust in the logic expert explains the gap to DNN.

- [Discussion panel verification](reports/DISCUSSION_PANEL_VERIFICATION_20261117.md) — focused tests, stamped-run isolation and pending native visual acceptance.

- [Package format v3 Hub release](reports/V3_HUB_RELEASE_20261116.md) — the published 34-package batch (SDK 3.1): commits, feeds, archive hashes, Pages deployment and what was verified before publishing.
- [Optuna 1.1.1 Hub release](reports/OPTUNA_HUB_RELEASE_20261116.md) — production feeds, archive hash, successful deployment and real Hub-card installation verification.

- [UI refinement verification](reports/UI_REFINEMENTS_VERIFICATION_20261115.json) — focused form, Monitor/Results, terminal/navigation and native BlendRL checks; visual review and known failures.

- [Phase 5c verification evidence](reports/LAYER_BOUNDARY_PHASE5C_VERIFICATION_20261115.json) — full per-file comparison against 5b, sandbox reruns, counts and remaining failure IDs.

- [Layer boundary implementation history](reports/LAYER_BOUNDARY_HISTORY_20261115.md) — dated evidence for phases 0–4 (2026-11-13 to 11-15): commits, test counts and reasoning; superseded where later decisions differ.
- [Installed layer-boundary distribution evidence](reports/LAYER_BOUNDARY_DISTRIBUTION_20261113.json) — artifact hashes, bundled inventory, dependency versions, external-workspace/worker agreement and relocated checkpoint reload.
- [Phase 6 installed distribution evidence](reports/LAYER_BOUNDARY_PHASE6_DISTRIBUTION_20261116.json) — installed `theta`/`theta-ide` entry points, SB3 installed and trained from an external workspace by the CLI and by the desktop's job manager.
- [Compatibility requirements verification](reports/COMPATIBILITY_REQUIREMENTS_VERIFICATION_20261116.json) — full per-file run after the compatibility change (no new failure IDs), package integration and the installed-distribution probe against the release.
- [Phase 7 verification](reports/LAYER_BOUNDARY_PHASE7_VERIFICATION_20261116.json) — full per-file run after phases 5d–7: totals, failure IDs, the eight UI-batch failures confirmed at `9a3870e`, and earlier IDs now passing.

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

- [Project organization and workspace separation proposal](archive/PROJECT_ORGANIZATION_PLAN.md) — superseded by the layer boundary plan.
- [Local component development status](archive/LOCAL_COMPONENT_DEVELOPMENT_STATUS.md) — the pre-v3 checkout layout.
- [Execution refactor implementation status](archive/EXECUTION_ABSTRACTIONS_STATUS.md) — the 2026-11-11 execution refactor, `in/config` layout.
- [NCShare CPU Slurm implementation handoff](archive/SLURM_NCSHARE_IMPLEMENTATION_HANDOFF.md) — the first Slurm hardening pass, before bundles.
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
