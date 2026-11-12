# Execution abstractions first, then THEMES and the AL group

Status: proposed implementation plan, revised 2026-11-11 following explicit approval to break obsolete Hub component execution contracts. Supersedes the earlier THEMES integration draft's recommendation to add another custom training entrypoint. Production Hub publication remains outside this implementation scope. This document is a plan, not an implementation status report.

## Objective and scope

Restore one coherent execution contract in Theta, migrate existing training integrations through it, and then implement reusable THEMES components with a dedicated composition runner and an AL experiment group.

General workflow development is deferred. Do not implement a DAG executor, general artifact routing language, cyclic workflow semantics, workflow editor, or a generalized meta-paradigm framework for this project. THEMES will have an explicit, domain-specific phase schedule in its runner. Artifact contracts and independent numerical components keep future composition possible without predicting a workflow architecture now.

## Findings motivating the change

The inspected implementation has two meanings of runner:

- `src/app/pipeline/engine.py` accepts method `entrypoint`, `runner`, and `script` as alternative script selectors. `engine` maps `pytorch` and `lightning` to `src/app/train.py`, otherwise treating its value as a script reference. Current SB3 profiles use `entrypoint: package://sb3-suite/sb3_runner.py`.
- Paradigm YAML declares `runner`, `default_data_module`, and `eval_protocol`. `paradigm_loader.py` resolves these to classes, but the normal experiment path never invokes the resolved `runner_cls`.
- `run_pipeline.py` dispatches to method orchestration; the selected worker runs its script. The default `train.py` constructs components and directly calls Lightning's `trainer.fit`.
- `BaseParadigmRunner.run` currently duplicates experiment-level setup/method launch/plot orchestration. Some subclasses instead expose `train_model`. Existing tests sometimes instantiate these classes directly, bypassing real launch dispatch.
- Paradigm default data modules are used. Paradigm evaluation protocols are not automatically invoked simply because their class is resolved; actual evaluation also runs through algorithm validation, configured evaluator callbacks, and backend-specific evaluation. Some registered RL protocols are documented no-op placeholders.
- `provider_worker.py` already provides valuable workspace resolution, provider leases, snapshot verification, resource setup, and isolated execution. Preserve and consolidate this foundation rather than adding a second worker system.
- Workflow transform execution and meta-pipeline composition are incomplete. They are not prerequisites for this plan.

## 1. Target ownership and vocabulary

| Abstraction | Responsibility | Does not own |
|---|---|---|
| Experiment coordinator | Resolve recipe, expand methods/repeats/trials, allocate resources, launch workers, coordinate cancellation, plot completed results | Fitting a scientific model |
| Theta worker bootstrap | Load effective method config, verify providers, establish run context, resolve the paradigm runner, invoke it, finalize status | Selecting scientific update schedules |
| Paradigm | Declarative learning contract: input setting, required model/learner slots, supported budgets, default runner and evaluation behavior | Arbitrary script execution |
| Runner | Execute one method's learning lifecycle; assemble selected providers and schedule fit/evaluate/checkpoint operations | Launching all experiment methods, queueing cluster jobs, purging experiment roots |
| Model | Parameters, representation, prediction/inference, model state | Data splitting or experiment orchestration |
| Learner/algorithm | Objective, fitting/updates, optimizer/sampler state | Choosing cluster submission or log destinations |
| Execution backend | Integration with a training implementation such as Lightning or SB3; lifecycle adapters behind the runner | Replacing the Theta worker bootstrap |
| Data provider | Dataset access, fixed splits, preprocessing state, training batches or trajectory access | Constructing policy networks |
| Evaluator | Defined scoring on the designated split | Hidden model fitting or duplicate validation |

“Learner” is explanatory terminology; reuse Theta's existing algorithm capability kind. “Backend” is a responsibility, not a requirement to introduce another plugin kind, registry, or config pane. Start with concrete adapters and an existing component capability where needed.

A paradigm and a backend are separate axes: online RL can execute through Lightning or SB3. A paradigm runner must therefore accept a compatible backend/algorithm implementation without hardcoding one trainer for every online-RL method. THEMES can use a numerical fit adapter without pretending that TICC or EM is an epoch-based Lightning model.

## 2. One execution route

Target call path:

```text
GUI / CLI
  -> existing experiment coordinator
  -> one Theta worker bootstrap per method/trial/replicate
  -> resolve paradigm's registered runner
  -> runner.run(effective_method_config, run_context)
  -> resolve selected data/model/learner/evaluation providers
  -> execute lifecycle through the appropriate backend adapter
  -> return result; finalize status and artifacts
```

Preserve subprocess isolation, alternate Python environments, package imports, and the existing local/Slurm allocation model. “One entrypoint” means one Theta-owned training bootstrap used by normal executions; it does not prohibit internal functions, backend adapters, or a standalone THEMES CLI.

The experiment coordinator owns one-time experiment preparation and plotting. A method runner writes only inside its assigned run directories. This removes the recursive ownership hazard in the current `BaseParadigmRunner.run`, which calls method dispatch itself.

Define the runner's unit as **one fully resolved method execution**. Repeats and hyperparameter trials each invoke that same unit with their own identity, seed, budget, and directories. Do not re-expand methods inside the worker.

## 3. Minimal runtime contract

Define a small versioned contract and implement one example through it before changing dispatch:

- Effective method configuration is normalized once per launch; the worker receives or deterministically composes the same values used in preflight. Do not allow parent and worker to select different providers through different precedence rules.
- `RunContext` carries run identity, workspace/provider resolution, seed, assigned artifact directories, logging/telemetry, cancellation access, and resume/checkpoint selection. Existing resource and site mechanisms remain coordinator-owned.
- `RunResult` describes completion status, metrics with declared meanings/directions, artifact/checkpoint references, and phase progress summary. Failed runs must propagate exceptions/nonzero status; cancellation is distinct from success and failure.
- Runner provider is selected by the paradigm definition and loaded through existing provider discovery. Unknown, missing, or incompatible runners fail preflight and worker assembly; there is no silent fallback to Lightning after resolution failure.
- Runner assembles providers from explicit effective selections and declared paradigm defaults. Neural model children remain model-owned; loaders do not build models.
- Backend adapters expose only the lifecycle operations current integrations actually require. Preserve backend-native checkpoint formats; describe them in metadata rather than forcing every checkpoint through Lightning.
- Progress can declare epochs, transitions, solver iterations, EM rounds, or named phases. Tuning receives the declared objective and reports, without requiring every learner to have Lightning callbacks.
- Evaluator ownership is explicit: algorithm/backend-native validation or selected evaluator providers produce each metric once. A registered no-op placeholder is not evidence that evaluation ran.

Existing `BaseParadigmRunner` should converge on this method-level contract. Migrate direct `train_model` consumers and tests to the supported contract; do not retain wrappers solely to preserve the obsolete interface. Make normal dispatch exercise the same implementation. Do not retain a second default `.run` that redispatches an experiment.

## 4. Strict migration and local component development

Canonical target:

- `paradigm` selects the learning contract and registered runner.
- Method model/algorithm and input bindings select scientific components.
- Backend selection is component-owned metadata/defaults, with an explicit `backend` selector only where a real supported choice exists. Publish its precedence rule; do not infer it from architecture names.
- Reserve `runner` for the paradigm lifecycle provider. It is not a method-level script alias.
- Reject execution-selection `entrypoint`, method `runner`, `script`, and legacy `engine` settings in effective experiment, method, and selected algorithm profile configuration. Validate specifically these execution locations, not arbitrary domain parameters named script or engine.
- No legacy script adapter, runtime alias normalization, old selector precedence, silent Lightning fallback, or compatibility-only runner methods. Known old first-party profiles are migration inputs, not runtime exceptions.

Preserve legitimate capabilities rather than obsolete syntax: subprocess isolation, workspace/provider identity, backend-native fitting and checkpoints, configurable Python environments, telemetry, resource budgets, cancellation and site execution. Interpreter selection belongs to the execution-resource contract; converge redundant interpreter aliases through the same migration process rather than promising all old spellings forever.

Keep provider capability `entry_point` (module:function/class import metadata). It identifies importable implementations and is distinct from legacy config `entrypoint` (a script selector). Package resource loading and provider leases also remain necessary; removing arbitrary script dispatch must not remove them.

### Editable first-party components are the implementation path

Develop and validate against editable local first-party components before rebuilding production packages. Use the existing `src/usr/algorithms/sb3-dev` component and `sb3-dev/*` profiles as the concrete starting point. Inspect its active source, schemas, manifests and provenance; preserve existing unrelated changes.

For d3rlpy and other affected providers, use an existing editable dev component or create a properly declared local development package with its own identity, selectors and provenance. Do not assume the model-copy command supports algorithms; its documented deep-copy support is narrower. Use supported authoring services or an explicit local package layout and validate discovery and ownership.

Migrate dev manifests, provider implementations, parameter schemas and owned profiles together. Update active test/experiment recipes to explicitly select dev IDs. Do not shadow a production package, silently alias `sb3/*` to `sb3-dev/*`, or rewrite every historical experiment. Fresh-process integration must resolve the local provider identity and record it in run metadata.

Installed Hub components are protected runtime payloads and must not be patched in place. Retained package snapshots are not active development source. Do not build, publish, upload, or redeploy production Hub artifacts as a prerequisite for each local iteration. This user-approved development route takes precedence over generic guidance to author every iteration directly in the production source repository.

### Breaking obsolete Hub execution contracts is accepted

An old Hub component that still depends on a removed execution selector or lacks the new runtime contract may become non-runnable. This is an accepted consequence, not a reason to add compatibility infrastructure or block local development on a Hub redeployment.

Define explicit execution-contract compatibility for the relevant provider capabilities. Prefer an existing compatible metadata mechanism; otherwise add the smallest versioned declaration and validate it before execution. Do not repurpose package storage version or assume package version alone proves execution compatibility. Reject incompatible selected components with the package ID/version, unsupported declaration or field, and a concrete remedy: select the migrated dev component or install a future compatible release. Unselected obsolete components must not prevent catalog browsing or unrelated compatible runs.

Keep old saved configs readable for inspection and historical provenance, but do not promise unchanged execution or resume. Checkpoint use requires matching backend/model/preprocessing state; inference reload and exact training resume have separate contracts.

### Migration tooling and release handoff

Provide a read-only migration preview for known first-party recipes/profiles, showing proposed runner/backend bindings and explicit dev IDs. Unsupported external scripts require an authored registered adapter; do not guess their replacement. Conflicting old execution declarations fail preflight and are reported as requiring manual resolution rather than choosing the old precedence winner. Applied migrations affect only explicitly selected editable files and are reviewable diffs.

Maintain a migration report listing affected first-party dev IDs, production IDs, profiles and required source changes. Once local behavior is proven, prepare a reviewed change/patch for the canonical `theta-plugins` sources. Production archive builds, registry/feed changes, uploads and Hub deployment are a separate later release task following the production release workflow. They are not a completion gate for the local runtime refactor or initial AL demonstration.

## 5. Foundation implementation phases

### A. Specify and characterize existing execution

Deliver ownership documentation, normalized execution configuration, runner/context/result contracts, and characterization tests for supported scientific behavior and execution paths. Inventory old selector usage for migration; do not enshrine obsolete precedence as the new behavior. Record the evaluator invocation sites and checkpoint/log owners for Lightning, SB3, and d3rlpy in the migration report.

Files likely involved: `interfaces.py`, `paradigm_loader.py`, `pipeline/engine.py`, `provider_worker.py`, `commands.py`, `local_runner.py`, `repeated_task.py`, `train.py`, `lightning_builder.py`, provider manifests/profiles, and site launch integration. Confirm actual paths against the selected workspace.

Gate: regression fixtures cover supported scientific behavior, and code/docs define the target single-method contract. Test actual invocation, incompatible provider rejection, removed-field errors, and conflicting old declarations. Old alias-resolution tests must be replaced with rejection/migration tests.

### B. Wire the common bootstrap and Lightning runner

Refactor worker bootstrapping to resolve and invoke the registered runner. Extract the current Lightning assembly/fit/finalization into callable adapter code. Make supervised, unsupervised, and Lightning RL runner implementations use it while preserving their current supported losses, budgets, callbacks, data bindings, and evaluation behavior.

Remove experiment-level setup/method dispatch from the runner default. Make the worker own run finalization in `try/except/finally`, including backend cleanup and failure/cancellation status. Do not silently claim placeholder protocols supply extra evaluation.

Gate: actual `run_pipeline.py` launches invoke a sentinel selected runner exactly once; changing its registered implementation changes execution. A runner resolution error fails before fitting. Small supervised, unsupervised, and Lightning RL runs preserve artifacts and declared metrics. Keep default data module and configured evaluator selection working.

### C. Migrate editable first-party backend components

Refactor `sb3-dev` script bodies into importable backend/learner adapters. Identify d3rlpy provider registrations, profiles and training call sites, then implement the same contract in a local dev package with a separate explicit identity. Migrate owned schemas/profiles and active smoke recipes to those providers. Remove script-based training dispatch from the new runtime; historical script files may remain as records but are not alternate supported Theta training routes.

Gate: tiny SB3-dev and d3rlpy-dev executions launch through the common Theta bootstrap and selected paradigm runner, with backend-native checkpoint reload, telemetry, and cancellation verified. Effective configuration and run metadata identify the dev providers. Old selected Hub profiles fail preflight with migration guidance; unrelated compatible runs still work with obsolete Hub packages installed. No production package build or Hub redeployment is needed to pass this gate.

### D. Exercise coordinator integrations and publish the contract

Verify one direct run, one repeat, one small sweep, GUI launch, alternate interpreter selection, and allocation-based Slurm launch use the same worker path and maintain config/provider identity. Keep queue/resource/process supervision in their existing layer. Validate Slurm command composition locally; a real remote smoke is a separate gate before claiming cluster validation.

Update component authoring, execution API, relevant parameter-form documentation, and runner examples. Adapt existing tests that currently instantiate runners directly; retain useful unit tests but add launcher-level tests proving invocation. Do not change GUI layouts except where selection semantics require shared parameter controls.

Gate: no migrated first-party normal training path bypasses the bootstrap; no runner recursively launches all experiment methods; effective providers and backend are recorded; cancellation/failure reach the coordinator; no duplicate evaluations or log-directory collisions. Complete these gates before introducing THEMES execution. Publish the contract in local documentation, not the production Hub; report unperformed remote checks honestly and retain a separate release handoff for obsolete Hub providers.

## 6. THEMES learning scope

After the foundation works, add `apprenticeship` as the integrated learning contract and `AL` as the experiment group. Apprenticeship uses static demonstrations, a model bundle, and a learner/composition schedule. It does not inherit CQL/IQL transition-reward assumptions from `offline_rl`.

TICC is unsupervised stage discovery; EDM is imitation from observed actions with energy distribution matching; EM-EDM alternates latent policy assignments and EDM fitting. Full THEMES adds inverse reward inference and reward-regulated stage refinement. Offline describes data access, not a requirement to route all phases through the offline-RL runner.

Use the existing independent reconstruction under `THEMES/demo` for the first integration. It implements fixed-spacing multi-series TICC and hard EM-EDM, with saved output and tests. Original scripts reference missing packages and execute machine-specific workflows; they are research references rather than drop-in provider code.

The first milestone is explicitly “TICC + EM-EDM reconstruction.” Full reward-feedback THEMES is a later milestone. Real irregular-time MT-TICC also requires timestamps and a validated time-aware solver.

## 7. Reusable components and dedicated runner

Extract `themes_al`, an installable numerical Python package that imports neither Theta nor Qt/Hydra in its core modules. The standalone CLI and Theta adapters call the same implementation.

| Component | Independent behavior | Theta integration |
|---|---|---|
| Trajectory loader | Read visits/observations/actions with masks and lineage | Training data provider; MIMIC source adapter kept separate |
| Preprocessing | Train-only fit, frozen transform, persistence | Data-provider-owned fitted state |
| Stage estimator | Fit/predict/persist TICC stage models | Selectable model component with documented non-neural fit interface |
| Segment builder | Convert contiguous stage runs into recorded segments | Pure reusable operation; no new transform plugin system |
| Policy network | Produce action logits | Model capability; child backbones selectable with model registry |
| EDM learner | Policy updates, energy objective, SGLD and optimizer state | Algorithm capability |
| EM-EDM learner | Assign/score groups, update priors, fit/prune policies | Algorithm capability composing EDM and policies |
| AL evaluation | Frozen held-out scoring and baseline comparison | Evaluator capability |
| AL figures | Read persisted tables/arrays | Plotter capability |
| THEMES composition | Explicit phase schedule and optional outer refinement | Registered runner through the common bootstrap |

Author initial THEMES capabilities as an editable local first-party package with explicit local IDs and owned profiles. Use existing capability kinds and lifecycle installation. One initial provider package may expose several capabilities; avoid unnecessary separately versioned packages or registries. Validate schemas and provider resolution before selecting final public IDs.

The dedicated runner owns this explicit schedule:

```text
load demonstrations and fixed splits
 -> fit preprocessing and TICC on training data
 -> infer stages and build per-visit segments
 -> fit EM-EDM on training segments
 -> evaluate frozen models and export artifacts
```

Later add the explicit high-level MDP -> inverse reward fit -> reward weighting -> stage refinement loop inside this runner. It calls reusable numerical components, not other experiment coordinators or general workflow nodes. Keep the phase schedule in one clear implementation, not an embryonic generic graph engine.

## 8. Artifact and data boundaries

Define versioned, independently reloadable artifacts:

- Trajectory set: visit IDs, optional patient IDs/timestamps, observations/actions, masks, feature/action schemas, original slots, split membership, source checksum.
- Stage bundle: fitted scalers, stage means/precision matrices, windows/time convention, assignments, solver diagnostics, configuration and source/split fingerprints.
- Segment set: visit/stage/segment IDs, inclusive start/exclusive stop, arrays/references, lagged policy states and scoring validity mask, split membership.
- Policy mixture: network configuration and weights, priors, active component IDs/pruning history, likelihood convention, preprocessing identity.
- Run bundle: effective config, provider/code versions, seed, phase budgets, metrics/definitions, artifacts, status and fitted-state provenance.

Invariants: no window/lag/segment crosses a visit; every valid source row belongs to exactly one segment; masks follow the source `mask != -1` rule; only training visits fit models/scalers/priors; validation selects parameters and test data remains frozen. Split by actual person ID where available; source array indices do not establish patient independence.

Preserve singleton segments and mark segments with no scoreable prehistory. Membership is retrospective from whole-segment recorded states/actions. Stage inference is also retrospective. Prospective clinical action selection is outside this milestone.

For MIMIC, document the provisional `AntiInfectiveAdmin_max` action, 14 physiological means plus trailing min/max yielding 42 lagged policy features, rare-action imbalance, transformed measurements, unknown upstream normalization, and missing timestamps. Outcome annotations are not policy inputs or evidence of expertise. Keep clinical files out of component distributions.

Checkpoint bundles must restore all fitted state needed for scoring. Exact resume additionally requires optimizer, SGLD buffer, RNG, mixture assignment/prior state, and current phase/iteration state; either implement and validate these or reject resume explicitly. Never label inference-only reload as training resume.

## 9. AL experiment recipes

Proposed selectors, not currently runnable:

| Recipe | Purpose |
|---|---|
| `AL/themes_synthetic_smoke` | Small generated trajectories proving independent components and composition without clinical files |
| `AL/themes_mimic_demo` | 512-visit TICC + EM-EDM demonstration using a configured local dataset |
| `AL/themes_mimic_full` | Full-cohort profile after demo acceptance |
| `AL/ticc_only` | Standalone unsupervised segmentation with a TICC-compatible runner/evaluator |
| `AL/edm_only` | One imitation policy without latent groups |
| `AL/em_edm_only` | Fit policies from a saved segment artifact without refitting stages |
| `AL/behavior_cloning` | Supervised recorded-action baseline using the same features and split |
| `AL/themes_refinement` | Later full reward-feedback milestone |

Group `_base.yaml` provides shared dataset/seed/output conventions and an apprenticeship default; independent recipes explicitly override paradigm where appropriate. Do not force the stock reconstruction-based unsupervised runner on the TICC solver.

For parity, preserve the existing demo settings in a dedicated profile, including its original split, and compare extracted/old numerical code on identical inputs. New comparison recipes use a fixed train/validation/test split; never tune against the old demo's held-out test set.

Start the MIMIC demonstration with five stages, two-row stage windows, sparsity 0.01, switch penalty 4, up to three mixture policies, six EM rounds, 250 EDM updates/component/round, 10 SGLD steps, energy weight 0.1, and seed 42. Make other solver/pruning/network defaults explicit and provider-owned.

Report held-out mixture marginal action NLL using a documented aggregation, compared with a training-fitted constant-action-rate baseline and behavior cloning. Also report group support, stage durations/switches, pruning, action rates, class-sensitive action metrics where defined, and fit traces. A successful integration need not beat the constant baseline. Do not score by selecting a component using the test segment and reporting only that component's likelihood.

Produce stage/group timelines, contingency/support summaries, likelihood comparisons, segment tables, model bundles, and source-lineage verification. Saved exports are recorded segments, not simulated patient trajectories.

## 10. THEMES delivery phases and gates

### E. Extract the reconstruction

Move numerical/data/export functionality into `themes_al`, remove hardcoded paths and import-time execution, port meaningful mathematical/lineage tests, and preserve source attribution. No algorithm changes during extraction.

Gate: standalone synthetic run and saved-state reload succeed; extraction parity holds within declared numerical tolerances on matching inputs/config/seed; label comparisons tolerate arbitrary label permutations.

### F. Install providers and the dedicated runner

Add storage-v2 manifests, config schemas, local dev providers, apprenticeship paradigm, dedicated runner, and synthetic AL recipe. Discover authored providers or install local packages through the normal lifecycle as appropriate; keep independent selectors and provider defaults discoverable.

Gate: clean-workspace `run_pipeline.py AL/themes_synthetic_smoke` uses the common bootstrap and selected runner; actual provider substitution changes execution; missing/incompatible inputs fail preflight; finite metrics, run status, checkpoints and plots appear in assigned locations. Test negative paths and fresh-process reload.

### G. Demonstrate independent and integrated use

Add the MIMIC demo/full profiles and independent recipes. Compare standalone versus Theta output using identical inputs and fixed splits. Run the demo, validate held-out scoring/lineage, then run full cohort only after the smaller gates pass.

Gate: TICC-only exports can feed EM-EDM-only without hidden refits; EDM works without segmentation; baseline and integrated methods share comparisons fairly; simultaneous runs have isolated fitted state; synthetic and local clinical demonstrations produce reviewable results.

### H. Implement full THEMES reward feedback

Recover or independently specify missing discrete dynamics, expert/cohort policy, MLIRL conventions, reward weighting and regulated partitioning. Document deviations from the original source. Use fitted active policy count, handle unseen transitions/terminals, zero reward normalization and label remapping, and persist each outer round.

Gate: small known discrete-MDP tests validate likelihood/gradient/dynamics behavior; at least one outer refinement round completes reproducibly and preserves source lineage; the actual reward-regulated stage solver is tested before calling this full THEMES integration.

Before any public distribution, resolve original source redistribution/license status and publish through the existing Hub release workflow. Hub publication is not required to complete local research integration.

## Completion criteria and sequencing

Complete A–D before F. E can proceed after contracts stabilize, but no THEMES-specific alternate worker entrypoint may bypass unfinished foundation work. Complete F and G for the initial AL delivery; H is the explicit later scientific milestone.

The result is one Theta-owned method execution route, meaningful paradigm runner dispatch, migrated first-party dev backends under a strict execution contract, independent THEMES components and standalone usage, and a dedicated AL composition runner. Generalized workflows remain outside this implementation scope.


## Implementation handoff

Use [the implementation prompt](EXECUTION_ABSTRACTIONS_IMPLEMENTATION_PROMPT.md) to start the work. It adopts the strict migration and local-first policy above. The runtime refactor and initial AL milestone do not include production Hub publication or general workflow implementation.
