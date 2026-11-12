> **THEMES update:** The complete paper/source loop is now integrated under explicit `_port`/`paper_synthetic` recipes, with model-owned stage/policy/reward components, temporal data loader, round replay and full-loop monitors/PlotMan panels. Earlier demo results remain distinct. See [the source audit](../guides/THEMES_SOURCE_AUDIT.md).

# Execution refactor implementation status

Local development, 2026-11-11. The foundation, phase E extraction, and initial phase F synthetic AL integration
are implemented; this does not claim completion of all A–G gates. Phase H, generalized workflows and production
Hub publication remain deferred.

## Implemented foundation

- One registered runner invocation for a resolved method through the common
  bootstrap, including direct launches, repeats and Optuna trial workers.
- Versioned context/result contracts, cancellation/failure finalization, provider
  snapshots/leases, backend selection and selected algorithm compatibility checks.
- Lightning assembly extracted from `train.py`; base runners no longer redispatch
  experiments. Obsolete `train_model` production methods removed. Scientific unit
  fixtures train assembled modules directly; launcher tests exercise real dispatch.
- `sb3-dev` profiles/manifests/schema/backend migrated together, including active
  materialized `in/config/agent/sb3-dev` profiles. No production selectors shadowed.
- Separate editable `d3rlpy-dev` backend, profiles, schema, native encoder provider
  and provenance. Static-data losses are explicitly training metrics.
- Strict removed-field rejection and explicit preview/apply YAML migration command.
- Backend metadata now supplies monitor discovery instead of script-path matching.
- Local smoke recipes under `Execution/` for SB3, d3rlpy, supervised and unsupervised.

## Validation evidence

Validation ran in `/tmp/theta-execution.LUCKmV`, an editable copy of the actual
checkout `/Users/cameronegbert/Research/Theta/Theta-IDE`. The supplied original
workspace path no longer exists. The existing checkout's Python environment was
used for execution only; no environment or installed provider was edited.

Focused command (all paths relative to the editable checkout):

```sh
python -m pytest tests/test_execution_contract.py tests/test_engine_dispatch.py \
  tests/test_slurm.py tests/test_repeated_task_config.py \
  tests/test_supervised_tabular.py tests/test_paradigm_unsupervised.py \
  tests/test_tabular_data_module.py -q --disable-warnings
```

Result: **116 passed, 15 skipped**, 15 warnings, 23.14 seconds. Optional installed
Python-entry-point fixtures account for skips. The contract tests include a real
`run_pipeline.py` sentinel invocation exactly once, missing-runner rejection,
two-seed repeats, two-trial Optuna dispatch, old-field rejection, migration preview
logic, evaluator ownership and failure/cancellation finalization.

Real local scientific smokes, all through `run_pipeline.py`:

| Recipe | Observed result |
|---|---|
| `Execution/sb3_smoke` | Completed; 32 transitions (16 requested, four training envs/native rollout rounding), greedy final reward 13, matched-seed sampled evaluation, `.zip` checkpoints and telemetry |
| `Execution/d3rlpy_smoke dataset_path=/tmp/theta-d3-transitions` | Completed; four DiscreteBC updates, final training loss 0.9329935, native `.d3` bundle and telemetry |
| `Execution/supervised_smoke dataset_path=/tmp/theta-tabular.npz` | Completed; one epoch, validation loss 0.6788366, Lightning checkpoint and telemetry |
| `Execution/unsupervised_smoke dataset_path=/tmp/theta-tabular.npz` | Completed; one epoch, validation reconstruction loss 0.7866650, Lightning checkpoint and telemetry |

The tabular fixture had 64 Gaussian four-feature rows with seed 42 and binary
labels `x[:, 0] > 0`. The transition fixture had 64 Gaussian four-feature rows,
alternating integer actions, reward 1 and terminal every eighth row, seed 42.
These are wiring checks, not performance claims.

A fresh Python process reloaded SB3's saved native policy and d3rlpy's learnable
bundle and predicted actions successfully. A real SB3 worker interrupted by
SIGTERM exited nonzero with `execution.json` status `cancelled`. A two-seed SB3
repeat completed in separate version directories with separate telemetry.
Slurm command composition tests passed locally; no remote job was submitted.

The additional GUI/launch suite initially aborted inside Qt's embedded terminal
under the sandbox. After permission, the unsandboxed rerun passed **21 tests** in
3.42 seconds, including GUI shell launch and scheduling. No GUI layout changed.
Paired SB3 evaluation/telemetry checks passed **17 tests** in 3.31 seconds. A fresh
workspace with explicit editable SB3 and DNN fixture identities passed its real
training/checkpoint-reload integration test (8.68 seconds), without Hub builds.
Provider-organization tests passed **33 tests**. The expanded contract suite passed
**20 tests**. Four decorator-only mock registration tests in
`test_e2e_user_configurations.py` fail identically on the untouched checkout;
baseline recheck: four failures in 4.35 seconds. They are not execution regressions.

## Evaluator and checkpoint ownership

| Backend | Actual evaluator invocation sites | Checkpoint/log owner |
|---|---|---|
| Lightning | Module `validation_step`; `ConfiguredEvaluatorCallback` validation hooks or one `evaluate` at fit end; configured online environment callback | Backend saves native `.ckpt` in context directory; CSV/telemetry use method/trial/replicate identity |
| SB3 dev | Native `ThetaMetricsCallback._evaluate` at start, configured transitions, final update; greedy/sampled modes preserve training RNG | Native `.zip` in context directory; telemetry run owns log version |
| d3rlpy dev | Native fitter produces one epoch's training metrics; no held-out evaluator or simulator score claimed | Native `.d3` in context directory; telemetry owns version |

Registered RL paradigm eval protocols remain documented no-op placeholders and
are not invoked as additional scorers. Explicit evaluator providers are currently
supported by Lightning; other backends reject that unsupported selection.

## Remaining gates before THEMES

- Alternate-interpreter gate closed locally: the venv coordinator launched the
  actual worker under `/opt/anaconda3/bin/python3`, a different resolved Python
  environment. The selected sentinel runner ran exactly once and completed.
  Scientific Lightning parity across those environments is not claimed.
- Characterize a migrated first-party Lightning RL provider with an actual local
  RL smoke. No such editable provider is present in this checkout; existing old
  algorithm providers require their own explicit execution-contract migration.
- d3rlpy tuning/promotion of native `.d3` bundles and additional metric/pruning
  integration are not implemented. Reject that unsupported execution choice.
- Broaden negative/duplicate-evaluation coverage across all migrated backend paths;
  runner substitution, old-selected/unselected behavior and native fresh-workspace
  SB3 checkpoint reload are now checked.
- Exact SB3/d3rlpy training resume is intentionally rejected. Lightning requires
  explicit `resume_from`; cross-backend/model/preprocessing identity validation
  needs further characterization before claiming a general resume guarantee.
- Phase E and the initial phase F synthetic AL runner are implemented (see below).
  The numerical path now has its own launcher, preflight, reload and panel checks.
  The remaining Lightning RL characterization and broader backend guarantees
  remain open; this is not completion of every A–G gate or full THEMES delivery.

## Production source migration / release handoff

No production-source package, protected installed payload, retained snapshot,
registry/feed or archive was modified or published.

| Editable development identity | Production source / profiles | Required later source changes |
|---|---|---|
| `sb3-dev/*` | `theta-plugins/methods/sb3`; `sb3/*`; parent sb3-suite 1.1.1 | Port `sb3_runner.py:execute`, manifest capability `execution_contract`/`backend`/`entry_point`, schemas and profiles; remove owned script entrypoint declarations; update release docs |
| `d3rlpy-dev/*`, `d3rlpy-dev/vector` | `theta-plugins/methods/d3rlpy`; `d3rlpy/*`; parent d3rlpy-suite 1.0.3 | Port the new importable fitter/encoder, storage-v2 capability declarations and profiles; document static training metrics, native bundles and unsupported exact resume |
| App-owned base runners / Lightning adapter | Theta app source | Review `core/execution.py`, `core/lightning_execution.py`, `interfaces.py`, `paradigm_loader.py`, `lightning_builder.py`, paradigm implementations, pipeline engine/worker/commands/composition/repeat/trial code and monitor discovery |
| Historical first-party recipes | Explicitly selected editable configs only | Preview migration; author registered adapters for unknown scripts; resolve conflicts manually; choose dev IDs explicitly |

After local gates pass, prepare reviewed patches against canonical production
sources. Package versioning, builds, archives, registry/feed updates, upload and
Hub deployment are a separate authorized release task under the production
release workflow. None is a prerequisite for this local iteration.

Final staged regression run: **67 passed** in 26.81 seconds across execution, native
provider contracts, SB3 paired evaluation/telemetry/reload, engine rejection and
Slurm composition tests. All edits remain in app source, tests, documentation and
local editable providers/configs. Active materialized smoke recipes under
`in/config/experiment/Execution` are workspace-local and Git-ignored; SB3/d3rlpy
recipes also live in their versioned editable provider payloads.

The patch was applied to the moved checkout. Its focused execution/SB3 reload/
selector/Slurm verification passed **37 tests** in 23.12 seconds. Source comparison
against the tested staging tree also checks the preflight validation and bootstrap
files, preventing a partial runtime handoff. `git diff --check` passed.


## Phase E: standalone reconstruction extraction (2026-11-11)

The independent numerical package is now `src/usr/packages/themes-dev`. Its `core.py`
is byte-identical to `/Users/cameronegbert/Research/THEMES/demo/core.py`;
`PROVENANCE.json` records original checksums. The MIMIC-schema adapter,
preprocessing, explicit fit/export schedule and plots were extracted from the
demo. No original source or clinical dataset was changed. Core modules import
neither Theta nor Qt/Hydra, and the standalone CLI shares that implementation.

Added generated synthetic input, version-1 inference bundles, source/artifact
checksums, frozen preprocessing and stage/policy reload, explicit exact-resume
rejection, and independent lineage/membership/likelihood verification. Original
recorded observations retain float64 precision in exports. Supplied `patient_ids`
use grouped splits; without them, the demo's original outcome-stratified 80/20
visit split remains unchanged for parity and establishes no person independence.
No algorithm changes were made during extraction. The 42-dimensional policy
schema, action mapping, train-only fitting, boundary-safe windows, lagging,
singleton retention and mixture marginal likelihood convention are preserved.

Validation ran in `/tmp/theta-al.z0Qetu` using the actual checkout's existing
`venv/bin/python`; no installed package/environment was changed. Command:

```sh
python -m pytest tests/test_themes_reconstruction.py \
  tests/test_execution_contract.py tests/test_engine_dispatch.py -q --disable-warnings
```

Result: **51 passed** in 19.49 seconds, including **19 reconstruction tests**.
The committed artificial parity fixture was generated by the *original* demo,
with source hash and configuration recorded beside it. Checks compare fitted
scalers, stage partitions (permutation invariant), stage means/precision,
segment boundaries and EM-EDM marginal evidence at rtol 1e-5 / atol 1e-6.
Other tests cover unchanged train scalers after held-out perturbation,
person grouping, singleton priors, fresh-process reload without Theta/Hydra/Qt,
invalid versions, modified artifacts/source, and rejected training resume.

Standalone command (from the repository with dependencies available):

```sh
PYTHONPATH=src/usr/packages/themes-dev python -m themes_al synthetic --output /tmp/themes-smoke
PYTHONPATH=src/usr/packages/themes-dev python -m themes_al verify --output /tmp/themes-smoke
```

Actual smoke: `/tmp/theta-al-final`, seed 42, 40 artificial visits, 480 rows,
32 train / eight test visits, 60 recorded segments, two fitted stage labels,
440 scoreable rows, one remaining policy after pruning. Held-out mixture NLL
was **0.690734612** per scored step; this is a wiring check, not performance
acceptance. CSV/NPZ/model bundles and an inspected `overview.png` were produced.
Fresh-process source-lineage and model verification passed. A local wheel was
built without downloading dependencies and installed only into a temporary
target for a standalone import/reload check; nothing was published.

A real alternate-interpreter launcher check used the existing sentinel workspace
fixture: venv parent -> `/opt/anaconda3/bin/python3` worker -> registered runner,
exactly one call, completed `execution.json` with the alternate worker path.
Workspace: `/var/folders/kr/wj_rvspd2757z0lrtby6nbgw0000gn/T/theta-al-alternate-9_xcpu6e`.
This closes interpreter dispatch, not a cross-environment scientific comparison.

### Remaining boundaries and next implementation gates

Initial phase F now supplies the synthetic AL integration described below.
Remaining G work includes independent TICC/EDM/EM-EDM recipes, behavior cloning,
a fixed train/validation/test comparison contract, stable pruning identities,
and configured local MIMIC demo/full validation. Unrelated foundation limitations
above are retained rather than treating the synthetic AL acceptance as a general
Lightning RL or native-backend resume guarantee.

Full reward-feedback THEMES (H), irregular-time MT-TICC, exact training resume,
general workflows and production Hub publication remain outside this milestone.
The new numerical package is local research material; resolve redistribution
rights before public distribution. See [the standalone guide](../guides/THEMES_AL.md).


## Initial phase F: runnable AL group (2026-11-11)

Delivered an editable storage-v2 suite at `src/usr/packages/themes-dev`, containing
the single independent `themes_al` package (relocated from `packages/themes-al`),
registered runner, numerical algorithm adapter, TICC model, trajectory source,
recorded-action evaluator, owned schemas/profiles, artificial input and AL recipes.
Active owned config copies in `in/config/experiment/AL` make the group visible in
the experiment panel. `AL/themes_synthetic_smoke` and `AL/themes_synthetic` both
run without external data. Restart the app once to load the backend code.

Runs use the common provider worker and the selected apprenticeship runner.
The adapter resolves selected providers, schedules explicit phases, calls the
same standalone numerical implementation, evaluates once, exports backend-native
inference bundles/plots, and records normal telemetry. Checkpoint/result state
remains isolated by method/run; provider snapshots include the numerical source.
No custom script entrypoint, installed package changes or publication was added.

The app now accepts registered `numerical` algorithm metadata, invokes optional
read-only runner validation on fully composed methods, and resolves dataset
package URIs through the existing provider loader. Missing data/providers,
incompatible data/model selection, invalid budgets, tuning and exact resume
fail preflight. Existing Lightning/SB3/d3rlpy dispatch behavior is unchanged.

Validation: **108 passed, two skipped** in 50.62 seconds across AL integration,
reconstruction parity, execution/dispatch and the experiment panel config model.
An additional AL two-seed repeat test passed in 12.50 seconds, verifying
separate fitted bundles, distinct seeds and independent metric log versions.
Tests exercise a clean workspace, panel group/profile visibility, GUI launch
preparation, real bootstrap execution, selected stage-model substitution,
one evaluator invocation, fresh-process reload/source lineage, standalone parity
and negative preflight. Optional installed-component tests account for skips.

Real staged smokes through `run_pipeline.py`:

| Recipe | Result |
|---|---|
| `AL/themes_synthetic_smoke` | Completed; 40 visits, 60 segments, two stages, one active policy; mixture NLL 0.690734612 versus constant 0.696526439 |
| `AL/themes_synthetic` | Completed; same fixed data/split, two active policies after four EM rounds; mixture NLL 0.507928651 versus constant 0.696526439 |

NLL is marginal recorded-action sequence NLL per scoreable held-out row. These
are artificial wiring/results demonstrations and do not establish clinical benefit.
Metrics CSV, segment tables, model bundles and overview plots were produced.
No clinical dataset, remote job, Hub package or production registry was changed.


## Downloaded CSV integration via custom data loader (2026-11-11)

Added selectable `themes-dev/csv_trajectories` (entry point
`csv_loader.py:ThemesCsvLoader`) with owned profile/schema. All CSV mapping,
validation, visit/time ordering, `a_2` action selection, row/timestamp lineage,
split construction and train-only preprocessing are owned by that data loader.
It delegates independent numeric work to `themes_al/csv_data.py` and the shared
preprocessor. The scientific runner only invokes selected `load`/`prepare`
methods. Standalone artifact verification reuses the persisted source schema,
without importing Theta or refitting preprocessing.

Two active AL recipes reference the external downloads in place:
`themes_mimic_q10_120` and `themes_mimic_cleantrain_q10_116`. Their default folder
is `~/Research/THEMES/datasets`, overridable with `THEMES_DATASETS_DIR` or the
experiment dataset path. No clinical file is copied into versioned provider
payloads or tests. Tests use generated CSVs only.

The first real Q10_120 run used 120 visits / 12,219 rows, with 96 training and
24 held-out visits. Binary `a_2` was selected as in the original script; the
three-class `Action` and partly missing row-level `Shock` were excluded.
The cohort file's expert designation is not independently validated. Actual
person identifiers are absent: the split establishes visit disjointness only.
`MinutesFromArrival` and zero-based source CSV row indices are retained. Solver
windows remain fixed-row and policy min/max spans three rows with one-row lag;
this does not claim irregular-time TICC or a 60-minute feature reconstruction.

The completed staged run produced five stages, 446 segments, three policies,
12,099 scoreable rows, and held-out mixture NLL **0.470436738** versus constant
rate baseline **0.669531862**. All final precision solvers converged. Fresh-process
verification reproduced source lineage, timestamps, stage labels, memberships
and held-out likelihood. These are retrospective recorded-action likelihoods,
not clinical outcome or policy-quality evidence. The cleantrain CSV has its own
recipe; no training run on it was required for the Q10_120-first acceptance.


CSV integration regression: **120 passed, two skipped** (72.55 seconds) across
custom-loader tests, AL integration, extraction parity, execution/dispatch and
panel config-model tests. Tests use generated CSVs and cover selectable provider
resolution, configurable columns/person split, rejected leakage/missing/fractional
inputs, timestamp ordering/lineage, ignored Shock/Action fields, train-only fitted
state, persisted custom schema, fresh-process reload, and both external recipe
preflights plus a real reduced-budget CSV worker run. No clinical data is in CI.


## Project-relative dataset path repair (2026-11-11)

The user-visible Q10_120 failure was a saved `Users/...` dataset path without a
leading slash, not an unresolved editable component. Repaired the active recipe
to `../../THEMES/datasets/MIMIC_md_expert_Q10_120.csv` while retaining its other
edits. Both owned CSV recipes now default to `../../THEMES/datasets`, with the
existing optional `THEMES_DATASETS_DIR` override. Relative paths resolve from
the workspace root in preflight and workers, including draft previews from a
temporary config directory. The custom CSV provider is unchanged; its adapter
passes explicit workspace context to the shared file resolver.

The shared dataset picker now saves relative paths for external files as well
as project-local files. Its status hint uses workspace resolution and evaluates
path interpolations without rewriting them. Explicit absolute/tilde paths remain
valid. Missing explicit files fail immediately instead of falling back to another
dataset. Dataset validation runs before component inventory errors, preventing
misleading Hub installation guidance; inventory diagnostics retain the file path
rather than substituting the method name.

Validation included 106 passing GUI form, AL, CSV-loader and path tests, plus
focused updated draft-preview and execution/dispatch checks. A pre-existing
`test_validation.py` baseline has 13 failures because its MockCfg has no `items`;
those failures reproduce on the unmodified runtime and are unrelated to this fix.


## Composable THEMES figures and numerical monitors (2026-11-11)

Added native `themes-dev/themes` plotter capability and owned plot profile. Eleven
Axes-aware panels read only completed saved bundles, including existing runs:
likelihood, support, action rates, stage/policy support matrix, lengths,
memberships, two timeline panels, two convergence panels and policy training CE.
PlotMan's provider bridge now resolves metadata-only plotter selectors to their
actual classes, isolating optional provider import failures. No plot-panel fitting
or implicit recomputation of evaluation is performed.

Added optional observation callbacks in the independent numerical core, plus
provider-owned scalar telemetry and bounded rich diagnostics. Original core
source hashes are retained; a new current hash records instrumentation. Existing
golden parity and standalone-vs-Theta comparisons verify unchanged fitting.
EM component IDs persist through pruning and are included in snapshots/checkpoints
and as an additive segment column. Iteration snapshots use the existing atomic
monitor snapshot contract, saved training-only probes, model/provider/probe hashes,
actual optimizer update counts and native matrix/distribution renderers. The
shared matrix renderer now accepts declared row/column semantics instead of
labeling every matrix row as a probe sample.

The monitor and PlotMan work share existing declarations/telemetry rather than
new AL-specific UI panels. Theta-wide reproducibility, exact resume and formal
baseline/ablation recipes remain separate tasks. This step does not turn the
post-hoc assessment into a pre-registered benchmark or add the THEMES reward loop.


Validation: **137 passed** in 70.07 seconds across the THEMES provider/PlotMan
panels, real AL launch/reload, golden extraction parity, generic figure composition,
native Diagnostics snapshot views and monitor selection. The metadata-only
catalog test confirms discovery imports neither Torch nor Qt. The real launcher
checks actual scalar columns, monotonic telemetry steps, fixed train-only probes,
rich snapshot loading, disabled collection, final snapshot retention, provider
substitution and isolated repeats. Numerical observer-enabled runs match standalone
outputs. The monitor-selection fixture was updated from obsolete custom-script
precedence to the registered-backend contract; old selectors remain rejected.
A six-panel composite was rendered from the user's existing Q10_120 bundle without
fitting or changing that run. Existing data/model artifacts are preserved.
