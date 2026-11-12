# THEMES source audit and implementation

The earlier AL integration used `THEMES/demo`, not the original
`THEMES/THEMES/THEMES_src`. Its results do not validate original THEMES.
The complete paper/source port is selected explicitly through
`AL/codex/themes_paper_synthetic` or `AL/themes_mimic_q10_120`.
Existing legacy experiment configs are preserved under `AL/codex/`; the canonical
`AL/themes_mimic_q10_120` selector exposes the ready full method. Development
acceptance outputs are archived under the codex results subtree.

## Preserved source

`src/usr/packages/themes-dev/reference/original/` contains byte-for-byte copies
of all 23 available Python files, `config.yml` and `submit_cca_array.sh`.
`SOURCE_MANIFEST.json` records their hashes and original location. Cohort data,
quality indices and pickled parameter files are excluded. Reference scripts
remain inert: importing them wholesale would mutate paths/configuration and
perform file I/O or construct environments.

Missing original EDM, preprocessing/TICC and shared-header implementations were
not recovered. The user authorized implementations from the papers and available
source with documented differences. This port does not claim exact missing-code
parity or the paper's published performance.

## Method mapping

| Implemented part | Basis |
|---|---|
| Full outer loop (`source_run.py`) | THEMES Algorithm 1: segmentation → EM-EDM → inverse reward learning → feedback |
| TICC likelihood and structure (`rmt_ticc.py`, tested precision kernel in `core.py`) | Paper equations 1–2, graphical lasso with two-block Toeplitz constraint |
| Time/reward consistency | Paper equation 3 with explicit Gaussian-CDF interpretation and documented parameters |
| Segment reward feedback (`segment_rewards`) | Available GenerateRewardsWeights helper: posterior-weighted rewards constant within segments; legacy row_rewards remains for old bundles |
| Hard EM (`SourceEMEDM`) | Available `EM_EDM_utils.py`: random equal initialization, fresh M-step models, hard assignments and likelihood stopping |
| EDM objective and persistent SGLD | Jarrett et al., NeurIPS 2020; explicit coefficients/budgets, supplied policy model |
| Discrete transition estimates | Available `GenerateDiscreteSAP`; direct source-function parity test |
| Inverse reward learning | Active ordinary-optimal-Q likelihood path in `NV_MLIRL.py`; analytic greedy-policy fixed-point derivatives replace unavailable gradient helper |
| Minute-window preprocessing | Available scripts' trailing-60-minute aggregation, with explicit safer split/scaling/lag differences |

The selected model bundle owns **all** stage, policy and reward components.
Algorithms never select an architecture or supply hidden-width defaults.
The selected reward model supplies its initial values; IRL updates those values
within explicitly configured bounds. The runner supplies the selected model
bundle to the algorithm. No installed components are modified.

## Explicit choices and differences

- Gaussian regulation: the paper identifies a bivariate Gaussian but does not
  specify PDF versus CDF or a fully reproducible parameterization; the original
  wrapper is missing. This port uses the product of two independent normal CDFs
  evaluated at absolute reward differences and `log(e + gap/time_unit)`, with
  switching penalty `beta / max(CDF, cdf_floor)`. Larger gaps/reward changes relax
  switching penalties. Means, standard deviations, time unit and floor belong to
  the selected stage model and are saved. Defaults `[0,1]`, `[0.2,1]`, 1 minute
  and 0.001 are explicit port choices; the available script's `[1,1]` means are
  not silently claimed as equivalent. Initial segmentation uses the paper's
  time-only decay, with no learned reward regulator yet.
- Initial and feedback models use two- and three-row windows respectively,
  matching the selected THEMES_learn.py entry point. A general block-Toeplitz
  ADMM projection extends the kernel while preserving the two-block path. A configured covariance ridge and a small positive-
  definiteness shift stabilize precision estimates; diagonal entries are not
  L1-penalized. Solver convergence/residuals are retained. Temporarily empty stage
  components preserve their parameters and identities instead of renumbering.
- Feedback now uses **segment policy posteriors**, reproducing the available
  helper arithmetic and broadcasting each reward within its segment. Reward
  columns stay aligned; the source truncation workaround is not copied. The
  previous per-row equation-4 convention is retained for historical replay.
- Policy features use a real trailing-60-minute window. The loader's explicit
  zero-row lag, train-only min/max scaling to [0,1] and train-only physiology
  z-score/clipping use the script's current-row alignment but differ from its global-cohort scaler fitting.
  Those choices are explicit loader settings/metadata. The original expert-quality
  split is unavailable, so the supplied expert CSV is accepted as the cohort and
  split by person when configured, otherwise by visit.
- EDM uses an explicit persistent SGLD sampler without the demo's particle clamp.
  Uniform [-1,1] initialization and replay indices sampled with replacement follow
  the official EDM implementation. The objective now restores its density-loss
  square, explicitly configured by density_regularizer. Uniform positive-row
  sampling remains a difference from the THEMES script's balanced-buffer selection;
  the modified buffer implementation is unavailable. This is not the absent
  THEMES-modified EDMStudent code.
  The model owns logit scaling; EM owns the separate Boltzmann likelihood scale.
- EM uses stable log likelihoods, local RNG and fresh supplied policies each
  M-step, instead of the demo's KMeans initialization and warm starts. Empty/all-
  pruned configurations fail explicitly. Conditional convergence scores exclude priors, matching the source helper.
  Rejected likelihood-degrading candidates restore the previous accepted model
  state, deliberately correcting the available source. Final empty components
  are pruned while retaining stable component IDs.
- IRL uses bounded L-BFGS-B with analytic greedy-policy fixed-point derivatives
  and ordinary Bellman optimality. It retains equal visit weighting, final-pair
  exclusion, and the source's additional division by visit count. Zero observed transition rows stay
  zero and contribute zero transition-weighted Q. Bellman iteration budgets are
  enforced. Optimizer nonconvergence and unidentifiable cases are reported.
- Outer iterations use the configured fixed budget. Held-out likelihood is
  diagnostic; it does not select a checkpoint or silently stop training. Reward
  feedback is always part of the full method, with no disable option.

## Artifacts and validation

Version-2 bundles save **every outer round's** stage parameters, policy specs and
weights, priors/component axes, reward values and dynamics. Fresh inference
replays frozen rounds to compute feedback and final segmentation; it does not
refit. Temporal preprocessing, effective settings, input checksum and explicit
source lineage are preserved. Version-1 historical readers remain available.

Tests cover exhaustive dynamic-programming minima, known-stage recovery,
patient-boundary isolation, minute-based windows, held-out scaler isolation,
actual reward-induced partition changes, model selection/shape preservation,
strict parameter rejection, source helper parity, EDM gradients, IRL likelihood
improvement and convergence, complete worker execution, fresh reload, monitors,
and all PlotMan panels without fitting or writes.

Scientific limitations remain: clinical action semantics are not inferred,
retrospective membership uses recorded actions, and likelihood improvement is
not evidence of improved treatment outcomes. The implementation has not
reproduced the published experiment protocol or performance.

Sources: [THEMES paper](https://arxiv.org/html/2305.09070v1),
[EDM paper](https://proceedings.neurips.cc/paper/2020/file/524f141e189d2a00968c3d48cadd4159-Paper.pdf),
and the preserved `THEMES_learn.py`, `EM_EDM_utils.py`, `NV_MLIRL.py` and
`THEMES_helper.py`.

## Reproduction corrections and remaining limits

See [corrected defaults and implementation changes](THEMES_AL.md#corrected-reproduction-defaults-2026-11-11)
for the 2026-11-11 fix. The density-loss square was omitted from this port; the
official EDM code supplies evidence for restoring it, although THEMES's modified
EDMStudent is still missing. The earlier claim that rollback would reproduce the
available EM source exactly was incorrect: rollback is a deliberate correction.

The independent Gaussian-CDF wrapper has not been recovered. Its switching
penalty scale jump is now measured per round, not silently normalized away.
Neither exact Gaussian regulation nor published-performance parity is claimed.
The available script also changes window size, sparsity and consistency strength
for feedback passes, unlike the paper's single reported parameter set. Current
profiles use the paper's two-row window and sparsity 1e-5 throughout, with
explicit budgets from the script. The cohort and single split remain unchanged.

## Current source schedule (2026-11-12)

The [source-alignment correction guide](THEMES_AL.md#source-alignment-corrections-2026-11-12-supersedes-earlier-defaults)
supersedes the earlier hybrid paper/script defaults. New manifests explicitly
record the selected source entry point, feedback convention, missing dependencies
and unimplemented source arguments. Per-phase window/sparsity/beta, segment
reward construction, EM partition/student seed rules and IRL objective scale
and derivative path are corrected. The Gaussian and balanced-buffer semantics
remain unavailable; matching their call-site argument names is not parity.
