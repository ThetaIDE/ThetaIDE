# Run the complete THEMES paper/source port

Select **AL → codex → themes_paper_synthetic** for an artificial smoke, or
**AL → themes_mimic_q10_120** for the downloaded 120-visit CSV.
Both use Theta's normal experiment worker and the editable `themes-dev` suite.
Restart the app after changing provider implementations.

```sh
venv/bin/python run_pipeline.py AL/codex/themes_paper_synthetic
venv/bin/python run_pipeline.py AL/themes_mimic_q10_120
```

The CSV recipe saves the project-relative path
`../../THEMES/datasets/MIMIC_md_expert_Q10_120.csv`.
The source is read in place. Its clinical data is not included in the component.

The full recipe runs ten outer rounds. Each fits time/reward-aware TICC on
training visits, discovers policies with hard EM and EDM, estimates high-level
rewards with maximum-likelihood inverse RL, and feeds those rewards into the
next segmentation pass. Reward feedback is part of the method and has no
optional disable switch. This is a documented paper/source port, not a claim of
exact equivalence to the unavailable original dependencies.

## Components and configuration ownership

| Selection | Owner and purpose |
|---|---|
| Agent `themes-dev/themes` | Outer/EM/EDM/IRL training budgets, optimization and likelihood coefficients |
| Model `themes-dev/full_bundle` | Explicit stage, policy and reward child-model selections |
| Stages `themes-dev/rmt_ticc` | Separate initial/feedback Toeplitz models, sparsity, time/reward switching penalty, Gaussian regulator parameters |
| Policy `themes-dev/dense_policy` | Configured `[32,32]` ReLU architecture and logit scale; input dimension comes from the loader |
| Reward `themes-dev/tabular_reward` | State/policy reward table, initialized by the selected model and optimized by IRL |
| Data Loader `themes-dev/time_csv_trajectories` | CSV mapping, validation, chronological visits, temporal aggregation, split and frozen preprocessing |
| Plotter `themes-dev/themes_full` | Saved full-loop results and reward/outer-loop figures |

These are selected defaults, not architectures chosen by the algorithm.
Unsupported parameters fail preflight. Configured rates and model parameters
are not silently filtered or replaced. Inverse reward initialization comes from
the selected reward model. Existing recipes and their overrides are preserved;
the new full-loop recipes have explicit names.

The data loader selects `VisitIdentifier`, `MinutesFromArrival`, binary `a_2`
and the declared 14 physiology columns. It builds current/trailing maximum/
trailing minimum features over 60 **minutes**, with an explicitly configured
zero-row lag. Feature normalization is fitted on training visits only. Configure
`person_column` when an actual person identifier is available; otherwise the
80/20 split is by visit, without claiming patient independence. Outcomes, other
interventions and the multiclass `Action` are excluded from policy inputs.

## Results, monitors and reload

Results appear under
`results/checkpoints/AL/<experiment>/themes/<run>/bundle/`.
`manifest.json` records the effective algorithm settings, preprocessing, child
model specifications, source checksum, per-round diagnostics and differences.
`SUMMARY.md`, `overview.png`, segment/source-row tables and full round model
artifacts are saved alongside it.

The Monitor includes TICC solver residuals/support, EDM losses, EM membership,
and outer-round training/held-out NLL, inverse reward losses, reward range and
feedback variability. Stage partition disagreement uses optimal label matching
so arbitrary stage-number permutations do not appear as structural changes.
Unidentifiable reward fits (no stage transitions or only one policy), optimizer
budget/failure and successful convergence are distinguished in the manifest.

PlotMan offers **themes-dev/themes_full**, including the existing eleven
trajectory/policy panels and three full-loop panels: outer likelihood,
state/policy reward matrix, and reward-feedback evolution. Plotters read saved
artifacts and do not refit models.

```sh
PYTHONPATH=src/usr/packages/themes-dev python -m themes_al verify --output /path/to/bundle
```

Verification checks source lineage and replays all frozen outer rounds,
including intermediate policies/rewards needed to reproduce final segmentation.
It does not train. New data prediction uses `SourceBundle.predict` with aligned
physiology, recorded binary actions and per-visit timestamps. Exact training
resume and tuning remain unsupported and are rejected before launch.

## Interpretation and historical recipes

Action NLL measures likelihood of recorded actions with retrospective segment
membership. It is not prospective accuracy, reward recovery or clinical benefit.
A successful run does not guarantee that later feedback rounds improve NLL.

Only `AL/themes_mimic_q10_120` is exposed at the AL root. Development,
synthetic and legacy recipes live in `AL/codex/`. Acceptance training runs use
`AL/codex/themes_mimic_q10_120_port`; their results also live beneath
`results/.../AL/codex/`. Existing legacy YAML content is retained there rather
than discarded. Legacy recipes remain gated unless explicitly used by a
historical test fixture.

The GUI preview resolves experiment interpolation after excluding Hydra's
internal runtime bookkeeping. This keeps valid dataset paths from being
misreported as missing components by the Run-button guard.

## Corrected reproduction defaults (2026-11-11)

The canonical recipe keeps the same 120-visit CSV, seed 42 and one 80/20 visit
split. No repeats, cross-validation, cohort replacement or filtering are added.
The supplied file has not been established to be the paper's 195-visit cohort;
poor model fit is not evidence that the supplied dataset is low quality.

The active and packaged profiles now select 11 physiological stages, a two-row
window, sparsity 1e-5, consistency coefficient 4, raw-minute time gaps, 50 TICC iterations, ten outer
rounds, up to ten EM rounds, 500 EDM updates, 10,000 replay particles and
32-by-32 hidden layers. EM likelihood tolerances are 1e-5 for convergence and
100 for a decrease, as in the available training script. The paper and available
script disagree on several settings; these defaults follow the paper for the
stage model, with the available script supplying the optimization budgets.

Policy inputs use current measurements plus trailing-hour extrema (`lag_rows: 0`)
and train-fitted [0,1] min/max scaling. Scaling remains fitted on training visits
only. Historical saved bundles retain their recorded lag and scaling when loaded.

EDM uses `CE + density + density**2`, matching the official EDM implementation's
loss, with `density_weight: 1`, `density_regularizer: 1`, and
`energy_regularizer: 0`. The previous port omitted the density-loss square.
Reducing density_weight to 0.1 was a diagnostic ablation, not a source-alignment
fix. Unknown settings from THEMES's unavailable modified EDM dependency are not
claimed recovered.

EM now scores convergence using conditional action likelihood without the
mixture-prior term, matching `Calculate_EM_EDM_Cluster_LLH`. If a candidate exceeds
the configured likelihood-decrease tolerance, it restores the previous accepted
models, priors and component IDs. Rollback is an explicit correction to the
available source's behavior, which also retained the deteriorated models. Final
empty components are pruned before constructing reward-model axes. Manifests
record the accepted iteration, stop reason and rejected candidate count.

The Gaussian regulator wrapper remains unavailable. No newly invented neutral
normalization is substituted for the published formula. Each outer-round record
now includes training-set median switching penalties for time-only, zero-reward
and actually applied regulation. A penalty-scale change alone does not prove
that learned rewards improved segmentation. The current independent-CDF
interpretation remains a documented port difference. Time gaps now enter the
formula in minutes (`time_unit_minutes: 1`), matching the declared timestamp unit.

Reference for restored EDM loss and replay sampling:
[official EDMStudent](https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/alg/edm/student/edm_student.py),
retrieved 2026-11-11, Git blob `fe1932ad486e2043775a5cd8df354b36a4e200a9`.

## Source-alignment corrections (2026-11-12; supersedes earlier defaults)

The canonical experiment now explicitly follows the visible schedule in
`reference/original/THEMES_src/THEMES_learn.py`, retaining the same clinical CSV,
seed 42 and single 80/20 split. This is still a partial source reconstruction.
Other source entry points have different settings; these are not claimed to be
universal paper defaults. No sweep or additional data splits are introduced.

| Setting | Initial stage | Feedback stage |
|---|---:|---:|
| Window rows | 2 | 3 |
| Sparsity | 0.00011 | 0.011 |
| Switching coefficient | 4 | 10 |
| Stages / iteration budget | 11 / 50 | 11 / 50 |

`full_bundle.yaml` owns both `stage` and `feedback_stage`. The Toeplitz solver
now supports the three-row window; windows remain isolated within each visit.
Historical two-row numerical behavior is preserved.

`reward_feedback: segment_posterior` computes each segment's reward as its
normalized policy posterior dotted with the stage/policy reward row, divided by
policy count, and repeats that value for every row of the segment. Component
axes remain explicitly aligned. This matches the visible helper calculation;
its column-truncation workaround is not copied. The previous row/action-probability
convention remains available for replay and explicit comparisons. Every saved
round records its convention. Old bundles lacking the field replay their
historical row/action-probability convention without retraining.

EM initialization matches the source's strided equal partition with
`initialization_seed: 0`. Student seeds are derived locally from each group's
trajectory count as in EncapsulateEDM; the previous outer/EM arithmetic seed
offsets no longer determine policy fits. The experiment seed and split are
unchanged. Complete parity with RNG side effects inside missing dependencies
is not asserted.

IRL now supplies analytic derivatives of the ordinary optimal-Q fixed point
under a selected greedy policy, replacing finite differences through value
iteration. It matches the source's extra division by trajectory count. Gradient
checks use non-tied MDPs; at ties, one greedy policy defines the piecewise
derivative, rather than substituting a soft Bellman operator.

Policy features retain current-row/trailing-hour alignment and train-fitted
min/max scaling. `clip_policy_inputs: false` removes the unsupported saturation
of held-out values to training extrema. Old artifacts lacking this field retain
their historical clipping during replay. Stage z-score/clipping and covariance
ridge remain documented choices because upstream stage preprocessing is absent.

### Still unavailable; not silently guessed

- The REGTICCSplitTrajs implementation defining Gaussian PDF/CDF and threshold
  semantics. Source arguments `[1,1]`, `[.2,1]` and `rewards_thres=0.5` are recorded
  as unimplemented reference parameters, not passed to a guessed substitute.
  The independent-CDF approximation keeps its existing explicit parameters.
- THEMES's modified balanced buffer. Actual positive-row sampling is explicitly
  `demonstration_sampling: uniform_rows`. Requesting balanced sampling fails
  instead of quietly using uniform rows. Class balance is not inferred from
  the word balanced.
- Original TICC preprocessing/stabilization and missing serialized parameters.

These dependencies are recorded in new manifests. Exact source or published
performance parity is not claimed. The final round still saves an additional
reward fit for diagnostics; it does not feed another segmentation pass.
