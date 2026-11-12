# THEMES editable component suite

The complete paper/source port is runnable through `AL/codex/themes_paper_synthetic`
and `AL/themes_mimic_q10_120`. The selected composite model owns stage,
policy and reward model construction. The custom temporal CSV loader owns all
source mapping, minute-window aggregation, split and frozen preprocessing.

See the [run guide](../../../../docs/guides/THEMES_AL.md) and
[source audit](../../../../docs/guides/THEMES_SOURCE_AUDIT.md) for commands,
model/algorithm/data settings, full-loop monitors/figures and explicit differences
from unavailable upstream implementations. Version-2 artifacts replay all frozen
outer rounds without fitting. Original source is preserved with checksums under
`reference/original`; no downloaded clinical data or installed components are modified.

The following retained documentation describes the historical independent demo,
not the new full method. Its old Theta recipes are gated by default.

# TICC + EM-EDM reconstruction

This editable `themes-dev` suite also owns the runnable `AL/themes_synthetic_smoke`
and `AL/themes_synthetic` recipes. Its `theta_adapter.py` connects the independent
numerical package to Theta's registered apprenticeship runner and telemetry;
`themes_al` itself remains independent of Theta. Profile sources live in `configs`,
with active copies in `in/config` for the experiment panel. Backend snapshots
include the numerical source and artificial demonstration resource.

```sh
venv/bin/python run_pipeline.py AL/themes_synthetic_smoke
```


Standalone numerical extraction of the existing `THEMES/demo` reconstruction.
Its core imports NumPy, SciPy, scikit-learn and PyTorch. It has no Theta, Qt,
Hydra, environment, or experiment-coordinator dependency. The extraction
retains original source hashes in `PROVENANCE.json`. Optional observer callbacks
and stable policy IDs have since been added; numerical parity tests preserve
the original fitting behavior. The current core hash is recorded separately.
This package is for local research. Original source redistribution/license
review is required before public distribution.

From the Theta checkout, using an environment with the declared dependencies:

```sh
PYTHONPATH=src/usr/packages/themes-dev python -m themes_al synthetic --output /tmp/themes-smoke
PYTHONPATH=src/usr/packages/themes-dev python -m themes_al verify --output /tmp/themes-smoke
PYTHONPATH=src/usr/packages/themes-dev python -m themes_al run \
  --dataset /path/to/trusted-local-mimic.npz --output /tmp/themes-local --patients 512
```

The package can also be installed from this directory; `themes-al` is its CLI.
No dependency installation or installed Theta component changes are necessary
when running directly with `PYTHONPATH`. Choose empty output directories;
synthetic input is written beside the output as `<name>_synthetic.npz`.

`core.py` exposes `MultiSeriesTICC`, `make_segments`, `Policy`, `fit_edm`,
`EMEDM`, `segment_log_scores`, and `responsibilities` independently.
`data.py` owns the MIMIC-schema source adapter and train-fitted preprocessing;
`demo.run` owns the explicit fit/export schedule, with optional plotting.
`InferenceBundle` reloads frozen preprocessing, stage models, and policies:

```python
from themes_al.artifacts import InferenceBundle
bundle = InferenceBundle('/tmp/themes-smoke')
# Each visit: N x 14 array in PHYSIOLOGY order; actions: aligned binary vector.
result = bundle.predict(observation_visits, action_visits)
```

Artifacts have version 1 and checksums. `verify` checks the source checksum,
visit IDs, original slots, observations/actions, segment coverage, fitted stage
labels, policy inputs, memberships, and held-out mixture likelihood. Numerical
exports load with `allow_pickle=False`; policy weights load with
`weights_only=True`. The trusted source adapter accepts NumPy object feature
names and therefore uses `allow_pickle=True` for source data only.

The MIMIC adapter reads `X`, `mask`, `y`, `vids`, and `feature_names_x`. Padding
uses `mask != -1`; valid rows must be contiguous. The 14 physiological means
exclude interventions and outcomes. Actions provisionally use
`AntiInfectiveAdmin_max`; min/max augmentation and one-row lag produce 42 policy
inputs. The first row of each visit is unscored. If `patient_ids` is supplied,
visits split by that actual person ID; otherwise the original outcome-stratified
80/20 **visit** split is preserved for reconstruction parity. Source row indices
are lineage, not evidence of patient independence. The upstream normalization
and real measurement/action timing remain unknown. No timestamps are inferred.

Scalers, stages, priors and policies fit training rows/segments only. Held-out
NLL sums segment log mixture marginal likelihoods, including all policies and
training-fitted priors, then divides by the number of scored held-out rows.
The comparison is a Bernoulli rate fitted only on scored training rows. It does
not select the best test component. Singletons remain in exports and inherit
priors if unscoreable. Group membership and stage inference are retrospective.

This is fixed-spacing TICC and hard EM-EDM. It does not implement reward-feedback
THEMES or irregular-time MT-TICC. Inference reload is supported; exact training
resume is rejected because optimizer, persistent SGLD, RNG, and EM iteration
state are not all persisted. Original component numbering is compacted during
pruning; stable component lineage is still part of the later provider work.

The original demo settings and 80/20 split are retained for parity, including
five stages, two-row windows, sparsity 0.01, switching penalty 4, three initial
policies, six EM rounds, 250 updates per policy/round, ten SGLD steps, density
weight 0.1, and seed 42. Synthetic smoke budgets are intentionally tiny. New
parameter comparisons require a separate fixed train/validation/test contract;
this CLI does not tune against the reconstruction's held-out split.


## Downloaded CSV data loader

`themes-dev/csv_trajectories` is the selectable custom data provider for the
THEMES downloads. `csv_loader.py` owns dataset configuration and preprocessing;
`themes_al/csv_data.py` implements the independent CSV reader shared with the
standalone CLI. The runner receives the loader's data and preprocessing methods.
No CSV rows or clinical files are included in this package.

`AL/themes_mimic_q10_120` and `AL/themes_mimic_cleantrain_q10_116` point to the
respective CSVs under `~/Research/THEMES/datasets`; `THEMES_DATASETS_DIR` overrides
the folder. Column mappings and optional actual person ID are loader parameters.
Defaults use the binary `a_2`, not the multiclass `Action`. Time and source-row
lineage are preserved; the numerical model remains fixed-row TICC + EM-EDM.


## PlotMan and live monitors

The suite exposes the composable plotter **`themes-dev/themes`**. Select it in
PlotMan for likelihood comparisons, policy support/rates, stage-policy heatmaps,
segment lengths, membership distributions, stage/policy timelines, TICC/EM
convergence and per-policy training CE. These eleven panel types read completed
saved bundles; existing AL runs need no retraining. Each panel draws into a
caller-owned axes and can be composed, styled and exported with other plotters.
`split`, `max_visits`, `panels`, figure size and DPI are owned plot profile controls.

New runs stream TICC solver/assignment diagnostics, EM likelihood/support/priors,
and per-policy CE, density, total loss and SGLD gradient diagnostics. The numerical
core only emits observation callbacks; Theta-specific collection is owned by
`training_monitor.py`. Diagnostics reuse `theta.monitor-snapshot/v1` and the
existing matrix/distribution views. Policy responses and training-median feature
replacement use a fixed, saved bank drawn only from scoreable training rows.
They are model sensitivity measurements, not causal treatment effects.

The existing monitor/recorded-metric controls gate optional probe work before
allocation. `monitor_probe_count` defaults to 64 and `monitor_max_snapshots` to 8;
the cap reserves the final trained state. Policy IDs remain stable through EM
pruning, and exports add `policy_component_id` alongside compact `policy_group`.
Numerical parity is checked with and without observation enabled. Old runs have
saved convergence traces but cannot acquire historical probe snapshots after
training. Fresh runs are needed for live curves and rich snapshot playback.
