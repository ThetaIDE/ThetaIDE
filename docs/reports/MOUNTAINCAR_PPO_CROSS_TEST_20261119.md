# MountainCar PPO parameter transfer test (2026-11-19)

## Question and design

Do the different winning PPO settings from `MountainCar/sweep_DNN_v3` (trial 56)
and `MountainCar/sweep_BlendRL_v3` (trial 38) reflect reproducible model-specific
preferences, or do they transfer between DNN and vanilla BlendRL?

The recipe `BlendRL/MountainCar/PPO Cross Test` runs a 2 × 2 design:
both model architectures use each of the two exact saved PPO parameter sets.
BlendRL's extra model coefficients remain fixed at its winning values in both
conditions: `blender_lr=0.0002556303359452334`,
`logic_lr=0.00685570314014524`, and
`logic_confidence_floor=0.034220593418525326`.
Its Bets features remain at the vanilla defaults, as in the original PPO sweep.

There are 12 fresh training seeds, matched across all four conditions: 48 full
training runs, each with 120,000 transitions, four environments, bounds-scaled
observations and the original architectures. Evaluation uses 10 greedy and 10
sampled episodes at initialization and every 5,000 transitions. The held-out
evaluation seed is 23,000,000, shared by every condition. There is no tuning,
pruning or stopping based on interim results. Four workers run concurrently,
each with one numerical-library thread.

## Predeclared analysis

The primary endpoint is each seed's normalized sampled-return curve area from
0 through 120k transitions, using the original `return_curve` implementation:
−200 maps to 0, −100 maps to 1, endpoint values are clamped, and trapezoids are
integrated over the fixed budget. Higher scores reward faster learning and
better return throughout training. The original mean/worst-half aggregate
objective and final sampled/greedy returns are secondary descriptive checks.

The seed is the unit of inference. Three two-sided exact paired sign-flip tests
use all 4,096 sign assignments:

1. DNN's advantage from its own PPO settings versus BlendRL's PPO settings.
2. BlendRL's advantage from its own PPO settings versus DNN's PPO settings.
3. The model-by-PPO interaction: the benefit of BlendRL PPO over DNN PPO in
   BlendRL minus that same benefit in DNN.

The interaction is the sum of the two own-setting advantages. A positive
interaction means that relative preference shifts toward each model's own
settings; it does not by itself establish that both models prefer their own.
Holm adjustment controls the family of three tests at alpha 0.05. Paired
percentile bootstrap intervals use 50,000 resamples of complete seed blocks
with a fixed RNG. The reported 95% intervals are pointwise; significance claims
use adjusted p-values. Sign-flip inference assumes symmetric seed-level effects
under the null. Every run must complete and cover the evaluation budget before
analysis; failed or incomplete runs cannot be scored as ordinary observations.

The experiment tests transfer of these two selected parameter bundles. It does
not identify individual coefficient effects, establish unique/global optima,
or generalize to other training budgets. Inference conditions on this shared
evaluation schedule; lack of significance does not establish equivalence.

## Reproducibility

- Recipe: `src/usr/config/experiment/BlendRL/MountainCar/PPO Cross Test.yaml`.
- Execution: `826c457a05c643669a1d7de9cb94a038`.
- Saved run: `MountainCar/PPO Cross Test/20261119-090701_826c457a`.
- Protocol, objective source snapshot, launch log and analysis outputs:
  `src/usr/results/analysis/mountaincar_ppo_cross_test_20261119/`.
- Analysis: `scripts/analyze_mountaincar_ppo_cross_test.py`; pass the execution
  manifest under `src/usr/results/executions/BlendRL/MountainCar/PPO Cross Test/`.

The protocol was saved before training results arrived. The analyzer verifies
the complete 4 × 12 design, seeds, training budget, environment count, exact
PPO coefficients, fixed BlendRL coefficients and objective-source hash.

The recipe now draws a fresh matched set for each new launch. Run it from the checkout root:

```bash
MPLCONFIGDIR=/private/tmp/theta-ppo-cross-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 THETA_MANAGED_THREADS=1 venv/bin/python run_pipeline.py 'BlendRL/MountainCar/PPO Cross Test'
```

Analyze a new execution with `scripts/analyze_mountaincar_ppo_cross_test.py <manifest> --fresh-seeds`. Its analysis is saved under the new execution ID. The original report's seeds remain in its archived `protocol.json`.

## Results

All **48/48 runs completed successfully**, without retries or excluded seeds.
The analyzer's design/parameter/source checks passed. An additional check of
every evaluation event confirmed both greedy and sampled evaluations cover
initialization through the full budget, with 10 episodes at each event.
PPO finishes its last rollout slightly beyond the requested budget; areas and
endpoint returns below are truncated/interpolated at exactly 120,000 transitions.

| Model | PPO parameter source | Mean curve area ± seed SD | Original robust objective | Sampled return at 120k | Greedy return at 120k |
| --- | --- | --- | --- | --- | --- |
| DNN | DNN winner | 0.5492 ± 0.1938 | 0.4973 | −123.07 | −118.52 |
| DNN | BlendRL winner | 0.4566 ± 0.2743 | 0.3590 | −137.02 | −134.97 |
| BlendRL | DNN winner | 0.7372 ± 0.0468 | 0.7187 | −108.54 | −103.63 |
| BlendRL | BlendRL winner | 0.6881 ± 0.0614 | 0.6649 | −114.20 | −111.01 |

Higher is better in every metric. Curve areas are normalized scores, not raw
returns. The robust objective is the original 50% mean + 50% worst-half mean
over the 12 areas, reported descriptively.

| Predeclared contrast | Mean area difference | Paired bootstrap 95% CI | Exact p | Holm-adjusted p |
| --- | --- | --- | --- | --- |
| DNN: own PPO − BlendRL PPO | +0.0926 | [−0.0564, +0.2586] | 0.2866 | 0.5732 |
| BlendRL: own PPO − DNN PPO | −0.0491 | [−0.0871, −0.0129] | 0.0313 | 0.0938 |
| Model × PPO interaction | +0.0435 | [−0.0819, +0.1846] | 0.5649 | 0.5732 |

**These results do not establish that DNN and BlendRL need different PPO
settings.** The DNN-derived PPO bundle performs better on average for **both**
models, including under the original robust objective and the secondary
endpoint returns. The model-by-PPO interaction is not significant, and neither
within-model contrast passes the predeclared multiplicity correction.

BlendRL's advantage from using DNN PPO is suggestive: +0.0491 area, with an
uncorrected p of 0.0313 and a pointwise bootstrap interval excluding zero.
However, its adjusted p is 0.0938, so this is not a confirmed effect under the
predeclared decision rule. DNN itself has substantial seed variability and a
wide interval; failure to reject is not an equivalence result.

The original sweep winners therefore show numerical differences that **did not
reproduce as distinct model-specific preferences on these held-out seeds**.
The DNN PPO bundle is a reasonable shared baseline for subsequent experiments,
while its superiority is not established by the corrected tests. These data
also do not identify which individual PPO coefficient matters. Additional
independent seeds or a predeclared equivalence margin would be needed for a
stronger claim of interchangeability.

Machine-readable scores and tests are in `results.json`; all 48 seed-level
measurements are in `per_seed.csv`. `learning_curves.png` shows sampled-return
means and SEM bands across seeds. The bootstrap intervals in the table are
pointwise; their exclusion of zero does not override the adjusted p-values.
