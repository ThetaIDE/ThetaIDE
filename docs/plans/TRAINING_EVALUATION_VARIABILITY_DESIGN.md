# Training and evaluation variability design

Status: proposed design for review; not implemented.

## Purpose and terminology

Let a user measure variation between independently trained models and variation across evaluation episodes for each model. An existing experiment group remains a directory and inheritance mechanism; it does not become a collection of executions. An **execution** is one launch of an experiment. A **replicate** is one independent training run of one method within that execution. A **suning trial** is one sampled parameter configuration. These concepts are distinct from the existing experiment group.

For example, two methods, five training seeds and 20 evaluation episodes means ten trainings, each evaluated on 20 episodes at each scheduled evaluation point. The episodes measure variation conditional on a trained model; the five trained models per method measure training variability. They are not 100 independent trainings.

## Experiment editor

Add an optional **Training and Evaluation Variability** box under Add Overrides, inheritable like other experiment settings. Its proposed controls:

- Training repetitions: default 1; either count plus starting seed, or an explicit unique seed list. Show the resulting seeds and number of trainings per method and in total.
- Evaluation episodes per evaluation: a positive integer when evaluation is enabled. Reuse the existing `eval_episodes` setting rather than introduce a competing value; surface the effective inherited value and preserve existing configs.
- Evaluation seed policy: a reproducible base seed, independent of training seeds, with documented derivation for evaluation events and episodes. Offer a fixed evaluation seed set across replicates for comparable scenarios; do not imply this alone guarantees paired statistical independence.
- Parallel trainings: a resource limit, subject to the selected launcher's actual capabilities. Display total training and approximate evaluation workload before launch.

Evaluation frequency and final-checkpoint policy should reuse existing settings where supported. Identify unsupported runner/paradigm combinations in preflight rather than silently ignoring settings. For an absent box, preserve existing behavior exactly, including evaluation defaults; adding the box must not implicitly change an existing experiment's episode count. Leave disabled evaluation explicit. Seed control is best effort unless each component supports it; record what was passed and avoid promising deterministic GPU results.

Illustrative semantic configuration, with exact keys settled after auditing config composition:

```yaml
variability:
  training_seeds: [1, 2, 3, 4, 5]
  evaluation_seed: 1000
  parallel_runs: 1
eval_episodes: 20  # existing setting, surfaced in the box
```

Validate positive counts, unique seeds, valid seed ranges, compatible evaluation metrics, runner support, concurrency and resource requirements. Preserve unknown YAML and existing experiment inheritance. A run with one training seed and multiple episodes is a useful configuration, but cannot estimate training variability.

## Execution and records

Create an immutable execution manifest with a unique execution ID, experiment group/path, resolved configuration, method identities, planned replicates, training seeds, evaluation protocol, statuses and artifact locations. Each replicate receives a unique run ID and isolated telemetry, checkpoints, datasets and plot outputs. Do not rely on timestamp alone for uniqueness or overwrite prior executions. Existing single-run layouts remain readable; a compatibility adapter can expose a legacy run as a single-run execution without rewriting it.

Use the existing individual training path for each replicate, with a shared normalized plan for local and Slurm launchers. Bound concurrency, record queued/running/completed/failed/cancelled states, and allow inspection of each run. A failed replicate remains visible; never silently replace or discard it. Partial results must show completed versus planned counts. Define retry as a new attempt associated with the same planned replicate, retaining attempt history and excluding superseded attempts from aggregation. Implement resume only once its checkpoint and scheduling semantics are reliable.

Audit random seed propagation across supported runners, environments, libraries and vectorized workers. Evaluation must use a separate environment and reproducible seed policy where supported. The current Lightning callback seeds its evaluation environment from `cfg.seed + 100` and logs episode mean and standard deviation; this couples evaluation to the training seed and does not preserve episode outcomes. The SB3 runner has its own evaluation path. Standardize their observable contract without assuming they currently behave identically.

## Evaluation and statistical contract

Persist each evaluation event with run ID, method, checkpoint or training step, metric, episode count, evaluation seed information and individual episode outcomes when the runner can provide them. Keep existing scalar telemetry for live monitoring. Record which checkpoint is evaluated and whether actions are deterministic; use the same protocol for comparisons. If outcomes cannot be obtained, advertise only the summaries actually available.

Summarize episodes within each replicate first. For training variability, treat each replicate's evaluation summary as one observation; never pool all episodes as independent training runs. For episode variability, display the distribution or standard deviation of episodes for a selected model and evaluation event. Label every band and its unit: standard deviation describes observed spread; a confidence interval describes uncertainty in a specified aggregate. Show sample sizes and suppress a training-variability band for fewer than two completed replicates. Small-sample intervals can be unstable; do not imply that five seeds ensure precision.

For the initial aggregate curve, use the mean across replicate evaluation means and optionally show standard deviation. Align by an explicitly selected common training-step grid; interpolate only within each run's observed range, never extrapolate, and show contributing run counts as they change. Keep raw points accessible. For final comparisons use a declared common budget/checkpoint rule, report per-replicate scores and failures, and do not automatically select each run's best checkpoint. Confidence intervals and paired comparisons can follow after their assumptions and resampling unit are specified and tested.

## Plotters, results and tuning

The existing plotting phase invokes `plot/manager.py` for an experiment and plotters are pluggable. Audit the actual inputs and outputs of each selected plotter before changing its contract. Introduce a versioned, read-only execution data interface that exposes individual runs, their metadata and evaluation events; provide a legacy adapter for single-run plotters. Declare plotter capabilities such as single-run and replicate-aggregate. Unsupported plotters should still work on an individual run and clearly explain when an aggregate is unavailable. Do not feed multiple runs to an old plotter as though they were one.

Implement one aggregate learning-curve plot first, using the statistical rules above. Results should show an execution selector, replicate status and seed list, individual runs, and aggregate curves; the existing single-run monitor remains available. Keep episode variation and training variation separately selectable and labeled. Make partial data and plotting failures visible; do not block training completion on an aggregate plot.

Tuning also produces multiple runs, so it benefits from the same execution data interface and run identity. However, different tuning trials have different parameter values and must not be aggregated as repeated samples of one configuration. Preserve trial ID and sampled parameters. A future tuning design can nest replicates under each trial and define the objective as an aggregate of their evaluation scores; it must specify failure handling and compute cost before enabling this combination. The existing sweep design describes trial-level objectives and a separate Sweeper view; this proposal does not assume replicate-aware tuning already exists.

## Delivery sequence

1. Audit evaluation settings, seed propagation, artifact paths, launcher behavior and plotter inputs for each supported runner. Define capability reporting and finalize configuration and manifest schemas.
2. Add normalized repetition plans, validation, isolated run identities and local execution. Preserve one-run and legacy behavior; make per-run status and failures observable.
3. Standardize evaluation event recording and episode-level data for an initial supported RL runner; extend runner support only when seed and evaluation behavior are verified.
4. Add the editor box and execution/replicate navigation, including workload preview and individual monitoring.
5. Add the plotter data adapter and one aggregate learning curve; show per-model episode distributions separately and retain legacy plotters.
6. Bring Slurm to equivalent semantics where supported, document limitations and migration, and later design replicate-aware tuning and confidence intervals.

## Acceptance criteria

- Five seeds and 20 episodes produce five independently trained runs per method, each with its own artifacts and recorded evaluation protocol. Repeating a launch never overwrites the previous execution.
- Evaluation episode outcomes can be inspected per model; an aggregate across seeds uses one summary per model and labels its band and sample count.
- Failed or unfinished runs are visible and never silently treated as successful; partial curves show contributing counts and never extrapolate beyond observed data.
- A legacy experiment without the box retains its prior behavior, and existing single-run plotters and results remain usable.
- Seed/configuration validation and unsupported runner features fail clearly before launch. Local and supported Slurm executions use the same replicate semantics.
- Tuning trials remain distinguishable by configuration; trial results are not presented as seed variability.

Verify with focused tests for config inheritance and validation, seed and artifact isolation, retry/partial-failure accounting, event recording, aggregation/alignment on known data, legacy plotter compatibility, and supported launcher parity. Add a small end-to-end seeded RL experiment to check the complete workflow without requiring 100 trainings.
