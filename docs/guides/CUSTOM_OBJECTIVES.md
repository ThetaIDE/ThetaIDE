# Editable objectives

In **Hyperparameter Sweep**, use the **Objective** selector to choose
`final_metric`, `learning_curve`, or a workspace objective. Open its source with the pencil action, or
edit the selected workspace component's Python code in Theta's configured text editor. Its
parameters appear directly beneath the selector in the sweep box. There is no
separate Objective box. Existing metric-based sweeps show **Reported metric**;
their metric remains editable underneath, without changing their execution mode.
When variability is configured for a reported-metric sweep, each trial uses the
same shared seed executor and returns the mean final metric across its seeds.

Objective parameters use the shared recursive parameter form. Defaults are
displayed without being saved as overrides; their tooltip identifies them.
Use the + beside Objective to add parameters, including parameters supplied by
your component's schema. Nested objects have their own child adders.

Example experiment fragment (add this to an otherwise runnable experiment):

```yaml
objective:
  component: learning_curve
  params:
    curve:
      reward_scale: 500
      half_life: 0       # Uniform weighting; positive values emphasize early learning.
    aggregation:
      method: lower_tail
      fraction: 0.2
variability:
  training_runs: 20
  starting_seed: 100
  evaluation_seed: 10000
  parallel_runs: 2
eval_episodes: 10
eval_freq_steps: 1000
tuning:
  component: optuna
  direction: maximize
  n_trials: 40
  options:
    optuna:
      pruner: none
```

The existing variability settings are the source of truth for samples. An
Optuna trial now means one configuration evaluated across the whole seed
panel. Without tuning, the same objective runner scores an ordinary experiment
with the same sampling protocol. With no variability block, it uses one seed.

This implementation supports the local launcher, including desktop resource
management. Both ordinary repeated experiments and resolved tuning trials use
the variance executor. Unmanaged runs honor `parallel_runs`; managed seed
workers acquire their own resource permits. Trial coordinators do not reserve
training capacity. Recovery, workflow and Slurm launches are rejected.

## Component files

Workspace components live in `src/usr/objectives/<name>.py` with a neighboring
`<name>.json` descriptor. Built-ins live in `src/app/objectives/builtins`.
Descriptors are read without importing Python source into the GUI.

Minimal descriptor:

```json
{
  "api_version": 1,
  "source": "my_score.py",
  "function": "evaluate"
}
```

Optional descriptor fields:

- `defaults`: default parameter values, recursively merged with overrides.
- `schema`: JSON Schema for parameters, used for validation and GUI controls.
- `requires.metrics`: recorded metric names that must be enabled.
- `requires.metric_parameter`: parameter naming the required metric.
- `requires.episode_outcomes`: require periodic online evaluation with episode outcomes.

Schema and defaults are optional. A component without them can still accept
arbitrary nested parameters added through the GUI or YAML. JSON Schema's
`required` declarations protect required fields from GUI removal.

```python
from src.app.api.tuning import ObjectiveResult

def evaluate(results, params):
    values = []
    for run in results.runs:
        # Failed runs are retained. Explicitly choose how they affect your score.
        if run.status != "completed":
            values.append(0)
        else:
            values.append(run.metrics[params["metric"]])
    return ObjectiveResult(
        value=sum(values) / len(values),
        diagnostics={"per_run": values},
    )
```

`results.config` contains the resolved configuration. Each run exposes `seed`,
`status`, final `metrics`, raw `events`, `elapsed_seconds`, `error`, and artifact
paths. `run.curve(metric)` returns sorted `(transitions, value)` pairs. Evaluation
events retain episode outcomes; telemetry records retain step and timestamp data.
An objective may return a finite numeric scalar or `ObjectiveResult` with
JSON-serializable diagnostics. It does not receive an Optuna trial or launch jobs.

The built-in learning-curve scorer integrates piecewise-linear returns over a
fixed transition budget, normalizes by reward scale, and supports uniform or
exponential weighting. Aggregation supports mean, mean minus sample standard
deviation, or the mean of the worst `ceil(fraction * runs)` scores. Completed
curves must cover transition zero through the budget; overshoot is clipped by
interpolation. Failed runs receive `failed_run_score` (default zero). This
default assumes a nonnegative normalized reward scale; customize it for other
tasks. The final-metric scorer fails if any sample fails or omits its metric.

For convergence-focused selection, set `curve.start_fraction: 0.75` to score
only the last quarter of the budget. Window boundaries are interpolated; the
same complete-curve coverage requirement still applies. Combine overall
performance with robustness using `aggregation.method: mean_lower_tail` and
`aggregation.tail_weight: 0.25`: 75% of the mean score plus 25% of the lower-tail
score, using the existing `fraction` to select tail samples. Failed runs remain
in both aggregates. Defaults preserve full-budget scoring. Positive
`curve.half_life` emphasizes the beginning of the selected window.

Learning-curve diagnostics include `window_start`, `completed_final_values`,
and `completed_window_minima`, alongside `per_run_auc`. These help distinguish
an early reward peak from sustained late performance.

## Results and current boundaries

Each independent sample has its own log/checkpoint version and a worker log
referenced by the shared execution manifest under `results/executions`.
The parent version's `execution.json` links to this manifest. The queue and
monitor follow the individual seed sources while training is in progress;
completed seeds leave the live monitor while their artifacts remain on disk.
After training, the parent version contains `objective.json` with the resolved parameters,
seed plan, all run statuses and data, score, and diagnostics. Its scalar appears
in Monitor/Results as `objective/<component>`. Tuning summaries link to the
objective artifact. A winning seed panel retains all checkpoints and selects
the configuration; it does not choose or promote an arbitrary sample model.

Source freezing and intermediate scoring are intentionally deferred. Source is
loaded fresh when a configuration is scored, so finish an edit before launching
a study and avoid editing it during a running or resumed study. The stored
parameters are a record, not a frozen source snapshot. Objectives currently
require pruning disabled and receive the complete sample panel once training
finishes. Scoring code never starts workers, reserves resources, or publishes
training status. Existing experiments without an objective retain their scalar-metric
behavior.

Bundled objectives now live in `src/plugins/components/bundled/{final_metric,learning_curve}`; the API stays in `src/app/objectives`. Package objectives are discoverable through the shared catalog and standalone authored descriptors remain under `src/usr/objectives`. Colliding IDs require an explicit user override; see [provider storage](../reference/PLUGIN_STORAGE.md). New-format metadata listings do not execute objective code.
