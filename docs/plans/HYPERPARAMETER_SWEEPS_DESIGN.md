# Hyperparameter Sweep Design

## Status

Design draft for review. This describes the intended user experience and pipeline contract before implementation.

## Summary

An experiment becomes a sweep when the user adds the **Hyperparameter Sweep** box from **Add Overrides**. Every enabled method in that experiment is swept independently. Users opt individual parameters into the sweep from their parameter rows, configure each parameter's search distribution inline, and set the optimization metric and run-level sweep settings in the sweep box.

The Training Monitor shows live sweep progress. Results shows finalized trial artifacts and metrics only; it does not poll running trials. Refresh loads newly finished trials, including those from studies that are still running.

## Goals

- Make sweep mode explicit and discoverable in the experiment editor.
- Let users choose exactly which parameters of each method vary.
- Support common categorical and numeric distributions without requiring Hydra syntax.
- Run each enabled method as its own study with a shared objective and run-level settings.
- Reject malformed or unscoreable sweeps before launching training.
- Keep live trials in the Training Monitor and publish each trial to Results only after it ends.
- Preserve existing YAML and CLI sweep workflows during migration.

## Out of scope for the first release

- Promoting or copying the winning checkpoint into the normal checkpoint location.
- Conditional search spaces where one parameter changes which other parameters are active.
- Multi-objective optimization.
- Pruning trials before their configured training budget is complete.
- Distributed Optuna storage and cross-machine trial workers beyond the existing local and Slurm launch paths.

## User experience

### Add a sweep

The **Add Overrides** menu contains **Hyperparameter Sweep (Optuna)**. Adding it creates a visible sweep configuration box and turns the experiment into a sweep. Removing the box removes sweep mode and its run-level sweep configuration. Method parameter values remain unchanged.

The sweep box contains:

- **Objective metric**: metric to optimize, selected from metrics supported by the current paradigm and runner.
- **Direction**: maximize or minimize. Default is derived from the metric catalog where known, with paradigm defaults as a fallback. The user can override it.
- **Trials**: number of trials per method; default 40.
- **Parallel trials**: number of concurrent trials per method; default 1.
- **Advanced settings**: sampler, random seed, and storage URL. The first release can retain current defaults and expose these only if the advanced section is opened.

The objective is shared by the methods in one experiment. A method that cannot produce the selected objective must fail preflight with a method-specific explanation.

### Mark a parameter for sweeping

When a sweep box exists, hovering a parameter row reveals its remove control and a sweep icon next to it. The sweep icon toggles sweep editing for that parameter. Existing methods and parameters remain ordinary fixed values until explicitly selected.

Activating the sweep icon:

1. Preserves the current parameter value as its base value.
2. Adds a sweep editor with a distribution dropdown and the fields for that distribution.
3. Shows a **Remove sweep** action.

Removing a sweep editor clears only that parameter's search-space definition. The base parameter and its ordinary remove control remain. Removing the base parameter also removes its sweep definition.

### Parameter distributions

The initial distribution choices are:

| Type | Inputs | Meaning |
| --- | --- | --- |
| List | One or more values | Sample one listed value; the current base value starts as the first item. |
| Integer range | Minimum, maximum, optional step | Sample an integer in the inclusive range, respecting step. |
| Numeric range | Minimum, maximum, optional step | Sample a number in the range. |
| Log numeric range | Positive minimum and maximum | Sample on a logarithmic scale. |

The list editor starts with the base value and one empty additional value field, matching the requested “add another value” interaction. Users can add and remove list entries. Empty entries are not valid at launch. Numeric distributions replace the list-entry editor with their bound fields. A parameter's base value is retained in the experiment as its fixed/default value, even when the sampled value overrides it for each trial.

The editor should infer a likely distribution from the parameter's schema and current value, but the dropdown remains editable. For example, booleans and enum fields default to List; integer fields default to Integer range; floating-point fields default to Numeric range. Log range is offered for positive numeric types.

### Method scope

All enabled methods participate. Each method gets its own study, with only the parameters explicitly marked for that method included in that study. Shared parameters under `methods.params` apply to every method; if one is marked for sweeping, that same dimension is included in each applicable method's study.

Every swept method needs at least one active search dimension. Preflight reports the names of methods with no search dimensions and asks the user to add a parameter or disable the method. It must not silently run those methods once while presenting the experiment as a sweep.

## Configuration representation

Use the existing top-level `tuning` block for run-level settings and the existing method-level `tune` mapping for dimensions. The presence of `tuning` in the composed experiment means sweep mode, including settings inherited through Hydra defaults. A null value disables inherited tuning. Legacy `sweep` and `tuning.enabled` flags do not control activation.

Example:

```yaml
tuning:
  metric: eval/reward
  direction: maximize
  n_trials: 40
  n_jobs: 1
  sampler: tpe
  seed: 42

methods:
  PPO:
    agent:
      name: sb3/ppo
      learning_rate: 0.0003 # fixed/base value
    model: basics/mlp
    tune:
      agent.learning_rate:
        type: log_float
        low: 0.00001
        high: 0.003
      agent.n_steps:
        type: choice
        values: [512, 1024, 2048]
```

The exact serialized keys for dimensions should be finalized with the implementation, but the semantic form is a typed mapping rather than a Hydra-specific expression string. Parameter targets are explicit paths such as `agent.learning_rate`, `model.hidden_sizes`, or a declared nested component path. This removes the current `_MODEL_KEYS` guessing from sweep routing.

Legacy `tune` / `search_space` string expressions remain readable during migration. The editor may preserve an unsupported legacy expression as read-only until it can parse it safely; it must not silently rewrite or discard it.

## Pipeline design

### Normalize and validate

Before launching, normalize the experiment into a sweep plan containing:

- Run-level objective, direction, trial count, concurrency, sampler, seed, and storage.
- One method plan per enabled method.
- One typed search dimension per marked parameter, each with a fully qualified target path.

Validation checks:

- A sweep box exists or the legacy CLI switch is active.
- Every swept method has at least one valid dimension.
- All dimension fields parse and have compatible types.
- Bounds are ordered; steps are positive; log bounds are positive; choice lists are non-empty and contain no blank values.
- Every parameter target resolves to a supported config path for the selected agent/model/method.
- The objective metric is valid for this paradigm, is retained in telemetry, and can be returned by the selected runner.
- Direction is valid and the number of trials and workers are positive.

### Execute studies

Retain the current per-method execution model: each method runs as an independent Optuna study, using the common settings from `tuning` and its own search dimensions. Study identity includes the experiment, group, and method key so methods using the same agent profile cannot collide.

The command builder converts normalized typed dimensions into the backend's Optuna search representation. UI code does not generate Hydra expressions. Local and Slurm runners consume the same normalized sweep plan so they agree on method scope, objective, and study identity.

The existing Hydra Optuna integration can remain behind this plan if it can reliably return the objective value for each job. If runner return values prove inconsistent, use a pipeline-owned Optuna ask/tell loop around one ordinary training run per trial. Do not let local and Slurm paths implement separate search-space semantics.

### Objective reporting contract

Every training runner must expose the selected scalar objective for the completed trial. The contract should include:

- Metric name and direction supplied by the sweep plan.
- A final scalar result, with a clear error when it was never produced.
- Trial metrics still written to the normal run logs for inspection.

The Lightning runner currently returns a training metric. The SB3 runner logs telemetry but does not currently return an objective value, so it needs a defined final metric path (including evaluation rather than an implicit training-only proxy) before SB3 sweeps are considered supported. Never mark a trial successful with a missing or `None` objective.

## Results: Sweeper tab

Add a **Sweeper** tab to `ResultsPanel`. It is populated from Optuna study storage and associated with the selected experiment, not treated as a normal file artifact.

The first version should show:

- Study selector when the selected experiment has multiple method studies.
- Method name, objective metric, direction, and study status.
- Trial counts by state: completed, running, failed, and queued when available.
- A sortable trial table with trial number, objective value, duration, state, and sampled parameter values.
- Best completed trial summary, including its parameters.
- Refresh/polling while a selected study is active.

Study metadata should include group, experiment ID, method key, objective metric, direction, and start time. The panel should use those attributes to find the right studies rather than relying only on parsing study names. If storage is unavailable or has no studies for the selected experiment, show a clear empty state.

The Sweeper tab does not replace Curves or Summary. Trial metrics continue to appear in the normal run views where the pipeline already records them.

## Compatibility and migration

- Accept legacy `sweep=true` and `--multirun` inputs without letting them override composed tuning-component presence.
- Keep existing top-level `tuning` settings and extend them with an explicit `metric` field.
- Keep current method-level `tune` and `search_space` inputs readable.
- New UI-generated configs use typed dimensions and a single canonical representation.
- Old configs without `tuning` remain normal runs even if they contain a base parameter value.
- Saving an experiment without editing its sweep settings must preserve unknown or legacy sweep YAML.

## Implementation phases

1. **Config model and validation**: typed sweep plan, legacy normalization, metric compatibility, dimension validation.
2. **Experiment editor**: add/remove sweep box; objective and trial controls; hover-revealed per-parameter sweep controls and typed distribution editors.
3. **Pipeline execution**: compile search spaces, make per-method study identity stable, add objective reporting for supported runners, align local and Slurm behavior.
4. **Sweeper results tab**: study discovery, live trial table, best-trial summary, empty and error states.
5. **Documentation and compatibility review**: document new YAML and CLI behavior; verify old sweep configs remain usable.

## Acceptance criteria

- Adding the sweep box visibly enables sweep mode; removing it disables the mode without changing base parameter values.
- The user can opt a parameter into a sweep from its row, choose a supported distribution, add/remove choices, and remove the sweep definition while retaining the base parameter.
- All enabled methods with dimensions run as separate, non-colliding studies; methods without dimensions fail preflight clearly.
- A sweep cannot start without a valid objective metric and direction, or with malformed dimensions.
- Every completed trial has an objective value and remains inspectable in telemetry.
- Local and Slurm launch paths use the same settings and dimension interpretation.
- Results excludes running trials and stays static until refreshed. Completed trials remain available while later trials are running; live study progress stays in the Training Monitor.
- Existing non-sweep experiments and legacy CLI sweep commands continue to behave as before.

## Open decisions for review

1. Should **Numeric range** be continuous only, with step available as an optional input, or should step imply a discrete stepped distribution?
2. Should the objective metric dropdown include only known paradigm metrics, or also allow a custom metric name for third-party runners?
3. For a shared parameter dimension, should every method be required to support that parameter, or should the dimension apply only to methods where the target exists? The recommended default is to validate per method and report unsupported targets before launch.
