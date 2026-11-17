# Figure contract (`figure/vega-lite-v1`)

Plot components build the figures of the [Discussion tab](../guides/DISCUSSION.md) as Vega-Lite specs. The existing `run()` method still writes the pipeline's saved plots; the figure spec is an additional, opt-in capability.

## Opting in

A plotter capability opts in with three things:

1. `"provides": ["figure/vega-lite-v1"]` in `.theta_component.json`. The SDK exports the ID as `src.app.api.plotting.FIGURE_CONTRACT`.
2. A `parameter_schema`: a JSON Schema object of the figure parameters, with defaults.
   - A property marked `"x-theta-live": true` is a **live parameter**. Editing it updates the drawn figure without calling the component.
   - Every other property is a **regeneration parameter**. Editing it calls `figure_spec` again in the background.
   - A string property marked `"x-theta-metric": true` offers the metrics the experiment's runs logged as choices, when there are any. Otherwise it is a plain text field.
3. `figure_spec(self, exp_id: str, params: dict) -> dict` on the capability's `BasePlotter` subclass. It returns a Vega-Lite spec with its data inline (`data.values`).
   - `exp_id` is the experiment's config path without `.yaml` (`cartpole/sweeps/ppo_long`).
   - `params` is the schema defaults merged with the figure's saved values.
   - For every live parameter the spec declares a top-level `params` entry with the same name, whose `value` is the current value. Spec properties that accept expressions use it (`{"expr": "line_width"}`).
   - The spec has no `bind`. Theta's Parameters box owns the controls.
   - Raise an exception, such as `ValueError("No run of cartpole/ppo logged 'eval/reward'.")`, when there is nothing to draw. The message replaces the figure.

```json
{
  "id": "convergence", "kind": "plotter", "entry_point": "convergence.py:ConvergencePlotter",
  "provides": ["figure/vega-lite-v1"],
  "parameter_schema": {
    "type": "object",
    "properties": {
      "metric": {"type": "string", "default": "eval/reward", "x-theta-metric": true},
      "line_width": {"type": "number", "minimum": 0, "default": 2, "x-theta-live": true}
    }
  }
}
```

Long series should be thinned before they go into the spec. `src.app.api.plotting.thin_xy(points, limit=2000)` keeps each bucket's extrema and the latest point, as the Results charts do.

## What the host checks

- A returned spec that is not a mapping, misses a top-level param for a live parameter, or binds a param is an error, shown in place of the figure.
- The host never edits a spec's structure. It only sets live parameters' signals (`view.signal(name, value)`) and supplies theme colors through the embed `config`.
- The figure file stores only the parameters the user set explicitly, validated against the schema. Unknown keys are rejected. The spec is derived and never stored. See [the Discussion guide](../guides/DISCUSSION.md) for the storage layout.

## Compatibility

The **+** menu of the Discussion tree lists the figure plotters compatible with the selected experiment. It evaluates each capability's [`requires` rules](COMPATIBILITY_REQUIREMENTS.md) with `src.app.core.requirements.figure_findings` against the composed experiment, composed as validation composes it:

- `config` rules and `when` conditions read the composed experiment.
- A `kind` rule is satisfied when any enabled method's selection of that kind satisfies it.
- Preflight hooks never run.

The menu omits candidates with block findings and marks candidates with warn findings, with the warning icon and tooltip of the configuration pickers.

## Providers

`standard_figures/convergence` (1.1.0 and later) plots one metric's smoothed mean per method with a standard-error band. Its live parameters are `line_width`, `opacity` and `title`; `metric`, `smoothing_window` and `methods` regenerate the figure.
