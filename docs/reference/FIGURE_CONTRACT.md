# Figure contract (`figure/vega-lite-v1`)

Plot components build the figures of the [Discussion tab](../guides/DISCUSSION.md) as Vega-Lite specs. The existing `run()` method still writes the pipeline's saved plots; the figure spec is an additional, opt-in capability.

## Opting in

A plotter capability opts in with three things, plus optional card text:

1. `"provides": ["figure/vega-lite-v1"]` in `.theta_component.json`. The SDK exports the ID as `src.app.api.plotting.FIGURE_CONTRACT`.
2. A `parameter_schema`: a JSON Schema object of the figure parameters, with defaults.
   - A property marked `"x-theta-live": true` is a **live parameter**. Editing it updates the drawn figure without calling the component.
   - Every other property is a **regeneration parameter**. Editing it calls `figure_spec` again in the background.
   - A property marked `"x-theta-metric": true` offers the metrics the target's runs logged; `"x-theta-method": true` offers the target's logged method names. A string property becomes a dropdown, or a plain text field while nothing is logged. An array of strings becomes the shared choice list (entries with **×**, and a **+** offering the remaining choices or a typed one).
   - `"x-theta-group": "<name>"` places a property in a titled, collapsible group. Groups appear in first-use order after any ungrouped properties. Their collapsed state is remembered per plotter in the app settings.
   - A string property with default `""`, or a nullable property (`"type": ["number", "null"]`) with default `null`, shows **Auto** while unset. Clearing such a field removes the explicit value, so the provider's automatic choice returns. Providers turn an empty live text into their default with an expression, such as `{"expr": "title || \"Default title\""}` (`auto_text` in the SDK).
3. `figure_spec(self, exp_id: str, params: dict) -> dict` on the capability's `BasePlotter` subclass. It returns a Vega-Lite spec with its data inline (`data.values`).
   - `exp_id` is the target that owns the figure. It is an experiment's config path without `.yaml` (`cartpole/sweeps/ppo_long`) or a saved run's full result identity (`cartpole/sweeps/ppo_long/20261116-120000`). Read its metrics by default. A saved run's config supplies the methods actually executed.
   - `params` is the schema defaults merged with the figure's saved values.
   - For every live parameter the spec declares a top-level `params` entry with the same name, whose `value` is the current value. Spec properties that accept expressions use it (`{"expr": "line_width"}`).
   - The spec binds no input controls. Theta's figure parameter box owns the controls. The only allowed bindings are the view interactions `"bind": "scales"` (zoom and pan) and `"bind": "legend"` (legend toggling), at any nesting level.
   - Raise an exception, such as `ValueError("No run of cartpole/ppo logged 'eval/reward'.")`, when there is nothing to draw. The message replaces the figure.

Optional capability fields `label` and `description` name and describe the figure on its **Add figure** card.

Metric plotters can opt in to workspace data selection by declaring an array parameter with `"x-theta-sources": true` and an empty list default. Each item is an object with a required workspace-relative result ID `target`, an optional `method`, and an optional `version` within that method. The host offers nested result, method, and seed choices and filters the metric picker to the selected logs. An explicit source list supplies its own method selection and takes precedence over the home target's method filter.

`src.app.api.plotting.load_selected_metrics(paths, sources)` reads those selections into the shape accepted by `metric_runs`, with keys of the form `target/method` so identical method names from different results stay distinct. Plotters use their existing target loader when the list is empty. Learning curves can pass `common_x=True` to `metric_runs` to reject series that would use different x-axis quantities.

```json
{
  "id": "convergence", "kind": "plotter", "entry_point": "convergence.py:ConvergencePlotter",
  "label": "Learning curve",
  "description": "A metric over training: each method's mean across runs with an uncertainty band.",
  "provides": ["figure/vega-lite-v1"],
  "parameter_schema": {
    "type": "object",
    "properties": {
      "metric": {"type": "string", "default": "eval/reward", "x-theta-metric": true, "x-theta-group": "Data"},
      "methods": {"type": "array", "items": {"type": "string"}, "default": [], "x-theta-method": true,
                  "x-theta-group": "Data"},
      "y_min": {"type": ["number", "null"], "default": null, "x-theta-live": true, "x-theta-group": "Axes"},
      "line_width": {"type": "number", "minimum": 0, "default": 2, "x-theta-live": true, "x-theta-group": "Style"}
    }
  }
}
```

Long series should be thinned before they go into the spec. `src.app.api.plotting.thin_xy(points, limit=2000)` keeps each bucket's extrema and the latest point, as the Results charts do.

## Building blocks

`src.app.api.plotting` (SDK 3.3) exports headless helpers, so providers share one tested implementation:

- **Data.** `metric_runs(runs_data, metric, x_quantity)` returns one `Run(method, version, x, y)` per logged version. The x quantity falls back to `step`, `epoch`, then the logged-point index. `smooth` (moving average or EMA), `aggregate(runs, band)` (mean and `sem` / `ci95` / `std` band over the runs' common x range), `final_scores`, `STATISTICS` (`mean`, `median`, `iqm`), seeded `bootstrap_interval`, `performance_profile`, and `study_summaries(paths, exp_id)` for tuning summaries.
- **Specs.** `vega_lite_figure(values, live, data, title=…, container=True, **body)` builds the outer spec with the live params. Pass `container=False` for faceted specs, which cannot fill the card and scroll instead. `auto_text`, `zoom_param`, `legend_param` / `legend_opacity` (click a legend entry to keep it bright and dim the rest), `hover_param` / `hover_opacity` (nearest-point highlight), `color_encoding`, `NUMBER_FORMAT`, `LEGEND_POSITIONS`.

## Recommended parameter vocabulary

The host does not enforce these names; providers that share them stay consistent with one another and with `standard_figures`.

| Group | Parameters |
|---|---|
| Data | `metric`, `methods`, `x_quantity` (`transitions` / `step` / `epoch` / `time/total`), `smoothing` (`moving_average` / `ema` / `none`), `smoothing_window`, `band` (`sem` / `ci95` / `std` / `none`) |
| Axes | `x_label`, `y_label` (live, Auto), `y_scale` (`linear` / `log` / `symlog`), `y_min`, `y_max` (live, nullable Auto) |
| Style | `title` (live, Auto), `line_width`, `opacity` (live), `legend` (`right` / `bottom` / `top` / `left` / `none`) |

## What the host checks

- A returned spec that is not a mapping, misses a top-level param for a live parameter, or binds a param to anything other than `scales` or `legend` is an error, shown in place of the figure.
- The host never edits a spec's structure. It only sets live parameters' signals (`view.signal(name, value)`) and supplies theme colors through the embed `config`.
- The figure file stores only the parameters the user set explicitly, validated against the schema. Unknown keys are rejected. The spec is derived and never stored. See [the Discussion guide](../guides/DISCUSSION.md) for the storage layout.

## Compatibility

The **+** in the Results tree toolbar opens Discussion's **Add figure** box for the selected experiment or run. For an experiment, the picker evaluates each capability's [`requires` rules](COMPATIBILITY_REQUIREMENTS.md) with `src.app.core.requirements.figure_findings` against the same composed experiment used in validation.

- `config` rules and `when` conditions read the composed experiment.
- A `kind` rule is satisfied when any enabled method's selection of that kind satisfies it.
- Preflight hooks never run.

The box omits candidates with block findings and shows warn findings inline on the card, as `controls.compatibility_status` does for configuration selections. A card shows the capability's optional `label` (else its ID), optional `description`, package and version, and its parameter and live-parameter counts.

When a saved run is selected, compatibility reads its saved `results/logs/<group>/<experiment>/<stamp>/config.yaml`. The picker requires that file to offer figures for the run. Saved runs remain selectable in Results after their current recipe is removed.

The desktop loads run listings lazily through the same backend as Results. Run figures and discussion text live under `discussion/<group>/<experiment>/<stamp>/`; the full target path is passed unchanged to the provider. Specs remain derived and parameter values remain solely in the figure YAML.

## Rendering

The figure view loads pinned local Vega 6.4.0, Vega-Lite 6.5.0 and Vega-Embed 7.3.0 builds, and creates QtWebEngine only when first shown. Figure generation runs in a worker thread with a 400 ms debounce. New requests supersede old results; live edits made during a rebuild are applied to the completed chart. Providers must not access Qt widgets from `figure_spec`.

The view always enables the Vega tooltip handler, styled with the app theme; providers add only `tooltip` encodings. Double-clicking a zoomed chart, or **Reset view** in the Figure card, restores the original scales.

Built specs are cached in memory (the 32 most recent). The key includes the plotter, package ID and version, figure target, regeneration parameters, and a file stamp for each result that supplies data. The stamp records modification times and sizes for metric files, the recipe or saved config, and tuning summaries. Reopening an unchanged figure or editing a live parameter therefore never calls the provider. **Regenerate** bypasses the cache.

## Providers

`standard_figures` 1.2.0 provides these figures. Every one groups its parameters into Data, Axes and Style and has an Auto `title` and tooltips. All but the table and the parallel coordinates zoom and pan. Figures with a method legend toggle methods from it; run variability has one panel per method instead.

| Capability | Card label | Shows |
|---|---|---|
| `convergence` | Learning curve | A metric's smoothed mean per method with a `sem` / `ci95` / `std` band; optional faint per-run traces (`show_runs`) |
| `final_performance` | Final performance | Each run reduced to its final (last `final_fraction` of points), best or lowest value; bar with bootstrap interval, box or strip chart, aggregated by mean, median or IQM |
| `performance_profile` | Performance profile | The fraction of each method's runs scoring above each threshold |
| `run_variability` | Run variability | One panel per method with every run and the mean |
| `losses` | Training losses | One panel per logged `losses/*` column (or the chosen `metrics`), methods coloured, each with its own y scale |
| `sweep_scatter` | Sweep scatter | The objective of each finished trial against one swept parameter, optionally coloured by another; log x for parameters spanning 100× or more; pruned trials (their last intermediate value) on request |
| `sweep_parallel` | Sweep parallel coordinates | Every trial across the swept parameters and the objective, each axis normalised (log-spaced where appropriate) and coloured by the objective |
| `metric_table` | Metric table | Final metrics per method with std, SEM or bootstrap spread; the best method per metric in bold |

The two sweep figures require a tuning block (`requires` with `"config": "tuning"`), so the **Add figure** box omits them for targets without a sweep. They read the target's `tuning_summary.json` files (schema version 1). The figures write nothing from `run()`; the saved pipeline plots remain the job of `reports`, `losses` and `tuning`.
