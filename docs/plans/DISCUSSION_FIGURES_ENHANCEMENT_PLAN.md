# Discussion figures enhancement plan

Proposed 2026-11-17. Status: Phases 1–4 implemented on `feature/discussion-panel` (see [Implementation notes](#implementation-notes)); Phase 5 is next, and Phase 6 awaits a decision. Native visual acceptance remains pending. Builds on the [Discussion panel plan](DISCUSSION_PANEL_PLAN.md) and the [figure contract](../reference/FIGURE_CONTRACT.md).

## Where figures stood (before Phase 1)

- **One provider.** `standard_figures/convergence` is the only figure plotter. Its siblings `losses`, `reports` and `tuning` still produce only saved pipeline plots through `run()`.
- **Six flat parameters.** `metric`, `smoothing_window`, `methods`, `line_width`, `opacity`, `title`. Axes, scales, ranges, legend, size and the x quantity are fixed by the spec.
- **Weak pickers.** `metric` is declared `x-theta-metric`, but the panel never passes logged metrics to `ParametersBox(metric_choices=…)`, so it is always a plain text field. `methods` is a free string list with no picker.
- **A chart without interaction or output.** It has no tooltips, zoom, export or copy, and the drawn figure cannot leave the app.
- **Bare figure management.** Figures cannot be renamed, duplicated, deleted, reordered or titled from the UI. The tree leaf is the file stem.
- **Thin picker cards.** Cards show ID, package, version and parameter counts. No capability declares a `description` or a preview yet.

## Goals

1. Ship a small, useful set of figure types for the questions an RL experiment actually raises: learning curves, final performance, run-to-run variability, sweeps, and losses.
2. Make every figure's look controllable through the shared parameter form, grouped so long schemas stay readable.
3. Let a figure, and a discussion with its figures, leave the app as publication-ready files.
4. Keep the contract small. Theta owns controls, theme and storage; providers own data and specs. Every addition below is an opt-in schema annotation or an optional capability field.

## Phase 1 — Fix the basics (small, no contract change)

1. **Metric choices.** On figure selection, list the metrics the target's runs logged (the Results catalog already knows them) and pass them as `metric_choices`. Load them off the GUI thread and reuse them per experiment.
2. **Method picker.** New annotation `x-theta-method: true` on a string or array-of-string parameter. Offer the target's method names (from the composed or saved config) through the shared list editor with dropdown entries. Convert `convergence.methods` to it.
3. **Figure management in the tree.** Rename (inline, like the config viewer's title edit), duplicate, delete (moves `.yaml` and `.md` together), and context-menu equivalents. Rename keeps the discussion file paired. Clicking the parameter box title starts a rename, as `TitleLabel` does in the config viewer.
4. **Card metadata.** Add `label` and `description` to the `convergence` capability, and release `standard_figures` through the [Hub release workflow](../development/THETA_HUB_RELEASE_WORKFLOW.md).
5. **Spec cache.** Keep the last built spec per figure file in memory, keyed by plotter, target, values and the target's run-catalog stamp. Reopening a figure draws at once; Regenerate bypasses the cache.

## Phase 2 — Richer parameters (contract additions)

1. **Parameter groups.** New property annotation `x-theta-group: "Data" | "Axes" | "Style" | …`. The box renders each group as a titled, collapsible block (the shared `_collapsible_parameter_label` and `IndentedHierarchyBlock`), in schema order. Collapsed state persists per plotter, like parameter order.
2. **Common axis and style parameters.** Not enforced by the host; documented as a recommended vocabulary so providers agree:
   - Axes: `x_quantity` (`step` / `episode` / `wall_time`), `x_label`, `y_label`, `y_scale` (`linear` / `log` / `symlog`), `y_min`, `y_max` (nullable = auto).
   - Style: `legend` (`right` / `bottom` / `none`), `width` and `aspect` (export geometry; the view still fits its card), `palette` (`method` = the method styles Results uses, or a named Vega scheme).
   - Statistics: `band` (`sem` / `ci95` / `std` / `none`), `smoothing` (`none` / `moving_average` / `ema`) with its window or alpha.
   Live where possible: labels, scale, ranges and legend become Vega signals, so they update without a rebuild.
3. **Nullable "auto" values.** The editors show an explicit **Auto** state for `{"type": ["number", "null"]}` with `default: null`. Clearing a value returns to Auto instead of failing validation.

## Phase 3 — More figure types

Each type is a capability in `standard_figures`, opted into `figure/vega-lite-v1`, with compatibility `requires` where they apply.

| Figure | Shows | Notes |
|---|---|---|
| `convergence` (extend) | Smoothed mean per method with band | Phase 2 axes/statistics; optional per-seed traces (`show_runs`) |
| `final_performance` | Final or best-window score per method: bar + CI, box or strip plot | `aggregate`: `mean` / `median` / `iqm`; bootstrap CIs computed in the provider |
| `performance_profile` | Fraction of runs above each score threshold (rliable-style) | Robust comparison across seeds |
| `run_variability` | Every seed's curve, faint, with the mean on top | Shows the variability the variability work records |
| `losses` (convert) | Training losses per method, multi-panel | Facet by loss name; `log` y-scale by default |
| `sweep_scatter` | Objective vs one swept parameter; colour by a second | `requires` tuning; reads trial results |
| `sweep_parallel` | Parallel coordinates over swept parameters and objective | `requires` tuning |
| `metric_table` | A compact table of final metrics per method | Vega text marks; also exports as CSV |

Shared data helpers move into `src.app.api.plotting` (per-method series, final values, bootstrap CI, IQM) so providers share one tested implementation instead of copying the convergence code.

## Phase 4 — Interaction in the view

1. **Tooltips** on every line or point: method, x, y and band. The host enables the Vega tooltip handler, so providers add only `tooltip` encodings.
2. **Zoom and pan** through an interval selection bound to scales (`bind: "scales"` is a scale binding, not a control binding). The contract allows this one binding explicitly; double-click resets.
3. **Legend toggling.** Clicking a legend entry dims that method, using a point selection on the colour field. This is view state only and never saved.
4. **A figure toolbar** in the Figure card header, built from the shared tool-button style: Regenerate (moved from the tree), Reset view, Copy image, Export.

## Phase 5 — Export and reports

1. **Figure export** to SVG and PNG (`view.toSVG()` / `view.toImageURL(scale)`) at the figure's `width` × `aspect`, with a light "print" theme option independent of the app theme. PDF goes through SVG with QtSvg/QPrinter, so there is no new dependency.
2. **Data export.** Write the spec's inline data as CSV, so numbers in the discussion can be checked.
3. **Discussion report.** Combine the experiment's or run's figures, in tree order, with their discussion text into one HTML (self-contained, SVG inline) or PDF file. Markdown in the discussion renders in the report; the editor gains an optional preview toggle.
4. **Figure references.** `[[figure_name]]` in discussion text links to another figure of the same target. The report numbers them.

## Phase 6 — Cross-target figures (decision needed)

A figure currently belongs to one experiment or one run. Comparing two experiments, such as PPO vs DQN recipes or two dates of the same run, needs a figure whose data comes from several targets. Options:

- **A.** A `targets` parameter (`x-theta-target`, a multi-picker of experiments and runs) on figures stored under the group, with the provider receiving them through `params`. This keeps the contract shape and the one-file storage.
- **B.** A separate "comparison" figure kind with its own storage folder.

The recommendation is **A**. This is the one structural decision in the plan and is reserved for the user.

## Contract changes, summarised

| Addition | Where | Host behaviour |
|---|---|---|
| `label`, `description` | capability | Shown on picker cards (already read) |
| `preview` (SVG path in the package) | capability | Thumbnail on the picker card |
| `x-theta-method` | property | Method picker from the target's config |
| `x-theta-group` | property | Titled, collapsible parameter groups |
| `x-theta-target` | property | Experiment/run multi-picker (Phase 6) |
| `bind: "scales"` | spec | Allowed; all other `bind` still rejected |
| tooltip handler | host | Always on; providers add `tooltip` encodings |

All additions are optional and need no compatibility adapters. Figure files keep exactly `plotter` and `params`.

## Verification

- Add contract tests (`tests/test_figure_contract.py`) for each new provider, using fixture runs with several seeds and methods: data shape, live signals present, no forbidden bindings, and errors with no data.
- Add form tests (`tests/test_parameter_forms.py`) for grouped parameters, Auto values and the method picker, following the cross-box conventions.
- Add panel tests (`tests/test_discussion_panel.py`) for rename, duplicate, delete, the spec cache, export file contents (SVG parses; PNG has the requested size) and report assembly.
- Run native visual acceptance on macOS for tooltips, zoom/pan and export in both light and dark themes. Offscreen tests cannot cover WebEngine rendering.

## Suggested order

Phase 1 is independent and immediately useful. Phases 2 and 3 go together: the new types should use the grouped vocabulary from the start. Phase 4 follows once more than one type exists. Phase 5 is the largest user-visible payoff for writing up results. Phase 6 waits on the decision above.

## Implementation notes

Phases 1–4 landed on 2026-11-17 with `standard_figures` 1.2.0 and SDK 3.3. The [figure contract](../reference/FIGURE_CONTRACT.md) and the [Discussion guide](../guides/DISCUSSION.md) describe the result. Deviations from the plan:

- **Run choices are read synchronously.** `model.target_runs` reads only the header line of each metric file, and the panel caches it by the target's file stamp, so a worker thread was not needed.
- **Method names come from the logged runs**, not the composed config, so the picker offers only methods that have something to plot.
- **The spec cache key** uses the target's file stamp (metric files, recipe or saved config, tuning summaries) instead of the run-catalog stamp, and includes the package version. It holds the 32 most recent specs.
- **`legend` is a bound selection** (`bind: "legend"`), so the contract allows both `scales` and `legend` bindings, checked at every nesting level.
- **`losses` defaults to a linear y scale.** Losses can be negative (policy losses often are), which a log scale cannot show.
- **`width` and `aspect` are deferred to Phase 5**, where they set export geometry; the view fits its card.
- **The toolbar's Export is deferred to Phase 5.** The Figure card has Regenerate, Reset view and Copy image.
- **No `preview` capability field yet.** Cards show the label, description and parameter counts.
- **Capability `description`** was added to the package manifest schema (`src/app/core/manifest_schema.py`) as an optional string.
- **Hub release pending.** `standard_figures` 1.2.0 is prepared in the repository but not published; publish it through the [Hub release workflow](../development/THETA_HUB_RELEASE_WORKFLOW.md) when authorised.

Verification: `tests/test_standard_figures.py` builds every figure over a seeded run with three seeds, two methods, losses and a sweep. When Node is installed it compiles and runs each spec in the vendored Vega builds (`tests/vega_support.py`). The panel, model, contract and parameter-form suites cover rename, duplicate, delete, the cache, groups, Auto values and the choice list.
