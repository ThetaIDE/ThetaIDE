# Discussion

Open **Discussion** in the side tabs, after Results. Its tree shows experiment groups and subfolders, with timestamped saved runs beneath their experiments, using the same catalog and labels as Results. Select a stamped run and use the square **+** in the tree toolbar to add a figure for that run. Its parameters and discussion belong to that run, and plotting reads only its metrics. Saved runs remain available even if the current experiment recipe was removed.

Experiment-level figures remain directly under the experiment. Select a figure to reopen its chart and discussion. Use Reload to pick up new runs. The **+** replaces the parameters column with an **Add figure** box: one card per installed plot component that provides the figure contract and satisfies the selected target's compatibility requirements. Each card shows the component's package, version and parameter count; warnings appear on the card. Click a card (or focus it and press Enter) to create the figure and open its parameters. **×** or Escape closes the box without adding anything. Run compatibility reads the saved run config; if it has not been downloaded, sync it before adding figures.

The page is laid out like the Experiment tab: a title naming the selected experiment or run, the figure's parameter box (titled with the figure's name) on the left, and the **Figure** and **Discussion** cards on the right. Select a figure to open its parameters, chart, and plain-text discussion.

## Figure types

The bundled `standard_figures` package provides:

- **Learning curve** (`convergence`): a metric's mean per method over training, with an uncertainty band (`sem`, `ci95` or `std`) and optional faint per-run traces.
- **Final performance**: each run's final, best or lowest score per method, as bars with a bootstrap interval, box plots or strip plots, aggregated by mean, median or IQM.
- **Performance profile**: the fraction of each method's runs scoring above each threshold.
- **Run variability**: one panel per method with every run and the mean.
- **Training losses**: one panel per logged loss.
- **Sweep scatter** and **Sweep parallel coordinates**: a sweep's trials against its parameters. They are offered only for targets with a parameter sweep.
- **Metric table**: final metrics per method with their spread, the best method in bold.

## Parameters

Parameters use the shared configuration controls, grouped into **Data**, **Axes** and **Style**. Click a group's arrow to collapse it; collapsed groups stay collapsed for every figure of that type. Unset values display the component's defaults; editing creates an explicit override. Remove an optional override with the row's **×** to restore its default. The **+** beside the box title adds optional fields without defaults. Invalid drafts leave the saved file unchanged.

`metric` offers the metrics the selected target's runs logged, and `methods` offers its methods: pick entries with **+**, remove them with their **×**, or leave the list empty to include every method. Fields such as `title`, `x_label`, `y_min` and `y_max` show **Auto** while unset; clear one to return to the automatic value.

Live parameters (titles, labels, axis ranges, line width and opacity) update the chart immediately. Other parameters rebuild it in the background after a short pause. Reopening an unchanged figure draws at once from memory; new or changed runs are picked up automatically.

## The figure

Hover a line or point for its values. Drag to pan and scroll to zoom. Click a legend entry to highlight that method; shift-click adds more. Double-click the chart to reset zooming and highlighting. The Figure card's header has three tools:

- **Regenerate** rebuilds the figure, bypassing the in-memory copy.
- **Reset view** undoes zooming, panning and legend highlighting.
- **Copy image** puts a PNG of the chart on the clipboard at twice screen resolution.

Component errors appear in the chart pane.

## Managing figures

Right-click a figure in the tree to **Rename…**, **Duplicate** or **Delete…** it. **F2** renames the selected figure and **Delete** (or Backspace) deletes it after confirmation; clicking the parameter box title also renames. Names use letters, digits, spaces, `_` and `-`. Press Enter to apply a rename or Escape to cancel; a name already in use stays in the field with the reason. The discussion text always moves, copies and deletes with its figure. A duplicate is named `<name>_copy`.

## Discussion text

Discussion text saves after about half a second without typing, and immediately when switching selection or closing the app. It has no Markdown preview.

## Files

For experiment `group/subdir/example`, files live in:

```text
src/usr/discussion/group/subdir/example/convergence.yaml
src/usr/discussion/group/subdir/example/convergence.md
```

A stamped run adds its identity below the experiment, for example:

```text
src/usr/discussion/group/subdir/example/20261116-120000/convergence.yaml
src/usr/discussion/group/subdir/example/20261116-120000/convergence.md
```

Selecting this run before adding a figure scopes the chart to that run. Selecting another run and adding the same plotter creates a separate figure and discussion.

The YAML has exactly two keys:

```yaml
plotter: convergence
params:
  metric: eval/reward
  line_width: 3
```

Only explicit parameter values are stored. Figure specs are generated when opened and never saved. Discussion text uses the matching `.md` file, created on the first nonempty edit. Adding the same plotter again produces names such as `convergence_2`.

For component authors, see the [figure contract](../reference/FIGURE_CONTRACT.md).
