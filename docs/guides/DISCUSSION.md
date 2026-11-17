# Discussion

Open **Discussion** in the side tabs, after Results. Its tree shows experiment groups and subfolders, with timestamped saved runs beneath their experiments, using the same catalog and labels as Results. Select a stamped run and use the square **+** in the tree toolbar to add a figure for that run. Its parameters and discussion belong to that run, and plotting reads only its metrics. Saved runs remain available even if the current experiment recipe was removed.

Experiment-level figures remain directly under the experiment. Select a figure to reopen its chart and discussion. Use Reload to pick up new runs. The **+** menu lists installed plot components that provide the figure contract and satisfy the selected target's compatibility requirements. Warnings appear as icons with tooltips. Run compatibility reads the saved run config; if it has not been downloaded, sync it before adding figures.

Select a figure to open its **Parameters**, chart, and plain-text discussion. The bundled `convergence` plotter reads saved run metrics. Set `metric` to a logged metric name; use `methods` to limit the methods shown, or leave it empty to include all of them. `smoothing_window` controls the moving average.

Live parameters (`line_width`, `opacity`, and `title` for convergence) update the chart immediately. Other parameters rebuild it in the background after a short pause. **Regenerate** in the tree toolbar forces a rebuild to include new runs. Component errors appear in the chart pane.

Parameters use the shared configuration controls. Unset values display the component's defaults; editing creates an explicit override. Remove an optional override with the row's **×** to restore its default. The **+** beside Parameters adds optional fields without defaults. Invalid drafts leave the saved file unchanged.

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
