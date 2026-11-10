# Sweep dashboard comparison

Reviewed on 2026-11-03 against the official Optuna Dashboard documentation and UI source on its main branch. Optional features depend on the Dashboard configuration; this is a comparison of the native Theta sweep views, not of every feature in the Optuna Python plotting library.

## Available in Theta

The Training monitor and Results → Sweeps share a dashboard with a study selector, optimization history, best-so-far curve, intermediate values, numeric parameter-versus-objective plots, a sortable/searchable trial table, trial parameters, durations, and checkpoint availability. Results reads saved `tuning_summary.json` artifacts, including older studies, and can open the recorded trial metrics file. Select runs in the Results tree, then open Sweeps. Refresh re-reads saved summaries.

The **Parallel coordinates** tab shows numeric and categorical parameters with an objective axis. Drag on axes to select ranges, click categories to select them, and combine ranges across axes. The selected trial subset is shared by the highlighted lines, trial table, optimization history, intermediate plots, and numeric scatter plots. Text search combines with the axis selections. Clear filters restores all trials; right-clicking an axis clears just that axis. Live snapshots preserve the selected parameter values as ranges expand; switching studies resets the filters. Missing parameters are excluded only when that parameter is filtered. Results uses the same interaction on its saved snapshots and adds no live polling.

Parameter importance is supported in the **Importance** tab: ranked percentage bars use deterministic Optuna fANOVA estimates after at least five completed trials. The snapshot writer updates estimates when the completed trial set changes, and the live monitor reads those updates. Results shows the saved estimates with the completed results; it refreshes only on request. Older summaries with stored importance values also display the chart. Empty states explain insufficient trials or unavailable estimates. fANOVA compares parameters shared across completed trials and does not indicate whether increasing a parameter helps.

## Missing or incomplete analysis

| Feature | What Theta still needs | Official reference |
| --- | --- | --- |
| Contour plots | Visualize the joint relationship of two parameters with the objective. | [Analytics page](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/StudyDetail.tsx) |
| Rank plots | Color parameter-pair plots by trial rank; include categorical and logarithmic axes. | [Rank component](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/GraphRank.tsx) |
| Objective distributions | Empirical distribution function charts for objective values. | [Analytics page](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/StudyDetail.tsx) |
| Trial timeline | Show when trials started, finished, and overlapped. Theta currently shows duration in a table. | [Study history](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/StudyHistory.tsx) |
| Multi-objective analysis | Multiple objectives, objective selection, Pareto fronts, dominated-trial controls, and multiple best trials. Our saved summary and views assume one objective. | [Study history](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/StudyHistory.tsx), [trial selection](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/TrialSelection.tsx) |
| Constraint visibility | Distinguish feasible and infeasible trials and filter them in analysis views. | [Trial selection](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/TrialSelection.tsx) |
| Study comparison | Overlay optimization histories and objective distributions across studies. Theta can select multiple runs but shows one sweep study at a time. | [Compare Studies, experimental](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/CompareStudies.tsx) |
| History controls | Logarithmic objective axes and a switch to include/exclude pruned trials. Theta's history includes completed trials; intermediate plots include trials with reported values. | [Study history](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/StudyHistory.tsx) |

## Missing or incomplete inspection and interaction

| Feature | What Theta still needs | Official reference |
| --- | --- | --- |
| Sweep CSV export | Export the selected trial records, including parameters and objectives. Existing Results exports cover training curves and metric summaries. | [Trial selection](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/TrialSelection.tsx) |
| Rich trial inspection | Dedicated start/end timestamps, user attributes, best-trial-only and state controls, and selectable duration units. Some timestamps are saved but not displayed by our sweep view. | [Trial list](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/TrialList.tsx) |
| Study attributes | Display arbitrary study user attributes. Our snapshots retain selected Theta metadata. | [Study history](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/StudyHistory.tsx) |
| Study/trial notes | Save and render Markdown annotations. | [Dashboard API](https://optuna-dashboard.readthedocs.io/en/stable/api.html) |
| Study/trial artifact cards | Preview artifacts attached to an Optuna study/trial and to selected trials. Theta has a general Files browser, but no Optuna artifact-store integration or trial artifact cards. | [Study history](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/StudyHistory.tsx), [trial list](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/TrialList.tsx) |
| Custom study plots | Render user-saved Plotly figures in the study dashboard. | [Dashboard API](https://optuna-dashboard.readthedocs.io/en/stable/api.html) |
| Study administration | Create, rename, or delete Optuna studies through the dashboard. Theta's run archive/delete controls operate on result runs. | [Study list](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/StudyList.tsx) |
| Individual trial state changes | Mark a running Optuna trial failed through its detail view. This is different from cancelling an entire Theta job. | [Trial list](https://github.com/optuna/optuna-dashboard/blob/main/optuna_dashboard/ts/components/TrialList.tsx) |

## Optional Dashboard extensions we do not expose

- Human scoring and editable trial attributes through objective/user-attribute form widgets. [Human-in-the-loop tutorial](https://optuna-dashboard.readthedocs.io/en/stable/tutorials/hitl.html), [API](https://optuna-dashboard.readthedocs.io/en/stable/api.html).
- Preference comparisons and their history/analytics. [Preferential optimization tutorial](https://optuna-dashboard.readthedocs.io/en/stable/tutorials/preferential-optimization.html).
- Natural-language trial filters and generated Plotly charts, when an LLM provider is configured. [LLM integration](https://optuna-dashboard.readthedocs.io/en/stable/tutorials/llm-integration.html).

Deployment integrations such as Jupyter, VS Code, browser-only SQLite viewing, and OpenTelemetry server instrumentation are outside this list of results UI gaps.

Suggested next additions: contour plots and a trial timeline.
