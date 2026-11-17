# Discussion panel plan

Proposed 2026-11-16. Status: units 1–6 implemented on `feature/discussion-panel`; unit 7 verification and documentation in progress. [Implementation prompt](DISCUSSION_PANEL_IMPLEMENTATION_PROMPT.md).

A new side tab, **Discussion**, for editing the figures of each experiment and writing about them.

```text
┌──────────────┬────────────────────┬───────────────────────────────┐
│ [toolbar  +] │ Parameters         │ Rendered figure (Vega-Lite)   │
│ Experiment   │ (ConfigBox, shared │                               │
│ tree         │  parameter form)   ├───────────────────────────────┤
│ (mirrors the │                    │ Discussion (plain text,       │
│  Experiment  │                    │ autosaved)                    │
│  panel)      │                    │                               │
└──────────────┴────────────────────┴───────────────────────────────┘
```

## Scope

- The tree mirrors the Experiment panel's groups and subfolders. Each experiment is a folder, and its figures are leaves below it.
- A square **+** button in the tree's header toolbar opens a dropdown of the plot components that are compatible with the selected experiment. Choosing one creates a figure.
- Selecting a figure shows its parameters, the rendered figure, and its discussion text.
- Parameter edits update the figure. Parameters the plot component marks as live update the drawn figure instantly. Every other parameter makes the component rebuild the figure in the background. A **Regenerate** button forces a rebuild, for example to pick up new runs.

Out of scope: export to PNG, SVG or PDF; Markdown preview; renaming, moving or copying figures; and converting plot components other than the first provider.

## Plotting library: Vega-Lite

Plot components produce Vega-Lite specs, and the embed path accepts full Vega specs as well. Reasons:

- **Specs are data.** A spec is JSON that a component builds in Python, with its data inline.
- **Live parameters are built in.** A Vega-Lite top-level param is a signal, and `view.signal(name, value)` changes it without rebuilding the spec. This covers properties that accept expressions (colors, widths, opacity, title text, filters).
- **No new rendering stack.** The app already ships QtWebEngine and a QWebChannel bridge (the xterm.js terminal uses the same pattern).

Changes that need different data, such as a different metric, smoothing done in Python, or a different set of methods, go through regeneration. This is what makes deeper parameters possible: the component recomputes and returns a new spec.

The existing matplotlib plotters keep `run()` for the pipeline's saved plots. The figure spec is an additional, opt-in capability.

## Plot component contract: `figure/vega-lite-v1`

A plotter capability opts in to Discussion by doing three things:

1. It declares `"provides": ["figure/vega-lite-v1"]` in `.theta_component.json`.
2. It gives the capability a `parameter_schema`, a JSON Schema of the figure parameters with defaults. A property marked `"x-theta-live": true` is a live parameter. Every other property needs regeneration.
3. It implements `figure_spec(self, exp_id: str, params: dict) -> dict` on its `BasePlotter` subclass. The method returns a Vega-Lite spec with inline `data.values`.
   - `params` is the schema defaults merged with the figure's saved values.
   - For every live parameter, the spec declares a top-level `params` entry with the same name, whose initial `value` is the current value.
   - The spec has no `bind`. Theta's form owns the controls.

The host checks a returned spec:

- A live parameter with no matching spec param is an error, shown in the figure pane.
- The host never edits the spec's structure. It only sets signals.

Compatibility comes from the capability's existing `requires` rules (see [compatibility requirements](../reference/COMPATIBILITY_REQUIREMENTS.md)), evaluated against the composed experiment. Add `figure_findings(cfg, capability, *, paths)` to `src/app/core/requirements.py`:

- `config` rules read the composed experiment.
- `kind` rules are satisfied when any enabled method's selection satisfies them.
- Preflight hooks are never run from the form.

The **+** dropdown omits candidates with block findings. It marks candidates with warn findings, with the same icon and tooltip as `controls.mark_candidates`.

SDK: export `FIGURE_CONTRACT = "figure/vega-lite-v1"` from `src.app.api.plotting`. This is a minor addition (SDK 3.2): update `SDK_VERSION` and [the SDK reference](../reference/SDK.md).

## Storage

```text
src/usr/discussion/<group>/<subdirs>/<experimentID>/<figure>.yaml   # plotter + params
src/usr/discussion/<group>/<subdirs>/<experimentID>/<figure>.md     # discussion text
```

- The layout follows `src/usr/config/experiment/<group>/<subdirs>/<experimentID>.yaml`.
- An experiment's folder is created when its first figure is written. Until then the tree shows the experiment with no children.
- `<figure>.yaml` has exactly two keys and rejects any others:
  - `plotter`: the capability selection, resolved through the provider catalog.
  - `params`: explicit values only.
- Unset parameters show the schema default and are not written. This keeps inherited defaults distinct from overrides, as in other forms.
- The spec is derived. It is regenerated when a figure opens and never stored.
- `<figure>.md` is optional and is created on the first keystroke.
- `discussion` joins `RESERVED_RESEARCH_DIRECTORIES`. `WorkspacePaths.discussion` returns `src/usr/discussion`.

## Units

### 1. Model (no Qt)

`frontend/discussion/model.py`, headless like `config_model.py`:

- `experiment_tree(paths)` lists the groups, subfolders and experiment IDs under the config experiment directory. It skips `_base.yaml`, as the Experiment tree does.
- `figures(paths, experiment_rel)` returns the `<figure>.yaml` files.
- `FigureDoc` handles `load`, `save`, `set_param`, `unset_param` and `schema()` (from the selected capability).
  - It validates `params` against the schema with `jsonschema` and rejects unknown keys.
  - `live_params()` returns the names marked `x-theta-live`.
- `create_figure(paths, experiment_rel, capability)` writes `<capability>.yaml` with empty `params`, adding `_2`, `_3` and so on when the name already exists.
- `read_discussion` / `write_discussion` read and write the `.md` file.

### 2. Contract and first provider

- Add `figure_findings` and `FIGURE_CONTRACT` as described above.
- `compatible_figure_plotters(cfg, *, paths)` returns the plotter capabilities that provide the contract, each with its findings. Compose the experiment with the composition that validation already uses; don't write a new one.
- First provider: the bundled `standard_figures/convergence` (bump to 1.1.0).
  - `parameter_schema` live parameters: `line_width`, `opacity`, `title`.
  - `parameter_schema` regeneration parameters: `metric` (enum of logged metrics where knowable, otherwise a string), `smoothing_window`, `methods` (list).
  - `figure_spec` builds the spec from `load_metrics` and `moving_average`, and thins long series with the existing results helpers.

### 3. Tree widget

`frontend/discussion/tree.py`: `DiscussionTree` emits `figure_selected(path)` and `experiment_selected(rel)`.

- It's a new widget, not another `mode` of `ConfigTreeWidget`.
- It reuses `tree_tools` (`TreeHeaderWidget`, `TreeHeaderActions`, `TreeToolbarPersistence`, `DraggableToolButton`, `style_tool_button`, `folder_icon`, `GroupItemDelegate`, `group_heading_font`), so it looks and behaves like the Experiment tree.
- The toolbar uses square buttons, as the Experiment tree does:
  - `row1_left`: reload, collapse, expand, search.
  - `row1_right`: **+** (`plus`) and **Regenerate** (`reload`, or a new icon if the two are confusable).
- **+** is enabled while an experiment, or a figure inside one, is selected.
  - Clicking it opens a `QMenu` of compatible plotters, labelled by capability and marked when they have warnings.
  - With no compatible plotters it shows one disabled item: "No compatible plot components".
  - Choosing a plotter creates the figure and selects it.
- **Regenerate** is enabled while a figure is selected.

### 4. Parameters box

`frontend/discussion/params.py`: one `ConfigBox` titled **Parameters**, built from the capability's `parameter_schema` with `ParameterForm` / `render_nodes` / `value_editor`, following [the shared parameter form conventions](../reference/CONFIG_PARAMETER_FORMS.md).

- Use the ordinary behavior: **+** beside the title adds optional parameters, and × on row hover removes them. Required ones get no ×.
- Editing a live parameter calls `set_param`, saves, and pushes `setParam(name, value)`.
- Editing any other parameter calls `set_param`, saves, and schedules regeneration.
- Typed commits work as everywhere else: invalid drafts leave the file unchanged.

### 5. Figure view and regeneration

`frontend/discussion/view.py` plus `frontend/discussion/vega.html`:

- Vendor pinned `vega`, `vega-lite` and `vega-embed` builds into `frontend/discussion/vendor/`. Add them to package data and `MANIFEST.in`, and record them in [the third-party libraries reference](../reference/THIRD_PARTY_LIBRARIES.md) (BSD-3-Clause).
- Over QWebChannel the view provides `render(spec)`, `setParam(name, value)` and `error(message)`. Errors replace the figure with the message.
- Theme colors come from `theme_color` tokens through the embed `config`.
- The WebEngine view is created when the panel is first shown, to protect startup time.
- `FigureBuilder` runs `figure_spec` in a worker thread:
  - It debounces about 400 ms after a regeneration-parameter edit.
  - A newer request supersedes an older one.
  - A **Regenerating…** state shows over the old figure.
  - Exceptions from the component go to the figure pane as errors.

### 6. Discussion editor and panel

- `frontend/discussion/panel.py`: `DiscussionPanel` with the splitter layout above.
- The discussion box is a `QPlainTextEdit` with debounced autosave of about 500 ms, plus a save when the selection changes or the app quits.
- With no figure selected, the parameters and figure panes show one muted line: "Select a figure".
- `app.py` `make_center`: `addTab(..., "Discussion", "discussion", "Discussion", tab_id="discussion")` after Results. Add `frontend/icons/discussion.svg` in the existing icon style.

### 7. Tests and docs

Tests are hermetic, under `tests/fixtures/` only:

- Fixtures:
  - A fixture plot component that provides the contract: a tiny `figure_spec` over fixture metrics, with one live parameter and one regeneration parameter, plus one `requires` block rule and one `warn` rule.
  - A fixture experiment tree.
  - One saved figure with its `.md` file.
- `tests/test_discussion_model.py`:
  - The tree mirrors the experiment groups.
  - Figure YAML validation, including rejected unknown keys and parameters.
  - `set_param` / `unset_param` write explicit values only.
  - `create_figure` makes unique names.
  - Discussion read and write.
- `tests/test_figure_contract.py`:
  - `figure_findings` block and warn findings.
  - Candidate listing.
  - A live parameter missing from the spec is an error.
  - The convergence provider returns a valid spec for fixture metrics.
- `tests/test_discussion_panel.py` (offscreen Qt, figure view stubbed):
  - The + menu omits blocked candidates and marks warned ones.
  - Choosing a candidate creates and selects a figure.
  - A live edit calls `setParam` without regenerating.
  - A regeneration edit rebuilds once after the debounce.
  - Regenerate rebuilds.
  - The discussion text saves.
- Add the Parameters box to the cross-box cases in `tests/test_parameter_forms.py`.
- Docs:
  - `docs/guides/DISCUSSION.md`: storage layout and using the panel.
  - A figure contract section in `docs/reference/` (or a new `FIGURE_CONTRACT.md`).
  - Updates to SDK.md and the [docs index](../README.md).

## Open decision

`frontend/figures/` is an unmounted matplotlib figure composer, used only by tests, and it overlaps with this panel. Leave it untouched in this work. The user will decide afterwards whether to remove it.
