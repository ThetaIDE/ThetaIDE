# Discussion panel implementation prompt

Paste this into a new agent session at the repository root.

---

Build the **Discussion** panel for ThetaIDE (PyQt6 desktop app, `frontend/`) by following `docs/plans/DISCUSSION_PANEL_PLAN.md`. Read that plan first, then `AGENTS.md`, `docs/reference/CONFIG_PARAMETER_FORMS.md`, `docs/reference/COMPATIBILITY_REQUIREMENTS.md` and `docs/reference/SDK.md`.

What the user asked for, in their own terms:

- A side tab called **Discussion**.
- Its tree mirrors the Experiment panel's file tree. Experiments are folders that hold figures.
- Clicking a figure shows its parameters to the right, then the rendered figure (top right) with a discussion text box below it.
- Figures are Vega-Lite and come from plot components.
- Parameters the component marks live re-render immediately. Deeper parameters make the component regenerate the figure, and a Regenerate button forces a rebuild.
- Add a figure with a square **+** button in the tree's header toolbar, like the existing square icons on the Experiment tree. It opens a dropdown of plot components compatible with the selected experiment.
- Storage is `src/usr/discussion/`.

Build exactly that. Don't add features the plan lists as out of scope: export, Markdown preview, rename or move, or more providers. Don't add extra polish.

Work through units 1–7 in order, without check-ins. After each unit:

- Run its tests one file at a time (`pytest tests/<file>.py`).
- Commit on a feature branch (`feature/discussion-panel`) with a clear message.

Stop only for a decision the plan reserves for the user. The only one is whether to remove `frontend/figures/`; leave it untouched.

Constraints:

- **No legacy support.** The user is the sole user. Use clean breaks, with no compatibility shims or deprecations.
- **One home per setting.**
  - Figure parameter values live only in `<figure>.yaml` `params`.
  - The spec is derived and never stored.
  - There are no fallback chains.
  - Reject unknown keys.
- **Shared controls.** Follow the parameter form conventions and build the Parameters box from `frontend/config_panel` (`ConfigBox`, `ParameterForm`, `render_nodes`, `value_editor`, `ToggleSlider`, the shared spin boxes). Don't write new control styles.
- **Shared tree tools.** Build the tree from `frontend/tree_tools.py` so it matches the Experiment tree. Make it a new widget, not a new `mode` in `ConfigTreeWidget`.
- **Generic host.** Host code never branches on a plotter ID. Compatibility goes only through `requires` declarations and the new `figure_findings`. Forms never run preflight hooks.
- **Hermetic tests.** Tests read only `tests/fixtures`, never the user's workspace, installed packages, sibling checkouts or tracked `src/usr`. Build the fixture plot component the plan describes.
- **Offline Vega.** Vendor pinned vega, vega-lite and vega-embed builds (as with xterm.js in `frontend/terminal/vendor/`). Package them and record their licenses.
- **Startup time.** Create the WebEngine view only when the panel is first shown.
- **Docs.** Put new docs in the matching folder and update `docs/README.md`. Update the plan's status line as units land.

When you finish:

1. Launch the app and confirm, with a screenshot, that:
   - the tab appears;
   - **+** lists the convergence plotter for an experiment that has runs;
   - a live parameter edit updates instantly;
   - a metric change regenerates;
   - the discussion text persists.
2. Report what was verified and what wasn't, and list any test failures with their output.
