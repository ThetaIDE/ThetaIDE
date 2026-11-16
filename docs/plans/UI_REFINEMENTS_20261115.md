# UI refinements during the layer refactor

Requested 2026-11-15, after 5c stabilization (`e0a09a2`). Integrate before phase 7. Vega is deferred pending the user's specific design. This work includes both the config changes and the downstream changes below; only the unclear sync-button/tree-navigation notes await clarification. Results migration is still unapproved.

Status: implemented locally on `implement/layer-boundaries`, with the two unclear notes deferred. BlendRL 1.3.1 is prepared in `../theta-plugins-v3` and installed locally after an isolated installation check; nothing was published. [Verification evidence](../reports/UI_REFINEMENTS_VERIFICATION_20261115.json).

## Configuration

- Change BlendRL's saved schema to `blender: neural|symbolic`, with `blender_model` retaining model configuration. Update provider implementation, schema, defaults and tests; no core branches on the provider ID.
- Pad overflowing label reveals and measure label columns so Variability and other long-label forms remain legible.
- Reveal the sweep's top-level parameter adder on hover; child-parameter adders stay visible.
- Transparent edit-action wrappers; edit buttons occupy space only when visible.
- Square, aligned section move buttons.
- Preserve structured saved range/list rendering when the tuner is missing,
  disabled or unconfigured. Disable editing and floating actions until a provider
  is ready; show the provider diagnostic instead of flattening ranges into text.

## Monitor

- Show diagnostics and system monitoring on the main overview with conditional sections, consistent with sweeps.
- Remove Logs, Feed and Files views; keep errors actionable through their durable logs.
- Show generated artifacts in Results while training; Results monitors display an in-progress message directing users to Monitor, without streaming live curves.
- Correct repeated experiment names in stamped run display/storage where appropriate.
- Sync-button issue: unclear note, pending clarification.

## Results

- Shared chart styling across Theta; remove numeric corner readouts and Hide controls from saved curves.
- Move diagnostics into Curves and rename the combined page Monitors. Diagnostics appear only for runs with those outputs; Sweeps only for runs with sweep outputs.
- Files shows experiment contents directly, without an extra enclosing run directory.
- Remove Summary tab.
- Config shows the YAML viewer without its file tree/title.
- Directory-navigation note is pending clarification.

## Panels and terminal

- Remove Model Builder, Figures and Plots panels and their navigation/commands. Preserve plotting primitives used by Monitor/Results until their consumers are changed.
- Remove terminal header and Clear button. Put New Shell at the end of the shell list and per-shell close buttons on the right of each shell label. Preserve shell choice and lifecycle behavior.

## Validation

Use hermetic fixture providers, shared parameter-form tests, focused Monitor/Results/terminal/navigation checks and native visual inspection. Compare existing failures by ID against the [5c verification](../reports/LAYER_BOUNDARY_PHASE5C_VERIFICATION_20261115.json). Run test files separately. Update the current status/handoff when changes land.
