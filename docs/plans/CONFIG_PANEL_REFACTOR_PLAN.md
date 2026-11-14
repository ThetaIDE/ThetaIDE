# Config panel refactor implementation plan

Status: implementation delivered for shared document/presentation and section migration, 2026-11-13; final integration acceptance remains limited by missing installed-provider prerequisites and existing wider-suite failures. See the [implementation status and extension guide](../development/CONFIG_PANEL_IMPLEMENTATION_STATUS.md). Based on the [configuration panel audit](../reports/CONFIG_PANEL_AUDIT_20261113.md). Follow the [shared parameter form conventions](../reference/CONFIG_PARAMETER_FORMS.md); update that contract when adopting deliberately changed layout behavior.

## Intended result

One parameter row/group implementation owns ordinary control presentation. Section editors compose it and declare their parameter sets, layout and domain-specific capabilities. `ConfigViewer` coordinates document selection and section hosts. A document adapter owns typed edits, inheritance, validation and persistence.

Keep the Qt application and existing widget library. Use composition, with specialized editors as explicit extension points. Avoid building an unrestricted layout language or introducing a new YAML UI framework.

## Proposed boundaries

Proposed modules under `frontend/config_panel/` (names can be refined during extraction):

| Module | Responsibility |
| --- | --- |
| `document.py` | Explicit document, effective values, provenance, typed patch commands, edit notifications and save orchestration; reuse existing configuration services. |
| `bindings.py` | Stable tuple paths/IDs, schema lookup and capabilities for add, remove, reset, select and open actions. Domain adapters handle Hydra/component identity. |
| `controls.py` | Shared row/header/group presentation, typography and action behavior; reuse `frontend/widgets.py`. |
| `editors.py` | Schema/type-to-editor factory and validation; explicit hooks for domain and rich editors. |
| `layout.py` | Single-column and two-column regions, density, indentation and responsive placement. |
| `sections/` | Experiment, Methods, components, sweep, variability, monitors, figures and metrics composition. |

Keep document logic Qt-independent where practical. Represent edit results as value changes, structural changes or dependency changes, so the host can choose the smallest refresh. Existing `config_model.py`, `parameter_actions.py`, configuration services and provider catalogs remain useful dependencies.

A binding describes storage path, schema context, effective/explicit value, presence, requiredness, validation and action capabilities. The row renders it; it does not infer these from key names. Use a missing-value sentinel distinct from null. Domain bindings can synthesize a selector from Hydra `defaults` while exposing an ordinary row to the renderer.

## Layout contract

- `ParameterRow` owns label/editor/action slots. Child parameters each get a row, including child parameters currently compacted together by specialized method model code unless an explicit rich-editor exception applies.
- `ParameterGroup` owns a parent row and children, remaining one logical subtree for dragging. It declares its identity directly; never rediscover groups by inspecting adjacent Qt items.
- A nesting container adds one shared indent increment. Children span the local region width; a label column must never add another indent. Define alignment for labels/editors at the same depth.
- A two-column region declares parameter membership and placement. Experiment declares a context region and a separate runtime region, with Description above, a separator between them, and a wrapping boolean section below.
- Preserve the current context arrangement: Paradigm left, Environment right, Site beneath Paradigm beside Environment overrides. Represent the parent and child rows explicitly so row heights align where intended. Do not count a multi-row child container as one descendant row.
- Use explicit row slots for paired alignment; allow a subtree to occupy multiple slots while preserving its logical grouping. Rich editors may intentionally span slots. Recompute heights on resize/expansion without recreating editors.
- Collapse to one column when measured minimum widths cannot fit labels, editors, indentation and actions. Keep logical order stable across resize. Width policy belongs to the region; Description and boolean controls retain their intrinsic/expanding behavior.
- Preserve sibling-only drag behavior and cancellation. Recommended rule: ordinary reordering changes order within a declared region, while explicitly pinned context slots remain pinned. This deliberately constrains the current free context reordering and requires a documented contract/test update. Alternatively, retaining cross-column context movement requires explicit mutable slot assignments; settle this in phase 1 before replacing the grid.

## Delivery sequence

### 1. Freeze behavior and make decisions concrete

Inventory each current section, its YAML paths, required fields, dependencies, specialized controls and edit actions. Record normalization behavior on load/save. Decide context dragging policy, narrow-width collapse, rich-editor exceptions and YAML formatting fidelity. Full source-format preservation, if desired, is a separate persistence change.

Add regression cases for typed strings (`"123"`, `"false"`, `"null"`), absent/null/empty/zero/false, small and large floats, exact large integers, unknown fields and invalid edits. Use fixture component schemas instead of installed plugins for core form coverage. Reproduce the reported two-column bug with a focused fixture or native inspection before claiming it fixed.

Acceptance: executable behavioral baseline and explicit layout decisions; no dependency on external plugins for core row/group tests.

### 2. Establish typed document edits

Introduce document/binding adapters around existing state. Route a vertical slice of generic component edits through typed set/remove/add commands. Move parsing out of typed string commits; validate list/numeric drafts before writes. Centralize schema/domain bounds and preserve loaded values. Keep existing save normalization explicit in the adapter; do not silently change it during extraction.

Acceptance: targeted correctness regressions pass; selector-child edits preserve identity; inherited reads never create overrides; invalid edits leave stored values unchanged. Existing autosave/read-only behavior remains covered.

### 3. Extract shared rows and nesting

Move and consolidate existing card/header/row/hover primitives. Construct ordinary geometry in these primitives. Replace `IndentedHierarchyBlock`'s separate form setup with shared policy. Provide explicit rich-editor and extra-action slots. Migrate generic component and leftover experiment sections first.

Acceptance: adding a plain schema property needs no new widget/hover/spacing implementation; nested rows have equal indent increments; required fields have no deletion capability; hover/focus actions do not shift editors. Migrated sections no longer depend on descendant geometry repair.

### 4. Replace column layout and reorder inference

Build regions from explicit rows/groups and stable IDs. Update `parameter_reorder.py` to consume these directly while retaining live-widget movement, preview, cancellation and scroll support. Migrate saved display order from existing title/ancestor/index keys where identifiable; retain old preferences when ambiguous rather than overwriting them.

Acceptance: geometry holds at narrow, normal and wide sizes with deep children, long labels, both actions, expanding descriptions and wrapped sweeps. Reordering/resizing preserves values, widgets, dirty state and group membership. Test saved orders across reconstruction and migration.

### 5. Migrate Experiment and Group Defaults

Create one Experiment section composition parameterized by base/experiment context. It declares Description, context region, separator, runtime region and boolean flow. Move schema/value/dependency decisions into bindings and reuse the same row construction in both contexts. Replace bespoke budget controls and action rows with the editor factory.

Acceptance: the two contexts share one layout definition, retain their inheritance differences, and render all current parameter types. Site remains visible above the separator without creating an override. Scalar changes preserve editors; structural changes refresh only affected regions where possible.

### 6. Migrate specialized and provider sections

Adopt shared bindings/rows in evaluator, loader, objective, sweep, variability and monitor forms. Preserve Figures and Tables cards and Recorded Metrics picker as specialized compositions. Migrate Methods in a separate reviewable change: retain method card creation/rename/duplicate/drag and algorithm/model layout, while replacing shared parameter rows and both recursive model/group render paths.

Acceptance: domain selectors, nested model schemas, open actions, required controls, inherited method defaults, sweep storage/activation and provider dependencies behave as before. Specialized sections cannot bypass typed writes or reimplement ordinary action/nesting policy.

### 7. Remove legacy paths and verify the complete panel

Remove migrated manual rows, duplicate recursion and geometry repair passes. Keep temporary import re-exports only as needed for callers/tests. Turn `ConfigViewer` into orchestration; section builders should not reach into other section widgets. Document the extension recipe and update the shared form conventions and documentation index.

Run cross-box contracts, model/config composition and edit/save/launch-override tests. Exercise actual add/edit/remove/save/reload, autosave failures and read-only behavior. Perform native visual/keyboard review at narrow/normal/wide widths, both themes, scaling, expansion, drag and mixed-height rows. Verify representative real provider configurations in addition to deterministic fixtures.

Acceptance: a new ordinary box declares bindings and layout without introducing new geometry, hover, nesting, parsing or dirty-state code. The old renderer paths are removed, and all required checks pass with skips accounted for.

## Scope and effort

Expect multiple reviewable changes across the seven stages, with Methods and persistence carrying the most risk. The first useful delivery is phases 1–5: the shared foundation and Experiment composition. That is a milestone, not completion of the whole-panel refactor. Complete phase 6 before claiming the panel has one universal parameter implementation, and phase 7 before considering the migration finished. A calendar estimate should follow the phase-1 inventory rather than be inferred from line count.
