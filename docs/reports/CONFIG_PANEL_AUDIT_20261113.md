# Configuration panel architecture audit

Date: 2026-11-13. Scope: the Qt configuration editor, its document/edit paths, shared controls, layout/reordering implementation, and relevant tests. This is a source audit with offscreen test execution and targeted probes; it does not establish the exact cause of every reported visual bug in the running desktop app.

## Assessment

The proposed design is a good direction: one shared parameter presentation, with box-specific compositions. Much of its vocabulary already exists, but ownership is incomplete. This is a substantial, bounded refactor, rather than a new GUI implementation.

`frontend/config_viewer.py` has 7,672 lines. `ConfigViewer` has 191 directly defined methods; `_render_experiment_boxes` alone occupies 1,555 lines and constructs both Experiment and Methods. Splitting that file would improve navigation, but would not by itself resolve the architectural problem: rendering, schema discovery, inheritance, editing, persistence, geometry repair, and domain rules are interleaved.

Treat “each parameter gets its own box” as a self-contained **row component**, including a label, editor, actions, and optional children. A parameter need not acquire a visible card border. Keep `ConfigBox` as the outer section card. Prefer composing these shared pieces over a deep inheritance hierarchy of box classes.

Implementation proposal: [Config panel refactor plan](../plans/CONFIG_PANEL_REFACTOR_PLAN.md).

## Gap by responsibility

| Responsibility | Existing implementation | Remaining gap |
| --- | --- | --- |
| Card, heading, type controls | `ConfigBox`, `_parameter_header`, `_parameter_title`; reusable widgets in `frontend/widgets.py` | Small foundation gap; specialized code still assembles headers/styles independently. |
| Parameter row and actions | `_parameter_row`, `_RowHoverReveal`, schema actions | Medium/large adoption gap; many manual rows duplicate action placement and policy. |
| Recursive nesting | `_render_generic_dict_contents`, `IndentedHierarchyBlock` | Large integration gap; Methods has another recursive renderer and distinct group structures. |
| Two-column placement | `_two_column_parameter_layout`, `ColumnRows`, experiment budget grid | Large design gap; several layout algorithms, inferred groups, no explicit region/slot contract. |
| Experiment composition | Helpers for Description, Site, separator, trainer | Medium/large extraction; base and experiment selector construction still diverge, budget rows are bespoke. |
| Methods composition | Specialized card container, model recursion, sweeps | Largest migration; preserve specialized layout while replacing ordinary rows and edit plumbing. |
| Document edits and defaults | `raw_data`, resolved defaults, many edit handlers | Large boundary gap; typed writes, inheritance and save normalization remain in the viewer. |
| Verification | Extensive cross-box tests | Good starting point; fill correctness gaps and remove unnecessary dependency on installed plugins. |

## Findings

### 1. Shared geometry is applied after construction

Evidence: `config_viewer.py:1923`, `:2218`, `:2253`, `:934`.

The build ends with `_cap_form_fields`, `_size_method_card_fields`, and `ConfigBox.standardize_parameters`. These walk descendants and infer policy from widget type and dynamic properties. Builders also set their own widths. `ParameterForm` initially gives labels a 180px width, the standardization pass selects 180px or 120px, and the budget builder initially uses 145px labels. Methods bypasses ordinary standardization entirely.

Impact: layout depends on construction, ancestry, and pass order. A change to one shared pass can affect controls in unrelated boxes. The new row/group components should own their geometry when constructed; layout density belongs to a region, not the entire card.

### 2. Two columns have several meanings

Evidence: `config_viewer.py:356`, `:2557`, `:2780`; `parameter_reorder.py:61`, `:110`, `:194`.

The shared helper consumes a form and redistributes inferred groups. `ColumnRows` either balances by size hints or assigns physical rows by the number of group parts. A whole child block counts as one part even if it contains many descendant rows. Experiment runtime settings instead use a four-column label/value grid with a separate reordering adapter.

This explains architectural fragility, but is not a reproduction of the user's specific visual defect. A nested block is atomic in the outer grid: its descendants cannot align independently with opposite-column rows. Group membership is inferred from adjacency, labels and widget names. The target needs explicit row/group identity and placement, including what happens when a parent has many children.

### 3. Nesting uses shared indentation but inconsistent structure

Evidence: `config_viewer.py:709`, `:5288`, `:5844`, `:6060`.

`IndentedHierarchyBlock` already uses one `HIERARCHY_INDENT` increment; `level` names/styles the block rather than multiplying the increment. However, it creates a plain `QFormLayout`, and callers do not consistently place headers outside the block or insert children as spanning rows. Methods includes headers inside blocks and some empty-label child rows. Those placements can add label-column offsets beyond the intended indentation.

Make nesting a construction invariant: parent row followed by a child container, with one increment per logical depth. Ordinary descendant parameters each receive their own row.

### 4. Repeated edit behavior is distributed across boxes

Evidence: `config_viewer.py:5234`, `:5378`, `:5657`–`:5826`, `:7128`–`:7320`.

Add/remove, requiredness, inherited defaults, component identity and dirty/refresh decisions are implemented through separate generic, experiment, shared-method and method-model handlers. These have real domain differences, but the renderer should receive their capabilities through bindings. It should not decide domain semantics by checking a YAML key.

### 5. Typed string writes can change YAML type

Evidence: `_schema_value_widget` (`:4990`), `_on_generic_path_edited` (`:5378`), `_on_field_edited` (`:5414`), `_parse_yaml_value` (`:536`).

String editors emit strings, but downstream write handlers parse those strings again. Targeted probes confirmed that writing Description `"123"` through `_on_field_edited` stores integer `123`, and generic string `"false"` stores boolean `False`. A declared string schema does not prevent this write-time reinterpretation.

Introduce typed commits; reserve YAML parsing for an explicitly raw-value editor or custom-value dialog. Also distinguish absent, null, empty string, false and zero.

### 6. Numeric correctness still varies by editor path

Evidence: `number_field` (`:694`), budget controls (`:2563`), method batch size (`:3556`), schema numeric controls (`:4990`), `widgets.py:77`.

The exact-integer factory exists, but numerous specialized fields still instantiate bounded `SpinBox` controls. The generic float fallback uses six decimals and a maximum of 100,000,000; a targeted probe showed `number_field(1e-9).value()` becomes `0.0`. Nonnegative loaded values also acquire a zero minimum without a schema declaration. This demonstrates display rounding/restriction; it does not establish an immediate disk write on load.

Unify numeric editor selection and validation. Preserve loaded values even when invalid under a declared schema; show an error rather than silently correcting them. Genuine runtime bounds should remain explicit schema/domain rules.

### 7. The document lifecycle does more than render and write YAML

Evidence: `load_file` (`:1773`), `_clean_redundant_params` (`:2163`), `save_to_disk` (`:7584`).

Load cleans the in-memory document, and save performs further cleanup, group reconciliation and empty-method removal. Saving uses `yaml.safe_dump` plus retained leading directives, so arbitrary comments and source formatting are not round-tripped. These are existing semantics to inventory and deliberately preserve or revise, independently of the layout refactor.

The editor also represents Hydra selections, inherited values, component schemas and plugin catalogs. It can have a generic rendering core, but still needs a domain adapter between YAML and the UI.

### 8. Layout identity and refresh are inferred too late

Evidence: `_setup_parameter_reordering` (`:1943`), `_render_boxes` (`:1811`), `_refresh_sweep_box` (`:6573`).

Reorder scopes use visible titles, ancestor captions and occurrence indices. Refactoring headings/layout structure risks losing saved display order. Structural actions frequently rebuild the entire panel; local refresh already exists for sweeps. Scalar edits and dragging generally preserve editors, which should remain a hard requirement.

Use stable logical paths/region IDs, migrate order preferences, and carry focus, cursor, expansion and viewport state through necessary structural refreshes.

## Details to add to the proposed design

- **Ownership and identity:** each row needs a tuple YAML path, stable presentation ID, schema owner, and binding. Component-owned schema paths can differ from storage paths.
- **Presence and inheritance:** required versus optional is separate from inherited versus explicit. Deleting an override may reveal a default; explicitly setting null may mask it. Adding a child to a scalar selector must retain the selected component.
- **Column policy:** specify parameter sets, placement, fill order, parent/child alignment, responsive collapse and reordering boundaries. Content height should not silently move fields between columns. Deep nesting must remain usable at narrow widths.
- **Editor lifecycle:** hold draft text until valid, use typed commits, preserve unknown fields, and define validation feedback. Autosave should never persist intermediate invalid list/number edits.
- **Specialized editors:** Methods cards, Figures and Tables, Recorded Metrics, layer stacks, tag editors, hierarchical ruleset menus and sweeps remain supported. They should reuse typography/actions/bindings without being forced into ordinary row geometry.
- **Sweeps and dependencies:** swept fields retain their unswept value, inline bounds, wrapping lists, distinct hover policy and inherited tuning activation. Dependency updates must preserve unrelated editors.
- **Interaction and accessibility:** preserve label dragging, editor gestures, keyboard focus order, disclosure controls, accessible names and action visibility on keyboard focus. Ordinary remove buttons reserve width; sweep-list entry removals intentionally do not.
- **Read-only and persistence:** preserve Hub read-only gating, debounced autosave, error handling, preamble directives, and presentation-only display ordering. Decide separately whether full YAML formatting/comment preservation is required.

## Verification

Executed with the repository `venv` and `QT_QPA_PLATFORM=offscreen`:

```text
venv/bin/python -m pytest tests/test_parameter_forms.py tests/test_config_viewer_composite.py tests/test_config_model.py tests/test_parameter_actions.py -q
146 passed, 19 skipped in 21.84s
```

The composite viewer suite's 17 cases skipped behind its installed CartPole/BlendRL prerequisite; two model cases also skipped. Passing tests protect many current behaviors, not complete visual correctness. The form geometry tests include 960/1050/1400px cases; add narrower layouts, deeper nesting, long labels and mixed-height editors. Targeted probes above expose correctness gaps outside the current passing baseline. No runtime implementation was changed by this audit.
