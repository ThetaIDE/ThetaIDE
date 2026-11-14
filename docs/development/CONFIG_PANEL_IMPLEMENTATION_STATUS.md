# Config panel implementation and extension guide

Date: 2026-11-13. Implements the shared document/presentation/section migration from the [seven-stage plan](../plans/CONFIG_PANEL_REFACTOR_PLAN.md), informed by the [audit](../reports/CONFIG_PANEL_AUDIT_20261113.md). The final integration acceptance gate is **not fully green**: missing installed providers and existing wider-suite failures are recorded below.

## Ownership

| Module under `frontend/config_panel/` | Owns |
| --- | --- |
| `document.py` | Qt-independent explicit/effective reads, missing sentinel, typed set/add/remove results, validation, notification and normalized persistence. |
| `bindings.py` | Tuple storage identity, schema/action capabilities, component identity promotion, and legacy/named/keyed method storage paths. |
| `domain.py` | Hydra selections, inheritance and structural domain actions, using the existing configuration services; load/save normalization. |
| `catalogs.py` | Config-tree injection/cache and catalog read helpers. |
| `controls.py` | Cards, headers, `ParameterRow`, `ParameterForm`, labels, hierarchy containers, hover/focus actions and specialized method-card controls. |
| `editors.py` | Schema/type editor selection, exact integers, round-trip floats and validated drafts. |
| `rendering.py` | `ParameterNode` and the shared row/group tree renderer, including explicit subtree membership. |
| `layout.py` | Responsive regions and physical row slots. |
| `sections/` | Experiment/Group Defaults, Methods, shared-method and model-schema adapters, components, sweep, variability, monitors/metrics and leftover configuration. |

`ConfigViewer` hosts document selection, dirty/autosave state, section refresh, code/resource actions and display-order persistence. Its forwarding methods and import re-exports preserve existing callers; sections do not import the viewer module. Domain adapters receive the host explicitly rather than inheriting a viewer class. Structural domain actions retain their existing component/inheritance semantics; ordinary typed writes share the document adapter.

The former `_cap_form_fields`, `_size_method_card_fields`, `ConfigBox.standardize_parameters`, inferred `ColumnRows`/`GridRows`/`ListRows`, compact model-option renderer and shared-method widget recursion have been removed. The raw `QFormLayout` reorder adapter remains a compatibility boundary; migrated parameter subtrees declare identity and child membership through `add_parameter` / `add_children` or `ParameterNode`.

## Frozen behavior and deliberate changes

- Context dragging retains cross-column movement. Logical sibling order assigns mutable row slots. Resizing and expansion do not use content height to choose a new order. Context and runtime are separate reorder scopes.
- Row-action hover bounds stay within the owning column, including its label. Active sweep containers fit their contents (and still wrap when constrained), keeping the removal action beside the sweep rather than stretching it toward the method-card edge. Follow-up geometry/hover regressions verify both range and list sweeps at normal/wide widths; the form suite passes 115 tests.
- Regions collapse to one column when their measured row minimums, including actions and indentation, cannot fit twice. Wide placement counts every row in a nested child form. Site occupies the slot below Paradigm beside Environment children in the default order.
- Each nesting container contributes one indent. Group headers are outside their child containers. Ordinary labels/editors use 180/220 px; compact regions use 120/140 px. Method forms declare their specialized editor width. Labels remain on one line, clip at the field, and reveal above it only while the label is hovered. The reveal is mouse-transparent and never changes editor geometry. Configure-sweep buttons collapse their width while hidden; ordinary remove actions retain their reserved space.
- Layer stacks, tags, hierarchical rulesets, Description, sweep ranges/lists, method cards, Figures and Tables and Recorded Metrics retain their specialized controls/compositions. Ordinary children formerly compacted into model option rows now receive individual rows.
- Text emitted by a typed editor stays text, including `"123"`, `"false"`, `"null"` and empty strings. Raw-value dialogs retain their explicit YAML parser. Number/list drafts commit after validation; invalid drafts do not dirty or autosave the document. Loaded invalid values remain available for correction.
- Integer controls preserve arbitrary Python integers and loaded out-of-domain values; UI-only maxima were removed. Float fallback uses round-trip scientific notation. Provider-declared bounds remain validation rules. Expanding a consecutive seed count into an explicit list is limited to 100000 entries; larger scalar run counts remain representable.
- Reads do not create inherited overrides. Explicit null masks inherited children. Promoting a scalar component to a dictionary retains its identity. Required fields have no removal capability; inherited fields without explicit overrides have no ordinary remove action.
- Read-only document operations reject edits and persistence. Failed autosave retains dirty data and now prevents switching to another file. Ordinary scalar edits and display dragging preserve live editors.
- Existing normalization remains: load runs redundant-parameter cleanup; save runs cleanup, group reconciliation and empty-method cleanup in that order, followed by `ConfigurationFiles.write`. Group bases omit `group`/`experiment_id`, retain concrete default methods and reconcile forbidden paradigm fields. Existing experiment/component identity cleanup is retained.
- Persistence retains leading directives/comments and uses `yaml.safe_dump(sort_keys=False)`. Arbitrary comments, quotes and original YAML source formatting are not round-tripped; full source fidelity remains a separate persistence project.
- Stable region/path display-order scopes migrate identifiable legacy caption keys. Old preference entries and absent optional positions are retained. Ambiguous old scopes are not overwritten. Presentation changes do not dirty YAML.

## Section inventory

| Composition | Storage and dependencies | Special capabilities |
| --- | --- | --- |
| Experiment / Group Defaults | `description`, `paradigm`, Hydra environment/site `defaults`, `env`, `resources`, `trainer`, runtime/budget keys and unknown scalars; group/config/paradigm inheritance | Same context/runtime composition; site always visible without writing an override; paradigm constraints; static dataset source; checkpoint mode; wrapping runtime flags. |
| Methods | `methods.<name>`, `methods.params`, named/keyed/legacy-flat algorithm and model parameters, `tune`; group method defaults and component schemas | Create/rename/duplicate/card drag, enable control, algorithm/model selectors, component-owned schema actions; requiredness from the active schema. |
| Data Loaders / Evaluators | `data_module`, `symbolic_state_loader`, per-method loader roles, `evaluator`, component `params`; pipeline input contracts and provider metadata | Required loader bindings, inherited selection, dependency/status information, child adders and open actions. |
| Sweep / Objective | `tuning`, `tuning.options.<provider>`, `objective.component/params`, method tuning dimensions | Activation from composed tuning; null inheritance masks; provider `when` dependencies and local refresh; unswept values retained. |
| Variability | `variability`, `eval_episodes`, inherited seeds/evaluation defaults | Explicit seed lists versus consecutive runs, evaluation seeds, parallel limits. |
| Monitors / Recorded Metrics | `monitors`, `telemetry.monitors`, `telemetry.metrics`; provider stream catalogs | Group expansion/toggles, provider settings, nested output streams and the specialized metric picker. |
| Figures and Tables | `plots`, `plot_overrides`, inherited plot selections | Specialized cards with shared parameter rows, explicit selection/masks and per-plot defaults. |
| Components / Other Sections | Component profile/schema properties and unknown mappings/lists | Shared recursive nodes, schema-required controls and preserved unknown keys. |

## Adding an ordinary parameter or section

1. Resolve the actual tuple storage path, schema owner/path, explicit/effective value and requiredness in a binding or domain adapter. Use `MISSING` for absence; do not treat null/false/zero/empty as absence.
2. Use `value_editor` (or the host's schema/domain editor hook) and typed binding/document callbacks. A string editor must never call the raw YAML parser. Rich editors are explicit hooks.
3. Create `ParameterNode(identity, title, row=ParameterRow(editor, ...), children=...)` and call `render_nodes`, or declare an individual row with `ParameterForm.add_parameter`. Use `add_children(identity, block)` when a domain selector's children are constructed separately. Do not infer a parent by looking at the previous Qt row.
4. Give a sibling form a storage-based `parameterContext`. Use `two_column_form(..., identity=...)` for a region; declare membership in logical order. Reuse the shared density/nesting/action policy. Never add a descendant geometry repair pass.
5. Request the smallest refresh required by the edit result/domain dependency. Provider sweep settings already refresh locally; structural section changes retain the viewport and cancel an active drag safely. Keep ordinary scalar commits in place.
6. Extend `tests/test_parameter_forms.py` for the real add/edit/remove path, geometry, requiredness, hover/focus and reordering. Core coverage uses fixture schemas rather than installed plugins.

## Verification and remaining acceptance limits

The audit baseline was reproduced before editing: **146 passed, 19 skipped** in the four named audit suites. The expanded main form/document/model/actions/constraints/variability/sweep/composite run passes **231 tests, with 20 skips**. The skips are 17 installed CartPole/BlendRL composite prerequisites, two model prerequisites and one constraint prerequisite. Deterministic model-schema fixtures now exercise nested model selectors, ordinary option rows, required controls and child identity without those plugins.

Native Qt snapshots were inspected at 450/1050/1400 px in Gruvbox Dark and Light. They confirmed single-column collapse, two-column context placement and clean load/resize state. Focused native tests at `QT_SCALE_FACTOR=1.25` passed eight drag/resize/typed-edit cases. Their two Description cases initially exposed stale ancestor layout caches; after adding shared content-height notifications and deferred parent activation, both passed on native Qt. Offscreen coverage includes keyboard gestures, focus/hover actions, mixed-height sweeps and live drag cancellation. This is targeted review, not an exhaustive manual review of every installed provider/theme combination.

Wider checks reproduce these existing failures on the unchanged `HEAD` renderer (loaded into an isolated test process):

- `test_experiment_pane.py` / `test_panel_sections.py`: ten existing catalog/UI expectation failures (provider-defined text option discovery, component hub labels/close button, window controls, and fixed-key/heading expectations).
- `test_config_compose.py`, `test_pipeline_config.py`, `test_experiment_component_overrides.py`: seven existing execution/provider/merge failures (unresolved PPO, removed entrypoint contract, deep-merge expectation, evaluator registrations and plotter registration).

A combined nine-suite run (those five suites plus `test_pipeline_config_extended.py`, `test_repeated_task_config.py`, `test_config_tabs.py`, and `test_component_gui_enhancements.py`) produced **173 passed, 31 failed** on both implementations, with the same failed test names. In this ordering, the existing fixture-root state also produces missing-experiment failures, and the enhanced-widget fixture supplies a `MockSettings` without the existing `.get` interface. The narrower results above were obtained in separate suite groups; their failure totals should not be substituted for this combined result.

Those failures are not marked as passing or hidden with new skips. No installed CartPole/BlendRL workspace is available here (`in/config/experiment` contains no experiment YAML), so representative installed-provider launch validation remains unverified. The complete seven-stage acceptance gate must be revisited in a populated workspace with the wider-suite baseline issues resolved. No Hub component release or plugin installation was performed.

Hover-action follow-up: configure-sweep and sweep-list remove actions are now detached overlays rather than expanding layout items. They extend beyond entry/card bounds on the same line, retain pointer access outside the original entry, and hide with their owner. Hover visibility no longer calls layout invalidation/activation. Narrow-width stress coverage repeats hover cycles and asserts unchanged flow geometry before clicking the floating action.
