# Compatibility requirements: components declare, core enforces generically

Status: planned 2026-11-16 with the user; decisions below are confirmed. Implement before the v3 Hub publication, then publish. Part of the [layer boundary plan](LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md); its confirmed decisions (no legacy support, one home per setting, no fallback chains) apply.

## Design philosophy (user, 2026-11-16)

- Theta supports the minimum set of paradigms (and, later, workflows that combine them, for example THEMES active learning, which today uses a workaround). Components plug in and play.
- **Components define compatibility; paradigms never list components.** A component states what it needs from the rest of an experiment. Declaring nothing means it is compatible with everything.
- If something breaks, it is the user's responsibility: running unverified component code, using a component outside its declared purpose, or writing bad code. Core enforces what authors declare; it does not police beyond that.
- **The SDK and the IDE do nothing extra for any component.** No core code, SDK export, form section or workspace convention exists for one package or backend. A capability a component needs from core is offered generically, so the next component can use it without core changes.

## Confirmed decisions

1. **Paradigms describe only themselves:** which component kinds a method consists of (for example, algorithm and model required, an environment or a data module), the runner, the budget and the evaluation protocol. `allowed_agents` is removed, and so is `constraints.requires`, which duplicates `method.<kind>: required`.
2. **Compatibility is declared on the component's capability** in `requires`. Rules refer to the method's other selections by **component kind**, never by a core "slot".
3. **Components own their sub-components.** A component that composes or consumes another component (BlendRL's blender model and its symbolic state builder) declares it as one of its own parameters, using a generic component-reference parameter. Core keeps no slot for it.
4. **Two levels, chosen by the author per rule:** `block` (the form refuses the choice, validation fails) and `warn` (the choice stays available; the form, validation output and run record show the warning, and the run proceeds).

## Today: overlapping and component-specific mechanisms

| Mechanism | Where | Fate |
| --- | --- | --- |
| Paradigm compatibility in three homes: profile `compatible_paradigms`, capability `paradigms`, package `paradigms` | forms (`frontend/config_viewer.py`, `frontend/config_model.py`, `frontend/config_panel/domain.py`, `frontend/widgets.py`), `src/app/pipeline/validation.py`, `src/app/core/provider_metadata.py`, `src/app/core/data_loaders.py` | `requires` rule on `paradigm` |
| Paradigm `allowed_agents` (whitelist, fuzzy names) and `constraints.requires` | paradigm profiles, validation fallback | Removed |
| Core loader slots `data_module` + `symbolic_state_loader`, `LOADER_SLOTS`, `SLOT_ROLES`, data-module `role` (profile then capability), `create_state_loader` (SDK) | `src/app/core/data_loaders.py`, `src/app/pipeline/config.py`, `src/app/pipeline/commands.py`, `frontend/config_panel/sections/components.py`, `frontend/config_panel/domain.py`, `src/app/api/data.py` | `symbolic_state_loader`, `role` and `create_state_loader` removed; BlendRL declares its builder as a component-reference parameter. `data_module` stays only as a kind a paradigm's method can consist of |
| `input_requirements` (capability, then package) and `output_contracts` | `src/app/core/data_loaders.py`, `frontend/config_panel/sections/components.py` | `requires` with `contract`, and `provides` |
| `execution_features.input_types` | `src/app/core/execution_features.py` | `requires` on the data module kind |
| `execution_preflight` hook, run at execution only | `src/app/core/execution.py` | `preflight`, run during validation |

`compatibility` (software versions: theta, torch, python) is a different concern and stays.

## Declarations (capability-level only)

```json
{
  "id": "my_ppo", "kind": "algorithm", "entry_point": "agent.py:Agent",
  "provides": ["policy/discrete-v1"],
  "requires": [
    {"level": "block", "config": "paradigm", "in": ["online_rl"]},
    {"level": "block", "kind": "model", "contract": "policy/discrete-v1",
     "message": "my_ppo drives discrete policy networks."},
    {"level": "warn", "kind": "model", "capability": ["dnn"],
     "message": "Only tested with basics/dnn."},
    {"level": "warn", "config": "agent.n_steps", "min": 8, "when": {"agent.mode": ["on_policy"]}}
  ],
  "preflight": "agent.py:check"
}
```

- **`requires`**: rules for each method that selects this capability. Every rule names its `level` (`block` or `warn`); there is no default.
  - `config`: a dotted path into the composed configuration. A path starting with one of the method's component kinds (`agent.`, `model.`, `env.`, `data_module.`) reads that method's selection, including nested parameters (`model.blender_model.name`); other paths (`paradigm`, `tuning.component`) read the experiment. Exactly one comparator: `in` (list), `equals`, `min`, `max`, or `present` (boolean).
  - `kind`: the method's selection of that component kind. Exactly one of `capability` (allowed capability IDs; aliases resolve through the catalog) or `contract` (a contract ID the selected capability lists in `provides`).
  - `when`: optional conditions (same path rules), all of which must match for the rule to apply.
  - `message`: optional text shown in the form and in validation output; otherwise core describes the rule.
- **`provides`**: contract IDs this capability satisfies. Contract IDs are opaque strings agreed between packages; core never interprets them.
- **`preflight`**: optional `file.py:callable` for checks that need code. It receives the composed configuration, the method name and the workspace, and returns `{"block": [...], "warn": [...]}` messages. It runs in validation, not only at execution.
- **Component-reference parameters** generalize the existing `x-theta-domain`: in a capability's `parameter_schema`, `{"x-theta-domain": "<kind>"}` makes a parameter a picker of that component kind, optionally narrowed by `"x-theta-contract": "<contract>"`. The form renders it with the shared picker and child adder for any kind (today it special-cases `model` and `ruleset`). The referenced component's own `requires` apply to it. An SDK helper loads the selected component (`src.app.api.packages.load_selected(kind, selection, *, paths)`), replacing `create_state_loader`.

Removed from the schema (unknown keys afterwards; nothing reads them): profile `compatible_paradigms` and `role`; capability and package `paradigms`; capability `role`, `input_requirements`, `output_contracts`, `execution_preflight`, `execution_features.input_types`; package `input_requirements`; paradigm `allowed_agents` and `constraints`.

## Evaluation (one module)

`src/app/core/requirements.py` evaluates every selected capability's rules and preflight against a composed configuration and returns findings (capability, method, rule, level, message). Every consumer calls it, and none adds its own compatibility logic:

- **Validation** (`src/app/pipeline/validation.py`): `block` findings are `ConfigurationError`s; `warn` findings are printed and returned. Launch, `dry_run`, the desktop's preflight and bundle `check` all go through it.
- **Forms**: a picker omits candidates with `block` findings against the current draft and marks candidates with `warn` findings; a current selection shows its findings inline. Follow [the shared parameter form conventions](../reference/CONFIG_PARAMETER_FORMS.md); extend the shared controls, not individual boxes.
- **Run record**: the run's manifest stores the `warn` findings it ran with, so results show that a run went off-label.
- **SDK** (3.1, additive): `src.app.api.config.compatibility(cfg, *, paths)` returns the findings; `src.app.api.packages.load_selected`.

## Core and SDK do nothing extra for any component (audit)

Remove or generalize each of these; afterwards a search should find no component-specific behavior in `src/app`, `frontend` or the SDK:

- **Symbolic state slot**: as above.
- **Rulesets**: `src/app/api/rulesets.py`, `src/app/services/rulesets.py` (`symbolic_environment`), the `ruleset` domain in `frontend/config_viewer.py`, `frontend/config_model.py`, `frontend/parameter_actions.py` and `src/app/pipeline/workflow/model.py`, and the `rules` entry in `COMPONENT_DIRECTORIES`. Rules belong to the logic packages that read them (NSFR, NEUMANN, BlendRL). Generalize as authored resources: a package declares a workspace resource directory it reads (for example `rules` → `src/usr/rules`, so the user's files stay where they are), and `{"x-theta-resource": "<name>"}` makes a parameter a picker of entries there. The ruleset lookup the packages share moves into a package export.
- **Slurm script environment**: `src/app/pipeline/runtime.py` builds Slurm shell blocks (`SLURM_SUBMIT_DIR`, module loads). Move it into the Slurm component.
- **TensorBoard results root**: `src/app/services/catalog.py` and `frontend/results/data.py` special-case `tensorboard` directories and event files. Run loggers declare the results category and file patterns they write; the catalog reads those declarations.
- **Optuna sweeper adapter**: `hydra_plugins/theta_optuna_sweeper.py` names Optuna. Replace it with one generic Theta sweeper adapter that forwards to the selected tuner's `hydra_sweeper` export, and update the exception register.
- **Backend wording in core text**: for example, the SB3 note in `frontend/results/metrics_catalog.py`; metric descriptions specific to a backend come from that package's monitor catalog.
- Extend `tests/test_sdk_boundary.py` with the new removals (no `symbolic_state`, `ruleset`, `tensorboard` special cases or tuner names in core) as far as static checks reach. Behavior tests use differently named fixture providers: an algorithm blocking on a fixture model contract, a warn rule with `when`, a component-reference parameter of a kind other than model, a fixture logic package declaring its own resource directory, and a fixture run logger declaring its own results category.

Out of scope: checkpoint ranking by `monitor_metric`/`checkpoint.monitor` (user), software-version `compatibility`, workflows (later; rules are evaluated per method, so workflow steps will carry their components' rules), and the Lightning execution and four base paradigms already in the exception register.

## Units

1. **Requirements core**: schema (`src/app/core/manifest_schema.py`), `requirements.py`, validation, execution, bundle check, run record, SDK 3.1; delete every reader of the removed fields, `allowed_agents` and paradigm `constraints`.
2. **Component references**: generic `x-theta-domain` for any kind with `x-theta-contract`; remove the symbolic state slot, `role` and `create_state_loader`; `load_selected`.
3. **Forms**: filtering, warn markers and messages through the shared controls; profile editors stop offering `compatible_paradigms` and `role`.
4. **Audit**: rulesets as package resources, Slurm shell blocks into Slurm, run-logger result categories, the generic sweeper adapter, backend wording.
5. **Packages**: convert every maintained package in `../theta-plugins-v3`, `optional_components/`, `src/plugins/components/bundled/`, the tracked authored packages under `src/usr` (including `blendrl-dev`: its blender model and symbolic state builder as component-reference parameters; NSFR/NEUMANN declaring the `rules` resource) and the test fixtures. Each changed v3 package gets a new version and a release note; the v3 builder validates through the IDE schema, so removed fields fail the build. The scaffolder writes `requires` from its paradigm option.
6. **Docs**: new `docs/reference/COMPATIBILITY_REQUIREMENTS.md`; update `PLUGIN_STORAGE.md`, `EXECUTION_CONTRACT.md`, `DATA_LOADER_CONTRACT.md`, `SDK.md` (3.1), `LAYER_BOUNDARIES.md` (exception register) and the docs index. Report each of the user's files the change makes out of date in one line (experiments under `src/usr/config` that set `symbolic_state_loader`, profiles with `compatible_paradigms`).
7. **Verification**: full per-file run against `docs/reports/LAYER_BOUNDARY_PHASE7_VERIFICATION_20261116.json` (no new failure IDs), v3 `tests/test_native_integration.py`, the installed-distribution probe, then reinstall the converted packages in the repository workspace.
8. **Publication** (authorized by the user on 2026-11-16 once this is done and verified): see the handoff's checklist and [the Hub release workflow](../development/THETA_HUB_RELEASE_WORKFLOW.md).
