# Compatibility requirements: one declaration, checked everywhere

Status: planned 2026-11-16, user-requested. Implement before the v3 Hub publication, then publish. Part of the [layer boundary plan](LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md); its confirmed decisions (no legacy support, one home per setting, no fallback chains) apply.

## Goal

A package states what its capability needs from the rest of an experiment in one place, its capability's `requires`, and core enforces it the same way in the form, in validation and before execution. A plugin author can require a paradigm, a particular model or algorithm, a model or data contract, or specific configuration values (for example, an agent that only works with its own PPO variant's settings), without core knowing anything about that package.

## Today: six overlapping mechanisms

| Mechanism | Declared in | Read by | Problem |
| --- | --- | --- | --- |
| Paradigm compatibility | profile `compatible_paradigms`, capability `paradigms`, package-level `paradigms` | form filter (`frontend/config_viewer.py`, `frontend/config_model.py`, `frontend/config_panel/domain.py`), validation (`src/app/pipeline/validation.py` `_get_agent_supported_paradigms`), `src/app/core/provider_metadata.py`, `src/app/core/data_loaders.py` | Three homes; form prefers the profile, validation the capability, then the package |
| Paradigm `allowed_agents` | paradigm profiles (`src/app/config_defaults/paradigms/*.yaml`) | validation, only when the agent declares no paradigms, with fuzzy name matching | Reverse-direction fallback; core config naming algorithms |
| Data slot `role` | profile `role`, then capability `role` | `frontend/config_viewer.py`, `frontend/config_panel/domain.py`, `src/app/core/data_loaders.py` | Two homes |
| Input contracts | capability (else package) `input_requirements` (slot, contract, `when`) and `output_contracts` | `src/app/core/data_loaders.py`, `frontend/config_panel/sections/components.py` | Capability-then-package fallback; data slots only |
| Execution input types | `execution_features.input_types` | `src/app/core/execution_features.py` | Separate vocabulary for the same idea |
| Code check | capability `execution_preflight` hook | `src/app/core/execution.py`, at run time only | Errors appear only after launch, not in the form or dry run |

`compatibility` (package-level software versions: theta, torch, python) is a different concern and stays as it is.

## Design

### Declarations (capability-level only)

```json
{
  "id": "my_ppo", "kind": "algorithm", "entry_point": "agent.py:Agent",
  "provides": ["policy/discrete-v1"],
  "requires": [
    {"config": "paradigm", "in": ["online_rl"]},
    {"slot": "model", "contract": "policy/discrete-v1"},
    {"slot": "model", "capability": ["dnn", "my_policy_net"]},
    {"config": "agent.n_steps", "min": 8, "when": {"agent.mode": ["on_policy"]}},
    {"slot": "data_module", "contract": "trajectories/v1", "when": {"agent.offline": [true]}}
  ],
  "preflight": "agent.py:check"
}
```

- **`requires`**: a list of rules, all of which must hold for each method that selects this capability.
  - `config`: a dotted path into the composed configuration. A path that starts with a method slot (`agent.`, `model.`, `env.`, `data_module.` …) is read from the method that selects the capability; other paths (`paradigm`, `tuning.component`) are read from the experiment. The comparator is exactly one of `in` (list), `equals`, `min`, `max`, `present` (boolean).
  - `slot`: another selection in the same method. Exactly one of `capability` (allowed capability IDs, aliases resolved through the catalog) or `contract` (a contract ID the selected capability must list in `provides`).
  - `when`: optional conditions on the method's or experiment's configuration (same path rules), all of which must match for the rule to apply.
  - Each rule may carry `message`, shown verbatim in the form and in the validation error.
- **`provides`**: contract IDs this capability satisfies. Replaces `output_contracts`. Contract IDs are opaque strings agreed between packages; core never interprets them.
- **`preflight`**: optional `file.py:callable` for checks that need code. It receives the composed configuration, the method name and the workspace, and returns a list of error strings. Core runs it in validation (form preflight, `dry_run`, launch, bundle `check`), not only at execution. Replaces `execution_preflight`.
- **`role`** stays a capability-only field: it says which data slot a loader fills (`training` or `symbolic_state`). The profile `role` key goes.

### Removed

Profile `compatible_paradigms`; capability and package-level `paradigms`; package-level `input_requirements`; capability `input_requirements` and `output_contracts`; `execution_features.input_types` (express it as `requires` on the data slot); paradigm `allowed_agents`; profile `role`; `execution_preflight`. The schema rejects them as unknown keys; nothing reads them. Paradigm profiles keep only paradigm settings (whether an agent harness is required, defaults).

### Evaluation (one module)

A new `src/app/core/requirements.py` evaluates rules against a composed configuration and returns structured failures (capability, rule, method, message). Every consumer calls it:

- **Validation** (`src/app/pipeline/validation.py`) reports every failure as a `ConfigurationError`, replacing `_get_agent_supported_paradigms`, the `allowed_agents` fallback and the loader contract checks in `src/app/core/data_loaders.py`.
- **Forms** (`frontend/config_viewer.py`, `frontend/config_panel/domain.py`, `frontend/config_model.py`) filter a slot's choices by evaluating candidate capabilities' rules against the current draft, and show the rule's message for a selection that fails. Follow [the shared parameter form conventions](../reference/CONFIG_PARAMETER_FORMS.md); extend shared controls rather than adding a per-box pattern.
- **Execution** and **bundle `check`** call the same validation, so a remote bundle fails before upload.
- **SDK:** export the evaluator's result types and a `requirements_errors(cfg, *, paths)` helper under `src.app.api.config` (SDK 3.1, additive) so package preflight code and tools can reuse it.

Rules are data: no rule names a package, backend or institution in core. Behavior tests use differently named fixture providers (an algorithm requiring a fixture model contract, a model with a `when` rule, a loader filling a slot by `role`), in the style of the phase 7 `lattice` fixtures.

### Scaffolder and docs

The scaffolder writes `requires: [{"config": "paradigm", "in": [...]}]` from its paradigm option and no profile key. Document the contract in a new `docs/reference/COMPATIBILITY_REQUIREMENTS.md`, link it from `PLUGIN_STORAGE.md`, `EXECUTION_CONTRACT.md`, `DATA_LOADER_CONTRACT.md` and the docs index; remove the old fields from those documents. Bump the SDK reference to 3.1.

## Units

1. **Core:** schema (`src/app/core/manifest_schema.py`), `requirements.py`, validation, data loaders, execution and bundle check; delete every reader of the removed fields. Tests with differently named fixtures.
2. **Forms:** filtering and messages through the shared controls; profile editors stop offering `compatible_paradigms` (`frontend/widgets.py` chip editor use) and `role`.
3. **Packages:** convert every maintained package in `../theta-plugins-v3`, `optional_components/`, `src/plugins/components/bundled/`, the tracked authored packages under `src/usr` and the test fixtures. Each changed v3 package gets a new version and a release note. The v3 builder (`scripts/package_plugins.py`) validates through the IDE schema, so old fields fail the build.
4. **Verification:** full per-file run against `docs/reports/LAYER_BOUNDARY_PHASE7_VERIFICATION_20261116.json` (no new failure IDs), v3 `tests/test_native_integration.py`, the installed-distribution probe, then reinstall the converted packages in the repository workspace.
5. **Publication** (authorized by the user on 2026-11-16 once this is done): follow [the Hub release workflow](../development/THETA_HUB_RELEASE_WORKFLOW.md); see the handoff for the release checklist.

Out of scope: checkpoint ranking by `monitor_metric`/`checkpoint.monitor` (user), software-version `compatibility`.
