# Compatibility requirements

A component declares, on each capability, what it needs from the rest of an experiment. Core evaluates those declarations in one place (`src/app/core/requirements.py`). Validation, the configuration forms, the run record and the SDK all read their findings from it. Declaring nothing means the capability is compatible with everything.

Software-version `compatibility` (`theta`, `torch`, `python`) is a different concern and is described in [package storage](PLUGIN_STORAGE.md).

## Declarations

Declarations live on a capability in `.theta_component.json`; there is no package-level or profile-level form.

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

### `requires`

Each rule applies to every method that selects the capability. Every rule names its `level`; there is no default.

- `block`: the configuration is invalid. Validation raises a `ConfigurationError`, and pickers omit the candidate.
- `warn`: the configuration runs off-label. Validation prints the message, pickers mark the candidate, and the run records it.

A rule names exactly one subject:

- `config`: a dotted path. A path that starts with a selection key (`agent.`, `model.`, `env.`, `data_module.`, `evaluator.`) reads that method's selection, including nested parameters (`model.blender_model.name`). Any other path (`paradigm`, `tuning.component`) reads the experiment with the method's settings over it. The rule has exactly one comparator: `in` (a list), `equals`, `min`, `max` or `present` (a boolean). `min` and `max` apply only when the value is set.
- `kind`: the method's selection of that component kind. The rule has exactly one of `capability` (allowed capability IDs; aliases and selectors resolve through the catalog) or `contract` (a contract the selected capability lists in `provides`).

Optional fields:

- `when`: conditions as `{path: [accepted values]}`, using the same path rules. The rule applies only when every condition matches.
- `message`: the text shown in the form and in validation output. Without it, core describes the rule (`my_ppo requires paradigm in ['online_rl']; got 'offline_rl'.`).

A rule's own `key.` paths describe the owning capability's settings. A picker does not use those rules to hide a candidate, because selecting the candidate replaces those settings.

### `provides`

Contract IDs the capability satisfies. They are opaque strings agreed between packages (`themes.models/v1`, `symbolic_state/v1`); core only compares them.

### `preflight`

An optional package-relative `file.py:callable` for checks that need code. It is called as `hook(cfg, method_name, *, paths)` with the composed configuration and returns `{"block": [...], "warn": [...]}` message lists. It runs during validation for the selected capability only, so it must be read-only, idempotent and lightweight. Include the hook file in `payload.implementations`. Forms do not run preflight hooks.

## Component references

In a capability's `parameter_schema`, `{"x-theta-domain": "<kind>"}` makes a parameter a picker of installed components of that kind. Any kind works. `"x-theta-contract": "<contract>"` narrows it to capabilities that provide that contract, and a selection that does not provide it is a block finding. The referenced component is part of the method's selections, so its own `requires` apply too.

Package code loads the selected component with `src.app.api.packages.load_selected(kind, selection)`. The selected profile supplies defaults, the selection's values override them, and the call returns what the capability's entry point returns.

## Where findings appear

| Consumer | Behavior |
| --- | --- |
| Validation (`src/app/pipeline/validation.py`) | Selections that resolve to no installed component are reported first. Block findings fail launch, dry run, the desktop preflight and bundle `check`; warn findings are printed and returned. |
| Configuration forms | Agent, model, environment, data module and evaluator pickers omit candidates with block findings against the current draft and mark candidates with warn findings (the shared `mark_candidates` control). A candidate's rules about a component kind the method has not selected yet do not hide it; that kind's picker then offers only compatible candidates. A current selection shows its findings inline (`compatibility_status`). See [parameter form conventions](CONFIG_PARAMETER_FORMS.md). |
| Run record | A run's manifest stores the warn findings it ran with as `compatibility_warnings`. |
| SDK | `src.app.api.config.compatibility(cfg, *, paths)` returns every finding of every enabled method. |

## Workspace resources

A package that reads authored files declares them as a workspace resource:

```json
"workspace_resources": {"rules": {"path": "resources/rules", "depth": 2}}
```

The entries of resource `rules` are the directories `depth` levels below `src/usr/rules` and below the package's own `path` (optional), named by their relative path (`cartpole/anticipatory_v1`). Two packages declaring the same name must agree on `depth`. An entry ID found in more than one root with different contents is an error, not a choice; identical copies count once. `{"x-theta-resource": "rules"}` makes a parameter a picker of the entries. An `open-in-editor` action on it opens the selected entry, or the user's directory for that resource. Package code reads entries with `src.app.api.packages.resource_entries(name)` and `resource_entry(name, entry)`.

NSFR declares `rules` and exports `resolve_ruleset`, which NEUMANN and BlendRL use; MIMIC declares the same resource for its own rules.

## Run logger results

A run logger declares the results category and file patterns it writes:

```json
"results": {"category": "tensorboard", "patterns": ["events.out.tfevents*"]}
```

The results catalog adds the declared categories to the core roots (`logs`, `plots`, `checkpoints`, `datasets`) and classifies files by those patterns.
