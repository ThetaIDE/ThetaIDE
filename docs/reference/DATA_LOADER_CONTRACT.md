# Data loader ownership and selection

An environment describes the task and data source. Experiments bind input providers to methods. A package may supply a model and a loader as separate capabilities; ownership of a loader does not make it responsible for constructing the model.

There are two input slots in the **Data Loaders** box:

- `data_module`: training batches, splits, and sampling through the existing Lightning data-module interface.
- `symbolic_state_loader`: a callable converter used whenever a model needs symbolic inputs, including online actions, evaluation, monitoring, next-state values, and inference after checkpoint reload.

The symbolic converter accepts a batched observation tensor and returns a tensor with shape `[batch, objects, features]`. It preserves batch membership and device and must preserve gradients when the conversion is differentiable. It does not construct experts, reasoners, gates, critics, policies, or trainers. BlendRL retains those responsibilities, action mapping, expert blending, optimizer groups, and routing diagnostics.

## Configuration

```yaml
data_module:
  name: replay_buffer
  params:
    num_workers: 4

symbolic_state_loader:
  name: blendrl/cartpole_state

methods:
  blended_policy:
    agent: sb3/ppo
    model: blendrl
  another_policy:
    agent: sb3/ppo
    model: basics/dnn
    symbolic_state_loader: null
```

Selections can name a registered provider or an installed `data/` profile. The BlendRL package supplies `blendrl_symbolic_state`, a general object-state converter, plus `blendrl/symbolic_state` and `blendrl/cartpole_state` profiles. The editable dev package uses the `blendrl-dev` prefix. The CartPole profile declares two objects and four features; it requires no environment builder hook. Optional custom conversion functions use the `builder` parameter with a `module:function` or `package://package/path.py:function` reference.

Each method inherits the experiment selection unless it overrides that slot. Parameters merge for the same selector. A different selector replaces the previous selection and its parameters, including when selecting another profile of the same implementation. A method can set `symbolic_state_loader: null` to disable an inherited converter. Training modules retain the paradigm fallback. Environments do not select loaders. CLI provider selections replace inherited provider settings; apply parameter edits after the provider selection.

Dataset identity, split membership, and run seeds remain shared experiment settings; selecting a symbolic representation does not resplit or replace the dataset. Each run creates its own input-provider instance.

## Provider declarations

A state converter uses a `datamodule` capability with an explicit role:

```json
{
  "id": "my_state_converter",
  "kind": "datamodule",
  "role": "symbolic_state",
  "entry_point": "state.py:StateConverter",
  "output_contracts": ["symbolic_state/v1"],
  "paradigms": ["online_rl", "offline_rl"],
  "parameter_schema": {"type": "object", "properties": {}}
}
```

Training providers use `role: training`, which is the default for existing data modules. Their classes still implement the data-module lifecycle. State converters are instantiated with their configured parameters and called directly; they do not replace the training data module.

A model or algorithm capability declares `input_requirements`, for example:

```json
{
  "slot": "symbolic_state_loader",
  "contract": "symbolic_state/v1",
  "when": {"actor_mode": ["hybrid", "logic"]}
}
```

`when` keys are dotted paths in the effective component configuration, with accepted values in arrays. All conditions in one requirement must match. Every matching requirement applies. Omitting `when` makes a requirement unconditional. BlendRL declares requirements for hybrid/logic actors and symbolic gates. A fully neural actor with a neural gate needs no symbolic converter.

The authoring panel and launch preflight use the same metadata check. Missing providers, wrong roles, missing output contracts, incompatible paradigms, and invalid declared parameters produce actionable errors. The panel shows effective providers and their source for each enabled method. It surfaces required slots without silently writing a selection or marking the recipe dirty. Runtime validates the converted tensor shape and device.

## Migrating existing recipes

Preview a migration before applying it:

```sh
venv/bin/python scripts/migrate_data_loaders.py path/to/experiment.yaml \
  --state-provider blendrl_symbolic_state
```

Add `--write` to apply the displayed changes. Use `blendrl-dev_symbolic_state` for the editable dev package. The converter provider must be installed before running the migrated recipe.

Migration moves `env.data_module` into the experiment default and moves `env.symbolic_observation_builder` or `model.symbolic.observation_builder` into an explicit state-loader binding, including method-local hooks. Explicit new bindings take precedence. Existing builder functions and model construction settings are retained. Custom builders may specify `object_count: null` to preserve their own object count. Migration does not modify installed Hub packages or old checkpoint files. YAML is rewritten with its global Hydra package directive preserved; review the diff for comments and formatting.
