# Theta-IDE Component Plugin Authoring Guide

This guide describes how to author, register, package, and distribute research components and plugins for **Theta-IDE**. It details the protocols that let model architectures, algorithms, environments, data tools, evaluators, and complete experiments integrate with the IDE, execution engine, and Community Hub without modifying core code.

---

## 1. Overview & Architectural Principles

In Theta-IDE, all components are cleanly decoupled through declarative runtime protocols:
- **Models & Composite Architectures**: Define policy networks, Q-functions, feature extractors, and symbolic reasoners.
- **Agents & Algorithms**: Implement RL or supervised loss functions, update steps, and optimization procedures.
- **Environments & Groundings**: Define simulators, observation wrappers, potential-based reward shaping, and logic valuations.
- **DataModules & Samplers**: Manage trajectory sequence windows, offline datasets, and replay buffer sampling.
- **Evaluators & Verification**: Compute formal verification bounds, safety metrics, and policy attributions.
- **Experiment Recipes**: Bundle complete reproducible benchmarks (configs, envs, models, rules, weights) for 1-click sharing.

```
                    ┌──────────────────────────────────────────────┐
                    │               Theta-IDE Core                 │
                    │   (Pipeline, GUI Viewer, Slurm, Lightning)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                        Discovers via standard protocols & entry points
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
   [Model Plugins]                 [Agent Plugins]                 [Env & Data Plugins]
• build_model (model registry)  • AgentPluginProtocol           • EnvironmentPluginProtocol
• AuxiliaryLossProtocol         • Dynamic loss steps            • DataModulePluginProtocol
• OptimizerGroupsProtocol       • Paradigm validation           • TrajectorySamplerProtocol
```

---

## 2. Component Plugin Archetypes & Protocols Reference

### Tuning components

Tuning components are Python entry points in the `theta.tuners` group. An
experiment selects one with `tuning.component`; omitted means `optuna`. The
component owns its own settings, validation, job compilation, study setup and
result finalization. Its options are stored under
`tuning.options.<component>`, so switching components preserves their settings.
Older Optuna experiments with `tuning.n_trials`, `tuning.n_jobs`, or
`tuning.storage` remain supported.

The Optuna component optionally contributes a Lightning validation callback
for median pruning through `training_callbacks(cfg)`. It reports the selected
objective at validation epochs and records pruned trials separately from
failed trials. SB3 and multi-process Lightning do not yet support this
pruning path. Its time budget is checked between batches; already running
trials are allowed to finish.

```toml
[project.entry-points."theta.tuners"]
my_tuner = "my_package.tuning:MyTuner"
```

The entry point supplies a no-argument factory declaring `api_version = 1` and implementing
`src.app.pipeline.tuners.Tuner`. `settings()` returns field descriptions for the
editor (`key`, `label`, `type`, `default`, and, where applicable, `min`, `max`,
or `choices`). Supported field types are `int`, `float`, `text`, `choice`,
and `bool`. A field may declare `when: {other_key: value}` to show it only
when another setting has that value. `validate_config` and `validate_dimension` reject unsupported
settings or spaces. `compile` receives the base overrides and method context;
`job_entrypoint` may replace the training script or interpreter with a tuner
runner. `study_name`, `storage`, `prepare`, and `finalize` manage the run.
For Slurm, the component must be installed in the submission and execution
environments; `prepare` runs at submission and `finalize` runs in the job.

Tuning jobs must report a finite objective and should export a version 1
`tuning_summary.json` in each method's checkpoint directory. The built-in
`tuning` PlotMan figure consumes that summary without opening the component's
database. The Optuna implementation in `src/app/tuners/optuna/component.py`
and `tests/test_tuner_components.py` are working examples of the contract.
The summary contains `schema_version`, `component`, `study`, `direction`,
`objective_metric`, `best_trial_id` (null until one completes), and `trials`.
Each trial contains `id`, `state`, `value`, `parameters`, and `checkpoint`;
optional `intermediate_values` maps zero-based validation epochs to values.
An optional `stop_reason` records whether the latest run reached its trial
limit, time budget, failure limit, or encountered an interruption or launch error.

For Lightning jobs, an optional `training_callbacks(cfg)` method can supply
callbacks when the job sets `theta_tuning.component` to the component name.
A callback can signal a clean prune with an exception whose
`theta_trial_status` attribute is `"pruned"`; `train.py` returns a corresponding
status object for the tuner to interpret, and telemetry records a pruned run.

### Declaring parameter domains for the GUI

Components declare an optional `parameter_schema` JSON Schema object in their
`.theta_component.json` manifest. Runtime YAML profiles remain the source for
default values; the manifest schema declares the UI domain: `properties` names keys that can be added at each level,
`enum` supplies dropdown choices, and `x-theta-domain: model` supplies installed
model component choices for module-valued parameters.

```json
{
  "parameter_schema": {
    "type": "object",
    "properties": {
      "blend_mode": {"type": "string", "enum": ["gate", "threshold", "fixed"]},
      "neural": {"type": "string", "x-theta-domain": "model"},
      "optimizer": {
        "type": "object",
        "properties": {
          "name": {"type": "string", "enum": ["adam", "adamw", "sgd"]},
          "kwargs": {"type": "object", "properties": {"weight_decay": {"type": "number"}}}
        }
      }
    }
  }
}
```

The GUI uses declared properties in its add-parameter menus at the matching
nested level. Existing YAML defaults remain valid and continue to seed the editor.
For a component with several profiles, list each one in `config_paths` and use
`x-theta-config-schemas` to map each config path to its own JSON Schema. This
keeps profile-specific keys and choices attached to the profile the editor opens.
The editor reads this metadata from the installed component manifest; plugin
code is not imported to build the form.

Parameters can declare `x-theta-actions` to expose a hover/focus pencil. The IDE
owns the presentation and editor routing; the component owns all target paths.
For a static target use `{"type": "open-in-editor", "path": "notes/example.txt"}`.
For dynamic targets declare a lightweight, standalone component handler:

```json
"x-theta-actions": [{
  "type": "open-in-editor",
  "label": "Edit rules",
  "handler": {
    "path": "src/usr/models/my_model/editor_actions.py",
    "function": "rules_edit_targets"
  }
}]
```

The IDE loads this file only when clicked and calls `rules_edit_targets(value=value,
context=context)`. Context contains `workspace_root`, the effective current `config`,
the full `parameter_path`, and `config_path`. The handler returns a path, a
`{"path": ..., "label": ...}` target, or a list of targets. Relative paths are
relative to the workspace. One target opens directly; multiple targets appear in
a menu. Missing files and handler failures show an error without guessing another
target. Keep handlers free of training imports. The IDE has no ruleset, environment,
model-name, or filename logic for these actions. Opening does not create an override.

All plugin contracts are defined in [`src/app/core/protocols.py`](file:///Users/cameronegbert/Research/Theta-IDE/src/app/core/protocols.py).

### Local development in the IDE

On an installed component's Hub card, **Edit** opens its actual source folder and
main implementation in the selected editor. The Theta Text Editor includes a file
tree with owned profiles and assets at their installed paths. Ordinary Save affects
the next fresh run. `COMPONENT_CONTEXT.md` is regenerated on Edit; keep personal
notes elsewhere. CLI edits to profiles and manifest schemas refresh when returning
to the configuration view.

For an independent model, choose **Deep Copy** and type its name in the empty,
focused component row. Press Enter or finish editing to create/select it and populate
its parameter pane. Escape cancels; name conflicts appear inline. The runtime ID is
derived from the entered name. The supported first layout
is a directory-based model with one literal `@register_model(...)` registration.
The operation remaps registration aliases, package imports, profile selectors,
profile schemas and owned asset paths. Unsupported identity locations are reported
before installation. Shared child components stay shared. The copy is installed
immediately and appears in the Hub's **Local** filter without publication.

The default collection is workspace-local `.thetaide/components/packages/<id>`.
Those packages are retained snapshots; edit the installed files. **Hide** filters
ordinary discovery without moving code or breaking saved experiments. Use Hub
**Show hidden** and **Unhide** to restore discovery. **Uninstall** retains saved
source/profiles/assets for reinstall; **Delete local component** shows affected
locations and experiment references, then moves owned files to the OS Trash.
Neither action changes remote Hub contents. Updates refuse to overwrite modified
source, schemas or assets, including edits retained through uninstall/reinstall.

The experiment's model profile path (for example `my_fork/default`) selects YAML;
the profile's `architecture` selects its independent runtime registration
(`my_fork`). They are intentionally separate identities.

### 2.1 Model Architectures & Composite Models

#### Composing models

Composite models own their child model construction. Use
`src.app.core.model_registry.build_model` with child selectors from the component's
configuration. Declare child selector fields with `x-theta-domain: model` in the
manifest parameter schema so the shared form can offer installed model profiles.
`ModelSlot`, `CompositeModelProtocol`, and automatic slot resolution are not runtime APIs.
See `src/app/core/model_registry.py`, `src/app/core/protocols.py`, and the installed
component's constructor/forward methods for the actual contracts. Tensor shapes and
outputs depend on the consuming algorithm; a component must document those explicitly.

#### `AuxiliaryLossProtocol`
Models can declare auxiliary objectives (e.g., entropy regularization over blending weights, sparsity constraints, or representation consistency):

```python
class AuxiliaryLossProtocol(Protocol):
    def compute_auxiliary_losses(self, batch: Any, output: Any) -> dict[str, torch.Tensor]:
        """Return dictionary of named auxiliary losses added to the algorithm loss."""
        ...
```

#### `OptimizerGroupsProtocol`
Models configure custom parameter groups with per-module learning rates (e.g., slower learning rate for symbolic logic rules, faster for neural heads):

```python
class OptimizerGroupsProtocol(Protocol):
    def configure_optimizer_groups(self, base_lr: float, weight_decay: float, cfg: Any = None) -> list[dict[str, Any]]:
        """Return parameter group dictionaries for the PyTorch optimizer."""
        ...
```

#### `EnvironmentWrapperProtocol`
Models declare required observation wrappers or predicate extractors needed by the simulator:

```python
class EnvironmentWrapperProtocol(Protocol):
    @classmethod
    def get_env_wrappers(cls, env: Any, cfg: dict[str, Any]) -> list[Any]:
        """Return Gymnasium wrappers to wrap the environment upon initialization."""
        ...
```

---

### 2.2 Agent & RL Algorithm Plugins

Agents implement RL algorithms (e.g., SAC, TD3, AWAC, Offline Multi-Agent) as PyTorch Lightning Modules:

```python
from src.usr.algorithms.agent_registry import register_algorithm, register_agent
from src.app.core.protocols import AgentPluginProtocol

@register_algorithm("my_algorithm", "my_algo")
class MyCustomAgent(L.LightningModule):
    """Custom reinforcement learning agent."""

    def training_step(self, batch, batch_idx):
        loss_dict = self.compute_algorithm_loss(batch)
        # Auxiliary losses from models are automatically collected and added
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        # Queries model.configure_optimizer_groups() if implemented
        ...
```

---

### 2.3 Environment & Grounding Plugins

Environment plugins package Gymnasium simulators, vectorization hooks, domain logic rules, and potential-based reward shaping:

```python
from src.app.core.protocols import EnvironmentPluginProtocol

class RoboticsSimulatorPlugin:
    @classmethod
    def make_env(cls, env_id: str, **kwargs):
        import gymnasium as gym
        return gym.make(f"Robotics/{env_id}", **kwargs)

    @classmethod
    def get_default_wrappers(cls):
        return [NormalizeObservationWrapper, StateExtractionWrapper]

    @classmethod
    def get_reward_shaping_hook(cls):
        return potential_based_shaping_function

    @classmethod
    def get_logic_valuation_hook(cls):
        return predicate_valuation_function
```

---

### 2.4 DataModule & Trajectory Sampler Plugins

DataModules manage offline datasets, clinical cohorts, and sequence window sampling:

```python
from src.app.core.paradigm_loader import register_component
from src.app.core.protocols import DataModulePluginProtocol, TrajectorySamplerProtocol

@register_component("DecisionTransformerDataModule", "TrajectoryDataModule")
class TrajectorySequenceDataModule:
    def setup(self, stage=None):
        self.dataset = load_trajectory_npz(...)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=64, shuffle=True)

    def sample_trajectory_window(self, batch_size: int, context_len: int) -> dict[str, torch.Tensor]:
        """Returns sequence batches (s_t, a_t, r_t, s_{t+1}) of length context_len."""
        return self.dataset.sample_slices(batch_size, context_len)
```

---

### 2.5 Evaluator, Safety & Verification Plugins

Evaluator plugins log formal safety bounds, barrier certificates, and explainability attributions without altering the training loop:

```python
from src.app.core.protocols import EvaluationHookProtocol

class SafetyVerificationEvaluator:
    def on_eval_start(self, agent, env, stage="val"):
        self.violations = 0

    def evaluate_step(self, agent, batch_or_transition) -> dict[str, float]:
        obs = batch_or_transition["obs"]
        action = agent.act(obs)
        is_safe = self.check_barrier_certificate(obs, action)
        return {"eval/barrier_safety": float(is_safe)}

    def on_eval_end(self) -> dict[str, float]:
        return {"eval/safety_violation_rate": self.violations / self.total_steps}
```

All metrics returned by evaluation hooks automatically log into `metrics.csv` and render in the Theta-IDE Results Browser.

---

## 3. Universal Python Packaging via `pyproject.toml` Entry Points

Any component can be packaged as a standard Python library and published to PyPI (`pip install my-theta-plugin`). Theta-IDE discovers and registers them automatically at startup via [`src/app/core/plugin_discovery.py`](file:///Users/cameronegbert/Research/Theta-IDE/src/app/core/plugin_discovery.py).

Declare standard entry-point groups in your `pyproject.toml`:

```toml
[project]
name = "my-theta-plugin"
version = "1.0.0"
dependencies = [
    "torch>=2.0.0",
    "gymnasium>=0.29.0",
]

[project.entry-points."theta.models"]
my_composite_model = "my_plugin.models:MyCompositePolicy"

[project.entry-points."theta.agents"]
my_rl_algorithm = "my_plugin.agents:MyCustomAgent"

[project.entry-points."theta.environments"]
my_custom_env = "my_plugin.env:RoboticsSimulatorPlugin"

[project.entry-points."theta.datamodules"]
my_trajectory_dm = "my_plugin.data:TrajectorySequenceDataModule"

[project.entry-points."theta.evaluators"]
safety_verifier = "my_plugin.eval:SafetyVerificationEvaluator"

[project.entry-points."theta.plugins"]
custom_gui_panel = "my_plugin.ui:MyCustomPanelPlugin"
```

Once installed, your components appear immediately in:
- The Config Viewer dropdown selectors in the GUI.
- The pipeline execution engine (`python run_pipeline.py <group>/<exp>`).
- Built-in registries (`MODEL_REGISTRY`, `AGENT_REGISTRY`, `_COMPONENT_REGISTRY`).

An environment entry point may be a class or factory. Theta-IDE calls it with the configured environment name and supported runtime options. It must return an object implementing `reset()` and `step()`; it does not need to inherit from an IDE environment class.

---

## 4. 1-Click Experiment-to-Plugin Exporter

Researchers can package any experiment into a self-contained, shareable, installable Theta Hub component archive (`.zip`) with one click.

### 4.1 Exporting from the GUI

1. Open Theta-IDE and select any experiment recipe in the Config Tree.
2. Click the **Export** button on the Action Bar (or press `Ctrl+Shift+S`):
   - **Theta Hub Component Archive (`.zip`)** (Recommended): Packages the recipe YAML, base templates, referenced environment definitions, valuation rules, custom model architectures, and optional checkpoint weights.
   - **Standalone Recipe YAML (`.yaml`)**: Exports only the composed recipe YAML for manual reproduction.
3. Configure packaging options (include rules, models, weights) and click **Export**.

### 4.2 Exporting from the CLI

Export any experiment directly from the command line:

```bash
# Export experiment with all dependencies
python run_pipeline.py export cartpole/quick_test

# Include trained model checkpoints & specify output archive
python run_pipeline.py export cartpole/quick_test --include-weights --out results/exports/cartpole_release.zip

# Or via Hydra override during execution
python run_pipeline.py cartpole/quick_test export=true
```

### 4.3 What Gets Packaged

The [`ExperimentExporter`](file:///Users/cameronegbert/Research/Theta-IDE/src/app/services/experiment_exporter.py) automatically resolves the full dependency closure:
```
cartpole_quick_test_v1.0.0.zip
├── .theta_component.json                      # Hub component manifest
├── README.md                                  # Auto-generated reproduction guide
├── in/config/experiment/cartpole/
│   ├── quick_test.yaml                        # Experiment recipe
│   └── _base.yaml                             # Group base template
├── in/config/env/cartpole.yaml                # Environment configuration
├── in/envs/cartpole/                          # Custom environment code & hooks
├── in/rules/cartpole/                         # Domain logic rules
├── src/usr/models/blendrl/                    # Referenced model architecture
└── results/checkpoints/cartpole/quick_test/   # Checkpoint weights (optional)
```

### 4.4 1-Click Roundtrip Installation via Theta Hub

Any researcher receiving the `.zip` package can install it into their Theta-IDE in one click:
- Drag and drop or import the `.zip` in **Components &rarr; Hub &rarr; Install Local Package**.
- The [`HubInstaller`](file:///Users/cameronegbert/Research/Theta-IDE/frontend/hub/installer.py) automatically unpacks configs to `in/config/experiment/`, environments to `in/envs/`, and rules to `in/rules/`.
- The experiment is immediately runnable with:
  ```bash
  python run_pipeline.py cartpole/quick_test
  ```

---

## 5. Reference Case Study: BlendRL as a Component Plugin

**BlendRL** is an illustrative example of a composite, hybrid neural-symbolic component plugin. Its implementation is an independently installable plugin and is not part of the base Theta-IDE checkout.

### 5.1 Plugin Manifest (`.theta_component.json`)

```json
{
  "id": "blendrl",
  "name": "BlendRL",
  "kind": "model",
  "version": "1.0.0",
  "description": "BlendRL hybrid neural-symbolic policy architecture",
  "author": {"name": "Theta Research"},
  "target_path": "src/usr/models/blendrl",
  "compatibility": {"theta": ">=2.0.0"}
}
```

The manifest's `target_path` is an example install location chosen by the plugin author; this directory is not present in the base IDE checkout.

### 5.2 Implementation Highlights

```python
from src.app.core.model_registry import register_model

@register_model("blendrl", "blender")
class BlenderActorCritic(nn.Module):
    """Hybrid Actor-Critic combining neural and symbolic policy heads."""

    # Construct child modules in __init__ using build_model(child_config, ...).
    # Child selector schemas belong in the manifest parameter_schema.

    def compute_auxiliary_losses(self, batch, output) -> dict[str, torch.Tensor]:
        losses = {}
        be = getattr(output, "blend_entropy", None)
        if isinstance(be, torch.Tensor):
            losses["blend_entropy"] = be.mean()
        return losses

    def configure_optimizer_groups(self, base_lr: float, weight_decay: float = 0.0, cfg=None):
        groups = []
        logic_lr = float(self.get_cfg("logic_lr", base_lr))
        blender_lr = float(self.get_cfg("blender_lr", base_lr))
        if hasattr(self, "actor") and hasattr(self.actor, "policy_modules"):
            for m, m_type in zip(self.actor.policy_modules, self.module_types):
                lr = base_lr if m_type == "neural" else logic_lr
                groups.append({"params": m.parameters(), "lr": lr, "weight_decay": weight_decay})
        return groups
```

---

## 6. Authoring Checklist

1. **Choose your component archetype**: Model, Agent, Environment, DataModule, Evaluator, or Experiment.
2. **Implement the matching runtime protocol** in [`src/app/core/protocols.py`](file:///Users/cameronegbert/Research/Theta-IDE/src/app/core/protocols.py).
3. **Register your component**:
   - Models: `@register_model("<name>")`
   - Agents: `@register_agent("<name>")`
   - DataModules: `@register_component("<Name>")`
4. **Publishing options**:
   - **Local / Hub package**: Include `.theta_component.json` and export via `python run_pipeline.py export`.
   - **PyPI Python package**: Declare entry points under `[project.entry-points."theta.*"]` in `pyproject.toml`.
5. **Verify with test suite**: Run `pytest tests/test_plugin_discovery.py` to verify protocol conformance.

### SB3 policies and symbolic observations

SB3's DQN adapter calls the selected model's `get_q_values(obs)`; PPO and A2C
call `get_action_probs(obs)` and `get_action_and_value(obs, action=None)`. A
missing method is reported as a compatibility error. The selected model is
saved inside the SB3 policy checkpoint, and its class is recorded in run
metadata. Custom model authors can implement the methods required by the
algorithms they want to support.

BlendRL builds its symbolic input itself. For vector observations it looks for
`model.symbolic.observation_builder`, then
`env.symbolic_observation_builder`: each names a Python callable as
`module:function`. The callable accepts a batched observation tensor and
returns a batched tensor for the reasoner. Without a builder, BlendRL rejects
vector inputs rather than guessing their symbolic structure. An already
object-centric tensor can be supplied directly. An environment also provides
its `symbolic_name` (used to locate its rules and valuations) and
`action_meanings` in action-index order; when available, SB3 can obtain action
names from the Gymnasium environment's `get_action_meanings()` method instead.

For example, the CartPole plugin supplies its own builder, rule valuations,
and the action names `left` and `right`. Other environments can supply their
own domain-specific functions without modifying the SB3 adapter.

### Parameter validation

The experiment YAML contains the configured values; component manifests describe parameters and editor controls. The pipeline, GUI preview, and launch use the same composition and validation path. To opt a component's parameter schema into runtime validation, set `"x-theta-validate": true` on its `parameter_schema` (or on a profile schema in `x-theta-config-schemas`). The schema must describe the resolved runtime values, including explicit `null` defaults and nested component objects where applicable. Existing schemas remain editor metadata until they opt in.

Validation applies JSON Schema draft 2020-12 to the selected method's composed agent or model configuration. An opted-in field's sweep choices and numeric bounds are checked against the same field schema. The pipeline also composes a representative Optuna trial before launching; this detects missing Hydra paths, but does not exhaustively test combinations of sweep dimensions or guarantee that training will succeed. Components without a validation schema continue to work with the shared pipeline checks.

On component update or reinstall, existing user YAML values are retained and missing block-mapping defaults from the new component are added. Explicit nulls are values and are retained. Flow-style mappings and incompatible structures are preserved rather than rewritten; they may need manual reconciliation. Check the config and validation output after upgrading a component.
