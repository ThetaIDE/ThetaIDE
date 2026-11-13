# Experiment monitors

Runners produce training metrics; models, evaluators and tuners can produce
additional diagnostics. `src.app.telemetry.Run` transports them to sinks. The
desktop monitor reads `events.jsonl` (or falls back to CSV for older runners).

The experiment panel's **Monitors** box lists outputs declared by its selected
components, intersected with the hooks supported by each selected runner. It
does not need a previous run or import model/runner implementation code. Enabled methods are
resolved through the pipeline's shared method inheritance and runner dispatch.
For multiple methods the panel shows their union; a model output supported by
one runner is not a promise that every runner collects it.
Shared CPU, memory and GPU transport samples are not synthesized as component
monitor entries in this box. They retain their existing runtime delivery.
Installed standalone monitor providers do not appear until explicitly added to
the experiment using the `+` immediately beside the **Monitors** title. This
creates a named `monitors` instance. Expand its stream row to edit its settings
and individual outputs in the same **Monitors** box. Adding the same provider
again creates a separate instance. Settings stay editable when an instance is
disabled or incompatible, so its configuration can be corrected in place.

## Experiment configuration

```yaml
telemetry:
  monitors:
    enabled: true
    disabled: [system/, blend/train/agreement/, eval/reward_std]
```

This exclusion list removes those outputs from the experiment's active live
monitors. Exact names match one metric; a trailing `/` matches a namespace and
all its descendants. Missing settings or an empty exclusion list stream
everything, including future metrics. `['*']` removes all metric and system
streams. Lifecycle, status, errors, configuration and artifact records remain
available so a run can still be tracked and diagnosed.

The box is absent by default; add it from **Add Overrides → Monitors**. Missing
or null monitor settings stream all metrics. Its header toggle sits to the left
of the title. Setting `enabled: false` bypasses the monitor selections and
streams everything, while preserving the saved `disabled` list for re-enabling.
The body's controls retain their states and are unavailable while bypassed.

Each group is an ordinary parameter row with a toggle and indented metrics.
Its toggle is on when any nested metric is on. Toggling the group updates all
its metrics together and preserves other groups' selections.
The arrow to the left of the group label expands or collapses its metrics;
collapse state survives parameter edits without changing the experiment.
The hover `×` to the right removes that set from the box and live stream.
For a standalone monitor, `×` deletes its configured instance (or disables an
inherited instance). Re-adding it reuses the freed identity. Older saved hidden
instances are restored under their original identity when added again.
Invalid standalone providers report their own errors without hiding outputs
provided by the selected model, runner, or other valid monitors.
Removed output names are persisted in `telemetry.monitors.removed`, using the
same exact-name/prefix rules as `disabled`. Restore a removed set from the
title's `+` menu; its previous individual toggle choices are preserved.

The shared `ToggleSlider` controls add/remove exclusions. A parent namespace owns its subtree;
restore that parent before editing its children. A metric row's hover `×` removes that explicit
exclusion; the box's top-right trash action removes the box, removes its standalone
instances, and restores component streams. Deleting inherited selections writes `monitors: null` to mask the
group's settings. Group defaults and experiment overrides remain distinct.
An explicit `disabled: []` overrides inherited exclusions. Unavailable saved
exclusions remain reviewable and removable after changing components.

`telemetry.metrics` (**Recorded Metrics**) retains its existing meaning: it
selects metrics for all recording destinations and can skip work needed only
for those recordings. Normal `run.log()` values removed from the live monitor
remain in archival sinks and `run.summary`, so checkpoints, objectives, pruning
and later analysis retain their inputs. Live finish summaries are filtered too.
CSV fallback readers honor the same monitor exclusions.
Outputs excluded by Recorded Metrics appear unchecked and unavailable in the
Monitors box; stream choices cannot re-enable an output excluded from telemetry.

Built-in stream sinks declare `live = True`; custom live sinks must do the
same. Archival sinks receive the original records. CSV declares
`records_system = False`; custom sinks default to consuming resource records.
When no destination consumes system samples, the sampler is never started.
The IDE also stops its legacy external resource probe once a telemetry run
starts, including runs that deliberately omit system samples.

## Component and runner declarations

Add `monitoring` to a native manifest **capability**, alongside `id`, `kind` and
`entry_point`. The package validator checks this contract during discovery,
installation and export. Research-specific names and collection logic belong
to components, never to engine/UI branches keyed by component names.

Runner capability:

```json
"monitoring": {
  "supports": ["scalars.v1", "system.v1", "model_observations.v1"],
  "provides": [{"name": "eval/reward", "label": "Evaluation reward"}]
}
```

Model, evaluator, tuner or nested model capability:

```json
"monitoring": {
  "provides": [{
    "name": "routing/train/entropy",
    "label": "Routing entropy",
    "description": "Entropy of the model's routing weights.",
    "requires": ["model_observations.v1"]
  }]
}
```

Each `provides` entry has a stable metric name or namespace prefix, optional
label/description, and nonempty `requires` hook list (default `scalars.v1`).
The live monitor shows `description` when hovering over a chart title. Include
what is measured, units, sampling scope, and how to interpret the value in each
component's declarations or catalog. Exact-name descriptions take precedence
over namespace-prefix descriptions; the longest matching prefix wins. Older
metrics use the shared display catalog when available, otherwise the tooltip
states that the component has not supplied a description. Discovery is cached
until the displayed run configurations change.

Live chart sections follow the complete metric namespace: for example,
`blend/train/logic_0/weight_mean` appears under **blend → train → logic_0**.
Each section can be collapsed independently. Folding a parent preserves child
collapse choices, and new metrics respect those choices. Counts include all
descendant plots. Collapsing only changes display; collection and recording
continue. Components should keep stable slash-separated metric names so their
hierarchy stays consistent across runs.

Every required hook must be supported by the actual runner. A custom script
override takes precedence over the selected algorithm's runner. Nested model
selectors are discovered through `parameter_schema` fields whose
`x-theta-domain` is `model`. Tuner outputs are included only when tuning is
active. A tuner that only consumes objective values should not advertise an
output that it never publishes.

For outputs whose exact names depend on configuration (such as expert indices),
declare a lightweight catalog in the capability instead of namespace stand-ins:

```json
"monitoring": {
  "catalog": "monitor_catalog.py:describe"
}
```

The standalone module must use only the standard library. Its
`describe(component_config, run_config)` function returns the same `provides`
list, with exact metric names and optional `group` labels. Include the catalog
file in `payload.implementations`. The IDE validates its imports and executes
only this file, bypassing the component's `__init__`, model, runner and ML
dependencies. Catalogs must describe outputs without allocating models, probing
training processes or mutating inputs. They must enumerate individual scalars,
including every configuration-dependent expert and pair. The BlendRL copy
shares its expert-name function with its runtime summaries so the two agree.

Every ordinary configuration box retains the shared `ConfigBox` card identity.
Section identity lives in `configSection`; shared theme selectors also use the
`configBox` property so a custom object name cannot remove the background.

Currently defined hooks:

| Hook | Runner responsibility |
| --- | --- |
| `scalars.v1` | Deliver namespaced scalar metrics through the shared Run API. |
| `system.v1` | Deliver shared resource samples (`system/` in selection; bare keys in records). |
| `lightning_callbacks.v1` | Execute component-provided Lightning callbacks. |
| `model_observations.v1` | Call `ModelMonitoringProtocol.collect_monitoring(obs)` at train/eval observation boundaries and insert the phase after the first namespace segment. |

For example, a model returning `blend/entropy` produces
`blend/train/entropy` or `blend/eval/entropy`. Declarations use the final emitted
names. Dynamic expert names can use a prefix. Advertising a hook is a promise
that its runner implementation exists; metadata does not install callbacks or
execute collectors by itself. Undeclared older components keep streaming by
default, but do not gain speculative picker entries.

Future mechanistic-interpretability components can declare versioned hooks such
as `activations.v1`. Their outputs stay absent until a runner both implements
and advertises those hooks. Hook definitions must specify observation timing,
ownership, tensor lifetimes and sample-artifact schemas before runners claim
support; scalar transport alone does not imply access to activations.

## Optional collection work

Expensive observation diagnostics should be gated before computation:

```python
run.monitor("interp/activation_norm", lambda: {
    "interp/activation_norm": compute_activation_norm()
}, step=step)
```

`run.monitor` checks the monitor and recorded-metric selections before invoking
the collector. The collector must return only its declared name/prefix. This
API is for optional observations: removing one skips collection and its saved
samples/metrics as well. Continue using normal `run.log` for losses, objectives
and other required training inputs. A batched collector can check
`run.wants_monitor("namespace/train/")` before collecting its namespace;
removing one scalar filters delivery while remaining scalars still require
collection. Removing every declared scalar skips the batch collector. Do not
use that check to skip evaluation required by an objective.

The editable SB3 development copy gates model observation diagnostics this way
and checks recorded metrics before invoking them. Model sampling remains
controlled by `monitoring_interval` and `monitoring_max_samples`.

## Editable components in this workspace

Installed sources are unchanged. Deep copies live at:

- `src/usr/algorithms/sb3-dev` — agent profiles `sb3-dev/ppo`, `sb3-dev/dqn`,
  `sb3-dev/a2c`, `sb3-dev/sac`; runner hook/output declarations and optional
  model-collection gating.
- `src/usr/models/blendrl-dev` — model profile `blendrl-dev`; routing/phase/
  agreement monitor declarations, independent model and plotter identities.

Select those profiles in an experiment to exercise the new declarations.
Their manifests and config payloads are independent of Hub-managed storage,
and can be exported/repackaged later. Existing experiments keep their selected
components; no provider override redirects them to development copies. The
`Belndrl Bench/Cartpole/singleSpecced` recipe explicitly selects the two dev
profiles; all its other parameters are preserved.

Run targeted checks with:

```sh
QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest tests/test_monitor_selection.py tests/test_metric_selection.py tests/test_parameter_forms.py tests/test_monitor_streams.py
```

## Monitor components

Standalone and model-owned monitor providers now share a lifecycle. See
[Mechanistic monitoring](../guides/MECHANISTIC_MONITORS.md) for component configuration,
fixed probes, post-update timing, adapter capabilities, artifact schemas and
examples. The `telemetry.monitors` stream-selection contract above is unchanged.

## Rich output declarations (v1)

A `monitoring.provides` entry, including one returned by a lightweight catalog,
may declare a rich output alongside existing scalars:

```json
{
  "name": "interp/features/heatmap",
  "label": "Activation heatmaps",
  "kind": "matrix",
  "visual_schema": "theta.visual-output/v1",
  "artifact_schema": "theta.activation-matrix/v1",
  "axes": ["probe_sample", "feature"],
  "value_semantics": "Module output activation, model units",
  "requires": ["probe_forward.v1", "post_update.v1"],
  "retention": "snapshots"
}
```

Kinds are `scalar_series` (the default for every legacy declaration), `matrix`,
`distribution`, `network_graph`, and `intervention_graph`. Rich outputs require
an exact stable name, label, artifact schema, named axes, value semantics and the
version shown above. Discovery attaches the owning instance and component source;
capability intersection and catalog isolation apply exactly as for scalars.
Model-owned defaults and standalone instances use the same declarations and runtime.

Providers return `Result.visuals`, keyed by the output suffix, with array keys or
a graph payload. Matrices reference layer arrays and fixed sample/feature IDs;
distributions reference an array; network graphs reference parameter and activation
arrays; intervention graphs identify measured tests and effects. The runtime
validates declared schemas/references and saves these payloads in snapshot artifacts.
`monitor_snapshot` artifact telemetry carries declarations plus a file reference
and counters, without tensors or full graphs. The JSON file is the completion
marker: arrays are committed first and JSON is replaced atomically afterward.
The desktop loads metadata and needed arrays only for the selected visual/step.

Output names participate in both Recorded Metrics and Monitors selection. A rich
output that is excluded from either is omitted from publication/recording. A
provider with no selected outputs allocates no probes or provider; per-output
`Context.wants_output(suffix)` gates optional artifact work. Shared scalar/visual
calculations still run when another selected output needs them. Switching desktop
views, collapsing layers, playback, and following history are display operations;
they do not change collection. Change the experiment selection or disable the
instance to change collection cost. `retention: snapshots` retains every completed
selected output under the instance's bounded `max_snapshots` policy, including the
reserved final snapshot. Hiding later does not delete previously saved artifacts.
No unbounded tensor history is transported through scalar records.
