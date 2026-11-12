# Mechanistic monitoring

Experiments can select independent monitor components, and models can supply
architecture-owned monitors. Both use `MonitorManager` and the same provider
lifecycle. The existing `collect_monitoring(obs)` model protocol and its
train/evaluation routing diagnostics remain supported.

## Try the working experiment

`in/config/experiment/newMonitors/onlineRL.yaml` runs MountainCar with the existing
DQN method (displayed as `PPO`) and an additional `BlendRL` method using PPO.
The original method and shared training budget are preserved. Each method keeps
its own architecture-compatible monitor instances.

Run it normally:

```sh
venv/bin/python run_pipeline.py newMonitors/onlineRL no_plot=true
```

Or use a short integration run without changing the saved recipe:

```sh
venv/bin/python run_pipeline.py newMonitors/onlineRL total_timesteps=256 no_plot=true \
  '++methods.PPO.agent={name:sb3-dev/dqn,learning_starts:32,train_freq:4,gradient_steps:1,batch_size:16,buffer_size:1000}' \
  ++monitors.activation-evolution.interval_updates=10 ++monitors.causal-circuit.interval_updates=10 \
  ++monitors.network.interval_updates=10 \
  ++experiment_id=monitor_example
```

In the experiment editor, **Add Overrides → Monitor Components** adds the
component configuration. It uses the shared parameter forms, typed component
selectors, nested adders, hover removal and `ToggleSlider` controls. Each instance
has a unique name. **Monitors** continues to control output visibility through
`telemetry.monitors`; **Recorded Metrics** controls recording destinations.

```yaml
monitors:
  features:
    component: activation-evolution
    target: network
    enabled: true
    interval_updates: 100
    max_snapshots: 16
    failure_policy: raise
    probes: {count: 128, steps: 512, seed: 1729}
    params: {max_features: 128}
```

`enabled: false` on an instance disables its execution. This is distinct from
`telemetry.monitors.enabled: false`, which bypasses stream exclusions. Explicit
instances merge with model defaults by name, so `{enabled: false}` can disable
an inherited/model-owned instance. Multiple instances of the same component are
independent. Outputs live under `interp/<instance>/`.

## Components

### Activation evolution

`activation-evolution` records per-layer arrays and statistics: activation norm,
centered effective rank (entropy of normalized singular values), inactive-unit
fraction, saturation for tanh/sigmoid, and centered linear CKA against the first
and previous snapshots. Scalars are means over applicable layers; JSON contains
each layer's values. Degenerate CKA references produce no CKA value.

Targets and optional `params.layers` are PyTorch module paths, relative to the
selected Theta model. For the installed DNN, `target: network` inspects hidden
features while avoiding separate critic branches. Automatic selection chooses
activation modules, falling back to linear modules. Explicit targets must run
once and emit `[samples, features]`; recurrent/spatial/shared-execution targets
need a model adapter. The first configured number of features is saved; this is
recorded in metadata and must not be interpreted as a complete representation.

### Causal intervention map

`causal-circuit` tests neuron or whole-layer candidates (`granularity: unit` or
`layer`). `layers` selects target-relative modules. Other parameters are
`max_units`, `top_k`, `random_controls`, `action`, `reference_action`, and
`baseline: mean|zero`, and `max_edges`. Whole-layer tests reject widths above 512 rather than
silently intervening on only part of a layer.

It divides probes into discovery/held-out halves. Mean replacements and candidate
ranking use only discovery observations. It measures individual and selected-set
ablations, ablation of the unselected **sampled candidates**, and matched random
sets. Clean activations patched into real donor observations measure recovery;
donors stay within their split. Near-zero clean/donor score gaps are excluded
from normalized recovery and the valid count is reported.

Edges in the report are **measured total intervention effects on behavior**,
not direct synaptic edges. Sequential targets additionally test bounded pairs of
selected candidates: source ablation followed by downstream activation patching
and rescue measures path mediation. These paths may include intermediate layers.
Selected sets are candidate circuits. Complement
tests do not cover unexamined units or establish a complete causal circuit.
Reported standard errors describe variation across probe samples; correlated
trajectory states are not independent episode-level confidence intervals.

### BlendRL-owned reliance

The `blendrl-dev` model package includes `blendrl-dev/reliance`, declares an
optional `blend_reliance` default, and implements `get_monitor_specs()` plus
`monitoring_interventions(obs)`. It is disabled until explicitly enabled:

```yaml
monitors:
  blend_reliance:
    enabled: true
```

It requires the hybrid/softmax architecture and policy probabilities, and tests
each expert replaced by uniform action probabilities, each expert removed with
remaining gates renormalized, uniform gates, and mean gates. Artifacts contain
per-intervention action-flip rates and signed/absolute action log-probability
contrast changes. These are fixed-state effects, not rollout reward effects.

## Execution contract

V1 supports the editable SB3 runner's PPO, A2C and DQN with selected Theta
PyTorch models, discrete actions and finite flat observations. The runner
advertises `probe_forward.v1`, `post_update.v1`, `module_interventions.v1`.
Policy-probability and Q-value score capabilities are distinct. Architecture-owned
capabilities such as `blendrl.expert_interventions.v1` must be advertised by the
model and provided by its runtime adapter.
SAC, native SB3 policies without an adapter, Lightning, distributed training,
training-pass activations and training gradients do not acquire these hooks.

Optimizer step post-hooks count real `optimizer.step()` calls. Collection runs
after the enclosing SB3 `train()` call returns, not at rollout end. PPO epochs,
train calls, environment transitions and optimizer steps are different counters.
Schedules use optimizer steps in the current monitoring session. The first
eligible update is sampled; later samples respect `interval_updates`.
One `max_snapshots` slot is reserved for the final trained model (a budget of
one therefore saves only the final state). No updates means no model snapshots.

Probes execute under `no_grad` and eval mode on isolated model copies. Python,
NumPy and Torch RNG states are restored. Non-leaf tensor caches are detached on
the copy; diagnostic hooks are installed only there and always removed. Training
parameters, buffers, modes, hooks and gradients remain owned by training.
Large-model copy cost is an explicit v1 limitation, not a zero-overhead promise.
Models with existing module hooks require an architecture-specific `fork()`;
the default adapter refuses to silently remove hooks that might define behavior.

Absent/disabled instances allocate no providers or probes. Recording/stream
selections gate optional work before collection. Instance, probe, feature,
candidate, control and snapshot limits bound work and artifacts. `failure_policy:
raise` fails visibly; `disable` records the error and disables that instance.
Cleanup is idempotent, including failed setup and interrupted training.

## Probe and artifact formats

Probe banks come from a separate seeded environment, with random discrete
actions and bounded reservoir sampling. Training environments are never used
to generate them. Raw observations are supplied through the same model scoring
API used by the runner; external normalization wrappers require an adapter with
explicit preprocessing metadata. Fixed random-policy banks may have limited
state coverage. Supply an NPZ `probes.path` produced by `ProbeSet.save()` for a
different cohort. Keep loaded banks within the runner/model's observation space.

Legacy model observation diagnostics continue to use current rollout/evaluation
states. They are distinct from the new fixed-probe comparisons: a changing state
distribution is not evidence of a changing representation on fixed inputs.

Each run saves `monitors/<instance>/probes.npz`, immutable numbered snapshot NPZ
and JSON pairs, and a self-contained `index.html` report. Reports provide a
snapshot slider/playback, metric timeline, activation heatmaps or intervention
maps. They reopen without Python or a running experiment. The live monitor’s **Files** view exposes each report. **Diagnostics** renders
rich outputs natively during training and after completion.

- `theta.probes/v1`: finite observations, unique sample IDs, environment, seed,
  sampling and preprocessing provenance; content hash covers data and metadata.
- `theta.monitor-snapshot/v1`: provider version/content hash, instance/config,
  target and axes, exact model-state hash, probe hash, transition/update counters,
  execution boundary, score semantics, metrics and measurement details.
- `theta.monitor-state/v1`: non-pickle NPZ/JSON monitor state, including CKA
  references and scheduling counters. `load_monitor_state()` and
  `MonitorManager.load_state_dict()` require the matching model checkpoint,
  provider, probe bank and configuration. This is an API for coordinated resume;
  the runner does not automatically restore an earlier training job.

Model weights are identified by hash, not embedded in every snapshot. Save a
matching training checkpoint when later model replay or coordinated resume is
required. Diagnostic arrays and reports remain usable without that checkpoint.

## Author a monitor or model adapter

A native manifest capability has `kind: monitor`, an executable entry point,
required hooks and `monitoring.provides` or a stdlib-only output catalog. Catalogs
receive params plus `instance` and `target`, and enumerate final output names.
Package through the normal export/install lifecycle. The scaffold dialog also
supports monitor components. A provider implements `collect(Context) -> Result`;
optional `setup`, `close`, `state_dict`, `load_state_dict` share the same lifecycle.

Models can include a monitor capability in their own package, advertise
`capability.monitor_defaults`, and return matching defaults from
`get_monitor_specs()`. A model can return a `ModelAccess` subclass from
`get_monitoring_adapter(algorithm)` to supply architecture-specific scores and
targets. Its `fork()` must isolate all model access. A provider may require
architecture-specific capabilities in addition to the versioned runner hooks.

This requires the updated IDE and local Hub schema/validator support. Exported
archives do not upgrade an older IDE. Production publication is a separate
release operation following `docs/development/THETA_HUB_RELEASE_WORKFLOW.md`.

## Verification

```sh
QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest \
  tests/test_mechanistic_monitors.py tests/test_parameter_forms.py \
  tests/test_monitor_selection.py tests/test_metric_selection.py \
  tests/test_blendrl_monitoring.py tests/test_monitor_streams.py
```

The conformance cases cover known causal units, layer patching, held-out donors,
CKA, RNG/gradient/cache isolation, cleanup, disabled collection, state roundtrips,
native archive export/install, real experiment composition, and identical final
DQN/PPO weights with diagnostics enabled and disabled.

## Native Diagnostics and network inspection

Open **Diagnostics** in either the training monitor or the **Results** panel.
In Results, select a completed run, method and version in the explorer, then open
its **Diagnostics** tab alongside Files and Curves. The tab uses the same native
viewer, including historical steps, playback, layer/probe selectors, display modes
and checkpoint comparisons. Snapshot metadata and selected arrays load through
the Results backend on demand, so inspection does not require a training process
or the original training machine’s artifact paths. Incomplete snapshot pairs are
omitted; runs without rich snapshots retain their HTML reports in Files/Reports.
Use Results’ **Refresh** to reload saved artifacts or retry a failed load.

Select the run/method, monitor,
output, optimizer step, layer and fixed probe state. **Follow latest** advances as
completed snapshots arrive; selecting a historical step or **Play** disables
following. Playback visits saved snapshots only. Other selections and the viewport
survive refreshes. **Compare** opens the same visual at another saved step.
Comparison requires the same initialization lineage, topology, target and probe
bank; changing weight values does not change unit identity. The two panels identify
their steps explicitly. Scalar curves stay in **Overview**.

`network-structure` supports an explicit feedforward `Sequential` target (the
standard DNN’s `network`), including Linear, supported pointwise activations,
eval-mode Dropout, and flat LayerNorm. LayerNorm is shown as a coupled operation
over the full layer, with saved scale/bias and epsilon, including when units are
omitted. It has no invented independent synapses. Unsupported operations or
nonsequential targets request an architecture-owned `ModelAccess.topology_modules`
adapter. Its ordered modules must execute once on the selected scoring path and
have compatible contiguous tensors. This ordered-path adapter does not claim
arbitrary branching-graph support.

```yaml
monitors:
  network:
    component: network-structure
    target: network
    interval_updates: 100
    max_snapshots: 16
    probes: {count: 128, steps: 512, seed: 1729}
    params: {max_layers: 16, max_units: 64, max_edges: 4096}
```

Snapshots record stable absolute module/unit IDs, topology fingerprints, actual
effective target inputs, preprocessing provenance, activations, biases and weights
in `[destination, source]` orientation. Only the selected target is represented;
the DNN’s separate action/value heads are outside `target: network`. Bounds select
the first layers/units and destination-major unit edges. The view shows recorded
versus total layers/units/edges and the selection rule. LayerNorm’s coupled group
is distinct from these individual edges. An incomplete graph is always partial.

Large layers start collapsed; select a layer to expand it, or turn off **Collapse
other layers**. Use the wheel to zoom, drag to pan, click to select, hover for
values, **Fit** to reset, and **Min |edge|** to filter individual edges.

- **Weights:** signed parameter values; edge width/opacity represents magnitude.
- **Activations:** signed node activations for the selected fixed probe or the
  named fixed-cohort mean, with explicit per-layer scales.
- **Local contributions:** source activation × weight to the destination’s linear
  preactivation. Bias and nonlinear/normalization operations remain separate.
  An inactive source has zero contribution even if its weight is large.
- **Intervention evidence:** overlays measured behavioral effects using dotted
  gold links and a separate intervention panel. It requires an exact matching
  model-state hash, optimizer step and probe bank, with absolute module IDs for
  alignment. Missing evidence is disclosed; a nearby snapshot is not substituted.
  Physical edge encodings never acquire causal-effect widths.

Heatmaps identify fixed probes and saved feature coverage. Native views label
units, signed color scales, missing/unavailable data and partial coverage. Scales
are currently per snapshot and labeled as such; fixed comparison scales and richer
distribution/CKA workflows remain roadmap section 4. Reports remain self-contained
and now include network weight/activation/contribution modes with explicit display
limits. Production package publication and coordinated training resume remain
separate roadmap work.

Validation includes known-network signs, biases and linear contributions,
zero activation with a large weight, topology changes, unsupported operations,
model-owned and standalone providers, output gating, metadata completion, lazy
array references, snapshot following/history, and comparison identity checks.

## BlendRL method in newMonitors/onlineRL

The `BlendRL` method selects `blendrl-dev` with a neural expert, an NSFR expert,
a neural gate and the `sb3-dev/ppo` runner. It adds these saved rich diagnostics
in both live **Diagnostics** and **Results → Diagnostics**:

- `blend_reliance` / **Expert routing and action probabilities**: fixed-probe
  heatmaps of gate weights, each expert's action probabilities and the blended
  policy probabilities. Columns identify experts/actions. Probability scales stay
  fixed at 0–1 across snapshots.
- `blend_reliance` / **Expert/gate intervention effects**: a probe × intervention
  heatmap of signed changes in `log P(left) − log P(right)`, relative to the
  unmodified policy. The companion **Expert/gate reliance map** summarizes mean
  absolute effects and exposes signed effects, action-flip rates and test names
  on hover. These use all fixed probes, without a discovery/held-out split; they
  do not estimate rollout reward or validate a complete circuit.
- `neural_activations` and `gate_activations`: layer activation heatmaps and
  representation summaries for the neural expert and gate separately.
- `neural_network` and `gate_network`: bounded network views with weights,
  activations and local linear contributions for each explicit target.

All five instances use the same 128-probe bank, seed 0, an interval of 1,000 real
optimizer updates, and at most 32 snapshots including a reserved final snapshot.
The existing DNN monitor instances are disabled only inside the BlendRL method,
whose targets are `policy_modules.1.network` and `blender.network`. Method-level
`monitors` settings are forwarded to the worker's root configuration, not
`agent.monitors`. Input-provider selection is also method-specific.

MountainCar action names are explicitly `[left, noop, right]`. The BlendRL method
selects `mountaincar/monitoring_default`, a local copy of the NSFR default rule
text with its own explicit valuation. The converter duplicates raw position and
velocity into the agent and environment objects, preserving both required object
IDs. Negative velocity selects the left predicate, positive velocity selects
right, and zero selects noop; truth values are 0.99/0.01. This local ruleset
supplies the valuation missing from the installed environment package and leaves
managed packages unchanged. See `src/usr/rules/mountaincar/monitoring_default`.


## Numerical AL diagnostics

The editable THEMES suite also publishes provider-owned TICC, EM and EDM training
signals and bounded policy-response, feature-replacement and membership snapshots
through the same telemetry/Diagnostics formats. These are not arbitrary standalone
monitor plugins attached to an SB3-style adapter: the numerical schedule exposes
explicit observation boundaries. Fixed probes come from training rows only;
feature replacement is model sensitivity, not a causal treatment intervention.
See [THEMES/AL](THEMES_AL.md#plotman-figures-and-live-diagnostics) for controls and
composable PlotMan panels. Broader architecture support and engine-wide resume
or reproducibility are separate work.
