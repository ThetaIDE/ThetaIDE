# Mechanistic monitoring: remaining work

Status: P0 sections 1–3 implemented; P1/release work remains. See the implementation
guide for supported targets, bounded coverage and current comparison semantics.

## Goal

Make network structure, activations and intervention evidence inspectable inside
the IDE while an experiment trains, with reproducible comparisons across saved
model states. Preserve both model-provided monitors and independently selected
monitor components.

The next priority is rich visual output and its integration into the live monitor
panel. Adding more scalar curves alone will not deliver this experience.

## What already exists

The foundation includes component discovery, architecture/runner compatibility
checks, model-owned and standalone instances, optional-collection gating,
optimizer-step scheduling, isolated probe execution, fixed probe banks, versioned
snapshots, monitor-state persistence APIs, and native package export/install.

Initial providers cover activation statistics and CKA, neuron/layer ablation and
patching, bounded sequential-path mediation, and BlendRL expert/gate reliance.
Separate HTML reports contain heatmaps and intervention maps. Local Hub schema
support and release candidate archives exist; production publication is pending.

The live IDE panel retains scalar charts and provides native Diagnostics views for
heatmaps, interventions and bounded sequential networks, with saved-step playback
and compatible checkpoint comparison. Monitor
state can be restored through an API, but training resume is not orchestrated
automatically. Current execution is intentionally limited to compatible
feedforward PyTorch models, flat observations and discrete-action SB3 adapters.

See [the implementation guide](../guides/MECHANISTIC_MONITORS.md) and
[the monitor contract](../reference/MONITOR_CONTRACT.md) for current behavior.

## Implementation order

| Priority | Work | Depends on |
| --- | --- | --- |
| P0 | 1. Declare rich visual outputs | Existing component and snapshot contracts |
| P0 | 2. Render rich outputs inside the live IDE panel | 1 |
| P0 | 3. Capture topology/weights and build a layered network viewer | 1; integrates with 2 |
| P1 | 4. Add matrix and distribution views | 1–2; existing activation snapshots |
| P1 | 5. Add environment state-response maps | 1–2; model/environment adapters |
| P1 | 6. Strengthen causal inspection and validation | Existing interventions; 2–3 |
| P1 | 7. Wire coordinated training/monitor resume | Existing persistence APIs |
| P2 | 8. Extend architectures, runners and capture modes | Explicit capability contracts |
| Release | 9. Ship the updated IDE and publish compatible packages | Verified release scope |

P0 provides the native visual foundation. Next priorities are the P1 views and
validation below, before expanding backend coverage or publishing packages.

## 1. Declare rich visual outputs

- [x] Define a versioned declaration for visual outputs alongside scalar metrics.
  Initial kinds should cover scalar series, matrices/heatmaps, distributions,
  network graphs and intervention graphs. State-response views may use matrix
  outputs plus environment-specific axis metadata.
- [x] Give each output a stable identity, label, source instance, required
  capabilities, artifact schema, axes and value semantics.
- [x] Transport references to arrays/graph artifacts through telemetry. Keep
  large tensors and full graph payloads out of scalar metric records.
- [x] Define how collection, recording, visibility and artifact retention apply
  to each visual output. Hiding a view must have an explicit relationship to
  collection cost; optional work must remain gateable before computation.
- [x] Retain scalar catalogs and legacy model collection without migration.
- [x] Keep discovery independent of importing models or allocating networks.

**Acceptance:** a model-owned monitor and a standalone monitor can declare rich
outputs through the same contract. The IDE discovers their kinds before a run,
validates compatibility and preserves existing scalar monitoring behavior.

## 2. Render rich outputs inside the live IDE panel

- [x] Dispatch shared view components by declared output kind, rather than by
  hardcoded model or monitor names.
- [x] Embed activation heatmaps and intervention graphs as first-class views.
  Keep the self-contained report export path for offline inspection.
- [x] Add a training-step selector, snapshot playback and a follow-latest mode.
  New snapshots should refresh an open view without requiring the user to reopen
  an artifact. Browsing an older snapshot must remain possible during training.
- [x] Provide applicable run, monitor-instance, target, layer and probe-state
  selectors. Keep selections stable across refreshes and comparison views.
- [x] Display legends, measurement units, missing data and sampled/partial
  coverage. Read only completed snapshots and load large arrays on demand.
- [x] Follow the [shared parameter form conventions](../reference/CONFIG_PARAMETER_FORMS.md)
  for all configuration controls. Reuse shared cards and controls.

**Acceptance:** a running `newMonitors/onlineRL` experiment shows curves,
heatmaps and intervention maps in the IDE. Views update as snapshots arrive,
support historical inspection, and reopen after the run ends.

## 3. Capture topology and build a layered network viewer

### Collection and model access

- [x] Add bounded topology and parameter snapshots; current activation artifacts
  and model-state hashes do not contain all data needed for this viewer.
- [x] Start with explicit support for the standard feedforward DNN. Let
  architecture-owned adapters expose meaningful modules, units, connections and
  semantic groups for other models.
- [x] Record stable module/unit identifiers, topology fingerprints, biases,
  weight orientation, activation functions and relevant preprocessing.
- [x] Detect topology changes. Do not silently equate units or features across
  incompatible snapshots or different model initializations.
- [x] Bound layer, unit and edge counts. Record truncation/selection rules and
  omitted coverage instead of presenting a sampled graph as the whole network.

### Viewer behavior

- [x] Arrange supported units by layer; support grouping/collapsing large layers,
  zoom, selection, hover inspection and edge filtering.
- [x] Provide explicitly labeled display modes:
  - **Weights:** edge width/opacity represents magnitude; color represents sign.
  - **Activations:** node values correspond to a selected probe state or a named
    aggregate over a fixed cohort.
  - **Local contributions:** for an applicable linear connection, display
    `source activation × weight` as a contribution to the destination's
    preactivation, with bias and nonlinear operations identified separately.
  - **Intervention evidence:** overlay measured behavioral or mediation effects
    with their baselines and test definitions.
- [x] Allow checkpoint comparison while preserving layout and node identity.
- [x] Keep physical connections distinct from indirect effects. Large weights,
  large local contributions and causal importance are different quantities.

**Acceptance:** a small known network renders the correct connections, signs,
activations and linear contributions. An inactive source unit can have a large
weight and zero contribution. Unsupported architectures fail clearly or request
an adapter. The selected state and training step are always identifiable.

## 4. Add matrix and distribution views

- [ ] Expose existing per-layer activation arrays as heatmaps in the IDE.
- [ ] Add histograms or comparable distribution views for activations and,
  once collected, weights. Show saturation and inactivity where meaningful.
- [ ] Add layer/checkpoint CKA matrices for compatible fixed probe banks rather
  than relying only on aggregate initial/previous similarity curves.
- [ ] Provide fixed versus per-snapshot color/axis scaling with explicit legends.
  Automatic rescaling must not make unchanged values appear to change.
- [ ] Preserve sample/feature identities and display saved-feature coverage.

**Acceptance:** selecting a layer or checkpoint reveals its detailed response
distribution; repeated snapshots produce consistent displays under fixed scales;
degenerate similarity measurements appear as unavailable rather than invented
zero values.

## 5. Add environment state-response maps

- [ ] Start with MountainCar position × velocity and CartPole pole angle ×
  angular velocity slices.
- [ ] Display policy decisions, action probabilities or Q-value contrasts, and
  model-specific quantities such as BlendRL routing weights.
- [ ] Obtain axis names, units, ranges, action meanings and preprocessing from
  environment/model adapters rather than embedding these assumptions in the UI.
- [ ] Identify coordinates held fixed in higher-dimensional slices. Distinguish
  synthetic grid states from observed rollout states and show observed coverage.
- [ ] Support checkpoint comparison and, for BlendRL, aligned expert/gate views.

**Acceptance:** users can see where a policy changes its decision and how that
boundary evolves. A synthetic slice is not presented as a trajectory or evidence
that the agent visited those states.

## 6. Strengthen causal inspection and validation

- [ ] Integrate candidate selection, ablation, donor patching, downstream
  mediation and rescue results into the network/intervention views.
- [ ] Make the behavior score, baseline, discovery/validation split, tested
  candidates, random controls and excluded near-zero denominators inspectable.
- [ ] Show individual-unit effects separately from selected-set effects and
  unselected sampled-candidate effects. Preserve the limits of complement tests.
- [ ] Add independent held-out cohorts/seeds and suitable episode-level
  uncertainty estimates where data supports them. Correlated probe states must
  not be treated as independent episodes.
- [ ] Expand known-network conformance cases to cover interactions, cancellation,
  alternative paths and failed rescue, not only a single successful pathway.
- [ ] For each stronger circuit claim, define the faithfulness test and its
  limits. Sequential-path mediation can include intermediate layers; it does
  not establish a direct synaptic edge or a complete circuit.

**Acceptance:** every highlighted causal relationship can be traced to a saved
intervention and evaluated on held-out data. A visually compelling candidate
that fails controls is shown as weak/unsupported evidence.

## 7. Wire coordinated training and monitor resume

- [ ] Coordinate model/optimizer checkpoints with monitor references, schedules,
  probe banks and required RNG/preprocessing state at a defined boundary.
- [ ] Restore actual optimizer-step counters; do not substitute PPO epochs or
  environment-transition counters.
- [ ] Validate model, provider, framework, configuration, topology and probe
  compatibility before applying saved monitor state.
- [ ] Handle incomplete checkpoint/snapshot pairs without mixing references.
  An intentional new reference must begin an identifiable new segment.
- [ ] Define replay/rollout buffer and environment-state requirements per runner.
  If exact online training continuation is unsupported, disclose the new-rollout
  boundary instead of promising an identical resumed trajectory.

**Acceptance:** a deterministic supported test harness resumes with the correct
monitor references, counters and noncolliding snapshots. Incompatible state is
rejected; ordinary supported restart behavior is documented end to end.

## 8. Extend architectures, runners and capture modes

- [ ] Add explicit conformance runs for each advertised adapter, including A2C,
  and grow beyond the existing DQN/PPO integration coverage.
- [ ] Add model-owned adapters for other feedforward/composite architectures and
  architectures whose hooks are essential to their computation.
- [ ] Define continuous-action behavior scores before extending to SAC; the
  existing discrete action contrast cannot be assumed to apply.
- [ ] Add Lightning support using implemented, versioned lifecycle boundaries.
- [ ] Treat training-forward activations, isolated replay gradients and actual
  training-backward capture as distinct capabilities. Specify timing, tensor
  ownership, lifetimes, gradient preservation and cleanup before advertising
  any of them.
- [ ] Define distributed/rank aggregation and sharded-model semantics before
  claiming distributed support.
- [ ] Measure collection and artifact/rendering overhead separately. Extend
  budgets and execution strategies for larger models while preserving state
  isolation and disabled-collection guarantees.

**Acceptance:** a capability is advertised only after its real runner/model
path has passed relevant conformance checks. Unsupported combinations remain
explicitly unavailable.

## 9. Release the IDE and compatible Hub components

- [ ] Define the release boundary and actual minimum IDE/API compatibility for
  the new visual contracts and collectors.
- [ ] Ship required IDE support before, or with, component packages that need it.
  A component archive cannot upgrade an older IDE's runtime or UI.
- [ ] Build new versioned archives from authoritative component sources, preserving
  unrelated workspace/registry changes. Revalidate the exact exported payloads.
- [ ] Verify discovery, installation, update, configuration, execution and report
  reopening in a clean workspace.
- [ ] When production publication is authorized, follow the
  [Hub release workflow](../development/THETA_HUB_RELEASE_WORKFLOW.md), including the exact
  client feed, archive hashes and client-visible release verification.

**Acceptance:** the target IDE can discover, install and execute the released
components with the declared visual outputs. Production availability is verified
from the actual client feed, not inferred from local archives or a successful
deployment alone.

## Deferred research extensions

These are later research additions, not prerequisites for a useful native viewer:

- Learned feature dictionaries using sparse autoencoders or transcoders, with
  reconstruction quality and feature alignment across checkpoints evaluated.
- Transformer-specific attention/feature views and circuit tracing through
  appropriate architecture adapters.
- Concept probes and example-based unit/feature interpretation, with decodability
  kept distinct from causal use.
- Broader automatic circuit discovery with explicit coverage, task generalization
  and intervention-based faithfulness criteria.

## Next milestone: a usable native visual experience

The next milestone is complete when a user can run the existing demo, inspect a
layered network inside the live monitor panel, select a probe state and training
step, switch among clearly defined weight/activation/evidence modes, and inspect
heatmaps alongside scalar curves. New snapshots should arrive automatically,
and the same results should remain inspectable after training ends.

This milestone must retain model-owned monitoring, compatibility checks,
reproducibility, bounded cost and training-state isolation. A graph's appearance
must never imply stronger mechanistic evidence than its measurements support.
