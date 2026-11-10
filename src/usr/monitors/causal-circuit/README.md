Measures neuron ablations and donor patching with held-out candidate-circuit evaluation.

Select this component under the experiment `monitors` mapping. See docs/guides/MECHANISTIC_MONITORS.md in Theta-IDE for the lifecycle, configuration and limitations.

## Configure

Select this component in the experiment's `monitors` mapping, with a unique
instance name, `target: network` for the standard DNN, and a `params` mapping.
`interval_updates` counts optimizer steps; `max_snapshots` includes a reserved
final snapshot. Outputs use `interp/<instance>/`. Open the monitor report from
the run's Artifacts list to inspect snapshots over training.

Requires the updated Theta monitoring API and an SB3 runner advertising
`probe_forward.v1` and `post_update.v1`. V1 supports flat observations and
feedforward PyTorch targets; probes execute on isolated model copies. These
measurements do not establish a complete causal circuit. Exact configuration,
probe/model hashes and measurement assumptions are saved with every snapshot.
