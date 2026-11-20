# What a launch fixes at start

A launch (`run_pipeline.py`, a GUI or queue launch, or a scheduler submission) fixes its configuration, provider packages, declared workspace resources and seeds when it starts. Every method, sweep trial and replicate of the launch uses those, whatever changes afterwards.

## Configuration snapshot

The launch copies the workspace config tree (`src/usr/config`) to `.thetaide/launches/<id>/config` and exports `THETA_CONFIG_SNAPSHOT`. It also copies provider packages (including their code, profiles and package-owned assets), framework config defaults, and the workspace resource directories providers declare, such as `src/usr/rules`. `THETA_INPUTS_SNAPSHOT` identifies that private inventory, including provider overrides and extension enablement. Symlinks are dereferenced so they cannot lead workers back to editable input files.

In the launch's processes, configuration, provider discovery and resource resolution read these copies. Hydra re-composing a later sweep trial therefore reads the launch's original inputs. Editing, adding or removing live configs, rules or components applies to the next launch; those changes neither alter nor invalidate the active launch. Package integrity checks verify the private copies. A frozen worker does not hold a lease blocking live component installation or editing. Capturing inputs briefly holds the component transaction lock so installs cannot race the copy.

A launch started inside another (a workflow node) keeps the outer snapshot. Copies are removed when the launch ends; one left by a killed launch is removed after a week. The application runtime and Python environment are not copied: updating Theta itself or replacing Python dependencies during training remains outside this isolation boundary. Explicit external data paths continue to refer to their configured datasets; large datasets and results are not copied into launch inputs.

These semantics apply to new launches. A job started by the older hash-only implementation has no private resource copy and cannot retroactively recover its original inputs after they have been changed.

Implementation: `src/app/pipeline/launch_config.py`, `launch_inputs.py`, `WorkspacePaths`, and provider discovery/loading.

## Worker failures and objective scores

A tuning trial with any failed or incomplete training replicate fails before its objective component is called. A worker crash or snapshot-integrity error cannot become a valid numerical score, including zero. Explicit cancellation/pruning remains distinct from failure. Previously recorded invalid scores are not silently rewritten by this change.

## Saved results

The desktop and fresh terminal launches save by default under the experiment's group and recipe path. Within the `logs` and `checkpoints` roots, a training worker writes to

```text
<group>/<recipe path>/<experiment>/<stamp>/<method>/trial_<number>/seed_<training seed>/
```

An unswept method uses `trial_0`. A sweep uses the tuner's trial number. Trial scores and objective records are saved at the trial level, and the launch recipe and resolved config are saved at the stamp level. Scheduler launches use their execution ID as the stamp.

The Results browser also reads the legacy directory layout in place. Turning off **Timestamp experiment**, supplying an explicit terminal `experiment_id`, recovering a checkpoint, or resuming a study lets a launch reuse its named result directory.

## Seeds

Seeds are drawn, not chosen. The launch draws every seed the experiment config leaves unset, once for the whole launch:

| Setting | Drawn when | Count |
|---|---|---|
| `seed` | unset (the default is `null`) | 1 |
| `variability.training_seeds` | `variability` is present without it | `variability.training_runs` (default 1) |
| `variability.evaluation_seed` | `variability` is present without it, and every method supports evaluation seed control | 1 |

Methods in one launch share the training and evaluation seed sets so their results can be compared at matched seeds. Every configuration in a fixed-parameter sweep is scored using the same set. The tuner's own sampler seed is a separate setting.

Continuing a study restores the training and evaluation seeds from its trial records, including for single-seed studies. A conflicting explicit seed or training-run count stops the launch. The staged LunarLander workflow records its seed plan when it creates a study with queued configurations. For a started study, keep its execution manifests or worker records available. A legacy study whose original vector-environment seeds overlapped can continue with a notice. Start a new study to use an independent set.

To reproduce an individual run, copy its recorded `seed`, or its `training_seeds` and `evaluation_seed`, into the experiment. `training_runs` can accompany an explicit `training_seeds` list when the counts match. For new online RL launches, Theta checks that the initial worker seed ranges are disjoint and that the evaluation base falls outside those ranges.

### Drawing

Draws use the operating system's secure generator (`secrets`). Each draw is below 2^32 and is at least 2^16 away from other draws, explicit launch seeds, and base seeds found in execution manifests and saved launch configs. This gives runners room for offsets such as `seed + i` for vector worker `i` and `evaluation_seed + event_index` for periodic evaluation.

A method worker started on its own (`python -m src.app.train ...`, as component tests do) is a launch of one run: with `seed` unset it draws one the same way and records it in its effective config.

Implementation: `src/app/pipeline/seeds.py`.

### Held-out RL comparisons

Freeze the selected hyperparameters, then run a separate experiment with tuning disabled and all compared methods in the same recipe. Set `variability.training_runs` and leave the training and evaluation seed fields unset. With the tuning results available in the workspace, the launch draws a fresh set away from the recorded tuning seeds and shares it across the compared methods. Use the held-out execution manifest when reporting results and estimate uncertainty across training runs. [Eimer et al., §6](https://proceedings.mlr.press/v202/eimer23a/eimer23a.pdf#page=8)

### Records

- The launch prints the drawn values and saves them in the launch config (`results/logs/<group>/<experiment>/config.yaml`), with `theta_drawn_seeds` listing which settings were drawn.
- An execution manifest's `plan` records `training_seeds`, `evaluation_seed` and `drawn_seeds`; each run records its `training_seed`.
- Each evaluation event in `evaluation_events.jsonl` records its exact `evaluation_seed` and seed policy.
