# What a launch fixes at start

A launch (`run_pipeline.py`, a GUI or queue launch, or a scheduler submission) fixes its configuration and its seeds when it starts. Every method, sweep trial and replicate of the launch uses those, whatever changes afterwards.

## Configuration snapshot

The launch copies the workspace config tree (`src/usr/config`) to `.thetaide/launches/<id>/config` and exports `THETA_CONFIG_SNAPSHOT`. In the launch's processes, `WorkspacePaths.config` resolves to the copy, so Hydra re-composing the experiment for each sweep trial reads the YAML as it was at launch; edits in the IDE or on disk apply to the next launch. A launch started inside another (a workflow node) keeps the outer snapshot. The copy is removed when the launch ends; one left by a killed launch is removed after a week. Package profiles are versioned components and are not copied, and code is not snapshotted: a launch's new worker processes import the application as it is when they start.

Implementation: `src/app/pipeline/launch_config.py`.

## Seeds

Seeds are drawn, not chosen. The launch draws every seed the experiment config leaves unset, once for the whole launch:

| Setting | Drawn when | Count |
|---|---|---|
| `seed` | unset (the default is `null`) | 1 |
| `variability.training_seeds` | `variability` is present without it | `variability.training_runs` (default 1) |
| `variability.evaluation_seed` | `variability` is present without it, and every method supports evaluation seed control | 1 |

Every method, sweep trial and replicate of the launch receives the drawn values as explicit overrides. Methods in one launch therefore train and evaluate on the same seeds and stay comparable, and every trial of a sweep shares them. A resumed study is a new launch: it draws new seeds unless the config pins the original ones.

Set a seed in the config only to reproduce a recorded run: copy its recorded `seed`, or its `training_seeds` and `evaluation_seed`, into the experiment. `training_runs` may stay beside an explicit `training_seeds` list only when the counts match. There is no `starting_seed`; consecutive seeds made neighbouring runs share vector-environment streams.

### Drawing

Draws use the operating system's secure generator (`secrets`) and stay below 2^32, the limit of NumPy's legacy seeding that Stable-Baselines3 uses. Runners derive further streams with small offsets (a vector environment seeds worker `i` with `seed + i`; evaluation uses `evaluation_seed + event_index`). Each draw is therefore at least 2^16 from the launch's other draws and from every seed recorded under `results/` (execution manifests and saved launch configs), so no two runs share an environment or evaluation stream. Gymnasium hashes seeds through NumPy's `SeedSequence`, so distinct seeds give independent streams.

A method worker started on its own (`python -m src.app.train ...`, as component tests do) is a launch of one run: with `seed` unset it draws one the same way and records it in its effective config.

Implementation: `src/app/pipeline/seeds.py`.

### Records

- The launch prints the drawn values and saves them in the launch config (`results/logs/<group>/<experiment>/config.yaml`), with `theta_drawn_seeds` listing which settings were drawn.
- An execution manifest's `plan` records `training_seeds`, `evaluation_seed` and `drawn_seeds`; each run records its `training_seed`.
- Each evaluation event in `evaluation_events.jsonl` records its exact `evaluation_seed` and seed policy.
