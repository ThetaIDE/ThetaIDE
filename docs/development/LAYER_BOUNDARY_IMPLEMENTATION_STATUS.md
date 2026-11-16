# Layer boundary implementation status

Current state of the [layer boundary plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md) as of 2026-11-16. This page describes only what is true now and what is left; replace its contents as work lands. Dated evidence for phases 0–4 is in the [implementation history](../reports/LAYER_BOUNDARY_HISTORY_20261115.md). The [handoff prompt](LAYER_BOUNDARY_HANDOFF.md) lists the next work.

## Progress

All phases (0–7) and the [compatibility requirements plan](../plans/COMPATIBILITY_REQUIREMENTS_PLAN.md) are done, and the v3 batch is published. `implement/layer-boundaries` is merged into `main` together with the publication.

| Phase (delivery) | State |
| --- | --- |
| 0–3 — Contract, workspace/config ownership, runner features, site/tuner extraction, app-plugin SDK (A–E) | Done |
| 4 — Retire `in/` and every adapter (F) | Done; v3 batch published ([release](../reports/V3_HUB_RELEASE_20261116.md)) |
| 5 — Results in `src/usr/results`, remote bundles (G) | Done (`4d95ad5`) |
| 6 — Installed distribution (H) | Done (`8da3c4c`) |
| 7 — Enforcement (I) | Done (`252d3dc`) |
| Compatibility requirements | Done (`1fa0d57`–`9d0e7d3`) |

Gates: the real cluster smoke run is not a gate and is not run (user decision, 2026-11-15). The desktop is accepted as working; the interactive items at the end are unchecked.

## Repositories

- **IDE:** `main` includes the whole refactor. `implement/layer-boundaries` is pushed and equal to it.
- **Hub:** ThetaHub `main` is `570d30c` (34 listings, SDK 3.1). Package sources are theta-plugins `main` at `cf27f57`. The local `../theta-plugins-v3` and `../theta-hub-v3` worktrees (branch `v3-republish`) are the working copies; `../theta-plugins` and `../theta-hub` hold the user's own uncommitted work and are untouched.
- **Repository workspace:** runs the published releases, installed from the Hub feed: atari-env, autoencoder, blendrl 1.4.0, d3rlpy-suite 2.1.0, dl_basics_suite, dnn, dueling_resnet, gym_classic_control, image-data, lstm, mlp, nature_cnn, neumann, nsfr, optuna 1.2.0, replay-buffer, sb3-suite 2.1.0, slurm 2.0.0, standard-figures, tabular-data, thetalib, transformer. The earlier preparation's installed copies are in `.thetaide/retained_packages`. The dev forks (`sb3-dev`, `d3rlpy-dev`, `gym_classic_control_dev`, `blendrl-dev`) left `src/usr`: their work is in the published releases, and the test suite keeps them as fixtures. The user's own packages stay in `src/usr`: the three monitors, two objectives, three sites and `themes-dev`.

## Compatibility requirements

- A capability declares `requires` rules (`block` or `warn`, a `config` path or a `kind` subject, optional `when` and `message`), `provides` contracts and an optional validation-time `preflight` hook. `src/app/core/requirements.py` is the only evaluator. Validation, forms (`mark_candidates`, `compatibility_status`), the run record (`compatibility_warnings`) and SDK 3.1 (`src.app.api.config.compatibility`) read its findings. Reference: [compatibility requirements](../reference/COMPATIBILITY_REQUIREMENTS.md).
- Paradigms describe only themselves. Package/profile `paradigms`, `compatible_paradigms`, `allowed_agents`, paradigm `constraints`, the data slot `role`, `output_contracts`, `input_types`, `execution_preflight` and the `symbolic_state_loader` slot are gone; the manifest schema is strict at the top and capability level.
- Components own their sub-components through `x-theta-domain` parameters (narrowed with `x-theta-contract`); package code loads them with `load_selected`. BlendRL's symbolic state builder is the model parameter `symbolic_state`.
- Core, the SDK and the desktop do nothing extra for any component: rulesets are NSFR's `rules` workspace resource (`x-theta-resource` pickers; `resource_entries`), Slurm's shell helpers live in the Slurm package, run loggers declare their result categories, and the sweeper adapter is the generic `hydra_plugins/theta_sweeper`.
- Validation names selections that resolve to no component before evaluating compatibility. Pickers ignore a candidate's rules about a component kind the method has not selected yet.

## Kept as current behavior

Tuning has one shape: `tuning.{component, metric, direction, options.<component>}` plus each method's `tune`. Checkpoint ranking by paradigm/environment `monitor_metric` and `checkpoint.monitor` stays (user). A run logger's only switch is `logging.backends.<id>.enabled`. The resource pool's SQLite column upgrade stays. `requires-python` says 3.10, but core already imports `tomllib` (3.11).

## The user's files that are out of date

Updated with the user's approval (originals in `.thetaide/backups/experiments-20261116`): `NewMonitors/onlineRL`, `BlendRL/Cartpole/LearnFast` and `BlendRL/Cartpole/sweep_sustainedSolve` select the published IDs, `model.symbolic_state` and `blender`/`blender_model`; the sweep's `n_trials`/`n_jobs` moved under `tuning.options.optuna`. All three validate. Still failing validation in `src/usr/config/experiment`:

- `BlendRL/Cartpole/Offline Run` needs `model.symbolic_state` (for example `blendrl/cartpole_state`) for its default hybrid actor.
- `BlendRL/MountainCar/single run` and `sweep` extend `experiment/Belndrl Bench/_base`, which no longer exists.
- `Execution/supervised_smoke` and `unsupervised_smoke` select the CartPole env, which blocks outside online/offline RL; supervised and unsupervised runs take no env (`override /env: null`).
- `AL/codex/themes_synthetic`, `themes_synthetic_smoke`, `themes_mimic_q10_120` and `themes_mimic_cleantrain_q10_116` select the stage-only `themes-dev/ticc`; THEMES's preflight requires a composite model such as `themes-dev/bundle` (THEMES's own shipped recipes do the same).

## Tests

Full per-file run (2026-11-16, 225 files): **2,655 passed, 112 failed, 61 skipped**; the failures are the IDs in the [verification evidence](../reports/COMPATIBILITY_REQUIREMENTS_VERIFICATION_20261116.json), unchanged by the fixture move. Package integration in `../theta-plugins-v3`: 70 passed. The installed-distribution probe passed against the release.

Tests never read the developer's workspace: the base test workspace is built from `tests/fixtures/base_usr` (`conftest.base_workspace`, selected at import), and `tests/test_layout_guard.py` rejects reads of the checkout's `src/usr`. Run per file: `QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`. Fixtures: `tests/conftest.py`'s autouse `tracked_workspace`, `fixture_workspace`, `optuna_workspace`, `component_fixtures` (`install_optional`, `sweep_workspace`, `LocalHub` via `THETA_HUB_ROOT`), `tests/bundle_fixtures.py` for fake-project bundles.

The 112 failures: about 85 test UI that the UI batch replaced (results, queue and monitor panels, layout, tray, launcher, file tree, experiment pane, splash); about 25 exercise behavior the refactor removed (bare `ppo` selections, script `theta_execution`, missing `tuning.component`, undeclared providers, a storage-v2 suite manifest, `MockSettings` without `get`).

## Desktop items not yet checked interactively

1. Editing a profile owned by an installed package asks for an override, and the copy appears under `src/usr/config`; authored-package profiles edit in place.
2. The text editor, TensorBoard, documentation, arXiv and antigravity panels load, and survive disable and re-enable.
3. Results previews generated figures for a finished run in a populated workspace.
4. Settings → Terminal → Cluster sites: the experiment picker and the generated setup (`pip install -r .thetaide/requirements.txt`).
5. The NSFR `rules` picker and its open-in-editor action; compatibility markers in the agent and model pickers.
