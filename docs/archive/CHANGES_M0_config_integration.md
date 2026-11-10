# Changes: Environment Setup, M0 Config Integration and M1 Training Monitor

> **Historical implementation log.** This records an earlier ThetaIDE/NeSyRL architecture and dependency setup. Its install commands, project layout, and runtime descriptions are not current Theta-IDE guidance.

Date: 2026-09-28

This document records every change made to set up the project's virtual environments and connect the ThetaIDE desktop frontend to the NeSyRL backend. The integration covers **M0 (experiment configuration)**. In the milestone plan, M0 means a config built in the GUI can be shown to match a hand-written one. Section 7 covers **M1**: launching real training from the GUI and monitoring it live.

## Summary

- The backend and frontend each have their own virtual environment. The GUI never imports torch.
- The two communicate over HTTP. The frontend is a thin client of the FastAPI server.
- The Hydra config composition that lived inside `run_pipeline.main()` now lives in a shared module. The command line and the API both use it, so a config built in the GUI composes exactly like the same experiment run from the command line.
- The experiment builder turns its form into Hydra overrides. The backend composes and validates them, and the GUI shows the resolved config.
- New tests prove the configs match. The full suite passes: 138 tests, the original 125 plus 13 new ones.

## 1. Virtual environments

| | Location | Python | Install command |
|---|---|---|---|
| Backend | `venv/` | 3.12 | `pip install -e ".[dev,api]"` |
| Frontend | `frontend/.venv/` | 3.12 | `pip install -r frontend/requirements.txt` |

Both environments use Python 3.12 because torch, Hydra and Lightning support it reliably.

The root `requirements.txt` still fails to install, because it points to `src/usr/models/fyd_repo/requirements.txt`, which doesn't exist. Install the backend from `pyproject.toml` as shown above. The optional `atari` and `video` extras are not installed.

**Running (from the repository root, in separate terminals):**

```powershell
# Backend
.\venv\Scripts\Activate.ps1
uvicorn src.app.api.app:app --reload --host 127.0.0.1 --port 8000

# Frontend
.\frontend\.venv\Scripts\Activate.ps1
python -m frontend            # optional: --api-url <url> or THETAIDE_API_URL
```

The API reads configs by paths relative to its working directory, so it must be started from the repository root.

## 2. Backend changes

### New: `src/app/pipeline/compose.py`
This module is now the only place experiment configs are composed.

- `compose_experiment(experiment, overrides)` does what `run_pipeline.main()` did before:
  1. Resolves the experiment name.
  2. Filters out sweep and Hydra-internal overrides.
  3. Composes the config with Hydra.
  4. Fills in `experiment_id` and `group`.
  5. Builds the launch arguments passed to subprocesses.
  6. Detects sweep runs.

  It returns a `ComposedExperiment` that includes the equivalent command (`argv`).
- It uses `initialize_config_dir` with an absolute path to `in/config`. A lock stops concurrent API requests from interfering, because Hydra keeps global state.
- `validate_composed()` wraps the existing `validate_experiment_config`.
- `method_plans()` returns each method's merged settings (from `parse_methods_dict`) and the exact training arguments its subprocess receives (from `build_method_overrides`).
- `effective_config()` and `comparable_config()` convert a config to plain data. `comparable_config()` drops the `hydra` bookkeeping node and `experiment_name`, the two fields that name the recipe file rather than describe the experiment.

### Modified: `run_pipeline.py`
- Config loading now calls `compose_experiment()` and `validate_composed()`. About 35 lines of inline composition were removed.
- Removed the unused `hydra`, `compose` and `initialize` imports, and the direct import of `validate_experiment_config`.
- Behavior is unchanged: `csc510/cartpole_demo dry_run=true` still validates, and an invalid override still exits with the Hydra error.

### Modified: `src/app/api/app.py`
- **`GET /api/config/schema`** describes the experiment builder's fields: key, label, type, min/max/step, choices and help text. The key is the Hydra override path the field sets.
  - Defaults are read from the composed `thetaide/_base` config: `seed`, `total_timesteps`, `agent.lr`, `agent.batch_size` and `env.gamma`. The GUI no longer keeps its own copy of backend defaults.
  - The response also names the base recipe, the paradigm, the environment and the method.
- **`POST /api/config/compose`** composes and validates `{experiment, overrides}` without running anything.
  - It returns `valid`, `argv`, `config`, `config_yaml`, `methods` (settings plus training arguments), `methods_yaml`, `notices` and `errors`.
  - Invalid configs return HTTP 200 with `valid: false`. Each error has a `stage`: `compose` (unknown recipe or bad override), `validation` (paradigm rules) or `methods` (method parsing).
- The builder's contract is defined in `GUI_BASE_EXPERIMENT`, `GUI_METHOD` and `GUI_FIELDS`. For now it covers CartPole/PPO only.

### New config recipes: `in/config/experiment/thetaide/`
- `_base.yaml` is the base for GUI-built experiments: `env: cartpole`, `paradigm: online_rl`, `seed: 42`, `recover: false`.
- `cartpole_ppo_reference.yaml` is a hand-written CartPole/PPO recipe that the builder's default form must match.

## 3. Frontend changes

### New: `frontend/api.py`
`Backend` is a JSON client built on `QNetworkAccessManager`. It ships with PyQt6, so there's no new dependency, and requests don't freeze the UI.

- Each callback receives `(data, error)`.
- Requests time out after 10 seconds.

### New: `frontend/requirements.txt`
Contains `PyQt6>=6.7,<7`.

### Modified: `frontend/model.py`
- New constants: `BASE_EXPERIMENT = "thetaide/_base"` and `METHOD`/`AGENT`/`MODEL = ppo/ppo/dnn`.
- `Config.overrides()` returns the Hydra overrides the form stands for, for example `++experiment_id='name'`, `seed=42` and `++methods.ppo.lr=0.0003`.
- `Config.command()` returns the equivalent `run_pipeline.py` command line.
- `Config.recipe_yaml()` returns a standalone experiment recipe equivalent to the base plus the overrides.
- `_num()` formats floats so that both Hydra and YAML read them back as floats. For example, `1e-06` becomes `1.0e-06`.
- **Removed** `Config.yaml()`, which produced a config the pipeline rejected: `paradigm` and `methods` were missing, and it overrode config groups that don't exist.
- `Store.save()` now writes `recipe_yaml()` into each run's `config.yaml`.

### Modified: `frontend/app.py`
- **The window takes an `api_url`.** A `--api-url` command-line option was added; it defaults to `THETAIDE_API_URL` or `http://127.0.0.1:8000`.
- **Live compose:** a form edit restarts a 250 ms timer, which then sends the config to `/api/config/compose`. Each request is numbered so that responses to earlier edits are ignored.
- **The `config.yaml` tab** shows:
  - the equivalent command and any notices,
  - the settings each method trains with,
  - the full composed config, labeled to explain that `agent.*` holds defaults the method settings override,
  - a status line (valid, or N errors) and a list of errors.
- **The experiment builder:**
  - A validity line under the form shows a check mark, or the first error.
  - The experiment name only accepts `[A-Za-z0-9_-]`.
  - Once the schema loads, the Environment, Method and Training mode boxes show backend values. Field tooltips, ranges and choices come from the schema, and defaults are applied at startup.
- **The status bar** shows the backend state. When offline, a tooltip gives the command to start the backend.
- **Offline fallback:** without a backend, the preview shows the local `recipe_yaml()`, clearly marked as not validated. The schema is fetched again once the backend becomes reachable. At that point ranges and tooltips are applied, but the form values are left alone.
- **Export:** "Export YAML…" is now "Export recipe YAML…". It writes `recipe_yaml()` and logs the `run_pipeline.py` command to run it.
- **Console:** the `config` command now prints the equivalent command.
- Runs are still simulated. The Run button does not start real training.

### Modified: `frontend/theme.py`
- Added the `QLabel#configOk` and `QLabel#configError` styles. They use existing theme role colors, so they remap with every theme.

## 4. Tests

### New: `tests/test_config_compose.py` (13 tests)

| Test | What it proves |
|---|---|
| `test_default_builder_matches_hand_written_reference` | The default form's overrides produce the same config as `thetaide/cartpole_ppo_reference.yaml`, and that config passes validation |
| `test_exported_recipe_composes_like_builder_overrides` (×3) | For three sets of values, an exported recipe produces the same config as the builder's overrides. The values include edge cases (`lr=1e-6`, `gamma=1.0`, a name made only of digits). The test writes the recipe into `in/config/experiment/thetaide/` and removes it afterwards. |
| `test_builder_values_reach_the_method_config` (×3) | Form values arrive at `methods.ppo.*` with the correct types |
| `test_compose_endpoint_matches_direct_composition` | The API returns the same config and command as calling `compose_experiment` directly, and each method's training arguments carry the chosen `lr` and `batch_size` |
| `test_compose_endpoint_reports_invalid_override` | A bad override returns `valid: false` with an error at stage `compose` |
| `test_compose_endpoint_reports_paradigm_violation` | Using `cql` in an online run returns an error at stage `validation` |
| `test_compose_endpoint_reports_unknown_experiment` | An unknown recipe returns `valid: false` |
| `test_schema_defaults_come_from_backend_and_match_builder` | The schema's defaults come from the backend and match the builder's defaults |
| `test_builder_command_passes_pipeline_dry_run` | The builder's command passes `run_pipeline.py … dry_run=true` in a real subprocess |

The frontend was also checked by hand, driven offscreen against the running API, in three cases:
- **Backend running:** the config showed as valid, and edits updated the preview.
- **Backend offline:** the fallback draft appeared.
- **Invalid config:** a paradigm error was shown.

## 5. Documentation and repository housekeeping

- **`INSTALL.md`:**
  - Added Windows activation steps.
  - Fixed the API module path (`src.api.app` became `src.app.api.app`).
  - Added a desktop frontend section covering backend-first startup, `--api-url` and recipe export.
- **`docs/API.md`:**
  - Fixed the same module path.
  - Documented `/api/config/schema` and `/api/config/compose`.
- **`.gitignore`:** added `.venv/` for the frontend environment and `.thetaide/` for the frontend's local run storage.

## 6. Known limitations and next steps

- **Scope is CartPole/PPO.** Adding environments or methods means extending `GUI_FIELDS`, adding a compatibility check, and adding tests.
- **Errors are not tied to fields.** They are reported as a list, labeled by stage.
- **Launching (M1)** was connected afterwards; see section 7. That work fixed the launch problems listed here previously.
- **One lint issue predates these changes:** `run_pipeline.py` has an `f` prefix on a string with no placeholders (`print(f"Declared Methods:")`). It was left as is.

## 7. M1: launching and monitoring real training

The **Launch training** button now trains the builder's config on the local machine through the backend. The Training monitor follows it live.

### Backend: `src/app/api/app.py`
The job endpoints were rewritten:
- **Launch validates first.** It composes and validates the config with the shared compose module and returns `422` with the message if the config is invalid. It returns `409` if `results/logs/<group>/<experiment_id>/` already has results, because the pipeline purges an existing experiment's results. `"overwrite": true` allows it anyway. The job records its group, experiment ID, agents and `total_timesteps`, so its metrics can be found.
- **Live output.** The pipeline runs unbuffered (`-u`, `PYTHONUNBUFFERED=1`, `PYTHONIOENCODING=utf-8`), with stderr merged into stdout. A reader thread collects lines as they arrive and keeps at most 20,000 in memory. It also writes the full log to `results/jobs/<job_id>.log`. `GET …/status?since=N` returns only new lines, plus `log_total`.
- **Live metrics.** A new `GET /api/experiments/{job_id}/metrics?since=N` endpoint reads each agent's newest `metrics.csv` while training runs:
  - Rows come back as numbers.
  - A line still being written is skipped.
  - `reset` signals that Lightning rewrote the file with fewer rows.
  - Versions sort numerically, so `version_10` comes after `version_9`.
- **Cancel stops the whole tree.** A cancelled job's pipeline, `train.py` and plotting subprocesses are all terminated. On Windows the job runs in a new process group and is stopped with `taskkill /T /F`; elsewhere it gets its own session and is stopped with `killpg(SIGTERM)`. A lock prevents a race: a job cancelled before it starts never spawns, and a cancelled job can no longer end up marked `completed`.
- **New `GET /api/experiments/jobs`** lists the jobs this server knows about.
- **Jobs live in memory.** Restarting the API forgets them, but their logs and metrics stay on disk.

### Frontend
- **`app.py`:**
  - **Launch and Stop.** The toolbar has **▶ Launch training** (F5) and **■ Stop** (Shift+F5). Launch is enabled only when the backend is running and has validated the current config. The simulated demo moved to **Run → Start simulated demo** (Ctrl+F5).
  - **Unique experiment IDs.** Each launch uses `<name>_<YYYYmmdd-HHMMSS>`, so earlier results are never purged.
  - **Polling.** Once a second, the window requests the job status and new log lines, then new metric rows. Log lines stream into the console, prefixed with `│`. Metrics are fetched after the status, so a finished job's last rows are always collected.
  - **Monitor panel:**
    - The cards show the latest evaluation reward, the latest total PPO loss, and the number of transitions out of the training budget.
    - The progress bar tracks the budget.
    - Both charts are scaled to the full training budget.
    - A status line says what is happening: starting (about 15 s), live, generating plots, or final metrics with the location of the plots.
  - **Finishing.** Completed, failed (with the exit code), cancelled and interrupted runs are saved to the local run record. A `404` from the backend, meaning it was restarted, marks the run as interrupted.
  - **Closing the window** leaves training running. On the next start, the app reconnects to that run and keeps monitoring it.
  - **Results browser.** It lists trained runs by experiment ID, with source "Trained" or "Simulated". The console `status` command reports the active run.
  - **Wording.** Labels that assumed every run was simulated were updated: the loss card is now "TRAINING LOSS", the loss chart is "Training loss (total)", and the console title no longer says "demo session".
- **`model.py`:**
  - `new_run(config, simulated=...)` creates run records for both kinds of run.
  - `LIVE_STATUSES` and `FINAL_STATUSES` group job statuses.
  - `metric_points()` turns `metrics.csv` rows into chart points keyed by `transitions`: evaluation rows set `reward`, training rows set `loss`, and the other is `None`.
  - `latest()` returns the newest value of a metric.
  - The store writes empty cells for `None` values. On load, it marks only simulated runs as interrupted, so live runs can reconnect.
- **`api.py`:** error messages include the backend's `detail`, for example the reason for a `409` or `422`.
- **`widgets.py`:**
  - `Chart.set_series(series, xmax=None)` skips points missing the chart's metric and accepts a fixed x-axis extent.
  - `MetricCard` exposes `subtitle`.
  - The empty chart now reads "Launch training to see live metrics".
- **`plots.py`:** captions say "simulated" or "trained" per run instead of always "simulated".
- **`theme.py`:** a disabled primary button now looks disabled.

### Tests: `tests/test_training_jobs.py` (8 tests; 2 marked `slow` because they run real training)

| Test | What it proves |
|---|---|
| `test_launch_rejects_invalid_config` | An invalid config is rejected with `422` before anything starts |
| `test_launch_refuses_to_overwrite_existing_results` | Reusing an experiment ID with existing results returns `409` and leaves the files untouched |
| `test_job_cancelled_before_start_never_spawns` | A job cancelled before it starts never calls `Popen` and stays `cancelled` |
| `test_status_returns_log_lines_since_offset` | Log paging returns only new lines and hides internal fields |
| `test_metrics_reader_skips_partial_line_and_picks_newest_version` | A half-written last line is skipped, and `version_10` is chosen over `version_9` |
| `test_metric_points_keep_reward_and_loss_rows_separate` | Evaluation and training rows become separate chart points; epoch duplicates are dropped |
| `test_launch_streams_logs_and_metrics_until_completed` (slow) | A real 3,000-step run completes with exit code 0, streams evaluation logs, reaches the full budget in its metrics, and pages correctly |
| `test_cancel_stops_pipeline_and_training_subprocesses` (slow) | Cancelling a real run mid-training ends the pipeline, and the metrics file stops growing, which shows `train.py` was killed too |

The full suite passes: 146 tests.

The GUI was also driven offscreen against the running API:
- A 20k-step launch showed its first metrics at about 16 s and completed at about 34 s.
- Stop cancelled a run within about 2 s.
- Closing the window mid-run and reopening it reconnected to the run and followed it to completion.

### Remaining limitations
- **One run at a time in the GUI.** The backend accepts several.
- **Jobs don't survive an API restart.** A run in progress then shows as interrupted, though its files stay on disk.
- **Charts show only the first agent.** The builder only makes single-method experiments, but a recipe with several methods would chart just the first.
- **Startup takes about 15 s** (imports and environment setup) before the first metrics appear.

## 8. TensorBoard tab (embedded with QWebEngineView)

The GUI now has a **TensorBoard** tab that embeds a TensorBoard server run by the backend. TensorBoard stays in the backend environment; the frontend only needs `PyQt6-WebEngine`.

### Backend
- **`src/app/api/app.py`:**
  - Added `GET /api/tensorboard`, `POST /api/tensorboard/start` and `POST /api/tensorboard/stop`. They manage one TensorBoard process over `results/tensorboard/`:
    - It binds to 127.0.0.1 on a free port.
    - It runs in its own process group, like training jobs, so stop ends its children.
    - Its output goes to `results/jobs/tensorboard.log`.
    - `atexit` stops it when the API exits normally.
  - `start` returns immediately. The client polls `GET` until `ready`, which means `/data/environment` answers.
  - A deliberate stop is not reported as a crash. `exit_code` and `error` appear only when the server exited on its own.
  - The builder schema has a new boolean field, `tensorboard` ("Log to TensorBoard").
- **`in/config/experiment/thetaide/_base.yaml`:** adds `tensorboard: true`, so GUI-built runs also write TensorBoard logs through Lightning's existing `TensorBoardLogger` hook in `lightning_builder.py`.

### Frontend
- **New `frontend/tensorboard.py`:** `TensorBoardPanel`.
  - Opening the tab starts the server through the API, shows "Starting TensorBoard…" until it's ready, then loads it in a `QWebEngineView`.
  - The toolbar has Start/Stop, Reload and Open in browser ↗ buttons.
  - Clear messages cover the backend being offline, TensorBoard missing from the backend, and an unexpected exit.
  - Without `PyQt6-WebEngine`, the tab falls back to opening TensorBoard in the system browser.
- **`app.py`:**
  - Adds the TensorBoard tab, which starts the server when first shown, and a **View → TensorBoard** menu entry.
  - Adds an **Open TensorBoard →** button under the Training monitor.
  - Adds a **Log to TensorBoard** checkbox to the builder. Its default comes from the backend schema.
  - The compare dialog now handles run records from before this field existed.
- **`model.py`:** `Config.tensorboard` (default `True`) adds `tensorboard=true|false` to the overrides and to exported recipes.
- **`frontend/requirements.txt`:** adds `PyQt6-WebEngine>=6.7,<7`.

### Tests
- **`tests/test_tensorboard.py`:**
  - The status is correct before any start.
  - (slow) Start serves TensorBoard at a 127.0.0.1 URL, a second start reuses the server, and stop really shuts it down without reporting a crash.
- **`tests/test_config_compose.py`:**
  - One case now uses `tensorboard=False`, and every case checks the flag reaches the composed config.
  - The schema default is checked against the builder's default.
- **`tests/test_training_jobs.py`:** cleanup now also removes `results/tensorboard/…`.
- The full suite passes: 148 tests.
- The GUI was driven against the running API. A launch with the default settings wrote TensorBoard event files. Opening the tab started the server, the embedded page loaded (title "TensorBoard"), and its own `data/runs` request listed the runs.

### Notes
- **`QApplication` needs the program name.** QtWebEngine aborts inside `Qt6Core` (exception `0xC0000409`) if the `QApplication` gets an empty argument list, because Chromium needs it. `frontend.app.main()` already passes `sys.argv[:1]`. Any script or test that builds the window itself must do the same, not `QApplication([])`.
- **Offscreen screenshots.** Under `QT_QPA_PLATFORM=offscreen`, `grab()` captures the web view only with `QTWEBENGINE_CHROMIUM_FLAGS="--disable-gpu --disable-gpu-compositing"`. Normal desktop use is unaffected.
- **One server for everything.** A single TensorBoard server shows every run under `results/tensorboard/`. Use its run filter to focus on one run.

## 9. Vertical panel tabs with SVG icons

The five central panels (Training monitor, Results browser, config.yaml, Plot viewer and TensorBoard) are now selected from a vertical bar on the left of the panel area, not from horizontal tabs above it.

- **New `frontend/sidetabs.py`:** `SideTabs`, a column of icon-over-label buttons next to a `QStackedWidget`.
  - It implements the subset of `QTabWidget` the window uses (`addTab`, `setCurrentIndex`, `setCurrentWidget`, `currentIndex`, `currentWidget`, `widget`, `indexOf`, `count`, and the `currentChanged` signal). No other window code had to change.
  - Tabs show a short label (Monitor, Results, Config, Plots, TensorBoard), with the full name as the tooltip and accessible name.
- **New `frontend/icons/*.svg`:** `monitor`, `results`, `config`, `plots` and `tensorboard`.
  - They are 24×24 line icons drawn with `stroke="currentColor"`. The TensorBoard icon is a generic layers mark, not the TensorFlow logo.
  - `svg_icon()` renders each icon with `QSvgRenderer` for the normal, hover and selected states. It replaces `currentColor` with theme role colors (`muted`, `text`, `accent`) and renders at the screen's pixel ratio.
  - `Window.theme_changed()` re-renders the icons, so they follow theme switches, including custom themes.
- **`theme.py`:** adds `QWidget#sideTabs` and `QToolButton#sideTab` styles. They use only role colors, so the theme remapping applies to them.
  - The bar is 70 px wide, with 20 px icons and 10 px labels; "TensorBoard" measures 54 px at that size.
  - Hover and the selected tab use a rounded (6 px) highlight, and the selected tab also gets accent-colored text and icon.
  - The buttons are inset 5–6 px from the bar's edges (in `sidetabs.py`), so the highlight never draws over the bar's 1 px right border.
- The Experiment builder / Notes tabs in the right dock are unchanged.

## 10. PPO training budgets are rounded up to whole rollouts

PPO collects `num_envs × num_steps` transitions (4 × 128 = 512 by default) before each update and never stops partway through a rollout. `lightning_builder.py` therefore runs `ceil(total_timesteps / 512)` rollouts, so a 10,000-step request trains 10,240 steps. The monitor used to show "10,240 of 10,000" without explaining why.

- **`compose.py`:** a new `ppo_rollout()` calculates the real budget from the composed config and each method's overrides (`num_envs`, `num_steps`). `method_plans()` includes it for each method as `rollout`.
- **`api/app.py`:**
  - `/api/config/compose` adds a notice when the budget is rounded up, for example "Method 'ppo' trains 10,240 steps, not 10,000 … so it runs 20 rollouts."
  - `/api/experiments/launch` returns `effective_timesteps`.
- **`frontend/app.py`:**
  - The builder's validity line adds "trains 10,240 steps (20 PPO rollouts × 512)".
  - The monitor measures progress and chart axes against the real budget, and the steps card says "of 10,240 environment steps / 10,000 requested, rounded up to whole PPO rollouts".
  - Records saved before this change fall back to the steps they actually ran.
- **Tests:**
  - The budget calculation is checked for several cases, including a method that overrides `num_envs`.
  - The compose notice appears only when the budget is rounded.
  - The slow launch test checks the prediction against real training: 3,000 requested steps is predicted as 3,072, and the metrics end at exactly 3,072.

## 11. Evaluation metrics reach every logger, on Lightning's step axis

- **`src/app/core/callbacks.py`:** `EnvironmentEvaluatorCallback.evaluate_and_log()` now writes to every logger in `trainer.loggers`. Before, it wrote only to `trainer.logger`, which Lightning resolves to the first configured logger (CSV), so with TensorBoard on, `eval/reward_std`, `time/eval`, `time/train` and `time/total` never reached TensorBoard.
- **Step alignment:**
  - **The problem:** once TensorBoard received both evaluation writes, `eval/reward` showed points on two step scales. The callback used `trainer.global_step`, which counts optimizer steps (8 per PPO epoch by default). Lightning's own logs, including losses and the callback's second `eval/reward` call, use `fit_loop.epoch_loop._batches_that_stepped`, which counts one per batch, with end-of-epoch logs one below it. That gave TensorBoard a zig-zag.
  - **The fix:** the callback now logs at the same step as Lightning (`_lightning_log_step`). It falls back to `global_step` if Lightning renames the private attribute.
  - **Side effect:** the CSV `step` column for evaluation rows now matches the loss rows. The monitor and the plotters read `transitions`, so they are unaffected.
- **Kept on purpose:** the callback's second call, `pl_module.log("eval/reward", ...)`, still feeds `trainer.callback_metrics`, which checkpoint monitoring (`monitor_metric: eval/reward`) and the progress bar use.
- **New `tests/test_callbacks.py`:** it checks that both loggers receive the full evaluation metrics, and that the step matches Lightning's, with the fallback. The first test fails without the fix. A real 2,048-step run was checked: TensorBoard now has `eval/reward_std` and `time/*`, with `eval/reward` and `losses/total_loss` on the same steps.

## 12. Training monitor: fitted reward axis, spread band, selectable second metric

- **`frontend/widgets.py`, `Chart`:**
  - **Fitted axes with round ticks** (`nice_step()`, `format_tick()`). Reward charts still start at 0. With `zero_based=False`, the axis fits the data so small changes such as entropy stay visible.
  - **`reference`** (for example CartPole's maximum reward of 500) caps the axis. It's drawn as a dashed "max 500" line once the data comes within 10% of it; until then the chart notes "max possible 500 ↑".
  - **`band`** (for example `reward_std`) draws a shaded ±1 band. The lower edge is clamped to the axis.
  - **Dots** mark each point of sparse series such as evaluations.
  - **`set_corner_widget()`** places a control in the top-right corner. `set_metric()` switches what the chart plots.
  - `MetricCard` now exposes `title`.
- **`frontend/model.py`:**
  - `METRIC_COLUMNS` maps run-record keys to `metrics.csv` columns. `metric_points()` now keeps reward, reward std, and total, policy and value loss, plus entropy and approximate KL.
  - `available_metrics()` lists the metrics a run actually has.
  - `Store.save()` writes whichever of those columns the run has.
- **`frontend/app.py`:**
  - **Reward chart:** it shows mean ± 1 std over evaluation episodes, under CartPole's 500 ceiling (`ENV_MAX_REWARD`). The title mentions the band only when the run has it.
  - **Second chart and card:** a selector (`SECOND_METRICS`) picks policy entropy (the default), approximate KL, or total, policy or value loss. Each has a card title, a short explanation and its own number format. Runs that lack a metric, such as simulated demos with only reward and total loss, offer only what they have. The second chart fits its axis to the data.
  - **Older records:** finished trained runs recorded before all these metrics were kept are fetched once from `/api/runs/{group}/{experiment_id}/{agent}/metrics` when selected, then saved (marked with `metrics_backfilled`).
  - The compare dialog uses the same band and ceiling.
- **`frontend/plots.py`:** the plot viewer offers every stored metric as an axis, not only step, reward and loss.
- **Tests:** `test_metric_points_keep_evaluation_and_training_rows_separate` covers the new keys.
- **Verified** in the GUI on a copy of a real run (`cartpole_ppo_baseline_20260928-152129`). Its old record was backfilled, and the cards match the run's CSV: entropy 0.638, approximate KL 0.00032, value loss 46.969. The reward axis spans 0–80 with the band. A simulated run falls back to total loss.

## 13. Job queue

Experiments can be queued to train one after another. The queue lives in the backend, so it keeps running with the GUI closed, and the GUI follows each queued job as it starts.

### Backend: `src/app/api/app.py`
- **`LaunchRequest.queue`:** a queued launch is composed and validated like a direct one, then stored with status `queued` in `queue_order`.
- **Queue worker:** a single daemon thread, `_queue_worker_loop`, waits on a condition until the queue is non-empty and no job is pending or running, then runs the next job. Jobs launched directly count too, so nothing runs concurrently with the queue. `run_experiment_task` notifies the condition when a job finishes and records `started` and `finished` times.
- **Duplicate IDs:** launching or queueing a `group/experiment_id` already used by a queued, pending or running job returns `409`, because the second run would purge the first's results.
- **New endpoints:**
  - `GET /api/queue` lists active, queued (with `position`) and the ten most recent finished jobs.
  - `POST /api/queue/{job_id}/move` moves a queued job.
- **Cancelling a queued job** removes it from the queue; it never spawns.

### Frontend
- **New `frontend/queue_panel.py`, `QueuePanel`:**
  - A table with ▶ for running, the position number for queued, and ✓/✗/■ for finished jobs, showing steps, status and timing (running time, queue time, duration, or "never started").
  - Buttons to move a queued job up or down, remove it, or open it in the monitor, plus a running/queued summary. The selection is kept across refreshes.
- **New `frontend/icons/queue.svg`**, for the **Queue** sidebar tab.
- **`app.py`:**
  - **＋ Add to queue** in the toolbar and Run menu (Ctrl+Shift+Q). It's enabled whenever the backend has validated the config, even while a job runs.
  - Queued runs get records with status `queued` and their queue position. The monitor shows "In the job queue (position N)…" for them.
  - **Queue polling:** the window polls `/api/queue` every 2 s while the Queue tab is open or any record is queued. When a queued job becomes active and nothing else is being followed, it becomes the active run, so the monitor follows it. Jobs removed or finished elsewhere update their records.
  - **Lost jobs:** a job missing from the listing is looked up individually. A `404` (the API restarted) marks it `interrupted`.
  - After a run finishes, the queue is checked immediately, so the monitor picks up the next job without waiting for the poll.
  - **Unique IDs:** `unique_experiment_id()` adds `-2`, `-3`, … when several jobs are created within the same second.
- **Tests:** `tests/test_job_queue.py` replaces training with jobs the test finishes on demand. It checks that queued jobs run strictly one at a time and in order, that a moved job runs first, that a removed job never starts, that a queued job waits for a directly launched one, that duplicate IDs are rejected, and that only queued jobs can be moved.
- **Verified end to end** in the GUI with real training. Three jobs were queued in the same second (IDs `…154043`, `…-2`, `…-3`). One was moved up and the other removed from the Queue tab. The monitor followed the first job, then switched to the moved job when it started (about 28 s); both completed. The removed job never started.

## 14. The job queue waits for Start

Adding a job used to start it immediately whenever nothing else was running, so a list could not be built up first. The queue is now paused until started.

- **Backend (`api/app.py`):**
  - `queue_state["running"]` starts `False`. The worker only starts jobs while it is `True`.
  - `POST /api/queue/start` starts the queue; starting an empty queue leaves it paused. `POST /api/queue/pause` stops new jobs from starting, and a job already training keeps running.
  - When the queue has drained and nothing is active, the worker pauses the queue itself, so jobs added later wait for the next start. Jobs added while the last job is still training are run.
  - `GET /api/queue` and queued launches report whether the queue is running.
- **Frontend:**
  - The Queue tab has a **▶ Start queue / ⏸ Pause queue** button and a status line: paused with N jobs waiting, running, or empty. The Run menu has **Start or pause queue** (Ctrl+Shift+R).
  - After Add to queue, the console and status bar say whether the job will run after the jobs ahead of it or is waiting for Start queue. The monitor's note for a queued run says when the queue is paused.
  - **Fix:** the window stopped polling the queue once no job was waiting and the Queue tab was hidden. It then never saw the queue pause itself after draining, and the toggle kept showing "Pause queue". Polling now continues while the queue is running.
- **Tests:** `test_queue_waits_for_start_pauses_on_request_and_pauses_itself_when_drained` and `test_starting_an_empty_queue_leaves_it_paused`. The existing queue tests now start the queue explicitly.
- **Verified in the GUI with real training:**
  - Two added jobs stayed queued for 4 s with nothing training.
  - Start queue ran them in order, with the monitor following each.
  - After both completed, the queue showed paused/empty.
  - A job added afterwards stayed waiting.
