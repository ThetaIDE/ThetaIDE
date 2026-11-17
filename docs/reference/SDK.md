# Theta SDK (version 3.2)

Packages import host functionality only from `src.app.api`. That covers components, app plugins and authored user packages. Everything else under `src.app` or `frontend` is private host implementation and may change without notice. The SDK became the supported surface in phase 3 of the [layer boundary plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md).

Every export is the canonical host object, so identity checks and pickles behave as before. Exports load on first attribute access: importing a headless SDK module never imports Qt, Optuna or SQLAlchemy. Lookups are not cached, so a host object patched on its defining module is what the next lookup returns.

| Module | Use | Exports |
| --- | --- | --- |
| `src.app.api.workspace` | Selected workspace; guarded edits | `WorkspacePaths`, `implementation_edit` |
| `src.app.api.errors` | Actionable configuration errors (`ValueError`) | `ConfigurationError` |
| `src.app.api.config` | Recipes, editable profiles and [compatibility findings](COMPATIBILITY_REQUIREMENTS.md) | `parse_methods_dict`, `ConfigurationFiles`, `compatibility` |
| `src.app.api.execution` | Algorithm adapters ([execution contract](EXECUTION_CONTRACT.md)), engines, worker processes, interpreters, resource admission | `RunResult`, `TrainingCancelled`, `algorithm_selection`, `execute_backend`, `resolve_engine`, `provider_worker_command`, `PROVIDER_WORKER_MODULE`, `task_worker_main`, `task_worker_command`, `TASK_WORKER_MODULE`, `saved_run_id`, `get_python_executable`, `ProcessOutput`, `kill_process_tree`, `cancellation_requested`, `ResourcePool`, `host_capacity`, `workload_estimate`, `uses_accelerator` |
| `src.app.api.tuning` | [Portable tuner execution](TUNER_EXECUTION_CONTRACT.md), objectives, trial artifacts | `ObjectiveResult`, `RunResult`, `TrialResults`, `TunerExecutionPlan`, `TunerArtifact`, `TunerLaunchBindings`, `prepare_plan`, `launch_bindings`, `apply_overrides`, `record_plan`, `record_config`, `record_overrides`, `sqlite_snapshot`, `check_artifact_admission`, `request_artifact_snapshot`, `commit_artifact`, `objective_metric`, `objective_value`, `trial_log_version`, `trial_checkpoint_dir`, `trial_checkpoint_dir_from_config`, `finish_trial_runs`, `trial_monitor`, `tuning_enabled` |
| `src.app.api.telemetry` | Metrics from training engines outside Lightning | `init`, `Run`, `settings_from_config`, `selection_from_config` |
| `src.app.api.results` | Read-only run results of the selected workspace (`WorkspacePaths.results`); see [result storage](RESULT_STORAGE.md) | `list_runs`, `run_metrics`, `run_manifest`, `read_run_file` |
| `src.app.api.monitoring` | [Mechanistic monitors](MONITOR_CONTRACT.md): monitor providers, and training engines outside Lightning | `MonitorManager`, `preserve_rng`, `Result`, `Context`, `ProbeSet`, `ModelAccess`, `action_difference`, `capture`, `effective_rank`, `layers_for`, `linear_cka`, `replace_units`, `validate_params`, `state_digest`, `write_snapshot`, `validate_visual_data` |
| `src.app.api.sites` | Remote sites for [scheduler components](SCHEDULER_CONTRACT.md) | `profiles`, `site_name`, `load_site`, `defaults_path`, `connection`, `push`, `ssh_arguments`, `rsync_shell`, `safe_path`, `digest`, `read_json`, `write_json`, `record_lock`, `transfer_lock`, `prepare_bundle`, `verify_bundle`, `current_bundle`, `extract_bundle`, `BUNDLE_ARCHIVES`, `BUNDLE_REQUIREMENTS`, `receipt_path`, `close_unfinished_runs`, `TERMINAL`, `pin_continuation`, `restore_continuation`, `resolve_policy`, `stage_inputs`, `own_work_root`, `assert_local_scratch`, `Publisher`, `site_cli_command`, `SITE_CLI_MODULE` |
| `src.app.api.models` | Model providers: registration, construction and the protocols the host detects (loads PyTorch) | `register_model`, `build_model`, `get_model_class`, `ActionResult`, `walk_model_modules`, `DynamicTopologyProtocol`, `ExtraStateProtocol`, `HasModelCallbacks` |
| `src.app.api.agents` | Lightning agents and training components of the base paradigms (loads Lightning) | `OnlineAgentBase`, `OfflineAgentBase`, `register_agent`, `get_agent_class`, `list_registered_agents`, `register_component` |
| `src.app.api.environments` | Environment providers (loads PyTorch) | `VectorizedBaseEnv`, `ActionCount` |
| `src.app.api.data` | Data module providers and offline datasets | `BaseDataModule`, `register_datamodule`, `DatasetReader`, `loader_selection`, `resolve_dataset_for_method` |
| `src.app.api.evaluation` | Evaluator providers | `register_evaluator` |
| `src.app.api.plotting` | Plotter capabilities, shared figure styling and [Discussion figures](FIGURE_CONTRACT.md) | `BasePlotter`, `PanelSpec`, `QUALITATIVE_PALETTE`, `clean_label`, `get_style`, `get_style_info`, `get_method_aliases`, `get_canonical_method_name`, `moving_average`, `deep_update`, `FIGURE_CONTRACT`, `thin_xy` |
| `src.app.api.packages` | [Declared cross-package exports](PLUGIN_STORAGE.md#cross-package-exports), exports a configuration names (`export://<package>/<name>`), environment resources, selected capabilities, [component references and workspace resources](COMPATIBILITY_REQUIREMENTS.md) | `provider_export`, `configured_export`, `environment_resource`, `selected_capability`, `load_selected`, `resource_entries`, `resource_entry` |
| `src.app.api.desktop` | App plugins and contributed Qt widgets (loads Qt) | `Plugin`, `PluginManifest`, `PluginContext`, `label`, `ComboBox`, `SpinBox`, `DoubleSpinBox`, `svg_icon`, `theme_color`, `current_theme`, `style_tool_button`, `refresh_tool_icons`, `TreeHeaderWidget`, `DraggableToolButton`, `TreeToolbarPersistence` |

Scheduler components own submission, polling and cancellation. The host keeps transport, snapshots, receipts, publication and the user's resource and site policy. Start host processes with the command helpers (`provider_worker_command`, `task_worker_command`, `site_cli_command`) rather than writing host module paths. Use the `run_pipeline.py site ssh SITE` command for interactive login shortcuts.

Plotter constructors take the host's workspace and pass it on: `def __init__(self, *, paths): super().__init__("name", paths=paths)`. The host always supplies `paths`; `BasePlotter` has no default workspace.

Discussion passes the selected experiment or stamped run's full result identity to `figure_spec`. See the [figure contract](FIGURE_CONTRACT.md) for run scoping and saved-config compatibility.

## App plugin host facade

`Plugin.activate(context)` receives a `PluginContext`. Use its services, never attributes of the host window. Everything registered through the context is withdrawn when the plugin is disabled. The context does not expose the host window.

| Services | Use |
| --- | --- |
| `add_sidebar_tab`, `remove_sidebar_tab`, `select_sidebar_tab`, `contribute_to_slot`, `remove_from_slot` | Contribute panels and slot actions |
| `register_command`, `unregister_command`, `command_hotkey` | Launcher commands with an application-wide hotkey. `default_hotkey` applies until the user sets `[shortcuts] <command_id>` (a dotted ID nests); the binding follows settings changes, and an empty setting disables it |
| `add_dock_widget`, `remove_dock_widget` | Dock a panel beside the editor area (right side by default). The plugin may show, hide and resize the returned `QDockWidget`; removing it detaches the plugin's widget without deleting it |
| `get_setting`, `set_setting` | Plugin-scoped persistent values |
| `spawn_process`, `track_process`, `parent_widget` | Supervised child processes; a parent for dialogs and Qt objects |
| `workspace_paths`, `get_workspace_dir`, `get_data_dir` | The host's selected workspace and state directory |
| `register_editor`, `unregister_editor`, `activated_explicitly` | Offer a text editor for host file actions. Pass `make_default=context.activated_explicitly`, so restoring saved state never overrides the user's default |
| `editor_preference`, `set_editor_preference`, `ui_zoom`, `add_appearance_listener`, `remove_appearance_listener` | Follow the user's editor, zoom and theme settings |
| `config_catalog`, `notify_config_files_changed`, `is_managed_file` | Read the parsed configuration catalog, report edited config files, and avoid editing installed package files |
| `hub_components`, `add_hub_index_listener`, `remove_hub_index_listener` | Read the Component Hub index (`id`, `name`, `kind`, `tags`, `description`) and follow updates |
| `tree_toolbar_store` | Persistence for `TreeToolbarPersistence` widgets, limited to the shared tree-toolbar settings section |
| `get_current_experiment`, `get_active_config_path`, `add_experiment_change_listener`, `open_settings`, `open_component_hub`, `log`, `show_status_message`, `mark_startup` | Follow the open experiment, open host pages and report to the user |

## Versioning and compatibility checks

`src.app.api.SDK_VERSION` is `3.2`. Minor versions only add exports or context services; a major version may remove them. Version 2.1 added the environment, data, evaluation, ruleset and telemetry modules, the model protocols, `environment_resource`, the trial helpers, docks and bound command hotkeys. Version 2.2 added the algorithm-adapter types (`RunResult`, `TrainingCancelled`), the monitoring module and `selected_capability`. Version 2.3 added what monitor providers and paradigm runners need: `Result`, `Context`, `ProbeSet`, `ModelAccess`, the measurement and snapshot helpers, `algorithm_selection`, `execute_backend`, `loader_selection`, `create_state_loader`, `resolve_dataset_for_method`, `tuning_enabled`, and the providing package's version and digest on a selected capability. Version 2.4 added the read-only results module. Version 3.0 replaced the source snapshot exports `freeze` and `extract` with the [bundle](SCHEDULER_CONTRACT.md#bundles) exports, and `push` takes the experiment to bundle. Version 3.1 added `compatibility`, `load_selected`, `resource_entries` and `resource_entry` for [compatibility requirements](COMPATIBILITY_REQUIREMENTS.md). It removed `create_state_loader` (use `load_selected`), the `rulesets` module (rulesets are NSFR's `resolve_ruleset` export over the `rules` workspace resource) and the Slurm shell helpers `get_shell_env_block` and `get_shell_python_cmd` (now in the Slurm component). Those removals are a deliberate exception to the minor-version rule: they were component-specific, no app plugin used them, and every maintained package moved off them in the same Hub release. Version 3.2 added `FIGURE_CONTRACT` and `thin_xy` for plot components that build [Discussion figures](FIGURE_CONTRACT.md). Backend-specific glue, such as the SB3 post-update monitor hook, belongs to the backend's package, not the SDK. Every app plugin declares the range it was written for, such as `"sdk": ">=3.0,<4"`, in `plugin.json` or in its capability metadata. Before importing any plugin code, the host checks the range and refuses activation with a logged reason when it is missing, malformed or unsatisfied. All four bundled plugins declare `>=3.0,<4`.

## Enforcement

`tests/test_sdk_boundary.py` parses every Python file in `optional_components/`, `src/plugins/components/bundled/` and `src/plugins/app/bundled/` and rejects private host imports. It covers function-local imports, `__import__`/`import_module` calls and string module paths such as `-m src.app.x`. It also rejects `context.window` and `context._window` access, and checks that the exports are canonical and that the headless modules stay free of Qt and optional libraries. `tests/test_app_plugin_sdk.py` installs an independent app plugin that imports only the SDK. It covers activation, settings, commands, editor routing, appearance listeners, workspace context, cleanup, refusal of an incompatible SDK range before import, and Qt-free headless discovery.

The same test checks every package the repository tracks (optional components, bundled packages, the authored packages under `src/usr` and the test fixtures) with `src.app.components.sdk_boundary`:

- `package_errors`: the implementation files (the manifest's `payload.implementations`, without `test_*` files) import no private host module, import no other package's module by its location (`src.usr`, `src.plugins`, `optional_components`; packages share code through declared exports, `provider_export`), and do not reach into the host window.
- `undeclared_imports`: every unconditional third-party import is provided by the application's requirements (with the `gui` extra, for desktop contributions) or the package's `dependencies.pip`. Imports under `try/except ImportError` or an `if` are optional. A module maps to its distribution through the environment's metadata, or to its own name when that distribution is not installed.

It also checks that core, the desktop and the Hydra adapters import no package by path (the provider loader excepted) and name no optional or authored package ID (the registered Hydra sweeper adapter excepted). Behavior tests with differently named providers (a scheduler `lattice-batch`, tuners `lattice` and `fixture-grid`) cover dispatch, forms and bundles.

The installer refuses an archive whose implementation code fails `package_errors`, and the exporter refuses to export one. The maintained Hub package sources are checked when they are built: `theta-plugins/scripts/package_plugins.py` runs both checks on every package. Scaffolded packages pass them.
