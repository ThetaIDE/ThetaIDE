# Theta SDK (version 1.0)

Packages import host functionality only from `src.app.api`. That covers components, app plugins and authored user packages. Everything else under `src.app` or `frontend` is private host implementation and may change without notice. The SDK became the supported surface in phase 3 of the [layer boundary plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md). At consolidation it moves to `src.ide.api`, with forwarding modules.

Every export is the canonical host object, so identity checks and pickles behave as before. Exports load on first attribute access: importing a headless SDK module never imports Qt, Optuna or SQLAlchemy. Lookups are not cached, so a host object patched on its defining module is what the next lookup returns.

| Module | Use | Exports |
| --- | --- | --- |
| `src.app.api.workspace` | Selected workspace; guarded edits | `WorkspacePaths`, `implementation_edit` |
| `src.app.api.errors` | Actionable configuration errors (`ValueError`) | `ConfigurationError` |
| `src.app.api.config` | Recipes and editable profiles | `parse_methods_dict`, `ConfigurationFiles` |
| `src.app.api.execution` | Engines, worker processes, interpreters, resource admission | `resolve_engine`, `provider_worker_command`, `PROVIDER_WORKER_MODULE`, `task_worker_main`, `task_worker_command`, `TASK_WORKER_MODULE`, `saved_run_id`, `get_python_executable`, `get_shell_python_cmd`, `get_shell_env_block`, `ProcessOutput`, `kill_process_tree`, `cancellation_requested`, `ResourcePool`, `host_capacity`, `workload_estimate`, `uses_accelerator` |
| `src.app.api.tuning` | [Portable tuner execution](TUNER_EXECUTION_CONTRACT.md), objectives, trial artifacts | `TunerExecutionPlan`, `TunerArtifact`, `TunerLaunchBindings`, `prepare_plan`, `launch_bindings`, `apply_overrides`, `record_plan`, `record_config`, `record_overrides`, `sqlite_snapshot`, `check_artifact_admission`, `request_artifact_snapshot`, `commit_artifact`, `objective_metric`, `objective_value`, `get_sweep_direction`, `trial_log_version`, `trial_checkpoint_dir`, `finish_trial_runs` |
| `src.app.api.sites` | Remote sites for [scheduler components](SCHEDULER_CONTRACT.md) | `profiles`, `site_name`, `load_site`, `defaults_path`, `connection`, `push`, `ssh_arguments`, `rsync_shell`, `safe_path`, `digest`, `read_json`, `write_json`, `record_lock`, `transfer_lock`, `freeze`, `extract`, `receipt_path`, `close_unfinished_runs`, `TERMINAL`, `pin_continuation`, `restore_continuation`, `resolve_policy`, `stage_inputs`, `own_work_root`, `assert_local_scratch`, `Publisher`, `site_cli_command`, `SITE_CLI_MODULE` |
| `src.app.api.desktop` | App plugins and contributed Qt widgets (loads Qt) | `Plugin`, `PluginManifest`, `PluginContext`, `label`, `ComboBox`, `SpinBox`, `DoubleSpinBox`, `svg_icon`, `theme_color`, `current_theme`, `style_tool_button`, `refresh_tool_icons`, `TreeHeaderWidget`, `DraggableToolButton`, `TreeToolbarPersistence` |

Scheduler components own submission, polling and cancellation. The host keeps transport, snapshots, receipts, publication and the user's resource and site policy. Start host processes with the command helpers (`provider_worker_command`, `task_worker_command`, `site_cli_command`) rather than writing host module paths. Use the `run_pipeline.py site ssh SITE` command for interactive login shortcuts. Shortcuts created earlier with `-m src.app.sites.slurm` keep working through the API-1 adapter.

## App plugin host facade

`Plugin.activate(context)` receives a `PluginContext`. Use its services, never attributes of the host window. Everything registered through the context is withdrawn when the plugin is disabled. `context.window` remains for older third-party plugins but is unsupported, and bundled plugins do not use it.

| Services | Use |
| --- | --- |
| `add_sidebar_tab`, `remove_sidebar_tab`, `select_sidebar_tab`, `contribute_to_slot`, `remove_from_slot` | Contribute panels and slot actions |
| `register_command`, `unregister_command` | Launcher commands and hotkeys |
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

`src.app.api.SDK_VERSION` is `1.0`. Minor versions only add exports or context services; a major version may remove them. An app plugin declares the range it was written for as `"sdk": ">=1.0,<2"`, in `plugin.json` or in its capability metadata. Before importing any plugin code, the host checks the range and refuses activation with a logged reason when the range is malformed or unsatisfied. A plugin without a declaration is treated as written for 1.0. All five bundled plugins declare `>=1.0,<2`.

## Enforcement

`tests/test_sdk_boundary.py` parses every Python file in `optional_components/` and `src/plugins/app/bundled/` and rejects private host imports. It covers function-local imports, `__import__`/`import_module` calls and string module paths such as `-m src.app.x`. It also rejects `context.window` and `context._window` access, and checks that the exports are canonical and that the headless modules stay free of Qt and optional libraries. `tests/test_app_plugin_sdk.py` installs an independent app plugin that imports only the SDK. It covers activation, settings, commands, editor routing, appearance listeners, workspace context, cleanup, refusal of an incompatible SDK range before import, and Qt-free headless discovery.

Not yet covered: bundled component packages under `src/plugins/components/bundled` still import `src.app.core.model_registry` and `src.app.objectives` (provider registration APIs), and authored packages are not checked. Phase 7 extends enforcement to every package and to declared package dependencies.
