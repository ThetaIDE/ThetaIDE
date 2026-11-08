# Native provider integration checks

Provider verification covers the full source-package workflow: validate declarations,
build archives, install into a disposable workspace, execute selected providers,
and exercise editable copies, export, update and uninstall. Maintained source
packages live in the sibling `theta-plugins` repository. The public package and
ownership contract is documented in [PLUGIN_STORAGE.md](PLUGIN_STORAGE.md).

## Repeatable checks

From the parent directory containing the two repositories:

```sh
cd theta-plugins
../Theta-IDE/venv/bin/python scripts/package_plugins.py --validate-only
../Theta-IDE/venv/bin/python scripts/package_plugins.py --output /tmp/theta-native-dist
../Theta-IDE/venv/bin/python -m pytest tests/test_native_integration.py -q --disable-warnings
```

The integration suite builds and installs every actual archive, then tests lazy
headless listing, all declared imports, resource/rules resolution, workspace
config precedence, independent editable copies, edit/export/reinstall/update,
transaction failure recovery, uninstall/archive/restore, every scaffolding kind,
experiment dependency exports, all four real paradigm workers, checkpoints,
a real Optuna trial/checkpoint promotion, native two-seed objective and replicate
workers, Slurm command construction and Obsidian activation.
Dependencies come from the existing IDE environment; tests do not download them.

The IDE regression command (from `Theta-IDE`) is:

```sh
QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q --disable-warnings tests/test_native_contract.py tests/test_rulesets.py tests/test_experiment_plugin_export_and_install.py tests/test_component_exporter.py tests/test_validation.py tests/test_plotter_registry.py tests/test_provider_app_plugins.py tests/test_provider_organization.py tests/test_plugin_scaffolder.py tests/test_objective_components.py tests/test_tuner_components.py tests/test_slurm.py tests/test_local_components.py tests/test_component_development.py tests/test_agent_registry.py tests/test_datamodule_registry.py tests/test_eval_registry.py tests/test_engine_dispatch.py tests/test_workspace_defaults.py tests/test_sb3_component_installation.py tests/test_variability.py
```

These are targeted integration checks. They do not replace the complete desktop
and scientific test suite. Run tests against disposable workspaces; preserve the
checkout's installed packages, authored implementations and active configs.

## Deliberately retained contracts and limits

The only historical import support is the fixed, distribution-owned API-1
forwarding set: agent registry/base contracts, Optuna, sequential specification,
standard figures and the five bundled frontend plugins. It preserves canonical
object identity for explicitly supported saved targets/checkpoints and internal
callers. There are no per-archive generated module/resource aliases, legacy
layout discovery or automatic checkpoint conversion. Retire each forwarding
module at provider API 2 after its maintained callers and saved fixture targets
use canonical imports and fresh-process checkpoint/import tests pass without it.
Native workspace-specific checkpoint namespaces remain part of the new loader.

Full MIMIC experiments require external clinical data/checkpoints. Atari,
MuJoCo/robotics and DOOM require their own runtime assets/services; headless class
loading does not exercise their complete simulations. Interactive DOOM and
Antigravity browser services and remote Slurm submission were not executed. Short
training runs prove installation/execution/checkpoint paths, not convergence or
scientific result equivalence. GUI tests require a compatible Qt environment. Optional provider tests require
the matching native packages; isolated integration tests install their fixtures
into disposable workspaces.

No dependency solver or environment manager was added. Filesystem transactions
do not undo pip side effects. External editors may bypass runtime leases.
Historical pre-native installed copies in this checkout are intentionally not
made discoverable by automatic fallback; rebuild/install maintained native
archives into the workspace where they are wanted. The local artifact builder
never performs that installation automatically.
