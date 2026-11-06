# Component development from the GUI or terminal

Theta keeps its current checkout layout: active research code in `src/usr/`, configurations in `in/config/`, and existing environment, rule, plot, and output directories. This feature does not move user files or introduce a separate workspace lifecycle.

The desktop and terminal share headless component services in `src/app/components/`, the path policy in `src/app/workspace.py`, and existing services in `src/app/services/`. Installer/model imports under `frontend.hub` remain compatibility aliases to the same classes.

## Inspect and copy

Run these commands in the Python environment used by Theta:

```bash
python run_pipeline.py component list
python run_pipeline.py component inspect blendrl
python run_pipeline.py component copy blendrl my_blendrl --name "My BlendRL"
python run_pipeline.py component inspect my_blendrl
python run_pipeline.py component context my_blendrl
```

Successful operations emit JSON on stdout. Handled operation failures end with a JSON error on stderr and return exit status 2; libraries may also emit diagnostic lines on stderr. Invalid command syntax uses normal argparse help/errors. Inspection reports installed component identity, active source, manifest, profiles/selectors, dependencies, parent provenance, a digest of Python source files, protection, and literal registration locations. It parses Python without executing component code. `list` lists installed manifests, not every remote or retained uninstalled package.

Independent copying assigns a new runtime identity, remaps supported imports/selectors/schema paths, preserves owned profiles/assets, and leaves shared dependencies as references. The copy is immediately editable in the existing active research directories. The GUI's Deep Copy calls the same operation. Copying currently supports directory-based models with safely transformable registrations/imports. Other kinds and unsupported layouts fail clearly; arbitrary Python refactoring is not attempted.

Edit the active files under `src/usr/models/my_blendrl/` with an editor or coding agent. Fresh experiment processes load saved changes. Packages retained under `.thetaide/components/packages/` are snapshots, not synchronized working copies. Generated context lives under `.thetaide/components/context/`, outside protected Hub payloads.

## Configure

Use a profile path returned by `copy` or reported by `inspect`; names vary by package. These examples assume `model/my_blendrl/default`:

```bash
python run_pipeline.py component config show model/my_blendrl/default
python run_pipeline.py component config set model/my_blendrl/default hidden_sizes='[128,128]'
python run_pipeline.py component config validate model/my_blendrl/default
python run_pipeline.py component config copy model/my_blendrl/default model/my_blendrl/wide
```

`config set` applies dotted `KEY=YAML` assignments and serializes the mapping while retaining opening comments, including Hydra package directives. Use a text editor to retain internal formatting/comments. Hydra command-line operators such as `+` and `~` belong on experiment invocations, not on `config set` keys.

A YAML copy still references shared code; a component copy creates an independent implementation. Copy operations refuse existing destinations. CLI config paths accept config-root-relative names, `in/config/...` paths, or absolute YAML paths within `in/config/`; directory and symlink escapes are rejected.

Hub originals are read-only in shared editing operations and retain filesystem protection. Local component/profile copies are editable. GUI config saves and Copy YAML call the same file/protection services. System Trash is injected by the desktop into the headless deletion service; core component operations never import Qt.

## Validate and run

Select the copied profile in an ordinary experiment YAML, through the GUI, or with an override:

```bash
python run_pipeline.py component validate-experiment benchmarks/cartpole \
  ++methods.ppo.model=my_blendrl/default
python run_pipeline.py benchmarks/cartpole \
  ++methods.ppo.model=my_blendrl/default dry_run=true
python run_pipeline.py benchmarks/cartpole \
  ++methods.ppo.model=my_blendrl/default
```

Replace the example experiment and method with existing ones. `validate-experiment` uses the existing pipeline composition/preflight and reports effective method model parameters and implementation sources after launch overrides. It starts no training. Profile validation checks YAML, static model registration/identity, and opted-in JSON Schema (`x-theta-validate`). Static preflight does not test forward-pass shapes or arbitrary runtime behavior; use component tests and a fresh-process run as well. Dynamic registration layouts may require manual runtime verification.

For scripts/tests, `--root` selects an existing checkout layout without creating or migrating a project. Place it before the command:

```bash
python /path/to/Theta-IDE/run_pipeline.py component --root /path/to/existing-checkout inspect my_blendrl
```

`python -m src.app.components` is the equivalent headless module entry point. Existing experiment launch syntax is unchanged.

## Export and lifecycle

```bash
python run_pipeline.py component export my_blendrl --out my_blendrl.zip
```

Existing GUI uninstall/update and visibility operations continue using the shared installer and local collection. Copying, editing, and exporting do not publish anything. Production publication remains a separate operation following `THETA_HUB_RELEASE_WORKFLOW.md`.

Uninstall removes the active installation, including active local source, after retaining saved content for reinstall. A retained snapshot is not an editable live source folder. Delete local component is a separate GUI action that moves owned local files to Trash. The terminal commands above do not currently expose remote Hub install/update/uninstall, Hide, or Delete actions; the shared Python services provide their core operations.

## Current storage map

All paths below are relative to the existing checkout, using the default app data location:

| What | Location |
| --- | --- |
| Active models, algorithms, data modules and evaluators | `src/usr/models/`, `src/usr/algorithms/`, `src/usr/data/`, `src/usr/eval/` (legacy methods may use `src/usr/methods/`) |
| Active environment, symbolic and plotting assets | `in/envs/`, `in/rules/`, `plot/`, plus package-declared assets |
| Experiment recipes and component profiles | `in/config/` by category; local files and protected Hub defaults share this tree |
| Local datasets | `in/datasets/`, or explicitly configured data paths |
| Research outputs | `results/` (logs, plots, checkpoints, exports and related outputs) |
| Hub GUI plugins | `.thetaide/plugins/<id>/` unless their install target/data directory specifies otherwise |
| Built-in GUI plugins | `frontend/plugins/core/` |
| Retained component packages, collection state and generated context | `.thetaide/components/packages/`, `collection.json`, `context/` |
| Hub cache and retained YAML/backups | `.thetaide/cache/hub/`, `.thetaide/retained_configs/`, `.thetaide/config_backups/` |
| IDE run records and queue | `.thetaide/runs/` (queue DB at `jobs/jobs.db` beneath the app data directory) |
| Workspace preferences and personal appearance | `.thetaide/settings.toml`, `.thetaide/runs/.appearance.json` by default |
| User-global preferences | `~/.config/thetaide/settings.toml` |

Backing up `.thetaide/` alone misses active research files, configurations, datasets and results. No unified package store, standalone research-workspace lifecycle, or application namespace move is implemented. `PROJECT_ORGANIZATION_PLAN.md` describes deferred design options.

Read-only file modes and IDE write guards protect Hub originals from ordinary edits. The account owning the files can change permissions or replace them; this is not a privileged store or plugin sandbox.
