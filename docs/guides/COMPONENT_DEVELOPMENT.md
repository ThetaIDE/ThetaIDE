# Component development from the GUI or terminal

Theta keeps authored research code in `src/usr/`, managed packages in `src/plugins/components/installed/<package-id>`, bundled providers in `src/plugins/components/bundled`, and active configurations in `in/config/`. See [the native storage contract](../reference/PLUGIN_STORAGE.md) for ownership, explicit manifests and package-relative exports.

Author component changes in the `../theta-plugins` source repository. Installed Hub payloads in this workspace are read-only runtime copies, not the component authoring source. Build and install a package to validate a component update; use an isolated workspace first.

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

Existing GUI uninstall/update and visibility operations continue using the shared installer and local collection. Copying, editing, and exporting do not publish anything. Production publication remains a separate operation following `docs/development/THETA_HUB_RELEASE_WORKFLOW.md`.

Uninstalling a Hub component removes its installed source and any obsolete saved Hub-source snapshot. Installing it again downloads the published Hub archive. Hub package source is read-only and updates replace it; edited YAML profiles are preserved separately. Make an independent component copy to develop editable source.

Private component copies retain their saved source for offline reinstall. A retained private snapshot is not an editable live source folder. Delete local component is a separate GUI action that moves owned local files to Trash. The terminal commands above do not currently expose remote Hub install/update/uninstall, Hide, or Delete actions; the shared Python services provide their core operations.

## Current storage map

All paths below are relative to the existing checkout, using the default app data location:

| What | Location |
| --- | --- |
| Authored models, algorithms, data modules and evaluators | `src/usr/models/`, `src/usr/algorithms/`, `src/usr/data/`, `src/usr/eval/` |
| Authored environments and independent plotters | `src/usr/environments/`, `src/usr/plotters/`; owned auxiliary sources stay inside their package |
| Managed research packages | `src/plugins/components/installed/<package-id>/` |
| Symbolic rules and ruleset-owned predicate code | `src/usr/rules/<environment>/<ruleset>/` |
| Experiment recipes and component profiles | `in/config/` by category; local files and protected Hub defaults share this tree |
| Local datasets | `in/datasets/`, or explicitly configured data paths |
| Research outputs | `results/` (logs, plots, checkpoints, exports and related outputs) |
| Managed GUI plugins | `src/plugins/app/installed/<id>/`; custom data directories hold state, not new source installs |
| Built-in GUI plugins | `src/plugins/app/bundled/` |
| Retained component packages, collection state and generated context | `.thetaide/components/packages/`, `collection.json`, `context/` |
| Hub cache and retained YAML/backups | `.thetaide/cache/hub/`, `.thetaide/retained_configs/`, `.thetaide/config_backups/` |
| IDE run records and queue | `.thetaide/runs/` (queue DB at `jobs/jobs.db` beneath the app data directory) |
| Workspace preferences and personal appearance | `.thetaide/settings.toml`, `.thetaide/runs/.appearance.json` by default |
| User-global preferences | `~/.config/thetaide/settings.toml` |

Backing up `.thetaide/` alone misses research packages, authored files, configurations, datasets and results. Package storage is unified by ownership and capability metadata; update maintained source packages, build their native archives and install normally. No workspace migration is needed.

Read-only file modes and IDE write guards protect Hub originals from ordinary edits. The account owning the files can change permissions or replace them; this is not a privileged store or plugin sandbox.

## Logic resources

Dropdown discovery, editor actions and NSFR/NEUMANN use `src.app.services.rulesets`.
The selected environment profile's `symbolic_name`, plus explicit experiment
overrides, determine the rules namespace. A scoped dropdown shows names such as
`default` and `anticipatory_v1`; unscoped component profiles use qualified names
such as `cartpole/default`. The open button opens that ruleset folder. Without an
environment or qualified selector, it opens the rules catalog once.

Each ruleset contains the usual `clauses.txt`, `preds.txt`, `neural_preds.txt`,
`consts.txt`, `bk.txt` and optional `blender_clauses.txt`. A `ruleset.yaml` may set
`format_version: 1` and `valuation: valuation.py`. This loads predicate functions
from a file inside the ruleset; otherwise the environment's valuation module is
used. Copy a ruleset directory under a new name to create editable user rules
without modifying a Hub-owned resource.

Authored rules use `src/usr/rules/`; installed rules remain assets of their owning package. The former `in/rules` location, including a
symlink at that location, is rejected. Package archives and manifests referencing
old rule paths are rejected before modifying the installation. There is no path
translation, discovery fallback or migration alias. Update source packages and
references explicitly before using them. Shared packaged files survive until
their last owner is uninstalled; unrelated user rules are not component-owned
or removed.

Installed source is protected; **Create editable copy** creates a new user ID and provenance. Ordinary lifecycle operations never replace authored code. Environment and plotter authoring uses `src/usr/environments` and `src/usr/plotters`. Exports use `storage_version: 2`; importing an export creates a managed installation. Explicit development archival is separate from uninstall.
