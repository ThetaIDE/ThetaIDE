# Native provider packages

The framework owns contracts, registries, execution and all four base paradigms.
Bundled providers resolve from the application distribution. Installed packages
resolve from the selected workspace under `src/plugins/components/installed/<id>`
or `src/plugins/app/installed/<id>`. Authored packages live under `src/usr`.
`in/config` holds active profiles, `in/datasets` holds input data, and `results`
holds run output. A custom data directory holds state, never implementations.

Use `THETA_WORKSPACE` and optionally `THETA_DATA_DIR` consistently in desktop,
CLI and workers. `WorkspacePaths` supplies the shared policy. The installer
persists workspace identity so native Python checkpoint namespaces survive a
workspace relocation and remain isolated between independent workspaces.

## Source → archive → installation

Update maintained source packages, build native archives, and install normally.
No workspace migration is required or exposed by the component CLI. Existing
old/user implementations are not rewritten, relocated or replaced automatically.
Create an editable copy when you want to change a managed implementation.

```sh
# From theta-plugins, with the IDE's Python environment:
../Theta-IDE/venv/bin/python scripts/package_plugins.py --output /tmp/theta-packages
# From the IDE, using a chosen workspace:
python -m src.app.components --root /path/to/workspace install /tmp/theta-packages/dnn-1.1.0.zip
python -m src.app.components --root /path/to/workspace inventory
python -m src.app.components --root /path/to/workspace copy dnn my_dnn
python -m src.app.components --root /path/to/workspace export my_dnn --out /tmp/my_dnn.zip
```

Install dependencies first or use the existing dependency installation workflow.
The local builder never synchronizes packages or registry metadata to a Hub.
Publishing and pushing remain separate actions.

## Required declarations

A root `.theta_component.json` explicitly declares `storage_version: 2`,
`api_version: 1`, capabilities, payload, and dependencies. Missing versions and
older formats are rejected before writes. Native packages cannot declare
`target_path`, `legacy_paths`, `module_aliases` or `legacy_archive`.

```json
{
  "id": "example-model", "name": "Example", "kind": "model", "version": "1.0.0",
  "storage_version": 2, "api_version": 1,
  "capabilities": [
    {"id": "example", "kind": "model", "entry_point": "model.py:Model", "aliases": ["basics/example"]}
  ],
  "payload": {
    "implementations": ["__init__.py", "model.py"],
    "resources": ["resources"],
    "configs": {"in/config/model/basics/example.yaml": "configs/model/basics/example.yaml"}
  },
  "dependencies": {"components": [], "pip": []}
}
```

Package IDs are case-sensitive ASCII letters, numbers, underscores and hyphens.
Capability selectors are case-insensitive and qualified by kind. Package IDs,
capability IDs, Hydra profile names and Python module names are distinct.
Declare every supported selector/alias. Unknown slash-qualified selectors are
not retried using their basename. Algorithm variant names retain declared
longest-prefix matching. Models require an exact declared selector/alias.

Executable capabilities use package-relative `file.py:Object` entry points.
Objective capabilities reference a JSON descriptor with an explicit source file.
Algorithm capabilities that record deterministic episode outcomes declare
`episode_outcomes: true`. Objective/replicate planning reads that runner contract;
a historical filename never proves evaluation support.
Config/dependency-only packages use an explicit empty capabilities list.
App capabilities use kind `plugin`; their `plugin.json` retains panel identity,
settings and defaults. The package ID may differ from the panel ID (`doom-sim`
provides `doom`; `obsidian-notes` provides `obsidian`). Bundled privileges come
from the distribution inventory, never `core: true` in a third-party manifest.

Store packaged profiles under `configs/<group>/<profile>.yaml`. Explicit payload
mappings control workspace materialization. Workspace configs override packaged
profiles; framework defaults always supply the four base paradigms. Runtime
listings and build validation read declarations without importing provider code.
Dependencies load when a provider is selected. Preflight verifies selected
providers can actually load; config files, decorators and directories alone
never establish availability. Duplicate package IDs and capability aliases fail
with ownership diagnostics. Authored overrides require an explicit owner in
`provider_overrides.json` in the selected state directory.

## Imports and resources

Use relative imports within a package. Cross-package helpers use explicit owned
files, for example `provider_object('nsfr', 'common.py:get_nsfr_model')` from
`src.app.core.provider_loader`. Resources use `package://ID/path`, `Package.resource`
or `resolve_resource`. `environment_resource` resolves an exact declared
environment capability and its explicit `resource_root`; it never searches
`in/envs` or unrelated directories.

Declare `rules_root` for package-owned rules. Authored rules remain under
`src/usr/rules/<environment>/<name>`. The rules service preserves environment
scoping, symbolic-name overrides and ruleset-owned valuation modules.
`in/rules` remains rejected. The maintained NSFR package owns shared CartPole
rules; NEUMANN depends on NSFR. MIMIC rules belong to the MIMIC experiment package.

Runners own their Hydra/workspace configuration. The generic worker establishes
workspace context, package-relative imports, leases and snapshot verification;
it does not inspect source text or inject config arguments for historical depth.

## Ownership and lifecycle

Install/update/uninstall stage complete native packages and materialize only
explicit config destinations. Protected application, bundled and authored roots
are never installer destinations. Dependencies have independent ownership.
Suites compose dependency packages instead of embedding duplicate providers.
`standard-figures` supplies profiles and depends on the bundled `standard_figures`
implementation; it does not register a second convergence/loss/report provider.

Workspace locks, transaction journals and implementation leases protect active
jobs, loaded app plugins and package/config updates. Failure recovery restores
source, config bytes, modes and ownership records. Edited releases are retained
before managed updates; user config values take precedence over added defaults.
Python dependency installation uses the existing workflow, outside filesystem
rollback. No dependency solver or per-plugin environments are introduced.

Editable copies receive new package/capability identities and provenance, retain
relative imports and owned resources, and export as native packages. Exports use
explicit declarations and never inspect decorators to manufacture entry points
or historical aliases. Loose configs can still be edited; exporting an
implementation requires its native manifest. Explicit authored archive/restore
is a development operation, separate from ordinary installation.

## Retained import support

Automatic legacy discovery, decorator/AST discovery, old archive adaptation,
resource alias lookup and generated module aliases are retired. Old package
manifests in authored roots produce diagnostics and are not providers. Old
`in/envs`, `plot`, frontend plugin and state-directory install roots are ignored.
Bare `usr.*` provider imports are unsupported.

Only the fixed distribution-owned forwarding modules/import map remains for
API-1 saved targets: agent/base contracts, sequential-spec, Optuna, standard
figures and the five moved bundled UI plugins. These preserve canonical object
identity. Native workspace checkpoint namespaces have a separate finder that
only resolves declared native packages. Retire each fixed adapter when provider
API 2 is adopted and its saved-target/checkpoint fixtures and downstream callers
have canonical imports; do not extend this map for maintained packages.

Wheels/sdists include bundled providers, native manifests, runtime assets and
framework defaults, and exclude installed/user payloads, datasets, state,
retained packages, virtual environments and `node_modules`.
