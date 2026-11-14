# Native provider packages

The framework owns contracts, registries, execution and all four base paradigms.
Bundled providers resolve from the application distribution. Installed packages
resolve from the selected workspace under `src/plugins/components/installed/<id>`
or `src/plugins/app/installed/<id>`. Authored packages live under `src/usr`.
`src/usr/config` holds the user's configurations and explicit overrides of package
profiles, `src/usr/datasets` holds input data, and `results` holds run output. A custom data directory holds state, never implementations.

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

A root `.theta_component.json` explicitly declares `storage_version: 3`,
`api_version: 1`, capabilities, payload, and dependencies. Missing versions and
older formats are rejected before writes. Native packages cannot declare
`target_path`, `legacy_paths`, `module_aliases` or `legacy_archive`.

```json
{
  "id": "example-model", "name": "Example", "kind": "model", "version": "1.0.0",
  "storage_version": 3, "api_version": 1,
  "capabilities": [
    {"id": "example", "kind": "model", "entry_point": "model.py:Model", "aliases": ["basics/example"]}
  ],
  "payload": {
    "implementations": ["__init__.py", "model.py"],
    "resources": ["resources"],
    "configs": {"model/basics/example.yaml": "configs/model/basics/example.yaml"}
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

Store packaged profiles under `configs/<group>/<profile>.yaml` and declare each one in
`payload.configs`, keyed by its `<group>/<profile>.yaml` selector. Profiles resolve
from the package and are never copied into the workspace. Resolution order is a
`src/usr/config` file, then installed and bundled package profiles, then the
framework defaults, which always supply the four base paradigms. Runtime
listings and build validation read declarations without importing provider code.
Dependencies load when a provider is selected. Preflight verifies selected
providers can actually load; config files, decorators and directories alone
never establish availability. Duplicate package IDs and capability aliases fail
with ownership diagnostics. Authored overrides require an explicit owner in
`provider_overrides.json` in the selected state directory.

## Imports and resources

Use relative imports within a package. Resources use `package://ID/path`, `Package.resource`
or `resolve_resource`. `environment_resource` resolves an exact declared
environment capability and its explicit `resource_root`; it never searches
unrelated directories.

Declare `rules_root` for package-owned rules. Authored rules remain under
`src/usr/rules/<environment>/<name>`. The rules service preserves environment
scoping, symbolic-name overrides and ruleset-owned valuation modules. The maintained NSFR package owns shared CartPole
rules; NEUMANN depends on NSFR. MIMIC rules belong to the MIMIC experiment package.

Runners own their Hydra/workspace configuration. The generic worker establishes
workspace context, package-relative imports, leases and snapshot verification;
it does not inspect source text or inject config arguments for historical depth.

### Cross-package exports

A package publishes objects for other packages with a manifest `exports` map. Each
target must be a payload implementation file:

```json
"exports": {"get_nsfr_model": "common.py:get_nsfr_model"},
"payload": {"implementations": ["common.py", "..."], "resources": [], "configs": {}}
```

Consumers declare the dependency (optionally with a PEP 440 version constraint) and load
the export by name:

```json
"dependencies": {"components": [{"id": "nsfr", "version": ">=1.0,<2"}], "pip": []}
```

```python
from src.app.api.packages import provider_export
get_nsfr_model = provider_export('nsfr', 'get_nsfr_model')
```

When called from provider code, the caller's package is identified from its module and
must declare the dependency; a declared version must match the installed one. Lookups
resolve in the workspace that loaded the caller. Undeclared names, missing packages and
version mismatches raise `ImportError` naming both packages. Host code may load any
declared export. Undeclared objects of another package cannot be loaded.

## Ownership and lifecycle

Install/update/uninstall stage complete native packages. They never write to
`src/usr/config`; overrides there survive every update and uninstall. Protected application, bundled and authored roots
are never installer destinations. Dependencies have independent ownership.
Suites compose dependency packages instead of embedding duplicate providers.
Uninstalling a suite leaves its dependency packages installed. Remove dependents
before uninstalling a package they require.
`standard-figures` supplies profiles and depends on the bundled `standard_figures`
implementation; it does not register a second convergence/loss/report provider.

Workspace locks, transaction journals and implementation leases protect active
jobs, loaded app plugins and package/config updates. Failure recovery restores
source, modes and ownership records. A managed package with local edits is copied
to `retained_packages` in the state directory before an update replaces it.
Python dependency installation uses the existing workflow, outside filesystem
rollback. No dependency solver or per-plugin environments are introduced.

Editable copies receive new package/capability identities and provenance, retain
relative imports and owned resources, and export as native packages. Exports use
explicit declarations and never inspect decorators to manufacture entry points
or historical aliases. Loose configs can still be edited; exporting an
implementation requires its native manifest. Explicit authored archive/restore
is a development operation, separate from ordinary installation.

## Imports and checkpoints

Saved Hydra targets and pickled checkpoints name canonical modules. There are no
forwarding modules or import aliases, so a checkpoint that pickled a module path
that has since moved does not load. Native workspace checkpoint namespaces have a
finder that resolves only declared native packages. Decorator or AST discovery and
bare `usr.*` provider imports are unsupported. A manifest that fails v3 validation, in
any root, produces a catalog diagnostic and is not a provider.

Wheels/sdists include bundled providers, native manifests, runtime assets and
framework defaults, and exclude installed/user payloads, datasets, state,
retained packages, virtual environments and `node_modules`.
