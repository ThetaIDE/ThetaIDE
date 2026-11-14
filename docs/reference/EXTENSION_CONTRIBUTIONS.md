# Extension contributions and readiness v1

`src.app.core.contributions` is the supported, headless contract for extension
contributions, workspace enablement, readiness and host-owned lifecycle scopes.
It does not expose private desktop objects or replace the app host facade planned
for the app-plugin SDK migration.

A native capability can declare:

```json
{
  "id": "example", "kind": "tuner", "entry_point": "provider.py:Tuner",
  "contribution_api": 1,
  "readiness": {
    "config": ["tuning.options.example.connection"],
    "distributions": ["example-runtime"]
  },
  "contributions": [
    {"id": "configure", "slot": "configuration.tuning", "availability": "enabled"},
    {"id": "ranges", "slot": "configuration.parameter_sweeps", "availability": "ready"}
  ]
}
```

`ContributionRegistry(paths)` reads manifest and distribution metadata only. It
never imports provider Python or Qt. A status is `absent`, `incompatible`,
`disabled`, `setup` or `ready`, with an actionable reason when appropriate.
Readiness config paths address the host-supplied composed config. Missing/null,
empty-string and empty collection values need setup; explicit zero/false values
are configured. Required distributions are checked through installed metadata,
without importing their modules. This is not a dependency installer or solver.

A contribution with `availability: enabled` is available during setup and when
ready. `ready` (the default) restricts it to configured capabilities with the
required dependencies. Disabled and incompatible extensions contribute nothing.
Capability ownership overrides apply equally to execution and contributions;
ambiguous owners fail instead of choosing scan order. Contribution IDs are unique
within a capability; slots and IDs are independent of package names.

## Factories and lifecycle

Slots are host-defined roles; declaring an unknown slot does not create a host
integration. The host chooses which slots to consume. An optional contribution
`entry_point` names a package-relative Python factory and must be included in the
declared package payload. Validation checks paths/existence without importing code.
`load_contribution(descriptor, context, paths=paths)` explicitly activates the
factory with the focused context supplied by that host slot. The factory's
capability kind does not alter factory loading.

`ContributionScope(mount)` reconciles desired descriptors. `mount(descriptor)`
returns a cleanup callback or `None`. Reconciliation closes removed, disabled,
readiness-changed, version-changed or relocated contributions before mounting
replacements. Unchanged descriptors remain mounted. A failed batch rolls back
its newly mounted contributions, attempting all cleanup callbacks even if one
fails. `close()` releases remaining resources and is idempotent.

Hosts must reconcile after relevant configuration, enablement, package or workspace
changes; this service is not a filesystem watcher. Mounts should return cleanup
for processes, commands, subscriptions and widgets they own. Cleanup errors are
reported instead of silently keeping stale registrations. Use explicit workspace
paths during activation and reconciliation.

## Workspace state and runtime checks

Enablement lives in `<selected state directory>/extensions.json`, schema
`theta.extension-state/v1`. Disabled keys identify origin, package, capability kind
and canonical capability ID. State is workspace-specific; reinstalling the same
owner retains the user's disabled choice. Package files and recipes are unchanged.
Writes use the existing workspace lifecycle lock and reject changes while that
package has an active provider/job lease. Reading state creates no files.

```sh
python run_pipeline.py component --root WORKSPACE extension status tuner example
python run_pipeline.py component --root WORKSPACE extension disable tuner example
python run_pipeline.py component --root WORKSPACE extension enable tuner example
```

CLI status has no selected recipe context, so config-dependent readiness reports
setup until a host supplies composed config. `ContributionRegistry.status(...,
config=cfg)` supports that context. Missing providers are inspectable as `absent`;
changing a missing provider is an error.

Public capability loading rejects disabled or incompatible extensions before
executing their code. `get_tuner(cfg, paths=paths)` additionally requires readiness
for an execution config; requesting a tuner by name for setup/settings can load an
enabled, unconfigured implementation. Remote tuner preparation uses the same check.
Legacy capabilities without contribution declarations retain their existing runtime
contracts and may be enabled/disabled, but do not implicitly gain new host slots.

GUI/config-preview fingerprints include enablement state. Standard source snapshots
also retain `.thetaide/extensions.json`; the selected-bundle/custom external state
cutover remains part of the remote bundle plan.

## Current host integration

The configuration editor consumes `configuration.tuning` and
`configuration.parameter_sweeps` for tuner capabilities. It offers setup only when
an enabled compatible tuner contributes it, and offers range controls only for the
selected ready tuner. Newly authored tuning blocks record an explicit provider and
its required field defaults under `tuning.options.<provider>`. Missing/disabled or
unconfigured providers in saved recipes show diagnostics; saved tuning and search
spaces are preserved. Unavailable ranges remain visible as read-only values.
Core objective scoring remains available when no tuner is installed.

A saved tuning block without a valid `component` shows **Select a tuner** and an
actionable diagnostic. Loading it never selects the first available provider,
imports its implementation or changes saved options/ranges. Existing ranges remain
read-only until the user chooses a ready provider. An explicit selection refreshes
dependent method controls immediately and preserves options for other providers.
New execution rejects unselected tuning, including dashboard-only requests; choose
the intended installed capability in the form or set `tuning.component` in YAML.
No recipe files are rewritten automatically to add a historical default.

Tuner field metadata supports `when`, `required_when`, `optional` and `legacy_flat`.
`legacy_flat` opts a field into reading/removing historical flat tuning overrides;
new writes use namespaced options. The provider declares these compatibility rules.
The shared parameter controls render fields and conditional children; host code does
not branch on a tuner package ID. Readiness-changing edits rebuild dependent controls.
Floating hover controls leave the host tree when their owning rows are destroyed.

The live monitor consumes the host-rendered `monitor.tuning` slot for tuner
capabilities. Declare a contribution with `availability: ready` and no factory
`entry_point`. The host requires ready state for this slot even if a declaration
uses `enabled`. Its shared study dashboard renders the provider-neutral summary
format in [portable tuner execution](TUNER_EXECUTION_CONTRACT.md). Providers can
adopt this slot without importing Qt or host desktop objects.

No tuning dashboard is constructed at startup. A selected recipe or study must
identify an installed, enabled, compatible, ready provider contributing the slot.
Study `component` identifies that provider; older summaries can use the recipe's
explicit `tuning.component`. There is no implicit provider selection for monitors.
Readiness uses the monitored launch's receipt configuration, process telemetry or
saved experiment `config.yaml`, rather than the compact desktop display record or
the recipe currently open in the editor.
Aliases resolve through ordinary capability ownership. Missing, unconfigured,
disabled and undeclared monitor providers show diagnostics while retaining recipe
and result files. Plain telemetry and queue process supervision remain available.

Each live refresh reconciles the contribution scope against current metadata,
workspace and saved configuration; the visible panel's existing clock also
reconciles once per second. Disabling/removal/readiness loss closes stale menus,
disconnects trial navigation, clears and detaches the last dashboard, and returns
to Overview. Workspace changes and package replacement release the old mount.
Closing the panel stops its timers and releases mounts; queued notifications cannot
remount it. Reopening resumes reconciliation. There is no background filesystem
watcher or extension code execution during contribution discovery.

Optuna declares its configuration/range/monitor contributions and dependency requirements.
The independent fixtures in `tests/fixtures/contribution_provider` and
`tests/fixtures/contribution_tuner` exercise generic site/command/setup slots and
non-Optuna configuration and monitor behavior respectively.

This is not the optional-component acceptance gate: Slurm setup, execution commands
and scheduler monitors still need migration to contributions; legacy app-plugin
settings still use their existing hosts. Optuna remains bundled.
Physical extraction, optional installation, full desktop lifecycle wiring and the
versioned app host facade remain outstanding.
