# Optuna 1.1.1 Hub release

Published 2026-11-16 at the user's request. In Components → Hub, click **Refresh**,
choose **All Components**, then install **Optuna tuner 1.1.1**. This release uses
the current IDE's native storage-v3 and SDK contracts.

## Published artifacts

- [Hub commit 8f80f5b](https://github.com/ThetaIDE/ThetaHub/commit/8f80f5b6cc1f38b73973f7675aee9bd9854695e0).
- [Default client feed](https://raw.githubusercontent.com/ThetaIDE/ThetaHub/main/dist/index.json),
  [tuner category](https://raw.githubusercontent.com/ThetaIDE/ThetaHub/main/dist/categories/tuner.json),
  and [Pages feed](https://thetaide.github.io/ThetaHub/index.json) verified at 1.1.1.
- [Archive](https://raw.githubusercontent.com/ThetaIDE/ThetaHub/main/packages/optuna-1.1.1.zip):
  24,207 bytes; SHA-256 `7993be5a185e2afe9809b47863bd3e1092a71e351ab85dfee09accac03369703`.
- [Pages deployment](https://github.com/ThetaIDE/ThetaHub/actions/runs/43874442422): success.
- Maintained source: this repository's `optional_components/optuna`, metadata
  commit `a8a33ae` (local). Packages the current `hydra/sweeper/theta_optuna.yaml`,
  public SDK imports and workspace-owned results paths.

The Hub schema/validator now describe tuner listings and storage v3 alongside the
existing published v2 catalog. No IDE runtime compatibility reader was added.
All previous 39 listings remain identical. The user's dirty `../theta-hub` and
`../theta-plugins` checkouts were inspected and left untouched; publication used
a fresh temporary clone. The prepared `-v3` branches and IDE main were not published.

## Verification

All 40 Hub manifests/current archives validated, the new listing passed JSON
Schema validation, and two builds produced identical hashes. Before and after
publication, an isolated real Hub client/view installed via its card's **Install**
button. The production check used the exact default feed and verified the package
hash, manifest, version and profile. Range controls became editable, command
compilation selected the correct sweeper, and the installed provider created and
resumed a persistent two-trial study and read its monitor snapshot. Screenshots
were inspected. No research training, cluster submission or installation into the
user's workspace occurred. Temporary evidence: `/private/tmp/theta-optuna-release/`.

Focused tests in separate processes: 64 SDK-boundary, 37 sweep/tuner-form,
14 Hub-client and 9 tuner-component checks passed (124 total). Four tuner-component
failures match the 5c baseline by ID and fail on unresolved `ppo` fixtures before
tuner execution: `test_ordinary_run_never_resolves_a_tuner`,
`test_ordinary_run_works_when_optuna_imports_are_unavailable`,
`test_other_tuner_owns_compilation_and_lifecycle`, and
`test_independently_installed_tuner_runs_and_produces_plotman_artifacts`.

Before publishing the remaining v3 batch, reconcile its stale Optuna 1.1.0 listing
with production 1.1.1. Preserve this release and never overwrite its archive.
