# Package format v3 Hub release

Published 2026-11-16 (`published_at` 2026-11-16T20:05:22Z), as authorized in the
layer boundary handoff once the compatibility work was verified. In Components → Hub,
click **Refresh** to see the new releases. They need an IDE with Theta SDK 3.1, which is
the code merged to `main` with this release.

## Published

- [Hub commit 570d30c](https://github.com/ThetaIDE/ThetaHub/commit/570d30cda656f32011aec799999097dec5e81d24)
  (fast-forward of `main`, also pushed as `v3-republish`). Release note:
  `releases/native-storage-v3-2026-11-16.md` in the Hub.
- Package sources: [theta-plugins `cf27f57`](https://github.com/ThetaIDE/theta-plugins/commit/cf27f572da4bf098251090802abf361a2ca27904)
  (fast-forward of `main`). Optuna 1.2.0 and Slurm 2.0.0 are built from this repository's
  `optional_components`.
- 34 listings: every maintained package at storage v3, plus Optuna 1.2.0 and Slurm 2.0.0.
  `ai_assistant`, `theta_mcp` and the MIMIC packages (`mimic`, `mimic-env`,
  `mimic-early-prediction`, `reward-shaping`, `mimic-clinical-suite`) left the feed; their
  manifests are in the Hub's `retired/registry/`. The stale Optuna 1.1.0 listing is
  replaced by 1.2.0; the published 1.1.1 archive stays.
- Unpublished archives that only earlier preparations referenced were removed; every
  previously published archive is still in `packages/`.

## Verification

- [Default client feed](https://raw.githubusercontent.com/ThetaIDE/ThetaHub/main/dist/index.json)
  (`DEFAULT_REGISTRY_URL`) is byte-identical to the commit's `dist/index.json`.
- Every one of the 34 listed archives was fetched from its published URL: SHA-256 matches
  the listing, and its `.theta_component.json` has the listed ID and version.
- [Pages deployment](https://github.com/ThetaIDE/ThetaHub/actions/runs/43947918651):
  success. The [Pages feed](https://thetaide.github.io/ThetaHub/index.json) has the same
  34 listings and hashes; only its generated `updated_at` differs.
- Before publishing: `validate_manifest.py --check-archives` passed for every entry, the
  package integration suite passed (70), and an installed IDE wheel installed DNN, Gym
  classic control and SB3 from the release, trained CartPole with `theta` and launched it
  from the `theta-ide` desktop
  ([evidence](COMPATIBILITY_REQUIREMENTS_VERIFICATION_20261116.json)).

Not exercised by these headless checks: Atari/MuJoCo runtime assets, the interactive
DOOM and Antigravity services, and Slurm on a real cluster.
