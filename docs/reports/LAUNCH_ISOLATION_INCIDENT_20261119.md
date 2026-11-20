# Launch input isolation incident, 2026-11-19

Adding unrelated LunarLander rules under `src/usr/rules/lunarlander` exposed a
global snapshot check in an already running MountainCar sweep. The old worker
bootstrap compared the entire live rules directory with its launch hash. New
replicates rejected the added files, even though their own MountainCar rules had
not changed. The objective coordinator then scored the failed replicas instead
of propagating failure. This is a launch-isolation and error-propagation defect.

## Affected records

Study `MountainCar/bets_sweep_RouteEpisode_v0`, trials **23–31**, had four failed
replicates per trial and was recorded as `COMPLETE` with invalid zero scores.
Those nine observations must be excluded from analysis. The next method,
FreeSignals, failed at bootstrap, stopping the remaining feature sweeps.
A scan of the BlendRL execution manifests for this launch window found the
resource-snapshot error in these nine trial executions only. Earlier trial
observations and all original study records have been left intact.

The exact affected manifests are listed in the local evidence artifact
`src/usr/results/validation/mountaincar_snapshot_incident.json`. The source log is
`src/usr/results/sweep_logs/MountainCar_bets_sweep_20261119-074205.log`.

## Recovery recipes

Two separate, preflight-validated local recipes preserve the old results:

- `BlendRL/MountainCar/Recovery RouteEpisode 23-31`: replay each affected trial's
  exact effective model parameters using the original four training seeds and
  evaluation seed, without a tuner. This writes replacement measurements in a
  separate experiment; it does not repair the old Optuna study's observations or
  undo the effect those invalid observations had on later TPE suggestions.
- `BlendRL/MountainCar/Recovery Remaining Bets`: run FreeSignals, ValueBets,
  ValueBetsRouted and GPIResidual using the original launch seeds and search spaces
  under new experiment/study identities.

Recovery studies have been prepared, not automatically launched. The first
recipe contains nine methods named Trial23 through Trial31. Its measured scores
must be associated with the original trial parameters when analyzing recovery.

## Implemented correction

New launches capture private copies of configs, provider code/profiles/assets,
declared workspace resources, framework config defaults, provider selections and
extension enablement. Late workers resolve those copies; live workspace changes
apply only to future launches. Symlinks are copied as content. Provider checks
still detect corruption of the captured inputs. Stable provider module names are
retained for checkpoints, and in-process module caches are cleared across launch
boundaries. Live component lifecycle operations are no longer blocked by workers
that already use private copies.

The objective coordinator rejects failed/incomplete replicates before scoring,
so tuning providers receive failure rather than an ordinary numeric objective.
Application code and the Python environment remain external runtime dependencies.
See [the launch contract](../reference/LAUNCHES.md).

Regression coverage includes a fresh worker after original provider deletion,
new component/ruleset additions, edits to rules/configs/provider selection and
enablement, config symlink changes, private-copy corruption, nested launches,
new-launch visibility, and failed-replicate objective rejection. The Box2D setup
also has provider/rule tests and a 12-method training/evaluation smoke recipe.

Verification completed: the focused 107-test regression batch passed, followed by
the additional active-snapshot cleanup test (all four launch tests passed). All
three Box2D provider/rule tests passed. The corrected runtime completed training
and greedy/sampled evaluation for all 12 methods at 4,096 transitions each; its
execution ID is `fdd5aa224af7422397528900624f8ea3`. Local evidence is saved in
`src/usr/results/validation/lunarlander_setup_verified.json`. All six LunarLander
recipes also passed preflight while resolving captured launch inputs.

## Authorized job cleanup, 2026-11-19

Process and queue inspection found no surviving training worker or queued job.
Twenty-eight dead provider leases were moved to the local validation archive;
the two leases for live desktop plugins were retained. A consistent SQLite backup
was made before study metadata cleanup, under
`src/usr/results/validation/job_cleanup_20261119/optuna_before_cleanup.db`.

The original study's nine zero values are preserved and explicitly annotated as
invalid. An analysis copy named
`MountainCar/bets_sweep_RouteEpisode_v0_audited_20261119` preserves its other trials
and represents trials 23–31 as `FAIL`, with no objective values. This copy avoids
treating worker errors as measurements while retaining the original evidence.
The cleanup report is `src/usr/results/validation/job_cleanup_20261119/cleanup.json`.
Historical unfinished studies unrelated to this incident were not changed.
