# Local PPO tuning examples

The HPO experiment package provides three Pendulum-v1 recipes. It declares
dependencies on the separately installable DEHB and PBT tuners, `sb3-suite`
with the PPO dynamic adapter, the `gym_classic_control` environment package,
and `dnn`. The tuner packages select runners through the Theta SDK. PBT has
no external Python requirements of its own. These packages require Theta SDK 3.5.

## Install local archives

Use a disposable workspace and the Python environment of a compatible Theta
build. Install the dependencies before the HPO archive. Run the following
commands from the directory containing the preview archives.

```sh
theta component --root "$workspace" install gym_classic_control-1.2.0.zip
theta component --root "$workspace" install dnn-1.2.0.zip
theta component --root "$workspace" install sb3-suite-2.3.0.zip
theta component --root "$workspace" install dehb-0.1.0.zip
theta component --root "$workspace" install pbt-0.1.0.zip
theta component --root "$workspace" install hpo-0.1.0.zip
theta-ide --workspace "$workspace"
```

Set `workspace` to the directory to use for the preview. Choose a recipe in the
HPO group. You can also launch the installed recipes with `theta`.

```sh
THETA_WORKSPACE="$workspace" theta HPO/dehb_pendulum
THETA_WORKSPACE="$workspace" theta HPO/ppo_replay_pendulum
THETA_WORKSPACE="$workspace" theta HPO/pbt_pendulum
```

## Fixed parameters with DEHB

Each evaluation starts a fresh policy for every seed in the launch panel.
The recipe uses 4,096, 8,192, and 16,384 training transitions per seed and caps
the study at 196,608 transitions across its two seeds. Evaluation uses separate
episodes. Each panel reserves its entire budget before launch. If a worker
stops without a progress record, the panel charges that worker's reservation.
The monitor shows the original reward and the per-seed budget. The tuner reads
progress and artifacts from Theta's execution records.

The study archive under `src/usr/results/checkpoints/HPO/dehb_pendulum/` contains
`tuning_summary.json` and, when eligible, `winner.json`. The winner comes from
completed full-budget panels. To retrain it, copy the recipe, remove `tuning`
and `methods.ppo.tune`, and set the frozen values and `total_timesteps`
from `winner.json`. Use held-out training and evaluation seeds.

## Replay and population training

`ppo_replay_pendulum` applies the learning rate and entropy coefficient at
step 4,096. Check `dynamic_schedule.json` for the applied values and the values
observed after the following optimizer update.

`pbt_pendulum` ranks four members across five rounds on matched evaluation
episodes. It assigns the lowest-ranked member the highest-ranked member's
training state and perturbs a mutable parameter within the recipe's bounds.
Continued members reset their environments on a stream derived from their own
seed and the completed milestone. This includes members continuing their own
checkpoints.

The PBT summary records parent trial IDs, transitions, the winner from the last
complete round, its replay schedule, and its native inference checkpoint.
Evaluate that checkpoint on fresh Pendulum episodes with SB3's `PPO.load`
and `evaluate_policy`. For a fresh training run, copy the replay recipe,
substitute `winner_schedule`, and set `total_timesteps` to the final milestone.
For another adaptive run with the same controller settings, launch the PBT
recipe again with a fresh launch seed.

## Compare tuners at matched training budgets

Calibrate PPO before comparing tuners. Calibration evaluates the same
Pendulum episodes under uniformly random actions, untrained PPO, and
PPO after several training budgets. The quality gate requires all three
training seeds to beat both controls by at least 200 return points.

```sh
python scripts/calibrate_pendulum_ppo.py calibration.json --profile short-horizon
```

The original short-budget pilot tested the launcher and produced timing
measurements. Its selected policies overlapped the no-learning controls.
Use a calibration with a qualified budget for learning comparisons.

Use a workspace with compatible PPO, environment, and tuner packages
plus Optuna. The benchmark writes named experiments into that workspace
and copies recipes, launch logs, and measurements into a timestamped
results directory.

```sh
python scripts/benchmark_pendulum_hpo.py --workspace "$workspace" --blocks 2 --calibration calibration.json
```

DEHB, Optuna TPE, and Optuna random search each receive six full-budget
equivalents in training transitions. They tune learning rate and entropy
with the same training seed and evaluation episodes within a block.
DEHB also uses lower fidelities. Each winner retrains on the same two
fresh training seeds. The population comparison starts each PBT member
and fixed PPO counterpart with matching seeds and parameters. It compares
the complete strategies, including PBT's extra evaluations and resets.

The benchmark measures random and untrained policies on the held-out
episode panel. The summary flags outcomes that clear the control floor.
It also separates end-to-end wall time from time reported inside SB3.
Use `--resume <results-directory>` to reuse completed launches.

For a fidelity sensitivity check, run the paired comparison below. It uses
fresh seed panels, compares the 8,192 and 16,384 DEHB minimum budgets to
full-budget random search, and keeps the total search allocation at 196,608
transitions. All three arms use the same learning-rate bounds. The comparison
records when each arm first reaches −500 return at the full budget on its
tuning episodes and checks the selected configuration using separate training
and evaluation seeds. Use one job for a wall-time comparison.

```sh
python scripts/benchmark_pendulum_fidelities.py --workspace "$workspace" --calibration calibration.json
```

Each block is an independent matched search across all three arms. Trials
inside one search belong to the same block. Extend a completed two-block
comparison with `--extend <results-directory> --blocks 10` for twelve
independent blocks at the same cap. This retains the completed measurements
and starts with fresh training, tuning, and holdout seeds.
