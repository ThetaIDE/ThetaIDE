# CartPole final sweep and best single run — 2026-11-09

## Current experiments

The Cartpole folder contains two self-contained YAML experiments. All model,
algorithm, and parameter declarations are explicit in each file; neither relies
on an inherited method. The former nested `_base.yaml` has been archived and
removed. The previous inherited layout composed correctly for training but did
not expose the chosen model and parameter rows in the configuration editor.

- **Single Runs:** one enabled method named `BlendRL`, one training seed
  (1,234,679), tuning disabled, no variability panel, and no comparison methods.
  Every agent/model hyperparameter is copied at full precision from the best
  completed historical sweep, `Cartpole Search_20261106-211938`, **trial 57**,
  score **0.9856621875**. The model is explicitly `blendrl`, with the winner's
  `default` rules, neural actor/critic 64x64/Tanh and gate 64x32/Tanh. The budget
  is 49,152 transitions with one environment. This is the current best-parameter
  recipe; replace it with the final sweep winner once that study is complete.
- **Final Sweep:** one enabled `BlendRL` method with explicit `model.name:
  blendrl`, the same exact trial-57 starting settings, and nine explicit search
  dimensions. It uses 64 configurations, four matched training seeds, and
  49,152 transitions per seed. Tuning is enabled and creates a new study.

Both use `sb3-dev/ppo` to record greedy and sampled evaluation curves. That
runner selection adds paired evaluation; the stored PPO hyperparameters are
unchanged from trial 57. This does not claim exact reproduction of the historical
component source: installed model/rule implementations have changed since the
historical study.

## Search dimensions

| Parameter | Range |
|---|---|
| Initial neural LR | log 0.0002–0.003 |
| Initial gate LR | log 0.000075–0.001 |
| Initial logic LR | log 0.0001–0.01 |
| Logic confidence floor | 0.05–0.25 |
| Minibatch | 32, 64 |
| Per-environment rollout | 1,024, 2,048 |
| PPO epochs | integer bounds 6–12 |
| PPO clipping | 0.18–0.30 |
| Gamma | 0.99, 0.995 |

The logic LR upper bound includes trial 57's exact value, 0.0058687294759989074.
Other fixed values now come directly from trial 57, including final neural LR
0.000001884762759224572, GAE lambda 0.9216124317732315, value coefficient
0.5509203415073732, and gradient norm 0.4411213427213416.

## Evidence and objective

The complete historical search has 100 configurations and 400 training runs.
Twenty-three configurations sustained greedy evaluation means >=495 on all four
seeds throughout the final quarter. Trial 57's per-seed sustained checkpoints are
20,480, 10,240, 10,240 and 12,288 transitions. Among the original top 20 trials,
18 used minibatch 32, 17 used rollout 1,024, and 18 used a Tanh gate. These are
correlated adaptive-search results, not isolated causal parameter effects.

The earlier objective mismatch remains relevant: a recent anticipatory-rule run
has greedy return 500 before training, but sampled return starts at 53.2, is
198.1 at 49,152 transitions, and finishes at 389.8 after an overshoot to 65,536.
Greedy reward therefore obscures learning by the sampled policy. That diagnostic
run uses different rules from the historical winner and is not the regenerated
single-run recipe. [SB3's evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
also distinguishes evaluation behavior from training behavior.

The final sweep retains the new `sustained_solve` objective on
`eval/sampled_reward`. It ranks the number of seeds sustaining >=495 throughout
the final quarter first. Its curve-quality tiebreak combines 50% sustained-solve
speed, 30% normalized AUC, and 20% lower-half late return, minus 20% normalized
time-integrated drawdown. Aggregate quality uses equal weights on the seed mean
and the worst-half seed mean. Failures stay in the panel. This changes selection
of hyperparameters; it does not change PPO's environment reward or guarantee
monotonic curves. Evaluation panels change between checkpoints, so observed
drops may also include sensitivity to evaluation starts.

![Recorded return curves](../../results/analysis/cartpole_final_20261109/curves.png)

## Validation and preservation

- Both regenerated files pass pipeline composition and preflight validation.
- Compiled final-sweep commands select `model=blendrl` and contain all nine
  Optuna search expressions. Single-run commands contain no search expressions.
- The actual Qt configuration viewer was loaded with each regenerated YAML.
  Both model selectors show `blendrl`; all nine final-sweep parameter editors
  exist; each experiment contains exactly one method, `BlendRL`.
- Every stored agent/model hyperparameter in Single Runs was compared directly
  with trial 57's objective artifact and matches at full precision.
- Earlier objective work passed 28 tests and a small real PPO runtime check.
  Five separate legacy `fast_stable` tests still have the prior loose-objective
  discovery issue; the new objective has its own registration metadata.
- No full sweep or best-parameter training run has been launched.
- The files immediately before regeneration are preserved under
  `.thetaide/experiment-archives/cartpole-regenerate-20261109/`. The original five
  recipes remain under `.thetaide/experiment-archives/cartpole-20261109/`.
- Read-only analysis and results are under
  `results/analysis/cartpole_final_20261109/`. The regeneration-specific evidence
  is `regenerated_validation.json` and `regenerated_ui_validation.json`; earlier
  resolved config snapshots describe the superseded inherited layout.
