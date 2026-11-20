# Lunar Lander baseline experiments

**Active variant tuning** is under **BlendRL → LunarLander → Bets →
Trial 12 Variants 20261120-082337 → Feature Sweeps 150k**. The eight variants
share vanilla trial 12's PPO settings, 16 environments, and full-controller
logic and gate settings. Each gets 24 trials on the same four seeds, with 20
episodes per mode every 10k transitions through a 150k budget. The coordinator
records each variant winner's difference from trial 12's sampled-return area
score of 0.747908 and prepares the fresh-seed comparison.

The completed **Vanilla Refinement 20261120-010837** sweep used the PPO settings
from DNN trial 45 and searched logic learning rate, gate learning rate, and
logic confidence floor. Trial 12 won after 18 completed and six pruned trials
and improved on the previous matched baseline of 0.723375. Both scores use the
350-ceiling objective on the same tuning panel.

The variant batch under **BlendRL → LunarLander → Bets →
Incumbent Variants 20261120-010433 → Feature Sweeps 150k**
was cancelled to prioritize vanilla refinement. The eight feature searches
in that batch shared the PPO and full-controller settings of
standard BlendRL trial 0. The plan replayed the incumbent comparison's four
training seeds and evaluation seed, with 20 episodes per mode every 10k
transitions and 24 trials per variant. The sampled-return area objective keeps its worst-seed
weighting and uses a return ceiling of 350. Standard BlendRL scores 0.723375
when its completed comparison runs are rescored with this objective.

On a completed variant sweep, the coordinator saves study winners and their
differences from this tuning baseline in the batch's `state.json`. It prepares **Tuned Held-out Ablations**
for standard BlendRL, the eight tuned variants, and fixed FreeSignals. The
held-out recipe draws ten fresh matched training seeds and a fresh evaluation
seed when launched. Earlier recipes and results are retained below.

The current baseline uses **`lunarlander/full_controller`**. The deliberately
partial `attitude_descent` experiment was withdrawn on 2026-11-19 at the user's
request. Its coordinator and all ten descendants were terminated; it must not
supply baseline parameters or automatically launch variant studies. The completed
DNN PPO sweep remains valid and is retained.

## Current experiment recipes

In Theta's Experiments panel, refresh and open **BlendRL → LunarLander → Full Logic**:

- **BlendRL Sweep**: 24 trials over logic learning rate, gate learning rate and
  logic confidence floor. PPO parameters are frozen at DNN trial 11 from
  `LunarLander/staged/20261119-090501/DNN`. Each trial uses the same four tuning
  seeds, 500k transitions, four environments per seed, and up to four concurrent
  seeds. Greedy and sampled evaluation each use ten episodes every 25k transitions.
- **Baselines**: DNN, learned Logic and vanilla BlendRL, ten fresh held-out training
  seeds, 1M transitions per seed. It is prepared, not launched. Apply the completed
  full-logic sweep's selected parameters before confirmatory evaluation.

The 500k sweep was subsequently cancelled at the user’s request after pilot trial 0 showed fast learning. The active recipe is **BlendRL Sweep 150k**, with 24 trials × four seeds × 150k transitions, evaluation every 10k, and pruning warmup of eight checkpoints (80k). The objective integration budget is also 150k. Its new study is **`LunarLander/full_logic_150k_20261119/BlendRL`**; the 500k study and results are preserved. PPO settings, rule code, seeds, search ranges and trial count are unchanged. This is an adaptive tuning-budget choice based on a pilot, not a claim that all configurations converge by 150k. The evidence is `src/usr/results/validation/lunarlander_150k_budget_decision.json`. No feature
sweep or imperfect-logic study follows automatically. The original 12-method
`Ablations` and other ordinary recipes now also select the full-controller rules;
these are available recipes, not automatically launched jobs.

## Complete controller

Code and clauses live in
`src/usr/environments/gym_box2d/resources/rules/lunarlander/full_controller/`.
The algorithm follows Gymnasium's complete discrete LunarLander heuristic, using
native observations in order: x, y, vx, vy, angle, angular velocity, left contact,
right contact. Horizontal position and velocity determine a target angle limited
to ±0.4; horizontal displacement determines target hover height. Angle/angular
velocity determine attitude correction; altitude/vertical velocity determine
thrust correction. Leg contact invokes the original settling override.

```prolog
noop(X):-attitude_priority(A),attitude_settled(A).
fire_left(X):-attitude_priority(A),rotate_left(A).
fire_main(X):-main_needed(A),descent_priority(A).
fire_right(X):-attitude_priority(A),rotate_right(A).
```

`main_needed` tests whether hover correction exceeds 0.05; `descent_priority`
tests whether it exceeds absolute angle correction. `attitude_priority` is the
complement of their conjunction, implementing main-engine priority. Left/right
conditions test angle correction against ±0.05; settled attitude lies between.
Predicate evidence is 0.99/0.01, consistent with the existing NSFR valuation
convention. No state feedback has been intentionally removed.

The fixed controller executes these conditions directly. NSFR learns clause
weights and BlendRL converts evidence into probabilities with a confidence floor
and noop fallback. Consequently the learned Logic expert's return is a separate
measurement; it is not assumed identical to the fixed heuristic.

## Validation and fixed-controller baseline

Four component tests pass, including direct-clause equivalence to Gymnasium's
heuristic on 1,000 generated states and checks that horizontal position and
velocity each affect the selected action. On 50 independent diagnostic episodes
(seed 1570463468), the fixed controller scored **250.13 mean return**, **91.44
standard deviation**, with **47/50 returns ≥200**. This is a descriptive controller
baseline, not a guarantee of optimal behavior or a learned-agent result.

Evidence: `src/usr/results/validation/lunarlander_full_logic_baseline.json`.

```bash
venv/bin/python -m pytest -q src/usr/environments/gym_box2d/tests
venv/bin/python src/usr/environments/gym_box2d/evaluate_rules.py \
  --controller full_controller --episodes 50 --seed 1570463468 \
  --out src/usr/results/validation/lunarlander_full_logic_reproduction.json
```

Keep final evaluation seeds distinct from tuning seeds. The DNN-first staged PPO
selection saves compute but does not claim independently optimal PPO settings for
every architecture. The environment is standard discrete `LunarLander-v3`, with
no reward shaping or modified dynamics; the heuristic is adapted from Gymnasium
under its included MIT license.

## Feature tuning launched after vanilla completion

The vanilla study finished with 17 complete trials, seven legitimate pruned trials,
and one interrupted failed attempt whose configuration was replayed. Trial 0 won
its unchanged sampled-return area objective (0.89669049). Its settings are frozen
for feature tuning: logic_lr 0.004561330581136724, blender_lr
0.00009969254661529107, and logic_confidence_floor 0.07326990869316641.
PPO remains fixed to the DNN winner. No architecture widths, activation functions,
rule semantics or shared base coefficients are searched in this stage.

Refresh Experiments and select **BlendRL → LunarLander → Bets → Feature Sweeps 150k**.
The following eight methods each get 24 trials × the original four tuning seeds ×
150k transitions. Methods run sequentially with four concurrent seed workers.
The worst-case training budget before pruning is 115.2M transitions. The objective,
evaluation frequency (10k), evaluation episode count, and pruning warmup (80k)
are unchanged from vanilla. Each method has a separate study named
`LunarLander/bets_150k_20261119/<method>`.

| Added parameter | Range | Applicable variants |
| --- | --- | --- |
| Gate entropy coefficient | 0–0.05 | BlendEntropy |
| Routing epsilon start | 0.1–1 | RouteStep, RouteEpisode, FreeSignalsRouted, ValueBetsRouted, GPI, GPIResidual |
| Routing epsilon end | 0, 0.05, 0.1 | Same routing variants |
| Routing decay fraction | 0.05–1 | Same routing variants |
| Value-head learning rate | 1e-5–0.003, log | ValueBets, ValueBetsRouted, GPI, GPIResidual |
| Value-head loss coefficient | 0.05–2 | Same value-bet variants |
| Value-bet ensemble size | 1, 3, 5 | Same value-bet variants |
| Pessimism coefficient | 0–2 | Same value-bet variants |
| GPI temperature | 0.1–10, log | GPI, GPIResidual |

Ensemble size is the uncertainty-estimation hyperparameter; each member's layer
widths remain 64×64. With one member, pessimism has no effect because ensemble
spread is zero. Routing modes, confidence-signal sets, arbitration modes and target
types define each named ablation and stay fixed. FreeSignals has no new numeric
hyperparameter under the fixed base and therefore has no redundant sweep.

The coordinator `scripts/run_lunarlander_ablations.py` records its state and logs
under `src/usr/results/sweep_logs/lunarlander_bets_150k_20261119/`. It promotes only
keys declared in each method's feature search, prepares **Tuned Held-out Ablations**
for all 12 methods (DNN, Logic, BlendRL plus nine extensions), and does not launch
held-out training automatically. The held-out recipe draws fresh training and
evaluation seeds. The current vanilla settings, feature ranges and tuning seeds
are recorded in `base_provenance.json`; source templates are retained for promotion.

Both recipes pass preflight. For every feature search, tests of the low and high
range endpoints pass the model's cross-field configuration validator, and each
method's inherited PPO parameters are checked against the completed vanilla recipe.

## Restaged tuning after the environment-count pilot

The 4/8/16-environment pilot improved the 150k sampled-reward mean from 144.9
to 163.4/168.2, but did not reach 200. The previous feature coordinator was
cancelled at the user's request before promotion; all its results remain under
the original paths. Its inherited PPO protocol was chosen for four environments
and a 500k horizon, so its downstream winners must not be mixed with this restart.

`scripts/run_lunarlander_restage.py --start` captures the current full-controller
templates and draws four fresh training seeds plus an evaluation seed for a new
batch. State, captured templates, reference configurations and logs live under
`src/usr/results/sweep_logs/lunarlander_restage/<batch>/`; recipes appear under
**BlendRL → LunarLander → Restaged → <batch>**.

Stages run sequentially, with up to four concurrent training seeds:

1. **DNN Sweep**: 48 trials searching 8/16 environments, rollout length
   (128/256/512/1024 steps per environment), learning rate and its constant/linear
   schedule, minibatch size, epochs, gamma, GAE lambda, entropy and clipping.
   Three queued reference trials include the published RL Zoo LunarLander-v3
   PPO settings (adapted to this budget, fixed 64×64 network and remaining runner
   defaults), and the prior incumbent with 8/16 environments at 2,048 transitions
   per rollout. The coordinator initializes a new study with these pending trials
   and attaches using resume; no old completed observations are reused.
2. **BlendRL Sweep**: 24 trials of vanilla logic/gate parameters, inheriting
   the new DNN winner's environment count and all selected PPO parameters.
3. **Feature Sweeps**: 24 trials for each of the eight existing feature variants,
   inheriting that PPO protocol and the newly selected vanilla BlendRL settings.

Each tuning run has a 150k-transition objective budget, evaluation every 10k with
20 episodes per mode, and the existing sampled-return area objective with
worst-seed weighting. Median pruning starts after 12 completed trials and eight
evaluation checkpoints. Maximum training before pruning is 158.4M transitions
plus normal PPO rollout overshoot. Only a study that ends at its trial limit can
promote parameters; errors or cancellation stop downstream execution.

The coordinator prepares **Held-out Ablations** for all 12 methods with ten fresh
training seeds drawn when launched, a 150k budget and 100 episodes per evaluation
mode every 50k. It never launches held-out training automatically. This remains a
DNN-first shared-PPO comparison, not independently optimal PPO tuning for every
BlendRL variant. Run a new batch only once; inspect its state before retrying.

Reference: [RL Zoo PPO configurations](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml).

## Boundary refinement batch 20261119-194116

The previous DNN winner hit the rollout, minibatch and epoch boundaries. The
previous downstream chain was cancelled and preserved. The replacement runs with
`--start --refine-from src/usr/results/sweep_logs/lunarlander_restage/20261119-142150/state.json`.
It replays the recorded four-seed panel for matched comparisons, retains Theta's
DNN and target_kl=0.05, and keeps the 150k-transition objective budget.

The 48-trial DNN search covers 8/16 environments, 64/128/256 rollout steps per
environment, minibatches 32/64, 10/20/30/40 epochs, and log learning rates
0.0003–0.003. Other PPO parameters are fixed to the preceding DNN winner. Five
queued trials replay that winner and separately probe smaller minibatches,
shorter rollouts, 40 epochs and a 0.0015 learning rate. Fixed PPO values are
promoted alongside searched values so the later stages inherit the whole protocol.

Vanilla BlendRL then searches logic learning rates 0.00001–0.01, blender rates
0.00001–0.001 (both log), and confidence floor 0–0.25. Queued references explicitly
include a zero confidence floor and lower-rate boundary probe. The eight feature
sweeps follow the newly selected DNN and BlendRL settings. Held-out evaluation
remains prepared only, at 150k transitions. No native-SB3 model replaces Theta's DNN.

## Corrected environment boundary: batch 20261119-195426

Batch 20261119-194116 was cancelled and preserved because it omitted the new
environment-count boundary. The replacement searches **8/16/32 environments**,
64/128/256 rollout steps per environment, batch sizes 32/64, 10/20/30/40 epochs,
and log learning rates 0.0003–0.003. All tuning and held-out budgets are 150k.
It starts with explicit 32×64 and 32×128 environment/rollout probes, then the
previous incumbent and the other boundary probes. 32×64 retains the incumbent's
2,048 transitions per rollout; 32×128 retains its per-environment horizon.

The four recorded training seeds and evaluation seed are replayed. Other PPO
settings, including target_kl=0.05, and Theta's DNN remain as in the reference.
There are 48 DNN trials, followed by 24 vanilla BlendRL trials with expanded
lower-rate bounds and explicit zero confidence floor, then the eight feature
studies. Environment count and fixed/searched PPO values propagate together to
every downstream method. Held-out training remains manual.
