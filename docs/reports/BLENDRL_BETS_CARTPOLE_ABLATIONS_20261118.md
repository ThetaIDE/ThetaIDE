# BlendRL Bets: CartPole ablations (2026-11-18)

Phase 4 of the [BlendRL Bets plan](../plans/BLENDRL_BETS_PLAN.md). Experiment
`BlendRL/Variant/Bets/Cartpole Ablations`, run `20261118-104640_46200ae7`:

- 12 methods × 5 training seeds (1234679–1234683), 49,152 transitions each.
- PPO settings of historical trial 57; `num_envs: 1`.
- Evaluation every 2,048 steps: 10 greedy and 10 sampled episodes.

All 60 runs completed without errors, taking about 80 minutes at 4 runs in parallel.
`blendrl_bets` 0.1.0 and sb3-suite 2.2.0 were installed locally.

## Results

The methods are sorted by mean sampled return over the second half of training (24,576
steps onward), with a bootstrap 95% interval over seeds.

| Method | Sampled, final | Sampled 24k+ [95% CI] | Sustained ≥ 475 (median step, seeds) | Policy entropy | Logic gate weight |
| --- | --- | --- | --- | --- | --- |
| DNN | 500 | 494 [486, 500] | 24,576 (5/5) | 0.53 | — |
| ValueBetsRouted | 473 | 489 [470, 500] | 17,408 (4/5) | 0.39 | 0.71 |
| ValueBets | 496 | 480 [456, 496] | 34,816 (5/5) | 0.39 | 0.63 |
| Logic | 491 | 478 [468, 488] | 40,960 (5/5) | 0.30 | 1.00 |
| FreeSignalsRouted | 449 | 473 [444, 496] | 30,720 (3/5) | 0.38 | 0.77 |
| GPIResidual | 450 | 471 [417, 500] | 18,432 (4/5) | 0.34 | 0.87 |
| RouteEpisode | 463 | 467 [427, 492] | 34,816 (3/5) | 0.44 | 0.62 |
| RouteStep | 483 | 467 [440, 489] | 41,984 (4/5) | 0.46 | 0.60 |
| BlendRL | 457 | 465 [435, 492] | 20,480 (3/5) | 0.43 | 0.61 |
| FreeSignals | 466 | 462 [440, 485] | 36,864 (3/5) | 0.38 | 0.70 |
| GPI | 462 | 462 [404, 497] | 26,624 (4/5) | 0.37 | 0.89 |
| BlendEntropy | 446 | 451 [414, 484] | 47,104 (4/5) | 0.45 | 0.55 |

Column definitions:

- **Sustained.** The first evaluation after which every later sampled evaluation is at
  least 475.
- **Policy entropy.** The negative of SB3's `entropy_loss` over the last three updates.
- **Logic gate weight.** `blend/train/logic/weight_mean`, mean of the last three samples.
- **Logic coverage.** `blend/train/logic/coverage` was 1.0 for every blended method except
  RouteStep (0.93).

## Findings

- **The experiment is at its ceiling.** Every blended method scores 500 greedy at step 0:
  the `cartpole/anticipatory_v1` rules solve CartPole before any training. Only the
  sampled return carries signal.
- **No method is distinguishable from vanilla BlendRL.** All confidence intervals overlap.
  The suggestive orderings are value bets with routed returns (ValueBetsRouted: highest
  second-half mean among blends, earliest sustained solve) and GPIResidual (early
  sustained solve). They would need more seeds or a harder task to stand.
- **Why the blends stay below DNN.** The blends' sampled returns sit near the logic expert's
  own level (Logic alone: 478). Their gates put 55–89% of the weight on the logic expert.
  - The logic expert does not abstain: coverage is 1.0.
  - The blends are not more random than DNN: their entropy is lower (0.30–0.46 against
    0.53).
  - So the gap is not the logic expert's 50/50 fallback injecting randomness. It is
    consistent with the gate trusting a slightly suboptimal expert in the states where its
    preferences are wrong.
  - Value-based arbitration raises trust in the logic expert further (GPI 0.89), because
    routed logic episodes return well early in training.

## Next

Move to MountainCar (plan phase 5) after checking that its rules do not also solve it
at initialization. Use `num_envs: 4`, sparser evaluation and a sustained-solve measure
with confidence intervals. On CartPole, a deliberately imperfect ruleset would expose
whether the gate learns to route away from wrong rules.
