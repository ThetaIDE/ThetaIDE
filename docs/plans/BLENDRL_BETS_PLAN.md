# BlendRL Bets: confidence-aware blending and module dropout

A variant of BlendRL in which each expert's own reliability informs the blend, and
early training occasionally hands a whole episode to one expert. Everything is one
integrated component, `blendrl_bets`; every new behavior is off by default, so the
default profile is vanilla BlendRL and each ablation is a configuration change.

Status (2026-11-18): phases 0–3 implemented and tested. The component is
`src/usr/models/blendrl_bets` (copied from blendrl 1.4.1; version 0.1.0), and sb3-suite 2.2.0
(`theta-plugins-v3/methods/sb3`) serves the protocols. Both are installed locally only, not
published. The ablation experiment
`src/usr/config/experiment/BlendRL/Variant/Bets/Cartpole Ablations.yaml` passes
`validate-experiment`. Phase 4 ran it ([report](../reports/BLENDRL_BETS_CARTPOLE_ABLATIONS_20261118.md)).

Phase 5, step 2 (MountainCar) is in progress:

- **Headroom.** The `mountaincar/monitoring_default` rules near-solve the task: the rule policy alone
  scores −113, and the optimum is about −100 to −105. At step 0 the models score worse only because
  NSFR's trainable clause weights start random. Weak weights hand most probability to noop through
  `_map_logic_output`, so on sparse-reward MountainCar a bad start never reaches the flag and never
  recovers (2 of 4 pilot seeds stayed at −200 for 150k–300k). BlendRL's random start is kept as is.
- **Observation scaling.** Plain SB3 PPO on raw MountainCar observations never learned in 300k (0 of 4
  seeds): velocity spans ±0.07 against position's 1.8. Scaling each dimension from its bounds onto
  [−1, 1] learned on 3 of 4 seeds. The setting is `observation_scaling: bounds` in the local dev copy
  `sb3_dev` 0.1.0 (`src/usr/algorithms/sb3_dev`, copied from sb3-suite 2.2.0; not published). It
  keeps velocity's sign and zero, so the rules are unchanged: Logic's step-0 returns match the raw run
  seed for seed.
- **Sweep.** `BlendRL/MountainCar/sweep` tunes for rapid convergence. It was relaunched on
  2026-11-18 as studies `MountainCar/sweep_{DNN,BlendRL}_v3`, run one after the other:
  - DNN and BlendRL studies over the same PPO ranges, including n_steps 32–1024 per environment and
    target_kl.
  - 64 trials × 4 seeds × 120k, num_envs 4, `sb3_dev/ppo` with bounds scaling; evaluation with 10
    episodes every 5k.
  - Median pruning (Optuna 1.3.0, `return_curve` intermediate scores): starts after 16 trials, with a
    warmup of 10 checkpoints (50k).
  - `_v2` used a 300k budget with 5-episode evaluations every 20k. Tuned seeds left −200 by 40k–160k
    and then plateaued, so most of the area was the shared plateau plus evaluation noise. The top
    BlendRL scores lay within 0.04 of each other, about the noise level, and configurations that
    converged twice as slowly scored nearly the same. `_v2` is superseded; `_v0` and `_v1` are
    aborted launches.
  - `_v2` also exposed a resume bug: the seeded TPE sampler restarted on resume and replayed trials
    0–3 as 13–16. Optuna 1.3.1 (theta-plugins `main`) fixes it but is not yet released to the Hub;
    with the installed 1.3.0, resume with a new sampler seed (`tuning.options.optuna.seed`).
  - `_v3` BlendRL stopped after trial 6: its YAML was edited mid-run and Hydra re-composes each
    trial from disk. It was resumed at trial 7 with sampler seed 4686 and the same training and
    evaluation seeds. Launches now compose from a config snapshot ([launches](../reference/LAUNCHES.md)).
  - Results (finished 2026-11-18): DNN best 0.652 (trial 56), median of completed trials 0.348;
    BlendRL best 0.703 (trial 38), median 0.435. Each study had 64 trials: DNN 45 complete and 19
    pruned; BlendRL 44 complete, 20 pruned and 1 failed (the mid-run edit).
    Both settle on batch 32, n_steps 256 and 10 epochs, with learning rates of 1.4–1.8e-3. They
    differ elsewhere, and each study's top five agree on these choices: BlendRL takes a wider clip
    range (0.28 against 0.15), gamma 0.999 against 0.99, a tenth of DNN's entropy coefficient
    (0.0009 against 0.008), a lower max_grad_norm (0.35 against 0.79), and a constant learning rate
    where DNN takes linear decay. BlendRL's best has blender_lr 2.6e-4, logic_lr 6.9e-3 and logic
    confidence floor 0.034. Each best is the top of 64 configurations scored on four seeds, so it is
    likely optimistic; the ablation measures it on fresh seeds.
- **Ablation.** `Variant/Bets/MountainCar Ablations` carries the best v3 settings: DNN uses DNN
  trial 56; Logic, BlendRL and every Bets variant use BlendRL trial 38. Each launch draws 10 fresh training seeds and an evaluation seed, kept
  clear of every recorded seed (including the sweep's), and records them in the execution manifest.
  - Run 2026-11-19 (execution `b25335dd`, 80 of 80 runs complete in 1 h 24 min). The figures below
    use sampled evaluation and matched seeds, with 95% bootstrap intervals; area is normalized so
    that −200 scores 0 and −100 scores 1.
    - BlendRL has the best early area: 0.743 over 0–120k against DNN's 0.652, a difference of
      +0.09 [0.05, 0.13]. By 300k the areas are level (0.809 against 0.796). DNN ends higher:
      −103 against −115, a final-return gap of 11.8 [5.4, 18.5]. The rules give BlendRL its head
      start, and BlendRL then plateaus near Logic's −118.
    - Logic is identical on every seed: −118, leaving −200 by 0–20k.
    - No Bets variant beats vanilla BlendRL on area.
    - ValueBetsRouted matches BlendRL's area and ends 7.7 [1.7, 13.9] higher, at −107, with the
      tightest worst seed (−119). It is the only variant ahead of BlendRL on final return.
    - RouteEpisode is slower early (−0.041 [−0.07, −0.01] area over 0–120k) but ends higher, at
      −107 (interval includes 0).
    - FreeSignals ends lowest (−121).
    - GPIResidual and ValueBets are level with BlendRL.
    - Caveats: the variants use vanilla BlendRL's tuned settings, and 10 seeds across 14
      comparisons leave the borderline intervals (Logic, GPIResidual early area) weak evidence.

Decisions made during implementation:

- **Bet standardization.** Value-head targets are standardized by their running mean and
  standard deviation, not the standard deviation alone. Uncentred return targets produced
  regression losses near 15, whose gradients compete with PPO's under SB3's joint gradient
  clipping; standardized, the loss is below 1.
- **Routing generator.** Routing draws come from a dedicated generator, seeded once from
  torch's global RNG (which SB3 seeds). Drawing from the global stream between action
  samples gave long runs of one expert in practice.
- **Training metrics.** The model reports training-state metrics through an optional
  `training_metrics()` in `training_progress.v1`. They are logged under `diagnostics/blend/…`,
  and auxiliary losses under `losses/train/aux_loss/…`.
- **Phase 5 keys.** `value_heads.checkpoint` and `freeze` are not in the schema yet; phase 5
  adds them, so until then they fail validation as unknown keys.

## Research questions

1. **Module dropout.** Does handing whole episodes to one expert early (decaying
   probability ε) give the logic module undiluted gradients and speed up joint training?
2. **Free confidence signals.** Does the blender improve when it also sees each expert's
   entropy, the logic module's rule strength and the experts' disagreement?
3. **Learned bets.** Do per-expert value heads, trained separately and fed to the
   blender (or used directly as soft generalized policy improvement), beat a blender that
   sees only the observation?
4. **Two stages.** Does a value predictor trained in a first run help a second run?

## What exists today (blendrl 1.4.1)

Facts the design depends on, from `theta-plugins-v3/models/blendrl` and
`theta-plugins-v3/methods/sb3`:

- The neural blender sees only the observation: `to_blender_policy_distribution` feeds
  `neural_state` to `blender_model`. It never sees the experts' outputs.
- The logic module already computes a confidence: `_map_logic_output` takes
  `evidence.max(...)` above `logic_confidence_floor` and mixes the rule distribution
  toward a default with it. That confidence never reaches the blender.
- The critic is `Σ_k w_k · V_k` (neural module critic, plus a logic critic only if the
  environment ships `mlp.py`; otherwise both share the neural critic).
- **SB3 PPO cannot train any auxiliary loss.** `ThetaActorCriticPolicy.evaluate_actions`
  returns only value, log-probability and entropy to SB3's own `PPO.train`.
  `compute_auxiliary_losses` is never called, and `validate_engine_compatibility`
  rejects `blend_ent_coef` for this reason.
- **SB3 cannot replay a per-sample routing decision.** Its rollout buffer stores
  observations and actions only, so `evaluate_actions` cannot know which expert was in
  control of a stored step.
- Rollouts call `policy.forward` → `get_action_and_value`; evaluation calls
  `policy.predict` → `get_distribution` → `get_action_probs`. Routing must apply only to
  rollouts, and that separation will be made explicit rather than relied on.
- The CartPole rules `cartpole/anticipatory_v1` have trivial blender clauses
  (`logic_agent(X) :- true(X)`), so CartPole experiments use the neural blender.

## Design

### 1. Module dropout (`routing`)

At each rollout step (`step`) or at each episode start (`episode`), with probability
ε(t) the step or episode is **routed**: one expert, chosen uniformly, acts alone. Otherwise
the ordinary blend acts. ε decays linearly from `epsilon_start` to `epsilon_end` over
`decay_fraction` of `total_timesteps`, then stays at `epsilon_end`.

- The route is stored with every transition as rollout context (one integer per step:
  −1 for blended, otherwise the expert index).
- When PPO re-evaluates a routed step, it uses that expert alone: log π_k(a|s), entropy
  of π_k, and value V_k. The PPO ratio is then exact, with π_old and π_new both the
  routed policy, and the gradient goes entirely to expert k.
- In `episode` mode a routed episode is controlled by one expert from start to end, so
  its returns are that expert's own returns. That makes them clean training targets for
  the value heads (section 3) and gives a free "standalone competence" curve per expert.
- Evaluation, `get_action_probs` and the reliance monitor never route.

### 2. Free confidence signals (`confidence.signals`)

Detached per-expert features concatenated to the neural blender's input:

| Signal | Per expert | Definition |
| --- | --- | --- |
| `entropy` | yes | Normalized entropy H(π_k)/log\|A\| |
| `rule_strength` | logic only | The existing evidence maximum from `_map_logic_output` (0 = abstaining) |
| `disagreement` | pairwise | Total-variation distance between the expert distributions |
| `value` | yes | Value-head estimate V̂_k(s) (section 3), standardized by its running mean/std |

The blender's first layer is built for observation size plus signal count. All signals
are detached, so an expert can never gain weight by changing its own reported confidence.
Signals require `blender: neural`; a symbolic blender would need these as predicates,
which is a later phase.

### 3. Learned bets: per-expert value heads (`confidence.value_heads`)

A separate small MLP per expert, V̂_k(s) (optionally an ensemble), on the observation.
They never share parameters with the experts or the critic, and they have their own
optimizer group. Two training targets:

- `routed_return`: regress V̂_k on the GAE returns of steps routed to k. This needs
  `routing.mode: episode`. It is unbiased for "how well does k do from here if it is in
  charge", but gets data only while ε > 0.
- `weighted_advantage`: regress Ĉ_k(s) on ρ_k · Â(s,a), with ρ_k = π_k(a|s)/π_b(a|s)
  (bounded by 1/w_k), for every blended step. This estimates E_{a~π_k}[A^{π_b}(s,a)]: how
  much better than the blend one step of expert k would be. It works without routing.
  Logged sanity identity: Σ_k w_k ρ_k = 1.

The loss is `loss_coef · MSE`. With `ensemble_size > 1`, each member trains on its own
bootstrap mask, and the reported bet is `mean − pessimism · std`.

### 4. Arbitration (`confidence.arbitration`)

How the gate logits are formed:

- `learned`: the ordinary BlendRL blender (with any signals as extra inputs).
- `gpi`: no learned blender; logits = bet_k(s)/`temperature`. This is soft generalized
  policy improvement over state values. A per-action argmax over Q_k would need
  per-expert Q-functions, which PPO does not learn.
- `learned_plus_gpi`: blender logits + bet_k(s)/`temperature`. The blender learns a
  residual correction to the value-based arbitration.

`gpi` and `learned_plus_gpi` build value heads even if `value` is not listed in
`signals`.

### 5. Two stages (phase 5)

`value_heads.checkpoint` loads value heads trained in an earlier run; `value_heads.freeze`
stops their updates. This supports the go/no-go check: take a trained `blendrl_bets`
checkpoint with episode routing, freeze both experts, set `arbitration: gpi`, and compare
its evaluation return with the same checkpoint's learned blender. Before adding an
expert-freezing or whole-model initialization key, check whether the pipeline already
owns model initialization from a checkpoint. If it does, use that; settings have one
home.

## Configuration surface

Additions to the model profile `model/blendrl_bets.yaml`. Every existing BlendRL key
keeps its meaning and its single home.

```yaml
blend_ent_coef: 0.0            # existing key, now trainable under SB3 (bonus on gate entropy)
routing:
  mode: none                   # none | step | episode
  epsilon_start: 1.0
  epsilon_end: 0.0
  decay_fraction: 0.25         # of total_timesteps
confidence:
  signals: []                  # subset of [entropy, rule_strength, disagreement, value]
  arbitration: learned         # learned | gpi | learned_plus_gpi
  temperature: 1.0
  value_heads:
    target: weighted_advantage # weighted_advantage | routed_return
    hidden_sizes: [64, 64]
    ensemble_size: 1
    pessimism: 0.0
    loss_coef: 0.5
    lr: null                   # null uses the agent learning rate, as logic_lr does
    # phase 5 adds checkpoint and freeze
```

Manifest `parameter_schema` requirements:

- `routing`, `confidence` and `value_heads` are objects with `additionalProperties: false`,
  so they render as nested parameter blocks with children (per
  [the form conventions](../reference/CONFIG_PARAMETER_FORMS.md)) and unknown keys fail
  validation.
- Every property has a `description`, `enum` for every choice, and `minimum`/`maximum`
  bounds (ε in [0, 1], `decay_fraction` in (0, 1], `temperature` > 0, `ensemble_size` ≥ 1,
  `pessimism` ≥ 0). `signals` is an array whose `items.enum` gives the four names, so the
  form uses the shared `ChoiceListEditor`.
- Cross-field rules are checked in the model constructor with direct messages:
  - `routed_return` requires `routing.mode: episode`.
  - `signals` and `gpi`/`learned_plus_gpi` require `actor_mode: hybrid` and
    `blender: neural`.
  - `rule_strength` requires a logic expert.
  - `routing.mode` other than `none` requires `actor_mode: hybrid`.
- The SB3 path rejects these features when the adapter lacks the protocols (an older
  sb3-suite) or the algorithm is not PPO. Expose them as blocking capability `requires`
  where the `requires` language can state them, as BlendRL already does for
  `symbolic_state`.

## Phases

Each phase ends with its tests passing and a short entry in the component's
`RELEASE_NOTES.md`. Nothing is published to Theta Hub; a later release would follow
[the release workflow](../development/THETA_HUB_RELEASE_WORKFLOW.md).

### Phase 0: copy the latest BlendRL

1. Build blendrl 1.4.1 from `theta-plugins-v3` (its working tree carries uncommitted
   monitoring changes beyond the installed 1.4.0) and install it into this workspace.
   This is a local build only.
2. `python run_pipeline.py component copy blendrl blendrl_bets --name "BlendRL Bets"`.
3. `component inspect blendrl_bets`. Confirm the registered model id, the profile path and
   whether the copied symbolic-state profiles were remapped. The ablation experiment
   selects `blendrl/cartpole_state` from the original package; switch it to the copy's
   selector if the copy owns one.

Acceptance: `Cartpole Ablations` method `BlendRL` (all features off) produces the same
evaluation curve as `blendrl` 1.4.1 on the same seed.

### Phase 1: SB3 adapter protocols (sb3-suite 2.2.0, `theta-plugins-v3/methods/sb3`)

Generic, opt-in, duck-typed model methods. Models without them behave exactly as today.

| Model method | Called by the adapter |
| --- | --- |
| `on_rollout_start(num_timesteps, total_timesteps)` | before each rollout (ε schedule) |
| `on_env_step(dones, infos)` | after each environment step (episode routing, routed-return logging) |
| `training_metrics()` (optional) | at each rollout end; logged as `diagnostics/<name>` |
| `get_action_and_value(obs, action=None, *, phase, context=None)` | `phase="rollout"` when sampling (may return `aux["rollout_context"]`); `phase="update"` with the stored context when PPO re-evaluates |
| `compute_auxiliary_losses(batch, output)` | in each PPO minibatch; `batch` holds observations, actions, old log-probabilities, advantages, returns and context; the returned losses are already weighted and signed, and are summed into the PPO loss and logged as `train/aux/<name>` |

Implementation:

- `ThetaRolloutBuffer` stores a `[n_steps, n_envs, C]` context array and yields it with
  each minibatch.
- `ThetaPPO.collect_rollouts` stores the context.
- `ThetaPPO.train` follows SB3 2.9's PPO loss exactly, plus context and auxiliary losses.
  The SB3 version pin is tightened so the copied loop cannot drift silently.
- A2C and DQN reject models that require these protocols.
- `get_action_and_value` keeps its current signature for `phase`-unaware models.

Tests (`theta-plugins-v3/tests`):

- A model without the protocols gets the same PPO loss as SB3's on a fixed batch.
- Context round-trips from rollout to minibatch, including shuffling.
- Auxiliary losses reach the optimizer and are logged.
- `on_env_step` sees the correct `dones` with several environments.

### Phase 2: routing (module dropout)

`blendrl_bets`:

- ε schedule, per-environment route state, rollout context and the routed
  forward/value/entropy path.
- `blend_ent_coef` is accepted under SB3 and sent through `compute_auxiliary_losses`.
- Monitoring under the existing `blend` namespace:
  - `blend/routing/epsilon`
  - `blend/routing/routed_fraction`
  - `blend/routed_return/<expert>` (episode return of routed episodes)
  - each added to `monitor_catalog.py` with hover descriptions.

Tests (component `tests/`, outside the payload, run explicitly):

- With routing to k, action probabilities, log-probability, entropy and value equal
  expert k's own.
- The schedule endpoints are correct.
- `episode` mode keeps one route through an episode and re-draws it after `done`.
- Evaluation and `get_action_probs` never route.
- `mode: none` matches the vanilla forward pass bit for bit.

### Phase 3: signals, value heads and arbitration

`blendrl_bets`:

- Signal features and the widened blender input.
- Value-head modules, ensemble and bootstrap masks, both targets, their optimizer group
  and the three arbitration modes.
- Monitoring:
  - `blend/bet/<expert>/mean`
  - `blend/bet/<expert>/loss`
  - `blend/bet/<expert>/calibration` (correlation between V̂_k at the start of a routed
    episode and its realized return)
  - `blend/bet/importance_identity` (mean of Σ w_k ρ_k; must stay 1)

Tests:

- Signals are detached: there is no gradient from the gate to the experts through a signal.
- Signal values match hand-computed entropy, rule strength and total-variation distance
  on fixed distributions.
- `weighted_advantage` targets on a constructed batch.
- `routed_return` uses only routed samples.
- With equal bets, `gpi` yields uniform gate weights.
- Every cross-field rule rejects its invalid combination with its message.
- Unknown keys fail schema validation.

### Phase 4: run the CartPole ablations

Run `BlendRL/Variant/Bets/Cartpole Ablations` (5 seeds per method). Compare in the
Discussion tab:

- Sampled and greedy evaluation curves, and the time to sustained solve.
- `blend/routed_return/*`: standalone competence of each expert during training.
- Gate-weight trajectories, to detect collapse.
- Bet calibration.

Report in `docs/reports/`.

### Phase 5: two stages and a harder environment

1. `value_heads.checkpoint`/`freeze`, then the frozen-experts go/no-go above, then a
   stage-2 run initialized from stage-1 heads.
2. Repeat the ablations on MountainCar (existing BlendRL experiments; sparse reward,
   where a logic expert should matter more).
3. Symbolic-blender predicates for signals (`logic_uncertain(X)`, `rule_fires(X)`), so
   the arbitration stays readable as rules.

## Ablation matrix (CartPole)

| Method | Routing | Signals | Value heads | Arbitration | Other |
| --- | --- | --- | --- | --- | --- |
| DNN | — | — | — | — | neural expert alone (reference) |
| Logic | — | — | — | — | `actor_mode: logic` (reference) |
| BlendRL | none | — | — | learned | vanilla baseline |
| BlendEntropy | none | — | — | learned | `blend_ent_coef: 0.01` |
| RouteStep | step | — | — | learned | |
| RouteEpisode | episode | — | — | learned | |
| FreeSignals | none | entropy, rule_strength, disagreement | — | learned | |
| FreeSignalsRouted | episode | entropy, rule_strength, disagreement | — | learned | |
| ValueBets | none | value | weighted_advantage | learned | |
| ValueBetsRouted | episode | value | routed_return | learned | |
| GPI | episode | — | routed_return | gpi | |
| GPIResidual | episode | all four | routed_return | learned_plus_gpi | full method |

Every method uses the same PPO settings (historical CartPole trial 57, as in
`Cartpole/LearnFast`), 49,152 transitions and the same five training seeds.

## Risks

- **Ceiling.** CartPole is solved quickly by the neural expert alone; differences may
  only show in time to solve. Phase 4 measures speed, and phase 5 moves to MountainCar.
- **Copied PPO loop.** `ThetaPPO.train` duplicates SB3's update; the pinned version
  and the equivalence test guard it.
- **Value-head staleness.** `routed_return` heads stop learning once ε reaches
  `epsilon_end`. A small nonzero `epsilon_end` or `weighted_advantage` keeps them current.
- **The gate learns only from blended steps.** A routed step trains one expert and gives
  the gate no gradient, so while ε is near 1 the blender barely moves. `decay_fraction`
  trades expert warm-up against gate training.
- **Routed collapse of an untrained logic expert.** A routed episode with poor rules
  ends quickly in CartPole. This is informative, but it lowers early evaluation.
  Evaluation never routes, so the reported curves stay blend-only.
