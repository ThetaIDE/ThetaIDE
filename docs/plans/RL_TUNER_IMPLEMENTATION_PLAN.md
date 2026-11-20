# RL tuner implementation plan

Deliver three milestones in order. First ship a usable local DEHB tuner for
fixed hyperparameters. Then add the runner and SDK infrastructure for changing
parameters during training. Finally, implement one example dynamic tuner using
that infrastructure. SB3 PPO is the first supported runner for all three.

## 1. Implement the DEHB tuner

Add an installable `dehb` tuning component selectable through `tuning.component`.
Use upstream DEHB's `ask` and `tell` interface and translate the existing
`methods.<name>.tune` dimensions into ConfigSpace. Theta launches and cancels
training workers, enforces resource limits, and records results. The component
owns DEHB settings and search state. Translate maximization scores for DEHB's
minimizing interface and retain the original metric in Theta's monitor.
[Upstream DEHB example](https://automl.github.io/DEHB/latest/examples/01.2_Optimizing_RandomForest_using_Ask_Tell/),
[Theta tuner contract](../reference/TUNER_EXECUTION_CONTRACT.md)

Use training transitions as the budget. Expose minimum and maximum transitions
per seed, the reduction factor, a total search cap, and an optimizer seed.
Validate the budget levels against PPO's rollout size before launch. Run each
candidate from scratch at its assigned budget and reuse the launch's tuning seed
panel across candidates. Reserve enough of the cap for each evaluation before
starting it and record actual transitions across all seeds, including any work
from failed evaluations. Record budget level, parameters, status, and objective
alongside the existing study artifacts. Keep failed or incomplete seed panels
out of winner selection.

**Done when** a local PPO recipe runs a DEHB study through the normal launcher,
evaluates candidates at multiple budgets, and stays within its search cap. The
study monitor can show each evaluation's budget and score. It promotes the best
completed full-budget configuration, or explicitly reports that no full-budget
configuration completed. The frozen winner can be retrained on held-out seeds.
Compare with Optuna TPE and random search at equal transition budgets before
recommending DEHB as a default. [ICML tuning guidance](https://proceedings.mlr.press/v202/eimer23a.html),
[launch seeds and records](../reference/LAUNCHES.md),
[shared parameter forms](../reference/CONFIG_PARAMETER_FORMS.md)

## 2. Build dynamic configuration infrastructure

Extend the execution SDK with a versioned capability that lets a runner report
progress at a safe boundary and accept validated parameter changes there. The
algorithm declares the mutable parameter paths, constraints, and when a change
takes effect. The controller receives the completed boundary's step, metrics and
effective parameters. Record each applied value and its effective step so a
saved sequence can be replayed. Keep controller policy in the tuning component
and parameter application in the runner. [Method execution contract](../reference/EXECUTION_CONTRACT.md)

Implement the adapter for SB3 PPO. Apply learning rate and entropy coefficient
after the optimizer finishes one rollout and before the next rollout starts.
Verify that SB3's learning rate schedule and optimizer groups agree. Provide a
backend-owned checkpoint for optional peer transfer at the same boundary. Save
model, optimizer, normalization state where used, and progress counters.
Validate the donor's algorithm, model, spaces and runtime compatibility.
Continue on the recipient's recorded streams with a fresh environment reset and
preserve cumulative progress. Record the reset policy and source checkpoint.

**Done when** one PPO worker can accept a scripted update and use the new values
in the next optimizer update. A fresh launch can replay the recorded schedule.
A second worker can load a compatible donor's training state and apply its own
parameters before continuing. Preflight rejects unsupported fields and values,
and the runner rejects incompatible or incomplete checkpoints.

## 3. Implement an example dynamic tuner

Implement a small synchronous Population Based Training (PBT) component with one
PPO method per population. Train and evaluate members at common transition
milestones on matching evaluation episodes. Rank them on the configured metric.
Before the next round, copy an eligible top member's training state into a
lower-ranked member and apply seeded, bounded perturbations to learning rate or
entropy coefficient. Use Theta's existing resource and cancellation controls.
Save scores, checkpoint ancestry, parameter changes, and actual training cost
for each round. Select the final winner from the last complete round.
[Population Based Training paper](https://arxiv.org/abs/1711.09846)

**Done when** one local recipe shows a completed round, a recorded donor-to-peer
transfer, and a subsequent PPO update using a changed parameter. The saved
history identifies the winner's checkpoint and the schedule along its ancestry.
The example evaluates the checkpoint on fresh episodes and can retrain the
exported schedule on fresh training seeds. A fresh population run with the
controller settings fixed tests the adaptive policy itself.
