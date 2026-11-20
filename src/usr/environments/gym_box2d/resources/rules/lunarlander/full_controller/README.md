# Full LunarLander baseline rules

Implements Gymnasium's complete heuristic: horizontal position and velocity determine a bounded target angle, horizontal displacement determines target hover height, vertical state controls thrust, and leg contact applies the original settling override. No state feedback has been deliberately removed.

The four clauses combine named conditions: main-engine demand and descent priority; attitude priority and a left/right correction or settled attitude. Each predicate evaluates that condition independently, returning 0.99/0.01 truth evidence. Native observation units and `[noop, fire_left, fire_main, fire_right]` action order are required.

`controller_action` executes the clauses directly for a fixed-controller benchmark. Tests compare it with Gymnasium's heuristic. NSFR learns clause weights and BlendRL maps evidence to action probabilities, so the learned Logic expert is evaluated separately from this deterministic baseline. The supplied heuristic is a baseline, not a guarantee of optimal control.

Algorithm adapted from Gymnasium's MIT-licensed heuristic; see LICENSE.Gymnasium.
