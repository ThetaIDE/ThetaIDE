# LunarLander attitude_descent

Native observations, in order: x, y, vx, vy, angle, angular velocity, left contact, right contact. Use `observation_scaling: none`, two duplicated objects, eight features, and action meanings `[noop, fire_left, fire_main, fire_right]`.

The partial controller stabilizes attitude and descent and deliberately ignores horizontal position and velocity. It provides a fallible expert for blending.

Predicates return 0.99/0.01 evidence, with one proposed action at each state. NSFR still learns randomly initialized clause weights; the actual Logic policy is not identical to the raw controller. The algorithm is adapted from Gymnasium’s MIT-licensed heuristic; see LICENSE.Gymnasium.
