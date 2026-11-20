"""Predicates for Gymnasium's complete LunarLander heuristic, in native units.

State: x, y, vx, vy, angle, angular velocity, left contact, right contact.
Includes horizontal position/velocity feedback and the original contact override.
Algorithm adapted from Gymnasium (MIT); see LICENSE.Gymnasium.
"""
import torch


def corrections(state):
    angle_target = (0.5 * state[..., 0] + state[..., 2]).clamp(-0.4, 0.4)
    hover_target = 0.55 * state[..., 0].abs()
    angle = 0.5 * (angle_target - state[..., 4]) - state[..., 5]
    hover = 0.5 * (hover_target - state[..., 1]) - 0.5 * state[..., 3]
    contact = (state[..., 6] > 0.5) | (state[..., 7] > 0.5)
    return (torch.where(contact, torch.zeros_like(angle), angle),
            torch.where(contact, -0.5 * state[..., 3], hover))


def _truth(condition):
    return torch.where(condition, 0.99, 0.01)


def main_needed(agent):
    _, hover = corrections(agent)
    return _truth(hover > 0.05)


def descent_priority(agent):
    angle, hover = corrections(agent)
    return _truth(hover > angle.abs())


def attitude_priority(agent):
    angle, hover = corrections(agent)
    return _truth(~((hover > 0.05) & (hover > angle.abs())))


def rotate_left(agent):
    angle, _ = corrections(agent)
    return _truth(angle > 0.05)


def rotate_right(agent):
    angle, _ = corrections(agent)
    return _truth(angle < -0.05)


def attitude_settled(agent):
    angle, _ = corrections(agent)
    return _truth(angle.abs() <= 0.05)


def controller_action(state):
    """Execute the four clauses directly, for the fixed-controller benchmark."""
    main = (main_needed(state) > .5) & (descent_priority(state) > .5)
    side = attitude_priority(state) > .5
    left = side & (rotate_left(state) > .5)
    right = side & (rotate_right(state) > .5)
    return torch.where(main, 2, torch.where(left, 1, torch.where(right, 3, 0)))
