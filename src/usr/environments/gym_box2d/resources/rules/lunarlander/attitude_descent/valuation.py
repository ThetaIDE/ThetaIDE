"""Native LunarLander state: x,y,vx,vy,angle,angular_velocity,leg1,leg2.

Attitude/descent controller deliberately ignores horizontal positioning.
The full-controller control adds the center-seeking target used by Gymnasium's
heuristic (Farama Gymnasium, MIT license).
"""
import torch


def controller_action(state):
    angle_target = torch.zeros_like(state[..., 4])
    hover_target = torch.zeros_like(state[..., 1])
    angle = 0.5 * (angle_target - state[..., 4]) - state[..., 5]
    hover = 0.5 * (hover_target - state[..., 1]) - 0.5 * state[..., 3]
    contact = (state[..., 6] > 0.5) | (state[..., 7] > 0.5)
    angle = torch.where(contact, torch.zeros_like(angle), angle)
    hover = torch.where(contact, -0.5 * state[..., 3], hover)
    main = (hover > angle.abs()) & (hover > 0.05)
    left = (~main) & (angle > 0.05)
    right = (~main) & (angle < -0.05)
    return torch.where(main, 2, torch.where(left, 1, torch.where(right, 3, 0)))


def _truth(agent, action):
    return torch.where(controller_action(agent) == action, 0.99, 0.01)


def idle(agent):
    return _truth(agent, 0)


def correct_left(agent):
    return _truth(agent, 1)


def thrust(agent):
    return _truth(agent, 2)


def correct_right(agent):
    return _truth(agent, 3)
