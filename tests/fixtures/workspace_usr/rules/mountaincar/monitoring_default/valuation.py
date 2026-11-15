"""Ground MountainCar predicates on duplicated [position, velocity] objects.

Uses a zero velocity threshold: negative → left, positive → right, zero → noop.
Truth values follow the classic-control 0.99/0.01 convention. Both the agent and
logical environment object hold the same raw observation; there is no scaling.
"""
import torch


def _prob(condition):
    return torch.where(condition, 0.99, 0.01)


def moving_left(agent):
    return _prob(agent[..., 1] < 0)


def moving_right(agent):
    return _prob(agent[..., 1] > 0)


def moving(environment):
    return _prob(environment[..., 1] != 0)


def not_moving(environment):
    return _prob(environment[..., 1] == 0)
