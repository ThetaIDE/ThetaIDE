"""A fixture multilayer perceptron: the `dnn` model tests select.

It implements the host's actor-critic and Q-value protocols, so SB3 and monitor tests can
train it without the Hub's dnn package.
"""
import torch

from src.app.api.models import ActionResult


class DNN(torch.nn.Module):
    def __init__(self, n_actions=2, num_in_features=4, hidden_sizes=(64, 64), out_size=None, **kwargs):
        super().__init__()
        self.n_actions = out_size if out_size is not None else n_actions
        sizes = [num_in_features, *(hidden_sizes or (64, 64))]
        layers = [module for a, b in zip(sizes, sizes[1:]) for module in (torch.nn.Linear(a, b), torch.nn.ReLU())]
        self.network = torch.nn.Sequential(*layers)
        self.actor = torch.nn.Linear(sizes[-1], self.n_actions)
        self.critic = torch.nn.Linear(sizes[-1], 1)

    def forward(self, obs):
        return self.actor(self.network(obs))

    def get_action_probs(self, obs):
        return torch.softmax(self.forward(obs), dim=-1)

    def get_q_values(self, obs):
        return self.forward(obs)

    def get_value(self, obs):
        return self.critic(self.network(obs))

    def get_action_and_value(self, obs, action=None):
        features = self.network(obs)
        dist = torch.distributions.Categorical(logits=self.actor(features))
        action = dist.sample() if action is None else action
        return ActionResult(action, dist.log_prob(action), dist.entropy(), self.critic(features))
