"""A fixture multilayer perceptron: the `dnn` model composition tests select."""
import torch


class DNN(torch.nn.Module):
    def __init__(self, num_in_features=4, n_actions=2, hidden_sizes=(64, 64), **kwargs):
        super().__init__()
        sizes = [num_in_features, *hidden_sizes]
        layers = [module for a, b in zip(sizes, sizes[1:]) for module in (torch.nn.Linear(a, b), torch.nn.ReLU())]
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(sizes[-1], n_actions)

    def forward(self, obs):
        return self.head(self.body(obs))
