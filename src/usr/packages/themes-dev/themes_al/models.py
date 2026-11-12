"""Model-owned historical demo architectures; never selected by the learner.

These are not a recovered implementation of upstream network.StudentNetwork.
"""
import copy
import torch
from torch import nn


class DensePolicy(nn.Module):
    def __init__(self, dimension, hidden_sizes, activation, output_scale, actions=2):
        super().__init__()
        if type(dimension) is not int or dimension < 1 or type(actions) is not int or actions != 2:
            raise ValueError('The historical binary evaluator requires positive dimension and actions=2')
        if not hidden_sizes or any(type(n) is not int or n < 1 for n in hidden_sizes):
            raise ValueError('hidden_sizes must contain positive integers')
        if activation not in ('relu', 'tanh'):
            raise ValueError('activation must be relu or tanh')
        if not isinstance(output_scale, (int, float)) or not torch.isfinite(torch.tensor(float(output_scale))) or output_scale <= 0:
            raise ValueError('output_scale must be finite and positive')
        layers = []
        for left, right in zip([dimension, *hidden_sizes[:-1]], hidden_sizes):
            layers.extend([nn.Linear(left, right), nn.ReLU() if activation == 'relu' else nn.Tanh()])
        layers.append(nn.Linear(hidden_sizes[-1], actions))
        self.net = nn.Sequential(*layers)
        self.output_scale = float(output_scale)

    def forward(self, x):
        return self.output_scale * self.net(x)


class Policy(nn.Module):
    def __init__(self, dimension, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 2))

    def forward(self, x):
        return 0.5 * self.net(x)



def historical_policy(dimension):
    """Explicit standalone demo model choice, outside EM/EDM."""
    return instantiate_policy({'architecture': 'themes-dev/dense_policy',
        'hidden_sizes': [32, 32], 'activation': 'relu', 'output_scale': 0.5}, dimension)


def instantiate_policy(spec, dimension, resolver=None):
    spec = copy.deepcopy(dict(spec))
    architecture = spec.pop('architecture', None)
    if not architecture:
        raise ValueError('A policy model architecture must be selected explicitly')
    if 'dimension' in spec:
        raise ValueError('Policy dimension comes from the data contract, not model overrides')
    if resolver is None:
        if architecture != 'themes-dev/dense_policy':
            raise ValueError(f'Model {architecture} requires its selected Theta provider to reload')
        factory = DensePolicy
    else:
        factory = resolver(architecture)
    # Bind explicitly: never filter or discard model parameters.
    import inspect
    inspect.signature(factory).bind(dimension=dimension, **spec)
    model = factory(dimension=dimension, **spec)
    if not isinstance(model, nn.Module):
        raise ValueError('Selected policy provider must return a torch.nn.Module')
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, dimension))
    finally:
        model.train(was_training)
    if output.shape != (2, 2) or not torch.isfinite(output).all():
        raise ValueError('Selected policy must return finite binary-action logits [rows, 2]')
    model.theta_model_spec = {'architecture': architecture, **spec}
    return model


def instantiate_stage(spec, seed, resolver=None):
    spec = copy.deepcopy(dict(spec))
    architecture = spec.pop('architecture')
    if resolver is None:
        if architecture == 'themes-dev/ticc':
            from .core import MultiSeriesTICC
            factory = MultiSeriesTICC
        elif architecture == 'themes-dev/rmt_ticc':
            from .rmt_ticc import RewardTICC
            factory = RewardTICC
        else:
            raise ValueError(f'Stage model {architecture} requires its selected provider to reload')
    else:
        factory = resolver(architecture)
    if 'params' in spec:
        if set(spec) != {'params'}:
            raise ValueError('Conflicting stage parameter forms')
        import inspect
        inspect.signature(factory).bind(seed=seed, **spec['params'])
        model=factory(seed=seed, **spec['params'])
        model.theta_model_spec={'architecture':architecture, **spec}
        return model
    if set(spec) != {'stages', 'sparsity', 'switch_penalty', 'ticc_iterations'}:
        raise ValueError('Stage specification contains missing or unsupported parameters')
    model = factory(spec['stages'], spec['sparsity'], spec['switch_penalty'], spec['ticc_iterations'], seed)
    model.theta_model_spec = {'architecture':architecture, **spec}
    return model
