"""Selected composite model owns stage and policy child providers."""
import copy
import inspect
from .themes_al.models import instantiate_policy, instantiate_stage


class ThemesModels:
    def __init__(self, stage, policy, reward=None, feedback_stage=None, *, resolver):
        self.stage_spec = copy.deepcopy(dict(stage))
        self.policy_spec = copy.deepcopy(dict(policy))
        self.reward_spec = copy.deepcopy(dict(reward)) if reward is not None else None
        self.resolver = resolver
        self.feedback_stage_spec=copy.deepcopy(dict(feedback_stage)) if feedback_stage is not None else None
        if self.feedback_stage_spec is not None:
            instantiate_stage(self.feedback_stage_spec,0,resolver)
        self.stage_name = self.stage_spec.pop('architecture', None)
        if not self.stage_name or not self.policy_spec.get('architecture'):
            raise ValueError('Model bundle must explicitly select both stage and policy architectures')
        if 'seed' in self.stage_spec:
            raise ValueError('Stage seed belongs to the run; duplicate model seed is not allowed')
        self.stage_factory = resolver(self.stage_name)
        if 'params' in self.stage_spec:
            if set(self.stage_spec) != {'params'}:
                raise ValueError('Stage params cannot be mixed with flat overrides')
            inspect.signature(self.stage_factory).bind(seed=0, **self.stage_spec['params'])
            # Validate values through the selected model constructor, before worker launch.
            self.stage_factory(seed=0, **self.stage_spec['params'])
        else:
            # Stage protocol deliberately explicit; reject ignored or misspelled keys.
            required = {'stages', 'sparsity', 'switch_penalty', 'ticc_iterations'}
            if set(self.stage_spec) != required:
                raise ValueError(f'Stage model requires exactly {sorted(required)}; got {sorted(self.stage_spec)}')
            for key in ('stages', 'ticc_iterations'):
                if type(self.stage_spec[key]) is not int or self.stage_spec[key] < 1:
                    raise ValueError(f'{key} must be a positive integer')
            import math
            for key in ('sparsity', 'switch_penalty'):
                if not isinstance(self.stage_spec[key], (int, float)) or not math.isfinite(self.stage_spec[key]) or self.stage_spec[key] < 0:
                    raise ValueError(f'{key} must be finite and nonnegative')
            inspect.signature(self.stage_factory).bind(self.stage_spec['stages'], self.stage_spec['sparsity'],
                self.stage_spec['switch_penalty'], self.stage_spec['ticc_iterations'], 0)
        factory = resolver(self.policy_spec['architecture'])
        inspect.signature(factory).bind(dimension=1, **{k:v for k,v in self.policy_spec.items() if k != 'architecture'})

    def create_stage(self, seed, *, phase='initial'):
        if phase not in {'initial','feedback'}:
            raise ValueError('Stage phase must be initial or feedback')
        if phase=='feedback' and self.feedback_stage_spec is not None:
            return instantiate_stage(self.feedback_stage_spec,seed,self.resolver)
        return instantiate_stage({'architecture':self.stage_name, **self.stage_spec}, seed, self.resolver)

    def create_feedback_stage(self,seed):
        return self.create_stage(seed,phase='feedback')

    def create_policy(self, dimension):
        return instantiate_policy(self.policy_spec, dimension, self.resolver)

    def create_reward(self, states, actions):
        if self.reward_spec is None:
            raise ValueError('Full THEMES requires an explicitly selected reward model')
        spec=copy.deepcopy(self.reward_spec)
        factory=self.resolver(spec.pop('architecture'))
        inspect.signature(factory).bind(states=states,actions=actions,**spec)
        model=factory(states=states,actions=actions,**spec)
        model.theta_model_spec=copy.deepcopy(self.reward_spec)
        return model
