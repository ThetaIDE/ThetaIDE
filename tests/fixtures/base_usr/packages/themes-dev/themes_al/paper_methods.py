"""Explicit paper/source ports, independent of Theta and of model architectures.

EDM: Jarrett et al., NeurIPS 2020, joint conditional/state-density objective.
MLIRL: available NV_MLIRL.py uses ordinary optimal Q (not soft value iteration).
Differences and unavailable upstream implementations are recorded in SOURCE_AUDIT.
"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import logsumexp
from scipy.optimize import minimize


def edm_objective(positive_logits, negative_logits, actions, *, density_weight, energy_regularizer, density_regularizer=0.):
    positive_energy = -torch.logsumexp(positive_logits, dim=1)
    negative_energy = -torch.logsumexp(negative_logits, dim=1)
    ce = F.cross_entropy(positive_logits, actions)
    density = positive_energy.mean() - negative_energy.mean()
    regularizer = energy_regularizer * (positive_energy.square().mean() + negative_energy.square().mean())
    return ce + density_weight * density + density_regularizer * density.square() + regularizer, ce, density


def validate_edm_parameters(*, updates, batch_size, learning_rate, adam_betas, buffer_size, sgld_steps, sgld_step_size, sgld_noise, reinit_probability, density_weight, energy_regularizer, density_regularizer=0., demonstration_sampling="uniform_rows"):
    import math
    if demonstration_sampling != 'uniform_rows':
        raise ValueError("Only explicit uniform_rows sampling is implemented; source balanced-buffer semantics are unavailable")
    integers = (updates, batch_size, buffer_size, sgld_steps)
    if any(type(n) is not int or n < 1 for n in integers):
        raise ValueError('Training counts must be positive integers')
    if any(not math.isfinite(v) or v < 0 for v in
           (learning_rate, sgld_step_size, sgld_noise, density_weight, energy_regularizer, density_regularizer)) or learning_rate == 0:
        raise ValueError('Training rates must be finite and nonnegative; learning rate must be positive')
    if not 0 <= reinit_probability <= 1 or len(adam_betas) != 2 or any(not 0 <= b < 1 for b in adam_betas):
        raise ValueError('Invalid reinitialization probability or Adam betas')


def train_edm(model, states, actions, *, updates, batch_size, learning_rate, adam_betas,
              buffer_size, sgld_steps, sgld_step_size, sgld_noise, reinit_probability,
              density_weight, energy_regularizer, seed, observer=None, density_regularizer=0., demonstration_sampling="uniform_rows"):
    """Persistent SGLD + joint EDM objective. The caller supplies the selected model.

    SGLD is unbounded; unlike the historical demo this does not clamp particles.
    Uniform [-1,1] is the explicit initialization law, not an input normalization.
    """
    validate_edm_parameters(updates=updates,batch_size=batch_size,learning_rate=learning_rate,
        adam_betas=adam_betas,buffer_size=buffer_size,sgld_steps=sgld_steps,
        sgld_step_size=sgld_step_size,sgld_noise=sgld_noise,reinit_probability=reinit_probability,
        density_weight=density_weight,energy_regularizer=energy_regularizer,density_regularizer=density_regularizer,demonstration_sampling=demonstration_sampling)
    x = torch.as_tensor(states, dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.long)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x) or not torch.isfinite(x).all():
        raise ValueError('Expected finite, aligned demonstration states and actions')
    if not np.isin(np.asarray(actions), [0, 1]).all():
        raise ValueError('Binary action contract required')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    def uniform(n):
        return 2 * torch.rand((n, x.shape[1]), generator=rng) - 1
    buffer = uniform(buffer_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=tuple(adam_betas))
    history = []
    model.train()
    for update in range(updates):
        ix = torch.randint(len(x), (batch_size,), generator=rng)
        bi = torch.randint(buffer_size, (batch_size,), generator=rng)
        negative = buffer[bi].clone()
        reset = torch.rand(len(bi), generator=rng) < reinit_probability
        negative[reset] = uniform(int(reset.sum()))
        for _ in range(sgld_steps):
            negative = negative.detach().requires_grad_(True)
            energy = -torch.logsumexp(model(negative), dim=1)
            gradient = torch.autograd.grad(energy.sum(), negative)[0]
            noise = torch.randn(negative.shape, generator=rng)
            negative = negative - sgld_step_size * gradient + sgld_noise * noise
        negative = negative.detach()
        if not torch.isfinite(negative).all():
            raise FloatingPointError('Nonfinite SGLD particles')
        buffer[bi] = negative
        loss, ce, density = edm_objective(model(x[ix]), model(negative), y[ix],
            density_weight=density_weight, energy_regularizer=energy_regularizer, density_regularizer=density_regularizer)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite EDM objective')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        record = dict(update=update+1, loss=float(loss.detach()), cross_entropy=float(ce.detach()),
            density=float(density.detach()), density_penalty=float(density_regularizer*density.detach().square()),
            sgld_gradient_norm=float(gradient.detach().norm(dim=1).mean()))
        history.append(record)
        if observer is not None:
            observer('edm', record)
    model.eval()
    return dict(buffer=buffer.detach(), history=history)


def discrete_dynamics(paths, states, actions):
    """Source GenerateDiscreteSAP convention: empirical transitions, zero unseen rows.

    Paths consist of stage/policy pairs, one path per visit. No intervisit edges.
    P0 is occupancy frequency, as in the source, rather than initial-state frequency.
    """
    transitions = np.zeros((states, actions, states))
    occupancy = np.zeros(states)
    for path in paths:
        for s, a in path:
            if not 0 <= s < states or not 0 <= a < actions:
                raise ValueError('Discrete component id outside selected model dimensions')
            occupancy[s] += 1
        for (s, a), (next_state, _) in zip(path[:-1], path[1:]):
            transitions[s, a, next_state] += 1
    totals = transitions.sum(axis=2, keepdims=True)
    np.divide(transitions, totals, out=transitions, where=totals != 0)
    if not occupancy.sum():
        raise ValueError('Discrete demonstrations must be nonempty')
    return transitions, occupancy / occupancy.sum()


def optimal_q(transitions, rewards, *, discount, tolerance, max_iterations):
    """Ordinary Bellman optimality, matching active vi() path in NV_MLIRL.py.

    Unobserved transition rows remain zero and have zero Q, including reward,
    matching a transition-conditioned reward expectation in the original source.
    """
    p = np.asarray(transitions, dtype=float)
    r = np.asarray(rewards, dtype=float)
    if p.ndim != 3 or p.shape[0] != p.shape[2] or r.shape != p.shape[:2] or not np.isfinite(p).all() or (p < 0).any() or not np.isfinite(r).all():
        raise ValueError('Invalid discrete transition/reward arrays')
    sums = p.sum(axis=2)
    if not (np.isclose(sums, 0) | np.isclose(sums, 1)).all():
        raise ValueError('Transition rows must sum to zero or one')
    if not 0 <= discount < 1 or not np.isfinite(tolerance) or tolerance <= 0 or type(max_iterations) is not int or max_iterations < 1:
        raise ValueError('Invalid Bellman convergence settings')
    q = np.zeros_like(r)
    for _ in range(max_iterations):
        new = sums * r + discount * np.einsum('sak,k->sa', p, q.max(axis=1))
        if np.max(np.abs(new-q)) < tolerance:
            return new
        q = new
    raise RuntimeError('Bellman iteration did not converge within max_iterations')


def optimal_q_gradient(transitions,rewards,*,discount,tolerance,max_iterations):
    """Piecewise derivative through the greedy-policy Bellman fixed point.

    As in source q_grad_fpi, select one greedy policy at action ties. This is
    not a soft Bellman operator and need not be differentiable at every tie.
    """
    p=np.asarray(transitions,dtype=float);r=np.asarray(rewards,dtype=float)
    q=optimal_q(p,r,discount=discount,tolerance=tolerance,max_iterations=max_iterations)
    states,actions=r.shape;mass=p.sum(axis=2);greedy=q.argmax(axis=1)
    direct=mass[:,:,None]*np.eye(states*actions).reshape(states,actions,-1)
    dv=np.linalg.solve(np.eye(states)-discount*p[np.arange(states),greedy],
                       direct[np.arange(states),greedy])
    dq=direct+discount*np.einsum('sak,kd->sad',p,dv)
    return q,dq


def irl_objective_gradient(rewards,transitions,paths,*,discount,beta,tolerance,value_iterations):
    """Source weighting: mean path NLL divided once more by path count."""
    from collections import Counter
    q,dq=optimal_q_gradient(transitions,rewards,discount=discount,
        tolerance=tolerance,max_iterations=value_iterations)
    logits=beta*q;log_probs=logits-logsumexp(logits,axis=1,keepdims=True)
    weights=np.zeros(q.shape)
    for path in paths:
        for (s,a),count in Counter(path[:-1]).items():weights[s,a]+=count/(len(paths)**2)
    probabilities=np.exp(log_probs)
    coefficients=weights.sum(axis=1,keepdims=True)*probabilities-weights
    return float(-np.sum(weights*log_probs)),beta*np.einsum('sa,sad->d',coefficients,dq)


def maximum_likelihood_irl(transitions, paths, *, discount, beta, reward_bound,
                           tolerance, value_iterations, optimizer_iterations, initial_rewards=None):
    """Bounded state/action reward fitting; no model construction or environment.

    Uses analytic greedy-policy fixed-point derivatives in place of q_grad_fpi.
    Source convention excludes the final pair and weights visits equally.
    """
    if not paths or not any(len(path)>1 for path in paths):
        raise ValueError('IRL requires a demonstrated transition')
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(reward_bound) or reward_bound <= 0 or type(optimizer_iterations) is not int or optimizer_iterations < 1:
        raise ValueError('Invalid IRL optimizer parameters')
    shape = np.asarray(transitions).shape[:2]
    for path in paths:
        if any(not (0 <= s < shape[0] and 0 <= a < shape[1]) for s,a in path):
            raise ValueError('IRL path component id outside transition dimensions')
    history = []
    def objective(flat):
        return irl_objective_gradient(flat.reshape(shape),transitions,paths,
            discount=discount,beta=beta,tolerance=tolerance,value_iterations=value_iterations)
    initial_values=np.zeros(shape) if initial_rewards is None else np.array(initial_rewards,dtype=float,copy=True)
    if initial_values.shape!=shape or not np.isfinite(initial_values).all() or (np.abs(initial_values)>reward_bound).any():
        raise ValueError('Selected reward model initial values must match the state/action axes and configured bounds')
    initial = objective(initial_values.ravel())[0]
    result = minimize(objective, initial_values.ravel(), method='L-BFGS-B',jac=True,
        bounds=[(-reward_bound,reward_bound)]*int(np.prod(shape)),
        options={'maxiter':optimizer_iterations}, callback=lambda x: history.append(objective(x)[0]))
    rewards = result.x.reshape(shape)
    scale = np.max(np.abs(rewards))
    return dict(rewards=rewards, normalized_rewards=rewards/scale if scale else rewards.copy(),
        initial_nll=initial, final_nll=float(result.fun), converged=bool(result.success),
        optimizer_message=str(result.message), history=history)


def source_partition_labels(count,groups,initialization_seed):
    """EqualPartition/InitEMEDMClusters without mutating global random state."""
    import random
    order=list(range(count));random.Random(count+initialization_seed).shuffle(order)
    labels=np.empty(count,dtype=int)
    for group in range(groups):labels[order[group::groups]]=group
    return labels


def source_training_seed(trajectory_count):
    import random
    return random.Random(trajectory_count).sample(range(trajectory_count),1)[0]


class SourceEMEDM:
    """Source-style hard EM: equal random initialization, refit each M-step.

    Stable log-space E-step replaces probability products. Convergence uses
    assigned-component log likelihood, as in EM_EDM_utils, not mixture marginal.
    There is no model selector, architecture default, or configuration mutation.
    """
    def __init__(self, *, policy_factory, trainer_params, groups, iterations,
                 cluster_threshold, beta, likelihood_tolerance, decrease_tolerance, seed, initialization_seed=0):
        import inspect
        if not callable(policy_factory):
            raise ValueError('Selected model must supply a policy factory')
        inspect.signature(train_edm).bind(None, None, None, seed=seed, **trainer_params)
        if any(type(n) is not int or n < 1 for n in (groups, iterations)) or type(cluster_threshold) is not int or cluster_threshold < 0:
            raise ValueError('Invalid EM counts')
        if any(not np.isfinite(n) or n < 0 for n in (beta, likelihood_tolerance, decrease_tolerance)) or beta == 0:
            raise ValueError('Invalid EM scoring/convergence parameters')
        self.policy_factory = policy_factory
        self.trainer_params = dict(trainer_params)
        self.groups, self.iterations = groups, iterations
        self.cluster_threshold, self.beta = cluster_threshold, beta
        self.likelihood_tolerance, self.decrease_tolerance = likelihood_tolerance, decrease_tolerance
        if type(initialization_seed) is not int:
            raise ValueError('initialization_seed must be an integer')
        self.initialization_seed=initialization_seed
        self.seed, self.observer = seed, None
        self.history = []

    def _scores(self, observations, actions, valid, segments):
        from .core import segment_log_scores
        # Boltzmann scale belongs to the likelihood, outside the selected model.
        return segment_log_scores([lambda x, m=m: self.beta*m(x) for m in self.models],
                                  observations, actions, valid, segments)

    def fit(self, observations, actions, valid, segments):
        import random
        self.dimension = int(observations.shape[1])
        from .core import responsibilities
        usable = [s for s in segments if s.split == 'train' and valid[s.start:s.stop].any()]
        if len(usable) < self.groups:
            raise ValueError('Too few train segments for EM groups')
        labels=source_partition_labels(len(usable),self.groups,self.initialization_seed)
        self.component_ids = list(range(self.groups))
        previous = None
        accepted = None
        self.history = []
        self.optimizer_updates = 0
        for iteration in range(self.iterations):
            count = np.bincount(labels,minlength=len(self.component_ids))
            active = np.flatnonzero(count > self.cluster_threshold)
            if not len(active):
                raise ValueError('All EM components fell below cluster_threshold; change the explicit configuration')
            if len(active) != len(count):
                # Do not silently drop demonstrations from a pruned component.
                if not hasattr(self,'models'):
                    raise ValueError('Initial equal partitions do not exceed cluster_threshold')
                scores = self._scores(observations,actions,valid,usable)[:,active]
                removed = ~np.isin(labels,active)
                # Preserve retained hard assignments and reassign removed ones
                # with the same posterior (including priors) used by the E-step.
                retained_labels = np.searchsorted(active,labels)
                if removed.any():
                    retained_labels[removed] = (scores[removed]+np.log(self.priors[active])).argmax(axis=1)
                labels = retained_labels
                self.component_ids = [self.component_ids[k] for k in active]
            self.priors = np.bincount(labels,minlength=len(self.component_ids))/len(labels)
            if (self.priors == 0).any():
                raise ValueError('Pruning produced an empty component; no implicit model replacement is allowed')
            # Fresh supplied models per M-step, rather than historical demo warm starts.
            self.models = []
            losses = []
            for group in range(len(self.component_ids)):
                trajectory_count=int(np.count_nonzero(labels==group))
                training_seed=source_training_seed(trajectory_count)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(training_seed)
                    model = self.policy_factory(observations.shape[1])
                rows = np.concatenate([np.arange(s.start,s.stop)[valid[s.start:s.stop]]
                    for i,s in enumerate(usable) if labels[i] == group])
                def observe(event,payload,group_id=self.component_ids[group],base=self.optimizer_updates):
                    if self.observer:
                        self.observer(event,{**payload,'policy':group_id,'round':iteration+1,'optimizer_updates':base+payload['update']})
                result = train_edm(model,observations[rows],actions[rows],
                    seed=training_seed, observer=observe, **self.trainer_params)
                self.models.append(model)
                losses.append(result['history'])
                self.optimizer_updates += self.trainer_params['updates']
            scores = self._scores(observations,actions,valid,usable)
            probabilities, _ = responsibilities(scores,self.priors)
            new_labels = probabilities.argmax(axis=1)
            # Source Calculate_EM_EDM_Cluster_LLH sums conditional action
            # likelihoods; priors select assignments but are not part of this score.
            llh = float(scores[np.arange(len(usable)),new_labels].sum())
            record = dict(iteration=iteration,conditional_log_likelihood=llh,
                component_ids=list(self.component_ids),priors=self.priors.tolist(),
                assigned_counts=np.bincount(new_labels,minlength=len(self.models)).tolist(),
                changed_fraction=float(np.mean(labels!=new_labels)),training_losses=losses,
                optimizer_updates=self.optimizer_updates, accepted=True)
            if previous is not None and llh < previous-self.decrease_tolerance:
                record['accepted'] = False
                self.models,self.priors,self.component_ids,labels = accepted
                self.stop_reason = 'likelihood_decrease_rollback'
                self.history.append(record)
                if self.observer:
                    self.observer('em',{**record,'learner':self,'probabilities':self.predict(observations,actions,valid,usable)[0]})
                break
            self.history.append(record)
            if self.observer:
                self.observer('em',{**record,'learner':self,'probabilities':probabilities})
            accepted = (list(self.models),self.priors.copy(),list(self.component_ids),new_labels.copy())
            self.accepted_iteration = iteration
            labels = new_labels
            if len(self.models)==1 or (previous is not None and abs(llh-previous)<self.likelihood_tolerance):
                self.stop_reason = 'likelihood_convergence'
                break
            previous, labels = llh, new_labels
        else:
            self.stop_reason = 'iteration_budget'
        # Prune final empty components too: stopping must not leave reward columns
        # for policies that received no demonstrations. Preserve component lineage.
        count = np.bincount(labels,minlength=len(self.models))
        active = np.flatnonzero(count > self.cluster_threshold)
        if not len(active):
            raise ValueError('All final EM components fell below cluster_threshold')
        self.models = [self.models[k] for k in active]
        self.component_ids = [self.component_ids[k] for k in active]
        self.priors = self.priors[active]/self.priors[active].sum()
        return self

    def predict(self,observations,actions,valid,segments):
        from .core import responsibilities
        return responsibilities(self._scores(observations,actions,valid,segments),self.priors)
