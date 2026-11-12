"""Source lineage, model ownership, and mathematical checks for the new ports."""
import ast
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/usr/packages/themes-dev'))
from themes_al.paper_methods import edm_objective, train_edm, discrete_dynamics, optimal_q, maximum_likelihood_irl
from themes_al.models import instantiate_policy
from themes_al.core import EMEDM
from test_themes_integration import workspace, launch

@pytest.fixture(autouse=True)
def thread_budget():
    previous=torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_original_recipes_fail_before_training_without_rewriting_config(workspace):
    from src.app.pipeline.compose import compose_experiment, validate_composed
    profile = workspace.config/'agent/themes-dev/em_edm.yaml'
    profile.write_text(profile.read_text().replace('implementation: demo_reconstruction','implementation: original'))
    before = profile.read_bytes()
    with pytest.raises(Exception,match='previous runner was demo-derived'):
        validate_composed(compose_experiment('AL/codex/themes_synthetic_smoke',workspace=workspace))
    assert profile.read_bytes() == before
    assert not list(workspace.results.rglob('execution.json'))


def test_learner_has_no_policy_architecture_fallback():
    with pytest.raises(TypeError,match='policy_factory'):
        EMEDM()
    source = (ROOT/'src/usr/packages/themes-dev/themes_al/core.py').read_text()
    tree = ast.parse(source)
    learner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='EMEDM')
    assert not any(isinstance(n, ast.Name) and n.id in {'Policy','DensePolicy','Linear','Sequential'} for n in ast.walk(learner))


def test_algorithm_cannot_override_model(workspace):
    from src.app.pipeline.compose import compose_experiment, validate_composed
    with pytest.raises(Exception,match='model|Unsupported|Additional properties|additional properties'):
        validate_composed(compose_experiment('AL/codex/themes_synthetic_smoke',
            ['++methods.ticc_em_edm.agent={name:themes-dev/em_edm,hidden_sizes:[400,400]}'],workspace=workspace))


def test_policy_parameters_are_not_silently_discarded():
    spec = dict(architecture='themes-dev/dense_policy',hidden_sizes=[7,5],activation='tanh',output_scale=1.)
    a = instantiate_policy(spec,9)
    b = instantiate_policy({**spec,'hidden_sizes':[11,3]},9)
    assert a.net[0].weight.shape == (7,9)
    assert b.net[0].weight.shape == (11,9)
    with pytest.raises(TypeError):
        instantiate_policy({**spec,'hidden_size':900},9)
    assert spec['hidden_sizes'] == [7,5]


def test_selected_policy_shape_survives_real_worker_and_reload(workspace):
    launch(workspace,overrides=['++methods.ticc_em_edm.model={name:themes-dev/bundle,policy:{hidden_sizes:[7,5],activation:tanh}}'])
    path = next(workspace.results.rglob('policies.pt'))
    checkpoint = torch.load(path,weights_only=True)
    assert checkpoint['artifact_version'] == 2
    assert checkpoint['model_specs'][0]['hidden_sizes'] == [7,5]
    assert checkpoint['policies'][0]['net.0.weight'].shape[0] == 7
    assert checkpoint['policies'][0]['net.2.weight'].shape == (5,7)
    from themes_al.artifacts import verify
    assert verify(path.parent)['status'] == 'passed'


def test_joint_edm_gradient_matches_joint_energy_likelihood():
    positive = torch.tensor([[1.,2.],[3.,-1.]],dtype=torch.float64,requires_grad=True)
    negative = torch.tensor([[-2.,1.],[.5,2.]],dtype=torch.float64,requires_grad=True)
    actions = torch.tensor([1,0])
    loss,_,_ = edm_objective(positive,negative,actions,density_weight=1.,energy_regularizer=0.)
    expected = -positive[torch.arange(2),actions].mean()+torch.logsumexp(negative,dim=1).mean()
    assert torch.allclose(loss,expected)
    g = torch.autograd.grad(loss,(positive,negative),retain_graph=True)
    e = torch.autograd.grad(expected,(positive,negative))
    assert all(torch.allclose(a,b) for a,b in zip(g,e))


def test_edm_trains_supplied_model_and_keeps_input_config():
    model = instantiate_policy(dict(architecture='themes-dev/dense_policy',hidden_sizes=[8],activation='tanh',output_scale=1.),3)
    before = model.net[0].weight.detach().clone()
    params=dict(updates=3,batch_size=4,learning_rate=.001,adam_betas=[.9,.999],buffer_size=12,
        sgld_steps=2,sgld_step_size=.001,sgld_noise=.01,reinit_probability=.05,
        density_weight=1.,energy_regularizer=0.,seed=4)
    saved=copy.deepcopy(params)
    result=train_edm(model,np.arange(18).reshape(6,3)/18,np.arange(6)%2,**params)
    assert not torch.equal(model.net[0].weight,before)
    assert result['buffer'].shape == (12,3)
    assert np.isfinite([h['loss'] for h in result['history']]).all()
    assert params == saved


def test_discrete_port_matches_original_function_exactly():
    # Execute only the available pure helper functions; never import upstream scripts.
    source=ROOT/'src/usr/packages/themes-dev/reference/original/THEMES_src/THEMES_helper.py'
    tree=ast.parse(source.read_text())
    keep=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'ConvertNumListToOneHot','GenerateDiscreteSAP'}]
    import itertools
    ns={'np':np,'itertools':itertools}
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(source),'exec'),ns)
    paths=[[(0,0),(1,1),(0,0)],[(1,0),(0,1)]]
    origins={i:(p,t) for i,(p,t) in enumerate(( (p,t) for p,path in enumerate(paths) for t in range(len(path))))}
    labels=[a for path in paths for _,a in path]
    splits=[[(s,1) for s,_ in path] for path in paths]
    _,original_p,original_p0=ns['GenerateDiscreteSAP'](origins,splits,labels,TICC_clusters=2)
    p,p0=discrete_dynamics(paths,2,2)
    np.testing.assert_array_equal(p,original_p)
    np.testing.assert_array_equal(p0,original_p0)


def test_bellman_analytic_solution_and_iteration_limit():
    p=np.ones((1,2,1));r=np.array([[1.,2.]])
    q=optimal_q(p,r,discount=.8,tolerance=1e-10,max_iterations=1000)
    np.testing.assert_allclose(q,[[9.,10.]],atol=1e-8)
    with pytest.raises(RuntimeError,match='did not converge'):
        optimal_q(p,r,discount=.99,tolerance=1e-12,max_iterations=1)


def test_irl_improves_demo_likelihood_without_perturbing_dynamics():
    p=np.ones((1,2,1));before=p.copy()
    result=maximum_likelihood_irl(p,[[(0,1)]*6],discount=.8,beta=.5,reward_bound=5.,
        tolerance=1e-8,value_iterations=1000,optimizer_iterations=40)
    assert result['converged'], result['optimizer_message']
    assert result['final_nll'] < result['initial_nll']
    assert result['rewards'][0,1] > result['rewards'][0,0]
    assert np.isfinite(result['normalized_rewards']).all()
    np.testing.assert_array_equal(p,before)


def test_source_em_uses_selected_factory_each_m_step_and_no_config_mutation():
    from themes_al.paper_methods import SourceEMEDM
    from themes_al.core import Segment
    calls=[]
    def factory(dimension):
        calls.append(dimension)
        return instantiate_policy(dict(architecture='themes-dev/dense_policy',hidden_sizes=[4],activation='tanh',output_scale=1.),dimension)
    trainer=dict(updates=1,batch_size=4,learning_rate=.001,adam_betas=[.9,.999],buffer_size=8,
        sgld_steps=1,sgld_step_size=.001,sgld_noise=.01,reinit_probability=.05,
        density_weight=1.,energy_regularizer=0.)
    original=copy.deepcopy(trainer)
    learner=SourceEMEDM(policy_factory=factory,trainer_params=trainer,groups=2,iterations=2,
        cluster_threshold=0,beta=.5,likelihood_tolerance=0.,decrease_tolerance=1e9,seed=4)
    states=np.random.default_rng(2).normal(size=(24,3)).astype('float32')
    actions=np.arange(24)%2
    segments=[Segment(i,i*4,(i+1)*4,0,'train') for i in range(6)]
    learner.fit(states,actions,np.ones(24,dtype=bool),segments)
    probabilities,evidence=learner.predict(states,actions,np.ones(24,dtype=bool),segments)
    assert len(calls)==sum(len(h['component_ids']) for h in learner.history)
    np.testing.assert_allclose(probabilities.sum(axis=1),1.)
    assert np.isfinite(evidence).all()
    assert trainer == original


def test_runner_accepts_any_selected_model_with_the_declared_contract(workspace):
    provider=workspace.root/'src/usr/packages/themes-dev'
    manifest_path=provider/'.theta_component.json'
    manifest=json.loads(manifest_path.read_text())
    selected=copy.deepcopy(next(c for c in manifest['capabilities'] if c['id']=='themes-dev/bundle'))
    selected['id']='themes-dev/alternative_bundle'
    manifest['capabilities'].append(selected)
    manifest_path.write_text(json.dumps(manifest))
    model_profile=workspace.config/'model/themes-dev/alternative_bundle.yaml'
    model_profile.write_text((workspace.config/'model/themes-dev/bundle.yaml').read_text().replace('architecture: themes-dev/bundle','architecture: themes-dev/alternative_bundle'))
    launch(workspace,overrides=['++methods.ticc_em_edm.model=themes-dev/alternative_bundle'])
    record=json.loads(next(workspace.results.rglob('execution.json')).read_text())
    assert record['metadata']['model']=='themes-dev/alternative_bundle'


def test_official_edm_density_square_matches_loss_and_gradients():
    positive=torch.tensor([[1.,2.],[3.,-1.]],requires_grad=True)
    negative=torch.tensor([[-2.,1.],[.5,2.]],requires_grad=True)
    actions=torch.tensor([1,0])
    loss,ce,density=edm_objective(positive,negative,actions,
        density_weight=1.,energy_regularizer=0.,density_regularizer=1.)
    expected=ce+density+density.square()
    torch.testing.assert_close(loss,expected)
    actual=torch.autograd.grad(loss,(positive,negative),retain_graph=True)
    wanted=torch.autograd.grad(expected,(positive,negative))
    for a,b in zip(actual,wanted):torch.testing.assert_close(a,b)
    # The old unbounded density shortcut must now increase the objective.
    extreme=positive.detach()+200
    bad,_,_=edm_objective(extreme,negative.detach(),actions,
        density_weight=1.,energy_regularizer=0.,density_regularizer=1.)
    assert bad>loss


def test_source_em_restores_models_priors_and_ids_after_likelihood_drop(monkeypatch):
    from themes_al import paper_methods
    from themes_al.core import Segment
    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__();self.logits=torch.nn.Parameter(torch.zeros(2))
        def forward(self,x):return self.logits.expand(len(x),-1)
    calls=[]
    def train(model,states,actions,**kwargs):
        logits=([3.,-3.] if len(calls)==0 else [-3.,3.]) if len(calls)<2 else [0.,0.]
        with torch.no_grad():model.logits.copy_(torch.tensor(logits))
        calls.append(model)
        return {'history':[]}
    monkeypatch.setattr(paper_methods,'train_edm',train)
    learner=paper_methods.SourceEMEDM(policy_factory=lambda d:Constant(),
        trainer_params={'updates':1},groups=2,iterations=3,cluster_threshold=0,
        beta=1.,likelihood_tolerance=0.,decrease_tolerance=0.,seed=2)
    x=np.zeros((12,1),dtype='float32');a=np.repeat([0,1],6);v=np.ones(12,dtype=bool)
    segments=[Segment(i,2*i,2*i+2,0,'train') for i in range(6)]
    learner.fit(x,a,v,segments)
    assert learner.stop_reason=='likelihood_decrease_rollback'
    assert learner.accepted_iteration==0
    assert learner.history[0]['conditional_log_likelihood']==pytest.approx(-12*np.log1p(np.exp(-6)),abs=1e-6)
    assert learner.history[-1]['accepted'] is False
    assert learner.models==calls[:2]
    np.testing.assert_array_equal(learner.priors,[.5,.5])
    assert learner.component_ids==[0,1]
    _,evidence=learner.predict(x,a,v,segments)
    assert -evidence.sum()/len(a)<.4


def test_final_empty_policy_is_pruned_without_renumbering_identity(monkeypatch):
    from themes_al import paper_methods
    from themes_al.core import Segment
    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__();self.logits=torch.nn.Parameter(torch.zeros(2))
        def forward(self,x):return self.logits.expand(len(x),-1)
    models=[]
    def train(model,states,actions,**kwargs):
        with torch.no_grad():model.logits.copy_(torch.tensor([0.,3.*len(models)]))
        models.append(model)
        return {'history':[]}
    monkeypatch.setattr(paper_methods,'train_edm',train)
    learner=paper_methods.SourceEMEDM(policy_factory=lambda d:Constant(),
        trainer_params={'updates':1},groups=2,iterations=1,cluster_threshold=0,
        beta=1.,likelihood_tolerance=0.,decrease_tolerance=0.,seed=2)
    x=np.zeros((8,1),dtype='float32');a=np.ones(8,dtype=int);v=np.ones(8,dtype=bool)
    segments=[Segment(i,2*i,2*i+2,0,'train') for i in range(4)]
    learner.fit(x,a,v,segments)
    assert learner.models==[models[1]]
    assert learner.component_ids==[1]
    np.testing.assert_array_equal(learner.priors,[1.])
    probabilities,_=learner.predict(x,a,v,segments)
    assert probabilities.shape==(4,1)
