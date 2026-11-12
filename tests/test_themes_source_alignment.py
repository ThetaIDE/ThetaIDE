"""Independent source-function parity and mathematical regression checks."""
import ast
import itertools
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import softmax

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/usr/packages/themes-dev'))
from themes_al.core import Segment,stack_windows,toeplitz_projection,sparse_toeplitz_precision
from themes_al.source_run import segment_rewards
from themes_al.paper_methods import (source_partition_labels,source_training_seed,
    optimal_q,irl_objective_gradient,validate_edm_parameters)


def source_function(filename,names,namespace):
    p=ROOT/'src/usr/packages/themes-dev/reference/original/THEMES_src'/filename
    tree=ast.parse(p.read_text())
    keep=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(p),'exec'),namespace)
    return namespace


def test_segment_rewards_match_original_helper_without_importing_upstream():
    segments=[Segment(0,0,2,1,'train'),Segment(0,2,5,0,'train'),Segment(1,5,8,1,'test')]
    posterior=np.array([[.1,.9],[.6,.4],[.8,.2]])
    rewards=np.array([[1.,-.5],[-.2,.7]])
    origins={0:(0,0),1:(0,1),2:(1,0)}
    split=[[(1,2),(0,3)],[(1,3)]]
    ns=source_function('THEMES_helper.py',{'GenerateRewardsWeights'},dict(
        np=np,itertools=itertools,
        PartitionTICCSubTrajs=lambda *a,**k:(None,None,{'trajs':[0,1,2]},origins,split),
        EM_EDM_Estep=lambda *a,**k:([1,0,0],posterior.copy(),None,2)))
    expected,_,_=ns['GenerateRewardsWeights'](None,None,[None,None],[],None,{},rewards,TICC_clusters=2,nrCl=2)
    result=segment_rewards(segments,posterior,rewards,np.array([0,5,8]))
    np.testing.assert_allclose(np.concatenate(result),expected)
    assert np.ptp(result[0][:2])==0 and np.ptp(result[0][2:])==0
    # Order must not affect row placement, and a segment must not cross visits.
    np.testing.assert_allclose(np.concatenate(segment_rewards(segments[::-1],posterior[::-1],rewards,[0,5,8])),expected)
    with pytest.raises(ValueError,match='boundary'):
        segment_rewards([Segment(0,0,8,0,'train')],np.array([[.5,.5]]),rewards,[0,5,8])


def test_em_equal_partition_and_student_seed_match_source_rules():
    ns=source_function('EM_EDM_utils.py',{'EqualPartition'},dict(random=random.Random()))
    for n in [7,11,24]:
        parts=ns['EqualPartition'](np.arange(n),3,init_seed=0)
        expected=np.empty(n,dtype=int)
        for k,part in enumerate(parts):expected[part]=k
        np.testing.assert_array_equal(source_partition_labels(n,3,0),expected)
        assert source_training_seed(n)==random.Random(n).sample(range(n),1)[0]


def test_three_row_windows_are_causal_and_do_not_cross_visits():
    windows,offsets=stack_windows([np.array([[1.],[2.],[3.]]),np.array([[8.],[9.]])],3)
    np.testing.assert_array_equal(windows,[[1,1,1],[1,1,2],[1,2,3],[8,8,8],[8,8,9]])
    np.testing.assert_array_equal(offsets,[0,3,5])


def test_three_block_precision_is_positive_definite_and_toeplitz():
    rng=np.random.default_rng(4);a=rng.normal(size=(6,6));cov=a@a.T+np.eye(6)
    theta,diagnostics=sparse_toeplitz_precision(cov,2,.01)
    np.testing.assert_allclose(theta,theta.T,atol=1e-12)
    np.testing.assert_allclose(theta,toeplitz_projection(theta,2),atol=1e-12)
    assert np.linalg.eigvalsh(theta).min()>0
    assert diagnostics['converged']


def test_irl_analytic_gradient_matches_central_differences_away_from_ties():
    p=np.array([[[.8,.2],[.1,.9]],[[.3,.7],[.9,.1]]])
    rewards=np.array([[.2,1.1],[-.7,.4]])
    paths=[[(0,1),(1,1),(0,0)],[(1,1),(0,1),(1,0)]]
    kwargs=dict(discount=.8,beta=.5,tolerance=1e-12,value_iterations=2000)
    loss,gradient=irl_objective_gradient(rewards,p,paths,**kwargs)
    numerical=[]
    for i in range(rewards.size):
        step=np.zeros_like(rewards);step.flat[i]=1e-5
        plus=irl_objective_gradient(rewards+step,p,paths,**kwargs)[0]
        minus=irl_objective_gradient(rewards-step,p,paths,**kwargs)[0]
        numerical.append((plus-minus)/2e-5)
    np.testing.assert_allclose(gradient,numerical,atol=1e-7,rtol=1e-5)
    # Execute the original NLL function with a controlled Bellman solution.
    class Policy:
        def __init__(self,q,scale):self.p=softmax(scale*q,axis=1)
        def prob_for_state_action(self,s,a):return self.p[s,a]
    ns=source_function('NV_MLIRL.py',{'bv_maxlikelihood_irl'},dict(np=np,
        Linear=lambda x:SimpleNamespace(theta=x),
        vi=lambda xtr,phi,r:(None,optimal_q(p,r.theta.reshape(2,2),discount=.8,tolerance=1e-12,max_iterations=2000)),
        BoltzmannExplorationPolicy=Policy))
    expected=ns['bv_maxlikelihood_irl'](rewards.ravel(),None,None,paths,nll_only=True)
    assert loss==pytest.approx(expected)


def test_unknown_balanced_sampler_is_rejected_instead_of_falling_back():
    with pytest.raises(ValueError,match='balanced-buffer semantics are unavailable'):
        validate_edm_parameters(updates=1,batch_size=2,learning_rate=.001,adam_betas=[.9,.999],
            buffer_size=4,sgld_steps=1,sgld_step_size=.001,sgld_noise=.01,
            reinit_probability=.05,density_weight=1.,energy_regularizer=0.,demonstration_sampling='balanced')


def test_policy_scaling_does_not_saturate_heldout_values_and_reloads_legacy_clip():
    from themes_al.temporal import scale_policy
    raw=np.array([[-1.],[2.]])
    np.testing.assert_array_equal(scale_policy(raw,np.array([0.]),np.array([1.]),'train_minmax_zero_one',clip=False),raw)
    np.testing.assert_array_equal(scale_policy(raw,np.array([0.]),np.array([1.]),'train_minmax_zero_one'),[[0],[1]])
