"""Time/reward-aware TICC paper port; model-owned structural parameters.

Multi-row Toeplitz precision estimation reuses the independently tested ADMM
kernel. Gaussian-CDF regulation is an explicit interpretation, not recovered code.
"""
import numpy as np
from scipy.special import ndtr
from sklearn.cluster import KMeans
from .core import MultiSeriesTICC,stack_windows,sparse_toeplitz_precision


def variable_viterbi(cost,penalties):
    cost=np.asarray(cost);penalties=np.asarray(penalties)
    if cost.ndim!=2 or not len(cost) or penalties.shape!=(len(cost),) or not np.isfinite(cost).all() or not np.isfinite(penalties).all() or (penalties<0).any():
        raise ValueError('Invalid emission costs or switching penalties')
    previous=cost[0].copy();back=np.zeros(cost.shape,dtype=int);k=cost.shape[1]
    for t in range(1,len(cost)):
        candidates=previous[:,None]+penalties[t]*(1-np.eye(k))
        back[t]=candidates.argmin(axis=0)
        previous=cost[t]+candidates.min(axis=0)
    labels=np.empty(len(cost),dtype=int);labels[-1]=previous.argmin()
    for t in range(len(cost)-1,0,-1): labels[t-1]=back[t,labels[t]]
    return labels


class RewardTICC(MultiSeriesTICC):
    def __init__(self,stages,sparsity,switch_penalty,ticc_iterations,window_rows,
                 covariance_ridge,time_unit_minutes,regulator_mean,regulator_std,cdf_floor,seed):
        super().__init__(stages,sparsity,switch_penalty,ticc_iterations,seed)
        if type(window_rows) is not int or window_rows<1:
            raise ValueError('window_rows must be a positive integer')
        self.window_rows=window_rows
        if any(type(v) is not int or v<1 for v in (stages,ticc_iterations)):
            raise ValueError('Stage counts and iteration budget must be positive integers')
        if any(not np.isfinite(v) or v<0 for v in (sparsity,switch_penalty)):
            raise ValueError('Stage penalties must be finite and nonnegative')
        if any(not np.isfinite(v) or v<=0 for v in (covariance_ridge,time_unit_minutes,cdf_floor)) or cdf_floor>1:
            raise ValueError('Invalid ridge, time unit or CDF floor')
        if len(regulator_mean)!=2 or len(regulator_std)!=2 or not np.isfinite(regulator_mean).all() or not np.isfinite(regulator_std).all() or min(regulator_std)<=0:
            raise ValueError('Regulator requires two finite means and positive standard deviations')
        self.ridge,self.time_unit,self.floor=covariance_ridge,time_unit_minutes,cdf_floor
        self.regulator_mean=np.array(regulator_mean);self.regulator_std=np.array(regulator_std)

    def penalties(self,times,rewards=None):
        times=np.asarray(times)
        if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or (np.diff(times)<=0).any():
            raise ValueError('Stage timestamps must increase strictly per visit')
        gaps=np.r_[0,np.diff(times)]/self.time_unit
        time_value=np.log(np.e+gaps)
        if rewards is None:
            result=self.switch_penalty/time_value
        else:
            rewards=np.asarray(rewards)
            if rewards.shape!=times.shape or not np.isfinite(rewards).all(): raise ValueError('Invalid aligned reward regulator')
            delta=np.abs(np.r_[0,np.diff(rewards)])
            z=(np.column_stack([delta,time_value])-self.regulator_mean)/self.regulator_std
            probability=ndtr(z[:,0])*ndtr(z[:,1])
            result=self.switch_penalty/np.maximum(probability,self.floor)
        result[0]=0
        return result

    def _assign(self,windows,offsets,timestamps,rewards):
        costs=self.costs(windows)
        return np.concatenate([variable_viterbi(costs[a:b],self.penalties(timestamps[i],None if rewards is None else rewards[i]))
            for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:]))])

    def fit(self,episodes,*,timestamps,rewards=None):
        if len(episodes)!=len(timestamps) or (rewards is not None and len(rewards)!=len(episodes)):
            raise ValueError('Stage time/reward arrays must align per visit')
        windows,offsets=stack_windows(episodes,self.window_rows);p=windows.shape[1]//self.window_rows
        if len(windows)<self.clusters*3: raise ValueError('Too few training rows for requested stages')
        labels=KMeans(self.clusters,n_init=5,random_state=self.seed).fit_predict(windows)
        if len(np.unique(labels)) != self.clusters:
            raise ValueError('Insufficient distinct training windows for requested stages')
        self.history=[]
        for iteration in range(self.iterations):
            # Keep model state identities stable, including temporarily empty states.
            means=[];precisions=[];diagnostics=[]
            for k in range(self.clusters):
                points=windows[labels==k]
                if not len(points):
                    means.append(self.means[k]);precisions.append(self.precisions[k]);diagnostics.append(dict(iterations=0,converged=True,primal_residual=0.,dual_residual=0.,pd_shift=0.));continue
                mean=points.mean(axis=0);delta=points-mean
                cov=delta.T@delta/len(points)+self.ridge*np.eye(self.window_rows*p)
                precision,diag=sparse_toeplitz_precision(cov,p,self.penalty)
                means.append(mean);precisions.append(precision);diagnostics.append(diag)
            self.means,self.precisions=np.array(means),np.array(precisions)
            new=self._assign(windows,offsets,timestamps,rewards)
            change=float(np.mean(new!=labels));labels=new
            self.history.append(dict(iteration=iteration,changed_fraction=change,
                counts=np.bincount(labels,minlength=self.clusters).tolist(),precision_solver=diagnostics))
            if self.observer:self.observer('ticc',self.history[-1])
            if change==0:break
        return self

    def predict(self,episodes,*,timestamps,rewards=None):
        windows,offsets=stack_windows(episodes,self.window_rows)
        labels=self._assign(windows,offsets,timestamps,rewards)
        return [labels[a:b] for a,b in zip(offsets[:-1],offsets[1:])]


class TabularReward:
    def __init__(self,states,actions):
        if min(states,actions)<1:raise ValueError('Reward model requires nonempty state/action axes')
        self.values=np.zeros((states,actions),dtype=float)
