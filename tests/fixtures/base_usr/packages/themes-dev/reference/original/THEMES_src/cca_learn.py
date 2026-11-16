"""
CCA: Contrastive Critical Apprenticeship — Dual-Contrast Architecture
=======================================================================
Extends CRUX (crux_learn.py) with a principled dual-contrast structure that
mirrors the natural paired comparison in the HALIDE quadrant taxonomy:

  High-risk arm  :  Q10 (shock→nonshock, expert)     vs  Q11 (shock→shock)
  Low-risk  arm  :  Q00 (nonshock→nonshock, subopt)  vs  Q01 (nonshock→shock)

Both arms share the same initial clinical state and diverge at outcome, so the
JSD between their action distributions at each state identifies the precise
moments where physician choices separated good from bad outcomes within each
risk stratum.

Key differences from CRUX:
  1. Phase 2 — Dual kNN criticality: separate Q10/Q11 and Q00/Q01 indexes.
     Each group's criticality reflects within-stratum discrimination, not
     cross-stratum JSD (which mixes different base-risk populations).
  2. Phase 3 — EM-EDM uses Q10 + Q00 only (generative model trained on
     expert/suboptimal data; Q11 and Q01 removed to prevent contamination of
     cluster models with failure-mode action patterns).
  3. Phase 4 — BT-IRL positives = Q10 ∪ Q00, negatives = Q11 ∪ Q01.
     Per-timestep weights routed to the correct arm:
       Q10: GP × P+(a|s)_hi  ;  Q00: GP × P+(a|s)_lo
       Q11: GP × P-(a|s)_hi  ;  Q01: GP × P-(a|s)_lo

Bug fix vs CRUX:
  Negative BT-IRL weights now correctly use scalar HALIDE weight when
  GP_CREDIT=0 (was falling back to uniform 1.0 — asymmetric ablation).

All CRUX / THEMES env-var controls are preserved unchanged.
Dual contrast activates automatically when DATA_MODE=all and both Q00/Q01
are present in the training fold.  Falls back to single high-risk arm for
DATA_MODE ∈ {e, ei, en}.

Result files: results_cca_cleantrain_{data}_{weight}{crit}{contrast}{...}_rp{N}.csv
"""

import sys
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import warnings
import os
import random
import copy
import csv
import itertools
import multiprocessing
import pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from preprocessing.EHR_demos_generator import PartitionTICCSubTrajs
from student.evaluation_utils import ModelEvaluate, AUCScore, APScore, JaccardScore
from EM_EDM_utils import (InitEMEDMClusters, EMEDMWarped, PredictClusterForTest,
                           SplitTrajsByLabs, SaveSplitTrajsToFile)
from NV_MLIRL import UpdatedMLIRL
from scipy.optimize import minimize
from scipy.special import expit as _expit

from THEMES_helper import (TICCSplitTrajs, REGTICCSplitTrajs,
                            GenerateDiscreteSAP, GenerateRewardsWeights,
                            PredictTICCClusters)
from IdentifyGoodBadTrajs import SliceGoodBadForTrainTest

from contrastive_criticality import (build_knn_criticality_index,
                                      compute_all_continuous_criticalities)

pd.options.mode.chained_assignment = None

base_path  = _REPO_ROOT + '/'
data_dir   = base_path + 'volume/'
model_dir  = base_path + 'models/'
result_dir = base_path + 'results/'

env = os.getenv('EHR_ENV', 'CCHS').strip()


# ---------------------------------------------------------------------------
# Env-var helpers
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ('1', 'true', 'yes', 'y')


def _seed_everything(seed: int, *, torch_deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

args = {'env_name': f'EHRs/{env}', 'trial': 1}

config_path = model_dir + args['env_name'] + '/EM-EDM_models/para_config.npy'
config = np.load(config_path, allow_pickle=True).item()

config['NUM_STEPS_TRAIN']          = 500
config['EMEDM_LLH_THRES_CONV']    = 1e-5
config['EMEDM_LLH_THRES_DES_REL'] = 0.005
config['BATCH_SIZE']               = 64
config['MLP_WIDTHS']               = 32
config['EMEDM_ITERS']              = 50

_nrcl     = int(os.getenv('NRCL', '3'))
_nrcl_tag = f'_nrcl{_nrcl}' if _nrcl != 3 else ''
config['EMEDM_nrCl']        = _nrcl
config['SGLD_BUFFER_SIZE']  = 10000
config['SGLD_LEARN_RATE']   = 1e-3
config['SAMPLE_BUFFER']     = 'balanced'
config['BASE_PATH']         = base_path
config['MODEL_LEARNER']     = 'EDM'
config.setdefault('ENV', env)
config.setdefault('GYM_ENV', ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
config.setdefault('ADAM_ALPHA', 0.001)
config.setdefault('ADAM_BETAS', [0.9, 0.999])
config.setdefault('SGLD_NOISE_COEF', 0.01)
config.setdefault('SGLD_NUM_STEPS', 20)
config.setdefault('SGLD_REINIT_FREQ', 0.05)

sampling_num = 10000
avg_patterns = 'weighted'

use_halide_weights = _env_bool('USE_HALIDE_W', True)
_weight_key        = os.getenv('WEIGHT_KEY', 'halide_weights').strip()

use_suboptimal  = _env_bool('USE_SUBOPTIMAL', False)
_data_mode_env  = os.getenv('DATA_MODE', '').strip()
data_mode       = (_data_mode_env if _data_mode_env in ('e', 'ei', 'en', 'all')
                   else ('ei' if use_suboptimal else 'e'))
_data_tag       = data_mode
_ticc_mode      = os.getenv('TICC_MODE', 'inductive').strip().lower()
_run_tag        = os.getenv('RUN_TAG', '').strip()
_tag_suffix     = f'_{_run_tag}' if _run_tag else ''
_halide_variant     = os.getenv('HALIDE_VARIANT', '').strip()
_halide_file_suffix = f'_{_halide_variant}' if _halide_variant else ''
_variant_tag        = _halide_file_suffix
_ne_ratio_env       = os.getenv('MAX_NONEXPERT_RATIO', '').strip()
max_nonexpert_ratio = float(_ne_ratio_env) if _ne_ratio_env else None

# CRUX-compatible env vars (unchanged)
use_gp_credit     = _env_bool('GP_CREDIT',      True)
_gp_weight_col    = os.getenv('GP_WEIGHT_COL', 'w_t_gp').strip()
use_criticality       = _env_bool('USE_CRITICALITY',        True)
_crit_phase3          = _env_bool('CRIT_IN_PHASE3',         True)   # False = criticality in Phase 4 only
_min_crit_support     = int(os.getenv('MIN_CRIT_SUPPORT',   '3'))
_knn_neighbors        = int(os.getenv('KNN_NEIGHBORS',      '50'))
_crit_weight_cap      = float(os.getenv('CRIT_WEIGHT_CAP',  '2.0'))
_crit_soft_cap        = _env_bool('CRIT_SOFT_CAP',          True)   # False = hard clip (legacy)
# CRIT_P4_MODE controls what criticality signal Phase 4 (BT-IRL) uses for pairwise weights:
#   'jsd'      (default) GP × C(s_t)           — JSD criticality score, same signal as Phase 3
#   'p_action'           GP × P+(a_t|s_t)      — kNN action probability (original, noisy on small n)
#   'none'               GP only               — same as NoCrit; criticality only affects Phase 3
_crit_p4_mode = os.getenv('CRIT_P4_MODE', 'jsd').strip().lower()
if _crit_p4_mode not in ('jsd', 'p_action', 'none'):
    raise ValueError(f"CRIT_P4_MODE must be 'jsd', 'p_action', or 'none'; got '{_crit_p4_mode}'")
_lambda_contrast  = float(os.getenv('LAMBDA_CONTRAST', '0.5'))
# NEG_QUADRANT controls the high-risk arm anti-expert (default Q11).
# The low-risk arm always uses Q01 as anti-expert.
_neg_quadrant_str = os.getenv('NEG_QUADRANT', '11').strip()
_neg_quadrant_map = {'00': 0, '10': 1, '01': 2, '11': 3}
if _neg_quadrant_str not in _neg_quadrant_map:
    raise ValueError(f"NEG_QUADRANT must be one of {list(_neg_quadrant_map)}; got '{_neg_quadrant_str}'")
_neg_quadrant_idx = _neg_quadrant_map[_neg_quadrant_str]
_csv_gp_suffix    = '_gp_weights' if use_gp_credit else ''

try:
    import torch as _torch
    _device = 'cuda' if _torch.cuda.is_available() else 'cpu'
    config['DEVICE'] = _device
    config['USE_TORCH_COMPILE'] = _env_bool('TORCH_COMPILE', False)
except ImportError:
    _device = 'cpu'
    config['DEVICE'] = 'cpu'
    config['USE_TORCH_COMPILE'] = False

config['MAX_CLUSTER_WORKERS'] = int(os.getenv('MAX_CLUSTER_WORKERS', '1'))
_parallel_repeats = int(os.getenv('PARALLEL_REPEATS', '1'))

print(f"[CCA CONFIG] EHR_ENV={env}, data_mode={data_mode}, NRCL={_nrcl}, "
      f"device={config['DEVICE']}, use_gp_credit={use_gp_credit}, "
      f"gp_weight_col={_gp_weight_col}, "
      f"use_criticality={use_criticality}, crit_phase3={_crit_phase3}, crit_p4_mode={_crit_p4_mode}, "
      f"knn_neighbors={_knn_neighbors}, crit_weight_cap={_crit_weight_cap}, "
      f"crit_soft_cap={_crit_soft_cap}, min_crit_support={_min_crit_support}, "
      f"lambda_contrast={_lambda_contrast}, neg_quadrant_hi=Q{_neg_quadrant_str}")


# ---------------------------------------------------------------------------
# Contrastive IRL  (identical to crux_learn.py)
# ---------------------------------------------------------------------------

def _weighted_feat_exp(traj, ts_weights, n_states, n_actions):
    feat_exp = np.zeros(n_states * n_actions, dtype=float)
    for t, (s, a) in enumerate(traj):
        if a is None:
            continue
        w = ts_weights[t] if t < len(ts_weights) else 1.0
        feat_exp[int(s) * n_actions + int(a)] += w
    return feat_exp


def _bt_loss_grad(theta, pos_feats, neg_feats, pos_w, neg_w):
    scores_pos = pos_feats @ theta
    scores_neg = neg_feats @ theta
    margins    = scores_pos[:, None] - scores_neg[None, :]
    pair_w     = pos_w[:, None] * neg_w[None, :]

    log_probs  = -np.logaddexp(0.0, -margins)
    loss       = (pair_w * log_probs).sum()

    residuals  = pair_w * _expit(-margins)
    grad       = (residuals.sum(axis=1) @ pos_feats
                  - residuals.sum(axis=0) @ neg_feats)

    return -float(loss), -grad


def ContrastiveMLIRL(
    input_MLIRL_args,
    demo_expert,
    demo_neg,
    expert_ts_weights=None,
    neg_ts_weights=None,
    expert_traj_weights=None,
    neg_traj_weights=None,
    bounds=None,
    verbo: bool = False,
):
    """Bradley-Terry Pairwise Contrastive IRL (identical to crux_learn.py)."""
    in_states, in_actions = input_MLIRL_args[0], input_MLIRL_args[1]
    n_states, n_actions   = len(in_states), len(in_actions)
    d = n_states * n_actions

    def _build(demos, ts_wts, traj_wts):
        feats, tw = [], []
        for i, traj in enumerate(demos):
            n = len(traj)
            if ts_wts is not None:
                w_ts = np.asarray(ts_wts[i], dtype=float)
                if len(w_ts) < n:
                    w_ts = np.pad(w_ts, (0, n - len(w_ts)), constant_values=1.0)
            elif traj_wts is not None:
                w_ts = np.full(n, float(traj_wts[i]))
            else:
                w_ts = np.ones(n)
            feats.append(_weighted_feat_exp(traj, w_ts, n_states, n_actions))
            tw.append(float(traj_wts[i]) if traj_wts is not None else 1.0)
        feats = np.array(feats, dtype=float)
        tw    = np.array(tw,    dtype=float)
        s = tw.sum()
        tw /= s if s > 0 else 1.0
        return feats, tw

    pos_feats, pos_w = _build(demo_expert, expert_ts_weights, expert_traj_weights)
    neg_feats, neg_w = _build(demo_neg,    neg_ts_weights,    neg_traj_weights)

    r_bounds = ([(-1.0, 1.0)] * d if bounds is None
                else [(bounds[0], bounds[1])] * d)

    res = minimize(
        _bt_loss_grad,
        np.zeros(d),
        args=(pos_feats, neg_feats, pos_w, neg_w),
        jac=True,
        method='L-BFGS-B',
        bounds=r_bounds,
        options={'disp': verbo, 'maxiter': 500},
    )

    if verbo:
        print(f'[BT-IRL] converged={res.success}  BT-loss={res.fun:.4f}')

    return res, np.array([res.fun])


# ---------------------------------------------------------------------------
# Weight attachment — dual-contrast routing
# ---------------------------------------------------------------------------

def _attach_weights_cca(comb_demos, orig_traj_idx, emedm_ticc_train_idx,
                         halide_weights, halide_ts_weights,
                         crit_hi_by_global, crit_lo_by_global,
                         q10_global, q00_global):
    """
    Attach per-timestep weights to comb_demos for EM-EDM.

    comb_demos contains only Q10 + Q00 trajectories (Q11/Q01 excluded).

    Weight:
      Q10 timestep: w(t) = GP(t) × C_hi(s_t)   [high-risk arm criticality]
      Q00 timestep: w(t) = GP(t) × C_lo(s_t)   [low-risk  arm criticality]

    If the low-risk criticality dict is None (dual contrast inactive),
    Q00 receives GP-only weight — no criticality boost.

    emedm_ticc_train_idx : list mapping EM-EDM-local index → global traj index.
    crit_hi/lo_by_global : dict {global_idx → crit_ts array} or None.
    """
    weights = []
    for i in range(len(comb_demos['trajs'])):
        local_idx  = orig_traj_idx[i][0]
        global_idx = emedm_ticc_train_idx[local_idx]
        traj_len   = len(comb_demos['trajs'][i])
        start      = orig_traj_idx[i][1]

        if use_gp_credit and halide_ts_weights is not None:
            w = halide_ts_weights[global_idx][start: start + traj_len].copy()
        elif use_halide_weights:
            w = np.full(traj_len, float(halide_weights[global_idx]))
        else:
            w = np.ones(traj_len)

        if use_criticality and _crit_phase3:
            if global_idx in q10_global and crit_hi_by_global is not None:
                traj_c = crit_hi_by_global.get(global_idx)
                if traj_c is not None:
                    end = min(start + traj_len, len(traj_c))
                    w[:end - start] *= traj_c[start:end]
            elif global_idx in q00_global and crit_lo_by_global is not None:
                traj_c = crit_lo_by_global.get(global_idx)
                if traj_c is not None:
                    end = min(start + traj_len, len(traj_c))
                    w[:end - start] *= traj_c[start:end]

        weights.append(w)
    comb_demos['weights'] = weights


# ---------------------------------------------------------------------------
# Per-repeat worker
# ---------------------------------------------------------------------------

def _run_one_repeat(kwargs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rp_idx              = kwargs['rp_idx']
    base_seed           = kwargs['base_seed']
    reseed_each_repeat  = kwargs['reseed_each_repeat']
    reseed_strategy     = kwargs['reseed_strategy']
    torch_deterministic = kwargs['torch_deterministic']
    norm_agg_split_df   = kwargs['norm_agg_split_df']
    idx_list            = kwargs['idx_list']
    halide_weights      = kwargs['halide_weights']
    halide_ts_weights   = kwargs.get('halide_ts_weights')
    agg_feats           = kwargs['agg_feats']
    ticc_num_proc       = kwargs['ticc_num_proc']
    ticc_train          = kwargs['ticc_train']
    reg_ticc_train      = kwargs['reg_ticc_train']
    TICC_clusters       = kwargs['TICC_clusters']
    te_perc             = kwargs['te_perc']
    final_df            = kwargs['final_df']
    sel_feats           = kwargs['sel_feats']
    time_col            = kwargs['time_col']

    print(f'[rp_idx={rp_idx}] Starting CCA repeat')

    if reseed_each_repeat:
        seed = base_seed if reseed_strategy == 'same' else base_seed + rp_idx
        _seed_everything(seed, torch_deterministic=torch_deterministic)

    idx_slice_return = SliceGoodBadForTrainTest(
        idx_list, te_perc=te_perc, rp_idx=rp_idx, base_seed=base_seed,
        data_mode=data_mode, max_nonexpert_ratio=max_nonexpert_ratio,
    )
    ticc_train_idx, ticc_test_idx = idx_slice_return[0], idx_slice_return[3]

    slice_tr_agg_df = [norm_agg_split_df[i] for i in ticc_train_idx]
    slice_te_agg_df = [norm_agg_split_df[i] for i in ticc_test_idx]

    # --- Phase 1: MT-TICC on all training data (all 4 quadrants when DATA_MODE=all) ---
    # TICC benefits from full state-space coverage; Q11/Q01 exclusion is only for EM-EDM.
    if _ticc_mode == 'transductive':
        TICC_return_cv     = kwargs['TICC_return_cv_all']
        reg_TICC_return_cv = copy.deepcopy(TICC_return_cv)
        tr_TICC_return     = [TICC_return_cv[1][i] for i in ticc_train_idx]
    else:
        _ticc_tr_data_path = (data_dir + args['env_name']
                              + f'/MTTICC/tr_seq_{_data_tag}{_variant_tag}_rp{rp_idx}.data')
        _MTTICC_tr_path    = (data_dir + args['env_name']
                              + f'/THEMES_{_data_tag}{_variant_tag}/'
                              + f'THEMESE_MTTICC_cleantrain_tr_rp{rp_idx}.npy')
        os.makedirs(os.path.dirname(_ticc_tr_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(_MTTICC_tr_path),    exist_ok=True)
        TICC_return_cv = TICCSplitTrajs(
            args['env_name'], slice_tr_agg_df, sel_feats, time_col,
            TICC_train=ticc_train, fixed_window=2, TICC_clusters=TICC_clusters,
            lambda_parameter=1e-5, beta=4, num_proc=ticc_num_proc,
            input_pattern='multiple', window_pattern='fixed', maxIters=100,
            save_TICC_results=True, TICC_cluster_path=_MTTICC_tr_path,
            ticc_data_path=_ticc_tr_data_path,
        )
        reg_TICC_return_cv = copy.deepcopy(TICC_return_cv)
        tr_TICC_return     = list(TICC_return_cv[1])

    # --- Phase 2: Dual-contrast kNN criticality ---
    # High-risk arm:  Q10 (expert) vs Q11 (anti-expert)     — same as CRUX
    # Low-risk  arm:  Q00 (subopt) vs Q01 (negative)        — CCA addition
    # Dual contrast activates when DATA_MODE=all and both Q00/Q01 are in training.
    _q00_global = set(idx_list[0])
    _q10_global = set(idx_list[1])
    _q01_global = set(idx_list[2])
    _q11_global = set(idx_list[3])

    _q00_in_train = any(gi in _q00_global for gi in ticc_train_idx)
    _q01_in_train = any(gi in _q01_global for gi in ticc_train_idx)
    _use_dual     = use_criticality and _q00_in_train and _q01_in_train

    if use_criticality:
        # High-risk arm (always built when criticality is on)
        nn_pos_hi, pos_acts_hi, nn_neg_hi, neg_acts_hi = build_knn_criticality_index(
            slice_tr_agg_df=slice_tr_agg_df,
            ticc_train_idx=ticc_train_idx,
            idx_list=idx_list,
            feature_cols=sel_feats,
            action_col='Action',
            n_neighbors=_knn_neighbors,
            n_actions=2,
            pos_quadrant_idx=1,
            neg_quadrant_idx=_neg_quadrant_idx,
            verbose=True,
        )
        crit_ts_hi, p_pos_ts_hi, p_neg_ts_hi = compute_all_continuous_criticalities(
            slice_tr_agg_df=slice_tr_agg_df,
            feature_cols=sel_feats,
            action_col='Action',
            nn_pos=nn_pos_hi, pos_actions=pos_acts_hi,
            nn_neg=nn_neg_hi, neg_actions=neg_acts_hi,
            n_actions=2,
            min_support=_min_crit_support,
            weight_cap=_crit_weight_cap,
            soft_cap=_crit_soft_cap,
            verbose=True,
        )

        if _use_dual:
            # Low-risk arm: Q00 vs Q01
            nn_pos_lo, pos_acts_lo, nn_neg_lo, neg_acts_lo = build_knn_criticality_index(
                slice_tr_agg_df=slice_tr_agg_df,
                ticc_train_idx=ticc_train_idx,
                idx_list=idx_list,
                feature_cols=sel_feats,
                action_col='Action',
                n_neighbors=_knn_neighbors,
                n_actions=2,
                pos_quadrant_idx=0,
                neg_quadrant_idx=2,
                verbose=True,
            )
            crit_ts_lo, p_pos_ts_lo, p_neg_ts_lo = compute_all_continuous_criticalities(
                slice_tr_agg_df=slice_tr_agg_df,
                feature_cols=sel_feats,
                action_col='Action',
                nn_pos=nn_pos_lo, pos_actions=pos_acts_lo,
                nn_neg=nn_neg_lo, neg_actions=neg_acts_lo,
                n_actions=2,
                min_support=_min_crit_support,
                weight_cap=_crit_weight_cap,
                soft_cap=_crit_soft_cap,
                verbose=True,
            )
        else:
            crit_ts_lo = p_pos_ts_lo = p_neg_ts_lo = None

        # Build global-indexed dicts for O(1) lookup in weight attachment and Phase 4.
        crit_hi_by_global  = {ticc_train_idx[j]: crit_ts_hi[j]  for j in range(len(ticc_train_idx))}
        p_pos_hi_by_global = {ticc_train_idx[j]: p_pos_ts_hi[j] for j in range(len(ticc_train_idx))}
        p_neg_hi_by_global = {ticc_train_idx[j]: p_neg_ts_hi[j] for j in range(len(ticc_train_idx))}
        if _use_dual:
            crit_lo_by_global  = {ticc_train_idx[j]: crit_ts_lo[j]  for j in range(len(ticc_train_idx))}
            p_pos_lo_by_global = {ticc_train_idx[j]: p_pos_ts_lo[j] for j in range(len(ticc_train_idx))}
            p_neg_lo_by_global = {ticc_train_idx[j]: p_neg_ts_lo[j] for j in range(len(ticc_train_idx))}
        else:
            crit_lo_by_global = p_pos_lo_by_global = p_neg_lo_by_global = None
    else:
        crit_hi_by_global = crit_lo_by_global = None
        p_pos_hi_by_global = p_neg_hi_by_global = None
        p_pos_lo_by_global = p_neg_lo_by_global = None
        _use_dual = False

    # --- Filter to Q10 + Q00 for EM-EDM ---
    # Q11 and Q01 are contrastive signal only; they must not contaminate the
    # generative cluster models.  TICC ran on all 4 quadrants for coverage;
    # EM-EDM trains on expert/suboptimal trajectories only.
    _emedm_global        = _q10_global | _q00_global
    emedm_ticc_local_idx = [j for j, gi in enumerate(ticc_train_idx) if gi in _emedm_global]
    slice_emedm_agg_df   = [slice_tr_agg_df[j]  for j in emedm_ticc_local_idx]
    emedm_ticc_train_idx = [ticc_train_idx[j]   for j in emedm_ticc_local_idx]

    print(f'[rp_idx={rp_idx}] EM-EDM subset: '
          f'{sum(gi in _q10_global for gi in emedm_ticc_train_idx)} Q10 + '
          f'{sum(gi in _q00_global for gi in emedm_ticc_train_idx)} Q00 trajectories '
          f'(dual_contrast={_use_dual})')

    # Initial sub-trajectory partition on Q10+Q00 subset
    emedm_ticc_return = [tr_TICC_return[j] for j in emedm_ticc_local_idx]
    _, _, comb_demos, orig_traj_idx, dis_TICC_split = PartitionTICCSubTrajs(
        slice_emedm_agg_df, agg_feats, emedm_ticc_return, TICC_clusters=TICC_clusters)
    _attach_weights_cca(
        comb_demos, orig_traj_idx, emedm_ticc_train_idx,
        halide_weights, halide_ts_weights,
        crit_hi_by_global, crit_lo_by_global,
        _q10_global, _q00_global,
    )

    iteration_num, max_iter = 0, 10
    itertation_results = []
    MTTICC_LLH, EMEDM_LLH, IRL_LLH = [], [], []

    while iteration_num < max_iter:

        # --- Phase 3: EM-EDM on Q10 + Q00 only ---
        config['ENV_VOLUME_PATH'] = base_path + 'volume/' + args['env_name']
        config['CLUS_BASE_PATH']  = (config['ENV_VOLUME_PATH']
                                      + f'/CCA_{_data_tag}{_variant_tag}/rp_{rp_idx}/')
        os.makedirs(config['CLUS_BASE_PATH'], exist_ok=True)

        clus_teachers = comb_demos
        clus_trajs, clus_returns = clus_teachers['trajs'], clus_teachers['returns']

        model_folder = model_dir + args['env_name'] + '/EM-EDM_models/'
        model_path   = model_folder + f'CCA_{_data_tag}{_variant_tag}_model_fold_{rp_idx}.sav'
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        config['MODEL_PATH'] = model_path

        nrCl      = config['EMEDM_nrCl']
        pred_labs = InitEMEDMClusters(clus_trajs, nrCl, sampling_num, init_mode='random',
                                      DTW_thres=1e5, max_iter=10, init_seed=base_seed)

        student_list, rho, LLH, nrCl, pred_labs, pred_probs = EMEDMWarped(
            clus_teachers, nrCl, pred_labs, config, decay_expert=False)

        pickle.dump((student_list, rho), open(model_path, 'wb'))
        EMEDM_LLH.append(LLH)
        fig_path = os.path.join(model_folder,
                                f'llh_cca_{_data_tag}{_variant_tag}_rp{rp_idx}_iter{iteration_num}.png')
        plt.figure(); plt.plot(LLH); plt.savefig(fig_path); plt.close()

        # --- Test evaluation (expert-only test fold, unchanged) ---
        if _ticc_mode == 'transductive':
            test_TICC_split = [reg_TICC_return_cv[1][i] for i in ticc_test_idx]
        else:
            test_TICC_split = PredictTICCClusters(
                slice_te_agg_df, sel_feats, time_col, 2, reg_TICC_return_cv)
        test_demos, _, sub_demos, _, _ = PartitionTICCSubTrajs(
            slice_te_agg_df, agg_feats, test_TICC_split, TICC_clusters=TICC_clusters)

        test_teachers = sub_demos
        test_trajs, test_returns = test_teachers['trajs'], test_teachers['returns']

        pred_test_labs   = PredictClusterForTest(student_list, test_trajs, nrCl, rho, config)
        unique_test_labs = sorted(np.unique(pred_test_labs))

        base_split_path = (base_path + 'volume/' + args['env_name']
                           + f'/CCA_split_{_data_tag}{_variant_tag}/rp_{rp_idx}/')
        Path(base_split_path).mkdir(parents=True, exist_ok=True)

        test_dict_array = SplitTrajsByLabs(test_teachers, nrCl, pred_test_labs)
        test_path_dict  = SaveSplitTrajsToFile(
            base_split_path, test_dict_array, unique_test_labs, sampling_num, rp_idx, mode='test')

        student_dict = {i: student_list[i] for i in range(len(student_list))}
        true_act, pred_act, pred_prob = [], [], []
        for clusIdx in unique_test_labs:
            if clusIdx in student_dict:
                _, tmp_true, tmp_pred, tmp_prob = student_dict[clusIdx].general_test(
                    test_path_dict[clusIdx], avg_patterns=avg_patterns)
            else:
                tmpIdx = random.sample(list(student_dict.keys()), 1)[0]
                _, tmp_true, tmp_pred, tmp_prob = student_dict[tmpIdx].general_test(
                    test_path_dict[clusIdx], avg_patterns=avg_patterns)
            true_act.extend(tmp_true)
            pred_act.extend(tmp_pred)
            pred_prob.extend(tmp_prob)

        _, metrics = ModelEvaluate(true_act, pred_act, np.arange(2),
                                    avg_patterns=avg_patterns, verbo=True)
        AUC     = AUCScore(true_act, pred_act, pred_prob, avg_patterns=avg_patterns)
        APR     = APScore(true_act, pred_act, pred_prob, avg_patterns=avg_patterns)
        Jascore = JaccardScore(true_act, pred_act, avg_patterns=avg_patterns)
        itertation_results.append(metrics + [AUC] + [APR] + [Jascore])

        # --- Phase 4: Dual-Contrast BT-IRL ---
        # Positives: Q10 (GP × P+_hi) ∪ Q00 (GP × P+_lo)
        # Negatives: Q11 (GP × P-_hi) ∪ Q01 (GP × P-_lo)
        if iteration_num < max_iter - 1:
            print(f'[rp_idx={rp_idx}] Applying Dual-Contrast BT-IRL...')
            state_num, act_num = TICC_clusters, nrCl
            dis_demos, P_a, P_0 = GenerateDiscreteSAP(
                orig_traj_idx, dis_TICC_split, pred_labs, TICC_clusters=TICC_clusters)

            irl_states, irl_actions  = np.arange(state_num), np.arange(act_num)
            irl_P0, irl_transitions  = P_0, P_a
            irl_terminals            = np.zeros(state_num)
            irl_gamma                = 0.8
            input_MLIRL_args = (irl_states, irl_actions, irl_P0,
                                 irl_transitions, irl_terminals, irl_gamma)

            # --- Positive demos: Q10 + Q00 from the EM-EDM subset ---
            # dis_demos is already filtered to Q10+Q00 (comb_demos contains only them).
            # Route each sub-trajectory to the correct arm for per-timestep weights.
            conv_dis_demos        = []
            expert_ts_weight_list = []

            for _ei in range(len(dis_demos)):
                _local_idx  = orig_traj_idx[_ei][0]
                _global_idx = emedm_ticc_train_idx[_local_idx]

                conv_dis_demos.append(
                    dis_demos[_ei][1][:-1] + [(dis_demos[_ei][1][-1][0], None)])

                _start    = orig_traj_idx[_ei][1]
                _traj_len = len(comb_demos['trajs'][_ei])

                if use_gp_credit and halide_ts_weights is not None:
                    _gp = halide_ts_weights[_global_idx][_start: _start + _traj_len].copy()
                elif use_halide_weights:
                    _gp = np.full(_traj_len, float(halide_weights[_global_idx]))
                else:
                    _gp = np.ones(_traj_len)

                # Phase 4 positive weight — mode controlled by CRIT_P4_MODE
                _phase4_pos_signal = None
                if use_criticality and _crit_p4_mode == 'jsd':
                    # GP × C(s_t): same JSD criticality used in Phase 3 — consistent signal
                    if _global_idx in _q10_global and crit_hi_by_global is not None:
                        _phase4_pos_signal = crit_hi_by_global.get(_global_idx)
                    elif _global_idx in _q00_global and crit_lo_by_global is not None:
                        _phase4_pos_signal = crit_lo_by_global.get(_global_idx)
                elif use_criticality and _crit_p4_mode == 'p_action':
                    # GP × P+(a_t|s_t): original kNN action probability (noisy on small n)
                    if _global_idx in _q10_global and p_pos_hi_by_global is not None:
                        _phase4_pos_signal = p_pos_hi_by_global.get(_global_idx)
                    elif _global_idx in _q00_global and p_pos_lo_by_global is not None:
                        _phase4_pos_signal = p_pos_lo_by_global.get(_global_idx)
                # 'none' or signal not found → GP only

                if _phase4_pos_signal is not None:
                    _sig_vals = np.array([
                        _phase4_pos_signal[_start + _t] if (_start + _t) < len(_phase4_pos_signal) else 1.0
                        for _t in range(_traj_len)
                    ])
                    expert_ts_weight_list.append(_gp * _sig_vals)
                else:
                    expert_ts_weight_list.append(_gp)

            if not conv_dis_demos:
                print(f'[rp_idx={rp_idx}] WARNING: no positive sub-trajs at '
                      f'iter {iteration_num}; skipping Phase 4.')
                IRL_LLH.append(np.array([0.0]))
                MTTICC_LLH.append(0.0)
                iteration_num += 1
                continue

            expert_traj_weights = np.array([w.mean() for w in expert_ts_weight_list])

            # --- Negative demos: Q11 (high-risk arm) + Q01 (low-risk arm) ---
            # Both serve as contrastive reference for their respective positive groups.
            # Re-partitioned from the full TICC training split (which includes Q11/Q01)
            # and cluster-assigned using the EM-EDM student models.
            _q11_ticc_local = [j for j, gi in enumerate(ticc_train_idx) if gi in _q11_global]
            _q01_ticc_local = [j for j, gi in enumerate(ticc_train_idx) if gi in _q01_global]
            all_neg_ticc_local = _q11_ticc_local + _q01_ticc_local

            conv_neg_demos        = []
            neg_ts_weight_list    = []
            neg_traj_weights_list = []

            if all_neg_ticc_local and _lambda_contrast > 0:
                neg_tr_agg = [slice_tr_agg_df[j] for j in all_neg_ticc_local]
                neg_ticc   = [tr_TICC_return[j]   for j in all_neg_ticc_local]

                _, _, neg_comb, neg_orig_idx, neg_dis_split = PartitionTICCSubTrajs(
                    neg_tr_agg, agg_feats, neg_ticc, TICC_clusters=TICC_clusters)

                neg_pred_labs = PredictClusterForTest(
                    student_list, neg_comb['trajs'], nrCl, rho, config)

                neg_dis_demos_raw, _, _ = GenerateDiscreteSAP(
                    neg_orig_idx, neg_dis_split, neg_pred_labs, TICC_clusters=TICC_clusters)

                for idx in range(len(neg_dis_demos_raw)):
                    conv_neg_demos.append(
                        neg_dis_demos_raw[idx][1][:-1]
                        + [(neg_dis_demos_raw[idx][1][-1][0], None)])

                for i in range(len(neg_comb['trajs'])):
                    local_j       = neg_orig_idx[i][0]
                    ticc_local_j  = all_neg_ticc_local[local_j]
                    global_neg_gi = ticc_train_idx[ticc_local_j]
                    start_n       = neg_orig_idx[i][1]
                    traj_len_n    = len(neg_comb['trajs'][i])

                    # Base weight — fixed bug vs CRUX: use HALIDE scalar when GP_CREDIT=0
                    if use_gp_credit and halide_ts_weights is not None:
                        ts_w = halide_ts_weights[global_neg_gi][start_n: start_n + traj_len_n].copy()
                    elif use_halide_weights:
                        ts_w = np.full(traj_len_n, float(halide_weights[global_neg_gi]))
                    else:
                        ts_w = np.ones(traj_len_n)

                    # Phase 4 negative weight — mode controlled by CRIT_P4_MODE
                    _phase4_neg_signal = None
                    if use_criticality and _crit_p4_mode == 'jsd':
                        if global_neg_gi in _q11_global and crit_hi_by_global is not None:
                            _phase4_neg_signal = crit_hi_by_global.get(global_neg_gi)
                        elif global_neg_gi in _q01_global and crit_lo_by_global is not None:
                            _phase4_neg_signal = crit_lo_by_global.get(global_neg_gi)
                    elif use_criticality and _crit_p4_mode == 'p_action':
                        if global_neg_gi in _q11_global and p_neg_hi_by_global is not None:
                            _phase4_neg_signal = p_neg_hi_by_global.get(global_neg_gi)
                        elif global_neg_gi in _q01_global and p_neg_lo_by_global is not None:
                            _phase4_neg_signal = p_neg_lo_by_global.get(global_neg_gi)

                    if _phase4_neg_signal is not None:
                        for t_sub in range(traj_len_n):
                            t_o = start_n + t_sub
                            if t_o < len(_phase4_neg_signal):
                                ts_w[t_sub] *= _phase4_neg_signal[t_o]

                    neg_ts_weight_list.append(ts_w.copy())
                    neg_traj_weights_list.append(float(ts_w.mean()))

                neg_traj_weights = (np.array(neg_traj_weights_list)
                                    if neg_traj_weights_list else None)
            else:
                neg_traj_weights = None

            reward_bound = [-5, 5]
            if conv_neg_demos and _lambda_contrast > 0:
                temp_reward, llh_list = ContrastiveMLIRL(
                    input_MLIRL_args,
                    conv_dis_demos,
                    conv_neg_demos,
                    expert_ts_weights=expert_ts_weight_list,
                    neg_ts_weights=neg_ts_weight_list if neg_ts_weight_list else None,
                    expert_traj_weights=expert_traj_weights,
                    neg_traj_weights=neg_traj_weights,
                    bounds=reward_bound,
                    verbo=False,
                )
            else:
                temp_reward, llh_list = UpdatedMLIRL(
                    input_MLIRL_args, conv_dis_demos,
                    bounds=reward_bound, scale=0.5)

            IRL_LLH.append(llh_list)

            tmp_max_range = max(abs(v) for v in temp_reward.x)
            if tmp_max_range >= 1e-10:
                norm_reward         = [v / tmp_max_range for v in temp_reward.x]
                state_action_reward = np.reshape(
                    norm_reward, (len(irl_states), len(irl_actions)), order='F')
                print(f'[rp_idx={rp_idx}] Rewards:\n', state_action_reward)

                # REG-TICC on all training data (full state-space coverage)
                reward_col = 'reward_weight'
                if _ticc_mode == 'transductive':
                    themes_w, _, _ = GenerateRewardsWeights(
                        norm_agg_split_df, agg_feats, reg_TICC_return_cv,
                        student_list, rho, config, state_action_reward,
                        TICC_clusters=TICC_clusters, nrCl=nrCl)
                    w_reg_df = copy.deepcopy(final_df)
                    tmp_num  = [0] + list(np.cumsum([len(i) for i in w_reg_df]))
                    for idx in range(len(tmp_num) - 1):
                        w_reg_df[idx][reward_col] = themes_w[tmp_num[idx]:tmp_num[idx + 1]]
                else:
                    themes_w, _, _ = GenerateRewardsWeights(
                        slice_tr_agg_df, agg_feats, reg_TICC_return_cv,
                        student_list, rho, config, state_action_reward,
                        TICC_clusters=TICC_clusters, nrCl=nrCl)
                    w_reg_df = copy.deepcopy([final_df[i] for i in ticc_train_idx])
                    tmp_num  = [0] + list(np.cumsum([len(i) for i in w_reg_df]))
                    for idx in range(len(tmp_num) - 1):
                        w_reg_df[idx][reward_col] = themes_w[tmp_num[idx]:tmp_num[idx + 1]]

                regMTTICC_path = (data_dir + args['env_name']
                                  + f'/CCA_{_data_tag}{_variant_tag}/'
                                  + f'CCA_regMTTICC_cleantrain_results_rp_{rp_idx}iter_{iteration_num}.npy')

                if iteration_num > 0:
                    reg_init_labs  = list(itertools.chain.from_iterable(reg_TICC_return_cv[0]))
                    reg_TICC_return_cv = REGTICCSplitTrajs(
                        args['env_name'], w_reg_df, sel_feats, time_col, reward_col,
                        TICC_train=reg_ticc_train, fixed_window=2,
                        TICC_clusters=TICC_clusters,
                        lambda_parameter=0.000001, beta=4,
                        num_proc=ticc_num_proc, input_pattern='multiple',
                        window_pattern='fixed', maxIters=100,
                        save_TICC_results=True, TICC_cluster_path=regMTTICC_path,
                        init_labs=reg_init_labs, rewards_thres=0.5,
                        smooth_reg_mu=[1, 1], smooth_reg_std=[.2, 1],
                    )
                else:
                    reg_TICC_return_cv = copy.deepcopy(TICC_return_cv)

                MTTICC_LLH.append(reg_TICC_return_cv[2][0])

                if _ticc_mode == 'transductive':
                    tr_TICC_return = [reg_TICC_return_cv[1][i] for i in ticc_train_idx]
                else:
                    tr_TICC_return = list(reg_TICC_return_cv[1])
            else:
                print(f'[rp_idx={rp_idx}] WARNING: IRL converged to zero reward — '
                      f'skipping REG-TICC update this iteration.')
                MTTICC_LLH.append(0.0)

            # Re-partition Q10+Q00 sub-trajectories with updated TICC labels.
            # kNN criticality dicts are state-space based and do not change.
            emedm_ticc_return = [tr_TICC_return[j] for j in emedm_ticc_local_idx]
            _, _, comb_demos, orig_traj_idx, dis_TICC_split = PartitionTICCSubTrajs(
                slice_emedm_agg_df, agg_feats, emedm_ticc_return, TICC_clusters=TICC_clusters)
            _attach_weights_cca(
                comb_demos, orig_traj_idx, emedm_ticc_train_idx,
                halide_weights, halide_ts_weights,
                crit_hi_by_global, crit_lo_by_global,
                _q10_global, _q00_global,
            )

        iteration_num += 1

    best_results = max(itertation_results, key=lambda r: r[4])
    return rp_idx, itertation_results, best_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    base_seed_env       = os.getenv('BASE_SEED')
    torch_deterministic = _env_bool('TORCH_DETERMINISTIC', False)
    reseed_each_repeat  = _env_bool('RESEED_EACH_REPEAT', True)
    reseed_strategy     = (os.getenv('RESEED_STRATEGY') or 'offset').strip().lower()

    base_seed = 1234
    if base_seed_env and base_seed_env.strip():
        try:
            base_seed = int(base_seed_env)
        except ValueError:
            print(f"Warning: could not parse BASE_SEED='{base_seed_env}'; using 1234.")

    _seed_everything(base_seed, torch_deterministic=torch_deterministic)
    print(f"[RNG] BASE_SEED={base_seed} (torch_deterministic={torch_deterministic}, "
          f"reseed_each_repeat={reseed_each_repeat}, reseed_strategy={reseed_strategy})")

    concat_final_df = pd.read_csv(
        data_dir + f'{env}_halide_filtered_cleantrain_dataframe_md{_halide_file_suffix}{_csv_gp_suffix}.csv')

    halide_quality = np.load(
        data_dir + f'{env}_halide_quality_scores_cleantrain_md{_halide_file_suffix}.npy',
        allow_pickle=True).item()
    q = halide_quality['quadrant_idx']
    idx_list = [
        list(q['nonshock_nonshock']),
        list(q['shock_nonshock']),
        list(q['nonshock_shock']),
        list(q['shock_shock']),
    ]
    halide_weights = halide_quality[_weight_key]

    final_df = []
    selected_vid = sorted(np.unique(concat_final_df.VisitIdentifier))
    for vid in selected_vid:
        final_df.append(concat_final_df.loc[concat_final_df.VisitIdentifier == vid])

    sel_feats = ['SystolicBP', 'MAP', 'RespiratoryRate', 'PulseOx', 'HeartRate',
                 'Temperature', 'WBC', 'BiliRubin', 'BUN', 'Lactate', 'Creatinine',
                 'Platelet', 'Bands', 'FIO2']
    time_col = 'MinutesFromArrival'
    max_sel_feats = ['max_' + f for f in sel_feats]
    min_sel_feats = ['min_' + f for f in sel_feats]
    agg_feats     = sel_feats + max_sel_feats + min_sel_feats

    agg_split_df = []
    for idx in range(len(selected_vid)):
        tmp_df = final_df[idx]
        min_max_time = [(i - 60, i) for i in tmp_df[time_col]]
        tmp_max = [list(np.max(tmp_df.loc[
            (tmp_df[time_col] >= min_max_time[i][0]) &
            (tmp_df[time_col] <= min_max_time[i][1])][sel_feats], axis=0))
            for i in range(len(min_max_time))]
        tmp_min = [list(np.min(tmp_df.loc[
            (tmp_df[time_col] >= min_max_time[i][0]) &
            (tmp_df[time_col] <= min_max_time[i][1])][sel_feats], axis=0))
            for i in range(len(min_max_time))]
        tmp_agg_data = np.concatenate(
            [np.array(tmp_df), np.array(tmp_max), np.array(tmp_min)], axis=1)
        tmp_agg_df = pd.DataFrame(
            tmp_agg_data,
            columns=list(tmp_df.columns) + max_sel_feats + min_sel_feats)
        tmp_agg_df['Action'] = np.max(np.array(tmp_agg_df[['a_2']]), axis=1)
        agg_split_df.append(tmp_agg_df)

    feat_max_min = {}
    for feat in agg_feats:
        tmp_vals = list(itertools.chain.from_iterable(
            [list(agg_split_df[i][feat]) for i in range(len(agg_split_df))]))
        feat_max_min[feat] = (np.max(tmp_vals), np.min(tmp_vals))

    norm_agg_split_df = copy.deepcopy(agg_split_df)
    for idx in range(len(norm_agg_split_df)):
        for feat in agg_feats:
            tmp_max, tmp_min = feat_max_min[feat]
            norm_agg_split_df[idx][feat] = (
                (agg_split_df[idx][feat] - tmp_min) / (tmp_max - tmp_min))

    # --- Phase 0: Load pre-computed InferGP per-timestep weights ---
    if use_gp_credit:
        if _gp_weight_col in concat_final_df.columns:
            halide_ts_weights = [
                np.array(final_df[i][_gp_weight_col].values, dtype=float)
                for i in range(len(final_df))
            ]
            print(f"[CCA] Loaded InferGP per-timestep weights ({_gp_weight_col}) "
                  f"for {len(halide_ts_weights)} trajectories.")
        else:
            halide_ts_weights = None
            print(f"[CCA] WARNING: GP_CREDIT=1 but '{_gp_weight_col}' not found "
                  f"— falling back to scalar HALIDE weights.")
    else:
        halide_ts_weights = None

    TICC_clusters  = 9
    ticc_num_proc  = int(os.getenv('TICC_NUM_PROC', '6'))
    ticc_train     = _env_bool('TICC_TRAIN', True)
    reg_ticc_train = _env_bool('REG_TICC_TRAIN', True)

    if _ticc_mode == 'transductive':
        MTTICC_path = (data_dir + args['env_name']
                       + '/All/CCA_MTTICC_cleantrain_results.npy')
        _ticc_data_path = (data_dir + args['env_name']
                           + f'/MTTICC/all_seq_multiple_{os.getenv("SLURM_ARRAY_TASK_ID", "local")}.data')
        TICC_return_cv_all = TICCSplitTrajs(
            args['env_name'], final_df, sel_feats, time_col,
            TICC_train=ticc_train, fixed_window=2, TICC_clusters=TICC_clusters,
            lambda_parameter=1e-5, beta=4, num_proc=ticc_num_proc,
            input_pattern='multiple', window_pattern='fixed', maxIters=100,
            save_TICC_results=True, TICC_cluster_path=MTTICC_path,
            ticc_data_path=_ticc_data_path,
        )
    else:
        TICC_return_cv_all = None

    te_perc    = 0.2
    repeat_num = 10

    results_dir_base = result_dir + args['env_name'] + '/CCA/'
    _weight_tag   = ('unweighted' if not use_halide_weights else
                     ('gp_weighted' if use_gp_credit else 'weighted'))
    _crit_tag     = '_crit' if use_criticality else ''
    _contrast_tag = f'_lc{_lambda_contrast}' if _lambda_contrast > 0 else ''
    results_file_name = (f'results_cca_cleantrain_{_data_tag}_{_weight_tag}'
                         f'{_crit_tag}{_contrast_tag}{_variant_tag}'
                         f'{_nrcl_tag}{_tag_suffix}.csv')
    results_file_path = os.path.join(results_dir_base, results_file_name)
    os.makedirs(results_dir_base, exist_ok=True)

    with open(results_file_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ('Round', 'Accuracy', 'Recall', 'Precision', 'F-score', 'AUC', 'APR', 'Jaccard'))

    _repeat_kwargs_list = [
        dict(
            rp_idx=rp_idx,
            base_seed=base_seed,
            reseed_each_repeat=reseed_each_repeat,
            reseed_strategy=reseed_strategy,
            torch_deterministic=torch_deterministic,
            norm_agg_split_df=norm_agg_split_df,
            idx_list=idx_list,
            halide_weights=halide_weights,
            halide_ts_weights=halide_ts_weights,
            agg_feats=agg_feats,
            TICC_return_cv_all=TICC_return_cv_all,
            ticc_num_proc=ticc_num_proc,
            ticc_train=ticc_train,
            reg_ticc_train=reg_ticc_train,
            TICC_clusters=TICC_clusters,
            te_perc=te_perc,
            final_df=final_df,
            sel_feats=sel_feats,
            time_col=time_col,
        )
        for rp_idx in range(repeat_num)
    ]

    _array_task_id = os.getenv('SLURM_ARRAY_TASK_ID')
    if _array_task_id is not None:
        _single_rp_idx = int(_array_task_id)
        _repeat_kwargs_list = [kw for kw in _repeat_kwargs_list
                               if kw['rp_idx'] == _single_rp_idx]
        _parallel_repeats = 1
        results_file_name = (f'results_cca_cleantrain_{_data_tag}_{_weight_tag}'
                             f'{_crit_tag}{_contrast_tag}{_variant_tag}'
                             f'{_nrcl_tag}{_tag_suffix}_rp{_single_rp_idx}.csv')
        results_file_path = os.path.join(results_dir_base, results_file_name)
        with open(results_file_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ('Round', 'Accuracy', 'Recall', 'Precision', 'F-score', 'AUC', 'APR', 'Jaccard'))
        print(f"[SLURM] Array task {_single_rp_idx} → {results_file_path}")

    all_repeat_results, repeat_results = [], []

    if _parallel_repeats > 1:
        print(f"[PARALLEL] {repeat_num} repeats, up to {_parallel_repeats} workers.")
        _ctx = multiprocessing.get_context('spawn')
        _results_map = {}
        with ProcessPoolExecutor(max_workers=min(_parallel_repeats, repeat_num),
                                  mp_context=_ctx) as executor:
            futures = {executor.submit(_run_one_repeat, kw): kw['rp_idx']
                       for kw in _repeat_kwargs_list}
            for fut in as_completed(futures):
                rp, all_iter, best = fut.result()
                _results_map[rp] = (all_iter, best)
                print(f"[PARALLEL] repeat {rp} finished.")
        for rp_idx in range(repeat_num):
            all_iter, best = _results_map[rp_idx]
            all_repeat_results.append(all_iter)
            repeat_results.append(best)
            with open(results_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for k, ir in enumerate(all_iter):
                    writer.writerow([f'rp{rp_idx}_iter{k}'] + ir)
                writer.writerow([f'rp{rp_idx}_best'] + best)
    else:
        for kw in _repeat_kwargs_list:
            rp_idx, all_iter, best = _run_one_repeat(kw)
            all_repeat_results.append(all_iter)
            repeat_results.append(best)
            with open(results_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for k, ir in enumerate(all_iter):
                    writer.writerow([f'rp{rp_idx}_iter{k}'] + ir)
                writer.writerow([f'rp{rp_idx}_best'] + best)

    avg_result = np.mean(np.array(repeat_results), axis=0)
    std_result = np.std(np.array(repeat_results), axis=0)
    agg_result = [f'{avg_result[i]:0.3f}({std_result[i]:0.2f})'
                  for i in range(len(avg_result))]

    with open(results_file_path, 'a', newline='') as f:
        csv.writer(f).writerow(['final'] + agg_result)

    print(f"\n[CCA] Final results saved to {results_file_path}")
    print(f"[CCA] Aggregate: {agg_result}")
