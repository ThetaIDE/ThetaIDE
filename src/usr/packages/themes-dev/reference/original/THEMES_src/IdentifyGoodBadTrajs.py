import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from EM_EDM_utils import *

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


# Select critical features by LASSO
def SelectFeaturesByLasso(X, y, alpha_range=[0.0005, 0.005, 0.0001], opt_alpha=np.nan, gs=True):
    if gs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        pipeline = Pipeline([('scaler', StandardScaler()),
                             ('model', Lasso())])

        search = GridSearchCV(pipeline,
                              {'model__alpha': np.arange(alpha_range[0], alpha_range[1], alpha_range[2])},
                              cv=5, scoring="neg_mean_squared_error", verbose=3)

        search.fit(X_train, y_train)
        opt_alpha = search.best_params_['model__alpha']
        print('Optimal alpha: ', opt_alpha)

        coefficients = search.best_estimator_.named_steps['model'].coef_
    else:
        clf = Lasso(alpha=opt_alpha)
        clf.fit(X, y)
        coefficients = clf.coef_

    importance = np.abs(coefficients)
    return importance


# Slice the good/bad trajectories for training/testing sets
# conv_final_idx - finalized trajs idx to be considered for training/testing
# idx_list - indexes based on LSTM results
# def SliceGoodBadForTrainTest(conv_final_idx, idx_list, te_perc=0.1, rp_idx=0, verbo=False):
#     [idx_00, idx_10, idx_01, idx_11] = idx_list
#     # REVISED HERE (11/30/2021)
#     good_group = sorted(idx_10) # sorted(idx_00+idx_10)
#     bad_group = sorted(idx_01) # sorted(idx_01+idx_11)
#
#     # Get the index for good & bad trajectories in pre_split_df
#     good_idx = [conv_final_idx[i] for i in good_group]
#     bad_idx = [conv_final_idx[i] for i in bad_group]
#
#     print(len(good_idx))
#     print(len(bad_idx))
#
#     # Determine the number of good trajs for training/testing
#     random.seed(len(good_idx)+1e3+rp_idx)
#     sel_good_group = sorted(random.sample(good_group, int(len(good_idx)*te_perc)))
#     rem_good_group = sorted(list(set(good_group) - set(sel_good_group)))
#     te_good_idx = sorted([conv_final_idx[i] for i in sel_good_group])
#
#     # Determine the number of bad trajs for training/testing
#     random.seed(len(bad_idx)+1e3+rp_idx)
#     sel_bad_group = sorted(random.sample(bad_group, int(len(bad_idx)*te_perc)))
#     rem_bad_group = sorted(list(set(bad_group) - set(sel_bad_group)))
#     te_bad_idx = sorted([conv_final_idx[i] for i in sel_bad_group])
#
#     print(len(te_good_idx))
#     print(len(te_bad_idx))
#
#     # Get the good/bad index for training data
#     tr_good_idx = sorted([conv_final_idx[i] for i in rem_good_group])
#     tr_bad_idx = sorted([conv_final_idx[i] for i in rem_bad_group])
#
#     # Decide the index for training/testing
#     tr_idx = sorted(tr_good_idx) # sorted(tr_good_idx)
#     te_good_idx = te_good_idx
#     te_bad_idx = te_bad_idx
#
#     if verbo:
#         print('* Good Trajs: Train-{}, Test-{}'.format(len(tr_good_idx), len(te_good_idx)))
#         print('* Bad Trajs: Train-{}, Test-{}'.format(len(tr_bad_idx), len(te_bad_idx)))
#         print('* All trajs: Train-{}, Test-{}&{}'.format(len(tr_idx), len(te_good_idx), len(te_bad_idx)))
#         print('-------------------------------')
#
#     return tr_idx, tr_good_idx, tr_bad_idx, te_good_idx, te_bad_idx, sel_good_group, sel_bad_group, rem_good_group, rem_bad_group


def SliceGoodBadForTrainTest(idx_list, te_perc=0.1, rp_idx=0, verbo=False, base_seed=0,
                              include_suboptimal=False, data_mode=None, max_ss_ratio=None,
                              max_nonexpert_ratio=None):
    """
    Split trajectories into train/test sets.

    data_mode (str | None) controls which quadrants enter the training set.
    When set it takes precedence over include_suboptimal.
      'e'   — expert only: shock→non-shock train trajectories  (default)
      'ei'  — expert + suboptimal: adds non-shock→non-shock (00) train trajectories
      'en'  — expert + negative: adds shock→shock (11) train trajectories
      'all' — all quadrants: shock→non-shock + non-shock→non-shock +
              non-shock→shock + shock→shock train trajectories

    max_nonexpert_ratio caps non-expert count to ratio × expert train count:
      'ei'  — caps the non-shock→non-shock (00) pool
      'en'  — caps the shock→shock (11) pool
      'all' — caps each of the three non-expert quadrants independently at
              floor(ratio/3 × expert_count), so total non-expert ≈ ratio × expert_count
              with equal representation from each semantic group

    Test set is always expert-only (shock→non-shock held-out portion).
    """
    # Resolve effective mode; data_mode takes precedence over include_suboptimal.
    if data_mode is None:
        data_mode = 'ei' if include_suboptimal else 'e'

    [idx_00, idx_10, idx_01, idx_11] = idx_list
    # Expert demonstrations: shock → non-shock (positive outcome)
    good_idx = sorted(idx_10)
    # Negative demonstrations: non-shock → shock
    bad_idx = sorted(idx_01)
    print(len(good_idx))
    print(len(bad_idx))

    # Use local RNG instances so global random state is not affected.
    # Split seed incorporates base_seed so changing BASE_SEED also changes splits.
    rng_good = random.Random(base_seed + len(good_idx) + 1000 + rp_idx)
    sel_good_group = sorted(rng_good.sample(good_idx, int(len(good_idx) * te_perc)))
    rem_good_group = sorted(list(set(good_idx) - set(sel_good_group)))
    te_good_idx = sorted(sel_good_group)

    # Determine the number of bad trajs for training/testing
    rng_bad = random.Random(base_seed + len(bad_idx) + 1000 + rp_idx)
    sel_bad_group = sorted(rng_bad.sample(bad_idx, int(len(bad_idx) * te_perc)))
    rem_bad_group = sorted(list(set(bad_idx) - set(sel_bad_group)))
    te_bad_idx = sorted(sel_bad_group)

    print(len(te_good_idx))
    print(len(te_bad_idx))

    # Get the expert/negative index for training data
    tr_good_idx = sorted(rem_good_group)
    tr_bad_idx  = sorted(rem_bad_group)

    # E+I mode: add non-shock→non-shock (idx_00) to training.
    # A separate RNG seed (offset 2000) keeps this split independent from good/bad splits.
    if data_mode in ('ei', 'all') and len(idx_00) > 0:
        subopt_idx = sorted(idx_00)
        rng_sub = random.Random(base_seed + len(subopt_idx) + 2000 + rp_idx)
        sel_sub_group = sorted(rng_sub.sample(subopt_idx, int(len(subopt_idx) * te_perc)))
        rem_sub_group = sorted(list(set(subopt_idx) - set(sel_sub_group)))
    else:
        rem_sub_group = []

    # Helper: build the shock→shock (idx_11) train split (shared by 'en' and 'all' modes).
    # RNG offset 3000 keeps this split independent from good/bad/subopt splits.
    if data_mode in ('en', 'all'):
        if len(idx_11) > 0:
            ss_idx = sorted(idx_11)
            rng_ss = random.Random(base_seed + len(ss_idx) + 3000 + rp_idx)
            sel_ss = sorted(rng_ss.sample(ss_idx, int(len(ss_idx) * te_perc)))
            rem_ss = sorted(list(set(ss_idx) - set(sel_ss)))
            # Legacy per-quadrant SS cap (superseded by max_nonexpert_ratio when set).
            if max_ss_ratio is not None and max_nonexpert_ratio is None and \
                    len(rem_ss) > int(max_ss_ratio * len(tr_good_idx)):
                cap = int(max_ss_ratio * len(tr_good_idx))
                rng_cap = random.Random(base_seed + len(rem_ss) + 5000 + rp_idx)
                rem_ss = sorted(rng_cap.sample(rem_ss, cap))
        else:
            rem_ss = []
    else:
        rem_ss = []

    if data_mode == 'all':
        # Cap each non-expert quadrant independently at floor(ratio/3 × expert_count).
        # Equal per-quadrant budget ensures each semantic group (weakly-positive NS→NS,
        # weakly-negative S→S, strongly-negative NS→S) contributes equally to training,
        # preventing the largest quadrant from dominating the combined sample.
        # Total non-expert ≈ ratio × expert_count, same budget as ei/en for fair comparison.
        if max_nonexpert_ratio is not None:
            per_cap = int(max_nonexpert_ratio / 3 * len(tr_good_idx))
            if per_cap > 0:
                if len(rem_sub_group) > per_cap:
                    rng_00 = random.Random(base_seed + len(rem_sub_group) + 7000 + rp_idx)
                    rem_sub_group = sorted(rng_00.sample(rem_sub_group, per_cap))
                if len(tr_bad_idx) > per_cap:
                    rng_01 = random.Random(base_seed + len(tr_bad_idx) + 8000 + rp_idx)
                    tr_bad_idx = sorted(rng_01.sample(tr_bad_idx, per_cap))
                if len(rem_ss) > per_cap:
                    rng_11 = random.Random(base_seed + len(rem_ss) + 9000 + rp_idx)
                    rem_ss = sorted(rng_11.sample(rem_ss, per_cap))
        all_nonexpert = rem_sub_group + tr_bad_idx + rem_ss
        if verbo:
            print(f'* Non-expert in Train: {len(all_nonexpert)} '
                  f'(per_cap={max_nonexpert_ratio}/3× {len(tr_good_idx)} expert — '
                  f'NS→NS={len(rem_sub_group)}, NS→S={len(tr_bad_idx)}, S→S={len(rem_ss)})')
        tr_idx = sorted(tr_good_idx + all_nonexpert)
    elif data_mode == 'en':
        # Expert + shock→shock: apply ratio cap to the SS (11) pool only.
        if max_nonexpert_ratio is not None:
            cap = int(max_nonexpert_ratio * len(tr_good_idx))
            if len(rem_ss) > cap and cap > 0:
                rng_ne = random.Random(base_seed + len(rem_ss) + 6000 + rp_idx)
                rem_ss = sorted(rng_ne.sample(rem_ss, cap))
        if verbo:
            print(f'* Shock→Shock (11) in Train: {len(rem_ss)} '
                  f'(cap={max_nonexpert_ratio}× {len(tr_good_idx)} expert)')
        tr_idx = sorted(tr_good_idx + rem_ss)
    else:
        # e / ei modes: apply ratio cap to suboptimal (00) pool if set.
        if max_nonexpert_ratio is not None:
            cap = int(max_nonexpert_ratio * len(tr_good_idx))
            if len(rem_sub_group) > cap and cap > 0:
                rng_ne = random.Random(base_seed + len(rem_sub_group) + 6000 + rp_idx)
                rem_sub_group = sorted(rng_ne.sample(rem_sub_group, cap))
        if verbo:
            print(f'* Non-shock→Non-shock (00) in Train: {len(rem_sub_group)} '
                  f'(cap={max_nonexpert_ratio}× {len(tr_good_idx)} expert)')
        tr_idx = sorted(tr_good_idx + rem_sub_group)

    if verbo:
        print('* Expert Trajs: Train-{}, Test-{}'.format(len(tr_good_idx), len(te_good_idx)))
        print('* Negative Trajs: Train-{}, Test-{}'.format(len(tr_bad_idx), len(te_bad_idx)))
        print('* Suboptimal (00) in Train: {}'.format(len(rem_sub_group)))
        print('* All trajs: Train-{}, Test-{}'.format(len(tr_idx), len(te_good_idx)))
        print('-------------------------------')

    return tr_idx, tr_good_idx, tr_bad_idx, te_good_idx, te_bad_idx, sel_good_group, sel_bad_group, rem_good_group, rem_bad_group


# Get the index of trajs with any actions
def RefinedIDsWithActions(df, idx_list):
    tmp_act_num, tmp_lab_sum = 0, 0
    tmp_idx_list = []
    for idx in idx_list:
        tmp_lab = list(df[idx]['Shock'])[-1]
        tmp_act_num = sum(df[idx]['a_0'])

        if tmp_lab == 1:
            tmp_lab_sum += 1
        if tmp_act_num < len(df[idx]):
            tmp_act_num += 1
            tmp_idx_list.append(idx)

    return tmp_idx_list



# ----------------------------------------------------------------------------------------------------
### Learn the weights for good/bad trajectories

# Manually decide the weights for state-action pairs
def ManualDesigndedSAWeight(action_tuple):
    if action_tuple == (1,0,0):
        weight = 0.5
    elif action_tuple == (1,0,1):
        weight = -0.5
    elif action_tuple == (1,1,0):
        weight = -1
    elif action_tuple == (1,1,1):
        weight = 1
    elif action_tuple == (0,0,0):
        weight = 1
    elif action_tuple == (0,0,1):
        weight = -1
    elif action_tuple == (0,1,0):
        weight = -0.5
    elif action_tuple == (0,1,1):
        weight = 0.5
    return weight


# Decide the LR learned weights
def LRLearnedSAWeight(action_tuple, lr_weights, pos_weight=1):
    if action_tuple == (1,0,0):
        weight = lr_weights[0]*pos_weight
    elif action_tuple == (1,0,1):
        weight = -lr_weights[1]*pos_weight
    elif action_tuple == (1,1,0):
        weight = -lr_weights[2]*pos_weight
    elif action_tuple == (1,1,1):
        weight = lr_weights[3]*pos_weight
    elif action_tuple == (0,0,0):
        weight = lr_weights[4]
    elif action_tuple == (0,0,1):
        weight = -lr_weights[5]
    elif action_tuple == (0,1,0):
        weight = -lr_weights[6]
    elif action_tuple == (0,1,1):
        weight = lr_weights[7]
    return weight


# Check the counts for binary tuples
def CheckActionTupleCounts(action_list):
    counts = []
    counts.append(action_list.count((1,0,0)))
    counts.append(action_list.count((1,0,1)))
    counts.append(action_list.count((1,1,0)))
    counts.append(action_list.count((1,1,1)))
    counts.append(action_list.count((0,0,0)))
    counts.append(action_list.count((0,0,1)))
    counts.append(action_list.count((0,1,0)))
    counts.append(action_list.count((0,1,1)))
    return counts


# Get the action tuple based on the learned good/bad student
def ActionTupleFeatureGenerator(trajs, good_student, bad_student):
    conv_feat, conv_action = [], []
    for trajIdx in np.arange(len(trajs)):
        tmp_traj = trajs[trajIdx]
        tmp_action = []
        for idx in range(len(tmp_traj)):
            tmp_good_return = good_student.select_action(tmp_traj[idx][0])
            tmp_bad_return = bad_student.select_action(tmp_traj[idx][0])
            tmp_action_tuple = (tmp_traj[idx][1][0], tmp_good_return[0].item(), tmp_bad_return[0].item())
            tmp_action.append(tmp_action_tuple)

        tmp_feat = CheckActionTupleCounts(tmp_action)
        tmp_feat = [i / sum(tmp_feat) for i in tmp_feat]
        conv_feat.append(tmp_feat)
        conv_action.append(tmp_action)
    return np.array(conv_feat), conv_action


# Learn the good/bad students and save the models to files
def GetGoodBadStudents(env, tr_good_trajs, tr_bad_trajs, tr_good_returns, tr_bad_returns, train_steps=10):
    good_dict_array = np.array({'trajs': tr_good_trajs, 'returns': tr_good_returns})
    good_teacher_path = 'voter/good_learner.npy'
    np.save(good_teacher_path, good_dict_array)

    bad_dict_array = np.array({'trajs': tr_bad_trajs, 'returns': tr_bad_returns})
    bad_teacher_path = 'voter/bad_learner.npy'
    np.save(bad_teacher_path, bad_dict_array)

    # Load the config file for EDM
    config_path = '../models/' + env + '/EM-EDM_models/para_config.npy'
    config = np.load(config_path, allow_pickle=True).item()

    # Save the good/bad trajs to file
    base_path = '../../'
    env_data_path = base_path + 'volume/' + env
    config['ENV_VOLUME_PATH'] = env_data_path
    config['NUM_STEPS_TRAIN'] = train_steps

    config['MODEL_PATH'] = 'voter/good_learner_model.sav'
    good_student, _ = EncapsulateEDM(config, base_path, good_teacher_path)

    config['MODEL_PATH'] = 'voter/bad_learner_model.sav'
    bad_student, _ = EncapsulateEDM(config, base_path, bad_teacher_path)

    return good_student, bad_student


