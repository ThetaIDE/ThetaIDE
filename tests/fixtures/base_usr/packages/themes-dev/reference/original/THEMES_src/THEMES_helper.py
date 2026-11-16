import sys
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import itertools

from preprocessing.EHR_demos_generator import *
from EM_EDM_utils import *

# Convert a numerical list to one-hot array
def ConvertNumListToOneHot(orig_list, num):
    orig_arr = np.array(orig_list)
    onehot_arr = np.zeros((orig_arr.size, num))
    onehot_arr[np.arange(orig_arr.size),orig_arr] = 1
    return onehot_arr


# Generate the discrete demos and dynamics based on TICC and EMEDM results
def GenerateDiscreteSAP(TICC_orig_traj_idx, dis_TICC_split, EMEDM_pred_labs, TICC_clusters=6):
    # Generate the discrete states
    dis_TICC_state = [[dis_TICC_split[j][i][0] for i in range(len(dis_TICC_split[j]))] for j in
                      range(len(dis_TICC_split))]

    # Generate the discrete actions
    dis_TICC_action = [[0] * len(dis_TICC_state[i]) for i in range(len(dis_TICC_split))]
    for idx in range(len(EMEDM_pred_labs)):
        dis_TICC_action[TICC_orig_traj_idx[idx][0]][TICC_orig_traj_idx[idx][1]] = EMEDM_pred_labs[idx]

    # Convert state to one-hot vector (Not used for now)
    onehot_TICC_state = [ConvertNumListToOneHot(dis_TICC_state[i], TICC_clusters) for i in range(len(dis_TICC_state))]

    # Get the discrete trajectories
    dis_demos = [[np.nan, [(dis_TICC_state[i][j], dis_TICC_action[i][j])
                           for j in range(len(dis_TICC_action[i]))]] for i in range(len(dis_TICC_action))]

    # Get the state-action-state transition probabilities
    state_num = TICC_clusters
    act_num = len(np.unique(EMEDM_pred_labs))
    P_a = np.zeros((state_num, act_num, state_num))

    tmp_state_pairs = [[(dis_TICC_state[i][j], dis_TICC_action[i][j], dis_TICC_state[i][j + 1])
                        for j in range(len(dis_TICC_state[i]) - 1)] for i in range(len(dis_TICC_state))]
    tmp_state_pairs = list(itertools.chain.from_iterable(tmp_state_pairs))

    for state in range(state_num):
        for action in range(act_num):
            tmp_sum = 0
            for next_s in range(state_num):
                tmp_count = tmp_state_pairs.count((state, action, next_s))
                P_a[state][action][next_s] = tmp_count
                tmp_sum += tmp_count
            if tmp_sum != 0:
                P_a[state][action][:] = P_a[state][action][:] / tmp_sum

    all_dis_TICC_state = list(itertools.chain.from_iterable(dis_TICC_state))

    P_0 = np.zeros((state_num))
    for state in range(state_num):
        P_0[state] = all_dis_TICC_state.count(state)

    P_0 = P_0 / sum(P_0)

    return dis_demos, P_a, P_0


# Generate the weights based on rewards
def GenerateRewardsWeights(agg_df, agg_feats, TICC_return_cv, student_list, rho, config,
                           state_action_reward, TICC_clusters=6, nrCl=3):
    # Get the sub-trajectories for all training data
    _, _, all_comb_demos, all_orig_traj_idx, all_dis_TICC_split = PartitionTICCSubTrajs(agg_df, agg_feats,
                                                                                        TICC_return_cv[1],
                                                                                        TICC_clusters=TICC_clusters)

    all_pred_labs, all_zij, _, _ = EM_EDM_Estep(all_comb_demos['trajs'], nrCl, student_list, rho, config)
    # EM_EDM_Estep may collapse clusters further, shrinking all_zij's column count
    # below state_action_reward's columns — truncate to avoid a dot-product shape crash.
    if all_zij.shape[1] < state_action_reward.shape[1]:
        state_action_reward = state_action_reward[:, :all_zij.shape[1]]

    # Get the sub-trajectory orders in the original trajectory
    all_orig_traj_tup = list(all_orig_traj_idx.values())

    whole_seq_idx = 0
    sub_orders, tmp_sub_orders = [], []
    for idx in range(len(all_orig_traj_tup)):
        if all_orig_traj_tup[idx][0] == whole_seq_idx:
            tmp_sub_orders.append(all_orig_traj_tup[idx][1])
        else:
            whole_seq_idx += 1
            sub_orders.append(tmp_sub_orders)
            tmp_sub_orders = [all_orig_traj_tup[idx][1]]

    sub_orders.append(tmp_sub_orders)
    sorted_sub_orders = [np.argsort(i) for i in sub_orders]

    # Get the labels and occurred numbers for each timestamp
    all_dis_TICC_labs = list(itertools.chain.from_iterable([[all_dis_TICC_split[i][j][0] for j in sorted_sub_orders[i]]
                                                            for i in range(len(all_dis_TICC_split))]))
    all_dis_TICC_nums = list(itertools.chain.from_iterable([[all_dis_TICC_split[i][j][1] for j in sorted_sub_orders[i]]
                                                            for i in range(len(all_dis_TICC_split))]))

    # Generate the weights for each timestamp based on the rewards
    all_r = list(itertools.chain.from_iterable([list(i) for i in state_action_reward]))
    w_mean = np.mean([abs(i) for i in all_r])

    tmp_seq_idx, start_idx = 0, 0
    themes_w = []

    for idx in range(len(all_orig_traj_idx)):
        tmp_subseq_idx = all_orig_traj_idx[idx][0]
        if tmp_seq_idx == tmp_subseq_idx:
            w = np.dot(state_action_reward[all_dis_TICC_labs[idx], :], all_zij[idx]) / np.shape(state_action_reward)[1]
            # w_list = [w] + [w_mean] * (all_dis_TICC_nums[idx] - 1)
            w_list = [w] * all_dis_TICC_nums[idx]
            themes_w.extend(w_list)
            start_idx += 1
        else:
            start_idx = 0
            w = np.dot(state_action_reward[all_dis_TICC_labs[idx], :], all_zij[idx]) / np.shape(state_action_reward)[1]
            w_list = [w] * all_dis_TICC_nums[idx]
            themes_w.extend(w_list)
            tmp_seq_idx += 1
            start_idx += 1

    return themes_w, all_dis_TICC_labs, all_dis_TICC_nums


