#!/usr/bin/env python
# coding: utf-8

import sys 
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../../IRL/unimodal-irl-master/unimodal-irl-master/unimodal_irl/')
sys.path.append('../../preprocessing/')

import warnings
warnings.filterwarnings('ignore')

import itertools

from preprocessing.EHR_demos_generator import *
from student.evaluation_utils import *
from EM_EDM_utils import *
from NV_MLIRL import UpdatedMLIRL
from THEMES_helper import *
from IRL.MLIRL import *
from IdentifyGoodBadTrajs import *

pd.options.mode.chained_assignment = None  # default='warn'

base_path = '../../' 
args = {'env_name': 'EHRs', 
        'trial': 1}


# Load the config file
config_path = '../models/' + args['env_name'] + '/EM-EDM_models/para_config.npy'
config = np.load(config_path, allow_pickle=True).item()

# Update the config values if necessary
config['NUM_STEPS_TRAIN'] = 500
config['EMEDM_LLH_THRES_CONV'] = 1e-5
config['EMEDM_LLH_THRES_DES'] = 1e2

config['BATCH_SIZE'] = 64
config['MLP_WIDTHS'] = 32
config['EMEDM_ITERS'] = 10 

config['EMEDM_nrCl'] = 3
config['SGLD_BUFFER_SIZE'] = 10000
config['SGLD_LEARN_RATE'] = 1e-3

config['SAMPLE_BUFFER'] = 'balanced'
config['BASE_PATH'] = base_path
config['MODEL_LEARNER'] = 'EDM'
print(config)

sampling_num = 10000
# avg_patterns = 'binary'
avg_patterns = 'weighted'


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Load the data
    concat_final_df = pd.read_csv('../../volume/lstm_filtered_dataframe.csv')
    #idx_list = np.load('../../volume/lstm_filtered_idx_list.npy', allow_pickle=True)
    idx_list = np.load('../../volume/lstm_filtered_idx_list_md.npy', allow_pickle=True)

    final_df = []
    selected_vid = sorted(np.unique(concat_final_df.VisitIdentifier))
    for vid in selected_vid:
        final_df.append(concat_final_df.loc[concat_final_df.VisitIdentifier == vid])

    sel_feats = ['SystolicBP', 'MAP', 'RespiratoryRate', 'PulseOx', 'HeartRate', 'Temperature', 
                'WBC', 'BiliRubin', 'BUN', 'Lactate', 'Creatinine', 'Platelet', 'Bands', 'FIO2']
    time_col = 'MinutesFromArrival'

    max_sel_feats = ['max_' + i for i in sel_feats]
    min_sel_feats = ['min_' + i for i in sel_feats]
    agg_feats = sel_feats + max_sel_feats + min_sel_feats


    # Data Pre-processing
    agg_split_df = []
    for idx in range(len(selected_vid)):
        tmp_df = final_df[idx]
        min_max_time = [(i - 60, i) for i in tmp_df[time_col]]

        tmp_max = [list(np.max(tmp_df.loc[(tmp_df[time_col] >= min_max_time[i][0]) &
                                (tmp_df[time_col] <= min_max_time[i][1])][sel_feats], axis=0))
                for i in range(len(min_max_time))]

        tmp_min = [list(np.min(tmp_df.loc[(tmp_df[time_col] >= min_max_time[i][0]) &
                                (tmp_df[time_col] <= min_max_time[i][1])][sel_feats], axis=0))
                for i in range(len(min_max_time))]

        tmp_agg_data = np.concatenate([np.array(tmp_df), np.array(tmp_max), np.array(tmp_min)], axis=1)
        tmp_agg_df = pd.DataFrame(tmp_agg_data, columns=list(tmp_df.columns) + max_sel_feats + min_sel_feats)
        tmp_agg_df['Action'] = np.max(np.array(tmp_agg_df[['a_2']]), axis=1)
        agg_split_df.append(tmp_agg_df)

    # Get the max/min for each feature
    feat_max_min = {}
    for feat in agg_feats:
        tmp_vals = list(itertools.chain.from_iterable([list(agg_split_df[i][feat]) for i in range(len(agg_split_df))]))
        tmp_max, tmp_min = np.max(tmp_vals), np.min(tmp_vals)
        feat_max_min[feat] = (tmp_max, tmp_min)

    # Normalize the data
    norm_agg_split_df = copy.deepcopy(agg_split_df)
    for idx in range(len(norm_agg_split_df)):
        for feat in agg_feats:
            tmp_max, tmp_min = feat_max_min[feat][0], feat_max_min[feat][1]
            norm_agg_split_df[idx][feat] = (agg_split_df[idx][feat] - tmp_min)/(tmp_max - tmp_min)



    # -----------------------------------------------------------------------------------
    # Training process
    #te_perc = 0.1
    te_perc = 0.2
    repeat_num = 10

    rp_idx = 0
    all_repeat_results, repeat_results = [],[]
    for rp_idx in range(repeat_num): 
        print('repeat idx: ', rp_idx)

        # Get the expert demonstrations for training
        idx_slice_return = SliceGoodBadForTrainTest(idx_list, te_perc=te_perc, rp_idx=rp_idx)
        ticc_train_idx, ticc_test_idx = idx_slice_return[1], idx_slice_return[3]

        ## -------------------------------------------------------------------------------------------------------------------------
        ## Phase I: MT-TICC to learn the high-level states
        # Slice the data
        slice_tr_agg_df = [norm_agg_split_df[i] for i in ticc_train_idx]
        slice_te_agg_df = [norm_agg_split_df[i] for i in ticc_test_idx]

        # Initialization: Apply MT-TICC to learn the sub-sequences
        TICC_clusters = 7
        MTTICC_path = '../../volume/' + args['env_name'] + '/All/' + 'THEMESE_MTTICC_results' + '.npy'
        TICC_return_cv = TICCSplitTrajs(args['env_name'], final_df, sel_feats, time_col, TICC_train=True, fixed_window=2,
                                        TICC_clusters=TICC_clusters,
                                        lambda_parameter=0.001, beta=10, num_proc=6,
                                        input_pattern='multiple', window_pattern='fixed', maxIters=50,
                                        save_TICC_results=True, TICC_cluster_path=MTTICC_path)
        reg_TICC_return_cv = TICC_return_cv

        # Slice the training trajectories
        tr_TICC_return = [TICC_return_cv[1][i] for i in ticc_train_idx]
        
        # Get the sub-trajectories for EM-EDM training 
        _, _, comb_demos, orig_traj_idx, dis_TICC_split = PartitionTICCSubTrajs(slice_tr_agg_df, agg_feats, 
                                                                                tr_TICC_return, TICC_clusters=TICC_clusters)

        iteration_num, max_iter = 0, 10
        itertation_results = []
        MTTICC_LLH, EMEDM_LLH, IRL_LLH = [], [], []
        while iteration_num < max_iter:
            ## Phase II: EM-EDM to learn the high-level actions 
            # Apply EM-EDM to learn the cluster of sub-sequences
            config['ENV_VOLUME_PATH'] = base_path + 'volume/' + args['env_name']
            config['CLUS_BASE_PATH'] = '../../volume/THEMES/'

            ### EM-EDM Training process
            # Load the expert trajectories
            clus_teachers = comb_demos 
            clus_trajs, clus_returns = clus_teachers['trajs'], clus_teachers['returns']
            traj_num = len(clus_trajs)

            model_folder = '../models/' + args['env_name'] + '/EM-EDM_models/'
            model_path = (model_folder + 'THEMES' + '_model_' + str(traj_num) + '_fold_' + str(rp_idx) + '.sav')
            Path(model_folder).mkdir(parents = True, exist_ok = True)
            config['MODEL_PATH'] = model_path
            
            # Learn different policies by EM-EDM
            print('** Applying EM-EDM to learn policies... ')
            nrCl = config['EMEDM_nrCl']
            
            # Initialize the clusters 
            pred_labs = InitEMEDMClusters(clus_trajs, nrCl, sampling_num, init_mode='random',
                                        DTW_thres=1e5, max_iter=10, init_seed=0)

            # EM-EDM to cluster the sub-trajectories
            student_list, rho, LLH, nrCl, pred_labs, pred_probs = EMEDMWarped(clus_teachers, nrCl, pred_labs, 
                                                                            config, decay_expert=False)

            # Save the learned models (policies per cluster) to file
            pickle.dump((student_list, rho), open(model_path, 'wb'))
            EMEDM_LLH.append(LLH)
            plt.plot(LLH)
            plt.show()

            ## ---------------------------------------------------------------------------------------------------------------------
            ## Phase III: IRL to learn the high-level rewards
            if iteration_num < max_iter - 1:
                # Apply MLIRL to learn the rewards for each state-action pair
                # Get the converted discrete trajectories and dynamics
                # dis_demos, P_a, P_0 = GenerateDiscreteSAP(orig_traj_idx, dis_TICC_split, pred_labs, 
                #                                         TICC_clusters=TICC_clusters)

                # Conduct MLIRL to learn the rewards 
                print('** Applying MLIRL to learn rewards... ')
                state_num, act_num = TICC_clusters, nrCl
                dis_demos, P_a, P_0 = GenerateDiscreteSAP(orig_traj_idx, dis_TICC_split, pred_labs, 
                                                        TICC_clusters=TICC_clusters)
                # state_num, act_num = TICC_clusters, nrCl

                # MLIRL Env
                irl_states, irl_actions = np.arange(state_num), np.arange(act_num)
                irl_P0, irl_transitions = P_0, P_a
                irl_terminals = np.zeros((state_num))
                irl_gamma = 0.8
                input_MLIRL_args = (irl_states, irl_actions, irl_P0, irl_transitions, irl_terminals, irl_gamma)

                conv_dis_demos = []
                for idx in range(len(dis_demos)): 
                    conv_dis_demos.append([dis_demos[idx][1][:-1] + [(dis_demos[idx][1][-1][0], None)]])
                conv_dis_demos = [conv_dis_demos[i][0] for i in range(len(conv_dis_demos))]

                reward_bound = [-5,5]
                temp_reward, llh_list = UpdatedMLIRL(input_MLIRL_args, conv_dis_demos, bounds=reward_bound, scale=0.5)
                IRL_LLH.append(llh_list)

                tmp_max_range = max([abs(i) for i in temp_reward.x])
                norm_reward = [i/tmp_max_range for i in temp_reward.x]
                state_action_reward = np.reshape(norm_reward, (len(irl_states), len(irl_actions)), order='F')
                print('Rewards: ', state_action_reward)

                # Get the reward for each timestamp
                themes_w, _, _ = GenerateRewardsWeights(norm_agg_split_df, agg_feats, reg_TICC_return_cv, 
                                                        student_list, rho, config, state_action_reward, 
                                                        TICC_clusters=TICC_clusters, nrCl=nrCl)

                # Apply reward-regulated MT-TICC to resign the sub-sequences
                print('** Applying MTTICC to refine partitions... ')
                reward_col = 'reward_weight'
                w_final_df = copy.deepcopy(final_df)
                tmp_num = [0] + list(np.cumsum([len(i) for i in w_final_df]))

                for idx in range(len(tmp_num)-1): 
                    w_final_df[idx][reward_col] = themes_w[tmp_num[idx]:tmp_num[idx+1]]

                regMTTICC_path = ('../../volume/' + args['env_name'] + '/All/' + 'THEMESE_regMTTICC_results_rp_' 
                                + str(rp_idx) + 'iter_' + str(iteration_num) + '.npy')

                if iteration_num > 0: 
                    reg_init_labs = list(itertools.chain.from_iterable(reg_TICC_return_cv[0]))
                    reg_TICC_return_cv = REGTICCSplitTrajs(args['env_name'], w_final_df, sel_feats, time_col, reward_col, 
                                                        TICC_train=True, fixed_window=2, TICC_clusters= TICC_clusters, 
                                                        lambda_parameter=0.001, beta=10, #5, 
                                                        num_proc=6, input_pattern='multiple', window_pattern='fixed', 
                                                        maxIters=50, #1000, 
                                                        save_TICC_results=True, TICC_cluster_path=regMTTICC_path, 
                                                        init_labs = reg_init_labs, rewards_thres=0.5,
                                                        smooth_reg_mu=[1,1], smooth_reg_std=[.2,1],
                                                        )
                else: 
                    MTTICC_path_0 = '../../volume/' + args['env_name'] + '/All/' + 'THEMESE_MTTICC_results' + '.npy'
                    reg_TICC_return_cv = np.load(MTTICC_path_0, allow_pickle=True)
                
                MTTICC_LLH.append(reg_TICC_return_cv[2][0])

                tr_TICC_return = [reg_TICC_return_cv[1][i] for i in ticc_train_idx]
                _, _, comb_demos, orig_traj_idx, dis_TICC_split = PartitionTICCSubTrajs(slice_tr_agg_df, agg_feats, 
                                                                                        tr_TICC_return, 
                                                                                        TICC_clusters=TICC_clusters)
            iteration_num += 1
            
            # --------------------------------------------------------------------------------------------------------------
            # Test the model
            test_TICC_split = [TICC_return_cv[1][i] for i in ticc_test_idx]
            test_demos, _, sub_demos, _, _ = PartitionTICCSubTrajs(slice_te_agg_df, agg_feats, test_TICC_split, 
                                                        TICC_clusters=TICC_clusters)

            # Load the test trajectories
            test_teachers = sub_demos
            test_trajs, test_returns = test_teachers['trajs'], test_teachers['returns']

            # Assign the test data into clusters
            print('Testing: ')
            test_model, test_rho = student_list, rho

            #BIC, _, _ = postCalculateBIC(test_model, max(LLH), test_demos)

            #print('BIC ----------------------------------------------: {}'.format(BIC))

            pred_test_labs = PredictClusterForTest(test_model, test_trajs, nrCl, test_rho, config)
            #pred_test_labs = PredictClusterForTest(test_model, test_trajs, len(test_rho), test_rho, config)
            unique_test_labs = sorted(np.unique(pred_test_labs))

            # Path to store the predicted clusters
            env_data_path = base_path + 'volume/' + args['env_name']
            base_split_path = env_data_path + '/PM_EM_EDM/'
            Path(base_split_path).mkdir(parents = True, exist_ok = True)

            test_dict_array = SplitTrajsByLabs(test_teachers, nrCl, pred_test_labs)
            test_path_dict = SaveSplitTrajsToFile(base_split_path, test_dict_array, 
                                                unique_test_labs, sampling_num, rp_idx, mode='test')
            # clf_thres = 0.3

            # Check the performance of each cluster
            student_dict = {i: student_list[i] for i in range(len(student_list))}

            clus_result = []
            true_act, pred_act, pred_prob = [], [], []
            for clusIdx in unique_test_labs:
                print('cluster: {}'.format(clusIdx))
                if clusIdx in student_dict.keys():
                    EDM_return = student_dict[clusIdx].general_test(test_path_dict[clusIdx], avg_patterns=avg_patterns)
                    result, tmp_true_act, tmp_pred_act, tmp_prob_act = EDM_return
                else: 
                    print('RANDOMLY SELECT A CLUSTER TO FIT ...')
                    tmpIdx = random.sample(student_dict.keys(), 1)[0]
                    result, tmp_true_act, tmp_pred_act, tmp_prob_act = student_dict[tmpIdx].general_test(test_path_dict[clusIdx], 
                                                                                                        avg_patterns=avg_patterns)   
                clus_result.append(result)
                true_act.extend(tmp_true_act)
                pred_act.extend(tmp_pred_act)
                pred_prob.extend(tmp_prob_act)

            print('total performance: ')
            _, metrics = ModelEvaluate(true_act, pred_act, np.arange(2), avg_patterns=avg_patterns, verbo=True)

            with open('all_metrics.txt', 'w') as x:
                x.write(str(iteration_num) + '-------' + str(metrics[0]))
                x.write('\n')

            AUC = AUCScore(true_act, pred_act, pred_prob, avg_patterns=avg_patterns)
            APR = APScore(true_act, pred_act, pred_prob, avg_patterns=avg_patterns)
            Jascore = JaccardScore(true_act, pred_act, avg_patterns=avg_patterns)
            tmp_results = metrics + [AUC] + [APR] + [Jascore]
            itertation_results.append(tmp_results)
        
        all_repeat_results.append(itertation_results)
        repeat_results.append(itertation_results[-1])
