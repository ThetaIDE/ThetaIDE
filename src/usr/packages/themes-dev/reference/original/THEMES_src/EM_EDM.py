import sys 
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../../IRL/unimodal-irl-master/unimodal-irl-master/unimodal_irl/')
sys.path.append('../../preprocessing/')

import warnings
import os
warnings.filterwarnings('ignore')

import itertools

from preprocessing.EHR_demos_generator import *
from student.evaluation_utils import *
from EM_EDM_utils import *
from NV_MLIRL import UpdatedMLIRL
from THEMES_helper import *
from IRL.MLIRL import *
from IdentifyGoodBadTrajs import *
import json

pd.options.mode.chained_assignment = None  # default='warn'

base_path = '../../'

env = 'CCHS' # Change this CCHS/Mayo/MIMIC

args = {'env_name': f'EHRs/{env}', 
        'trial': 1}

result_path = base_path + 'results/'

# Load the config file
config_path = '../models/' + args['env_name'] + '/EM-EDM_models/para_config.npy'
config = np.load(config_path, allow_pickle=True).item()

# Update the config values if necessary
config['NUM_STEPS_TRAIN'] = 500
config['EMEDM_LLH_THRES_CONV'] = 1e-5
config['EMEDM_LLH_THRES_DES'] = 1e2

config['BATCH_SIZE'] = 64
config['MLP_WIDTHS'] = 32
config['EMEDM_ITERS'] = 50

config['EMEDM_nrCl'] = 3
config['SGLD_BUFFER_SIZE'] = 10000
config['SGLD_LEARN_RATE'] = 1e-3

config['SAMPLE_BUFFER'] = 'balanced'
config['BASE_PATH'] = base_path
config['MODEL_LEARNER'] = 'EDM'
print(config)

sampling_num = 10000
avg_patterns = 'weighted'

if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Load the data
    concat_final_df = pd.read_csv(f'../../volume/{env}_lstm_filtered_dataframe_md.csv') #Mayo_lstm_filtered_dataframe_md.csv
    idx_list = np.load(f'../../volume/{env}_lstm_filtered_idx_list_md.npy', allow_pickle=True) #Mayo_lstm_filtered_idx_list_md.npy

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
    te_perc = 0.2
    repeat_num = 10

    results_dir_base = result_path + args['env_name'] + '/EM_EDM/'
    results_file_name = 'results.csv'
    results_file_path = os.path.join(results_dir_base, results_file_name)

    if not os.path.exists(results_dir_base):
        os.makedirs(results_dir_base)
        
    with open(results_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('Round', 'Accuracy', 'Recall', 'Precision', 'F-score', 'AUC', 'APR', 'Jaccard'))

    rp_idx = 0
    repeat_results = []
    for rp_idx in range(repeat_num): 
        print('repeat idx: ', rp_idx)

        # Get the expert demonstrations for training
        idx_slice_return = SliceGoodBadForTrainTest(idx_list, te_perc=te_perc, rp_idx=rp_idx)
        ticc_train_idx, ticc_test_idx = idx_slice_return[1], idx_slice_return[3]
        
        ## -------------------------------------------------------------------------------------------------------------------------
        # Slice the data
        slice_tr_agg_df = []
        slice_te_agg_df = []

        slice_tr_agg_df = [norm_agg_split_df[i] for i in ticc_train_idx]
        slice_te_agg_df = [norm_agg_split_df[i] for i in ticc_test_idx]
                    
        train_demos = {'trajs': [], 'returns':[]}
        for i in range(len(slice_tr_agg_df)): 
            tmp_s = [np.array(slice_tr_agg_df[i].iloc[k][agg_feats], dtype=np.float64) for k in range(len(slice_tr_agg_df[i]))]
            tmp_a = [np.array([slice_tr_agg_df[i].iloc[k]['Action'].astype('uint32')]) for k in range(len(slice_tr_agg_df[i]))]
            tmp_trajs = [(tmp_s[k], tmp_a[k]) for k in range(len(tmp_s))]
            tmp_returns = [np.nan for _ in range(len(tmp_s))]
            train_demos['trajs'].append(tmp_trajs)
            train_demos['returns'].append(tmp_returns)
        train_traj_num = len(train_demos['trajs'])

        # ---------------------------------------------------------------------------------------------------
        # EM-EDM Starting
        # Apply EM-EDM to learn the clusters
        config['ENV_VOLUME_PATH'] = base_path + 'volume/' + args['env_name'] + '/EM-EDM' + '/teachers/' + str(rp_idx) + '/'
        config['CLUS_BASE_PATH'] = config['ENV_VOLUME_PATH']
        Path(config['CLUS_BASE_PATH']).mkdir(parents = True, exist_ok = True)

        ### EM-EDM Training process
        # Load the expert trajectories
        clus_teachers = train_demos
        clus_trajs = clus_teachers['trajs']
        traj_num = len(clus_trajs)

        model_folder = '../models/' + args['env_name'] + '/EM-EDM_models/'
        model_path = (model_folder + 'EM-EDM' + '_model_' + str(traj_num) + '_fold_' + str(rp_idx) + '.sav')
        Path(model_folder).mkdir(parents = True, exist_ok = True)
        config['MODEL_PATH'] = model_path
        
        # Learn different policies by EM-EDM
        print('** Applying EM-EDM to learn policies... ')
        nrCl = config['EMEDM_nrCl']
        
        # Initialize the clusters 
        pred_labs = InitEMEDMClusters(clus_trajs, nrCl, sampling_num, init_mode='random',
                                    DTW_thres=1e5, max_iter=10, init_seed=0)

        # EM-EDM to cluster the trajectories
        student_list, rho, LLH, nrCl, pred_labs, pred_probs = EMEDMWarped(clus_teachers, nrCl, pred_labs, 
                                                                        config, decay_expert=False)

        # Save the learned models (policies per cluster) to file
        pickle.dump((student_list, rho), open(model_path, 'wb'))
        print(LLH)

        # --------------------------------------------------------------------------------------------------------------
        # Test the model

        # Load the test trajectories
        test_demos = {'trajs': [], 'returns': []}
        for i in range(len(slice_te_agg_df)): 
            tmp_s = [np.array(slice_te_agg_df[i].iloc[k][agg_feats], dtype=np.float64) for k in range(len(slice_te_agg_df[i]))]
            tmp_a = [np.array([slice_te_agg_df[i].iloc[k]['Action'].astype('uint32')]) for k in range(len(slice_te_agg_df[i]))]
            tmp_trajs = [(tmp_s[k], tmp_a[k]) for k in range(len(tmp_s))]
            tmp_returns = [np.nan for _ in range(len(tmp_s))]
            test_demos['trajs'].append(tmp_trajs)
            test_demos['returns'].append(tmp_returns)

        # Save the test trajs to file
        test_path = config['ENV_VOLUME_PATH'] + '/test_' + str(rp_idx) + '_trajs.npy'
        tmp_dict_array = np.array(test_demos)
        np.save(test_path, tmp_dict_array)

        # Assign the test data into clusters
        print('Testing: ')
        test_model, test_rho = student_list, rho

        pred_test_labs = PredictClusterForTest(test_model, test_demos['trajs'], nrCl, test_rho, config)
        unique_test_labs = sorted(np.unique(pred_test_labs))

        # Path to store the predicted clusters
        env_data_path = base_path + 'volume/' + args['env_name'] + '/EM_EDM' 
        base_split_path = env_data_path + '/PM_EM_EDM/'
        Path(base_split_path).mkdir(parents = True, exist_ok = True)

        test_dict_array = SplitTrajsByLabs(test_demos, nrCl, pred_test_labs)
        test_path_dict = SaveSplitTrajsToFile(base_split_path, test_dict_array, 
                                            unique_test_labs, sampling_num, rp_idx, mode='test')
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

        AUC = AUCScore(true_act, pred_act, pred_prob, avg_patterns=avg_patterns)
        APR = APScore(true_act, pred_act, pred_prob, avg_patterns=avg_patterns)
        Jascore = JaccardScore(true_act, pred_act, avg_patterns=avg_patterns)
        tmp_results = metrics + [AUC] + [APR] + [Jascore]

        repeat_results.append(np.array(tmp_results)) 
        with open(results_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([rp_idx] + tmp_results)

    avg_result = np.mean(np.array(repeat_results), axis=0)
    std_result = np.std(np.array(repeat_results), axis=0)
    agg_result = []

    for i in range(len(avg_result)):
        agg_result.append(f'{avg_result[i]:0.3f}({std_result[i]:0.2f})')

    with open(results_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['final'] + agg_result)

