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

pd.options.mode.chained_assignment = None  # default='warn'

base_path = '../../'

env = 'MIMIC' # Change this CCHS/Mayo/MIMIC

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
TICC_clusters = 9
student_path = '../models/' + args['env_name'] + '/EM-EDM_models/' + 'THEMES' + '_model_All_good.sav'
TICC_cluster_path = ('../../volume/' + args['env_name'] + '/All/' + 'THEMESE_regMTTICC_All_results_iter_1.npy')



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
    # Get the expert demonstrations for training
    [idx_00, idx_10, idx_01, idx_11] = idx_list
    good_idx = sorted(idx_10)
    bad_idx = sorted(idx_01)
    print(f'Number of Good Visits: {len(good_idx)}')
    print(f'Number of Bad Visits: {len(bad_idx)}')

    ticc_train_idx, ticc_test_idx = good_idx, bad_idx

    ## -------------------------------------------------------------------------------------------------------------------------
    ## Phase I: MT-TICC to learn the high-level states
    # Slice the data
    slice_tr_agg_df = [norm_agg_split_df[i] for i in ticc_train_idx]
    slice_te_agg_df = [norm_agg_split_df[i] for i in ticc_test_idx]
        
    # --------------------------------------------------------------------------------------------------------------
    # Test the model
    print('Testing: ')
    TICC_return_cv = np.load(TICC_cluster_path, allow_pickle=True)
    test_TICC_split = [TICC_return_cv[1][i] for i in ticc_test_idx]
    test_demos, _, sub_demos, _, _ = PartitionTICCSubTrajs(slice_te_agg_df, agg_feats, test_TICC_split, 
                                                TICC_clusters=TICC_clusters)

    # Load the test trajectories
    test_teachers = sub_demos
    test_trajs, test_returns = test_teachers['trajs'], test_teachers['returns']

    # Assign the test data into clusters
    with open(student_path, 'rb') as file:
        loaded_object = pickle.load(file)
    student_list, rho = loaded_object[0], loaded_object[1]
    test_model, test_rho = student_list, rho

    pred_test_labs = PredictClusterForTest(test_model, test_trajs, config['EMEDM_nrCl'], test_rho, config)
    unique_test_labs = sorted(np.unique(pred_test_labs))

    # Path to store the predicted clusters
    env_data_path = base_path + 'volume/' + args['env_name']
    base_split_path = env_data_path + '/THEMES/case_study/'
    Path(base_split_path).mkdir(parents = True, exist_ok = True)

    test_dict_array = SplitTrajsByLabs(test_teachers, config['EMEDM_nrCl'], pred_test_labs)
    test_path_dict = SaveSplitTrajsToFile(base_split_path, test_dict_array, 
                                        unique_test_labs, sampling_num, mode='test')

    # Check the performance of each cluster
    student_dict = {i: student_list[i] for i in range(len(student_list))}

    clus_result = []
    true_act, pred_act, pred_prob = [], [], []
    for clusIdx in unique_test_labs:
        print('cluster: {}'.format(clusIdx))
        if clusIdx in student_dict.keys():
            _, _, pred_act, _ = student_dict[clusIdx].general_test(test_path_dict[clusIdx])
        else: 
            print('RANDOMLY SELECT A CLUSTER TO FIT ...')
            tmpIdx = random.sample(student_dict.keys(), 1)[0]
            _, _, pred_act, _ = student_dict[tmpIdx].general_test(test_path_dict[clusIdx])

        tmp_path = (base_split_path + 'test_predicted_actions_Clus_' + str(clusIdx) + '.npy')
        np.save(tmp_path, pred_act)

    print('------------------------!!!Case-Study Completed!!!-------------------------------------')
