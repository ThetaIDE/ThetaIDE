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

env = 'MIMIC' # Change this CCHS/Mayo/MIMIC

args = {'env_name': f'EHRs/{env}',
        'trial': 1}

result_path = base_path + 'results/'

# Load the config file
config_path = '../models/' + args['env_name'] + '/EM-EDM_models/para_config.npy'
config = np.load(config_path, allow_pickle=True).item()

# Update the config values if necessary
config['NUM_STEPS_TRAIN'] = 500

config['BATCH_SIZE'] = 64
config['MLP_WIDTHS'] = 32

config['SGLD_BUFFER_SIZE'] = 10000
config['SGLD_LEARN_RATE'] = 1e-3

config['SAMPLE_BUFFER'] = 'balanced'
config['BASE_PATH'] = base_path
config['MODEL_LEARNER'] = 'EDM'
print(config)

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

   # ------------------------------------------------------------------------------------------------------------------------
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

    # Get the trajectories for EDM training
    train_demos = {'trajs': []}
    for i in range(len(slice_tr_agg_df)): 
        tmp_s = [np.array(slice_tr_agg_df[i].iloc[k][agg_feats], dtype=np.float64) for k in range(len(slice_tr_agg_df[i]))]
        tmp_a = [np.array([slice_tr_agg_df[i].iloc[k]['Action'].astype('uint32')]) for k in range(len(slice_tr_agg_df[i]))]
        tmp_traj = [(tmp_s[k], tmp_a[k]) for k in range(len(tmp_s))]
        train_demos['trajs'].append(tmp_traj)
    train_traj_num = len(train_demos['trajs'])

    ## Phase II: EDM to learn the policy
    config['ENV_VOLUME_PATH'] = base_path + 'volume/' + args['env_name'] + '/EDM/case_study/'
    config['TEACHER_PATH'] = config['ENV_VOLUME_PATH'] + 'teachers/teacher_trajs_All.npy'

    ### EDM Training process
    # Save the expert trajs to file
    tmp_dict_array = np.array(train_demos)
    np.save(config['TEACHER_PATH'], tmp_dict_array)

    model_folder = '../models/' + args['env_name'] + '/EDM_models/'        
    model_path = (model_folder + 'EDM_model_All_.sav')
    Path(model_folder).mkdir(parents = True, exist_ok = True)
    config['MODEL_PATH'] = model_path

    config['NUM_TRAJS_GIVEN'] = train_traj_num
    config['env_maker'] = eval('CustomizeEnv.' + config['ENV'] + '()')

    # ---------------------------------------------------------------------------------------------------
    # Make a student learner
    random.seed(config['NUM_TRAJS_GIVEN'])
    run_seed = random.sample(range(config['NUM_TRAJS_GIVEN']), 1)[0]
    student = make_student(run_seed, config)
    student.train(num_updates=config['NUM_STEPS_TRAIN'])
    
    # --------------------------------------------------------------------------------------------------------------
    # Test the model

    # Load the test trajectories
    test_demos = {'trajs': [], 'returns': []}
    for i in range(len(slice_te_agg_df)): 
        tmp_s = [np.array(slice_te_agg_df[i].iloc[k][agg_feats], dtype=np.float64) for k in range(len(slice_te_agg_df[i]))]
        tmp_a = [np.array([slice_te_agg_df[i].iloc[k]['Action'].astype('uint32')]) for k in range(len(slice_te_agg_df[i]))]
        tmp_traj = [(tmp_s[k], tmp_a[k]) for k in range(len(tmp_s))]
        tmp_return = [np.array(slice_te_agg_df[i].iloc[k][['VisitIdentifier', 'MinutesFromArrival']]) for k in range(len(slice_te_agg_df[i]))]
        test_demos['trajs'].append(tmp_traj)
        test_demos['returns'].append(tmp_return)

    # Save the test trajs to file
    test_path = config['ENV_VOLUME_PATH'] + 'test_trajs_All.npy'
    tmp_dict_array = np.array(test_demos)
    np.save(test_path, tmp_dict_array)

    # Assign the test data into clusters
    print('Testing: ')
    _, _, pred_act, _ = student.general_test(test_path, avg_patterns=avg_patterns)
    tmp_path = (config['ENV_VOLUME_PATH'] + 'test_predicted_actions.npy')
    np.save(tmp_path, pred_act)
