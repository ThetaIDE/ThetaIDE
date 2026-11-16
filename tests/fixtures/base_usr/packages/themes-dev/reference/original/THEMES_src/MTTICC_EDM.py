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
env = 'Mayo' # Change this CCHS/Mayo/MIMIC

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
    ## Phase I: MT-TICC to learn the high-level states
    # Initialization: Apply MT-TICC to learn the sub-sequences
    TICC_clusters = 9
    MTTICC_path = '../../volume/' + args['env_name'] + '/All/' + 'THEMESE_MTTICC_results' + '.npy'
    TICC_return_cv = TICCSplitTrajs(args['env_name'], final_df, sel_feats, time_col, TICC_train=True, fixed_window=2,
                                    TICC_clusters=TICC_clusters,
                                    lambda_parameter=1e-5, beta=4, num_proc=6,
                                    input_pattern='multiple', window_pattern='fixed', maxIters=100,
                                    save_TICC_results=True, TICC_cluster_path=MTTICC_path)
    # reg_TICC_return_cv = TICC_return_cv

    # -----------------------------------------------------------------------------------
    # Training process
    te_perc = 0.2
    repeat_num = 10

    results_dir_base = result_path + args['env_name'] + '/MTTICC_EDM/'
    results_file_name = 'results.csv'
    results_file_path = os.path.join(results_dir_base, results_file_name)

    if not os.path.exists(results_dir_base):
        os.makedirs(results_dir_base)

    with open(results_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('Round', 'Accuracy', 'Recall', 'Precision', 'F-score', 'AUC', 'APR', 'Jaccard'))

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

        # Slice the training trajectories
        tr_TICC_return = [TICC_return_cv[1][i] for i in ticc_train_idx]
        
        # Get the sub-trajectories for EDM training 
        _, _, comb_demos, orig_traj_idx, dis_TICC_split = PartitionTICCSubTrajs(slice_tr_agg_df, agg_feats, 
                                                                                tr_TICC_return, TICC_clusters=TICC_clusters)


        ## Phase II: EDM to learn the policy for each MTTICC partitioned cluster
        config['ENV_VOLUME_PATH'] = base_path + 'volume/' + args['env_name'] + '/MTTICC_EDM' + '/teachers/'
        config['TEACHER_PATH'] = config['ENV_VOLUME_PATH'] + 'teacher_' + str(rp_idx) + '_trajs.npy'

        ### EDM Training process
        # Load the expert trajectories
        trajs, returns = comb_demos['trajs'], comb_demos['returns']
        traj_num = len(trajs)

        # Save the expert trajs to file
        tmp_dict_array = np.array({'trajs': trajs, 'returns': returns})
        np.save(config['TEACHER_PATH'], tmp_dict_array)

        model_folder = '../models/' + args['env_name'] + '/EDM_models/'        
        model_path = (model_folder + 'EDM' + '_model_' + str(traj_num) + '_rp_idx_' + str(rp_idx) + '.sav')
        Path(model_folder).mkdir(parents = True, exist_ok = True)
        config['MODEL_PATH'] = model_path

        config['NUM_TRAJS_GIVEN'] = traj_num
        config['env_maker'] = eval('CustomizeEnv.' + config['ENV'] + '()')

        # ---------------------------------------------------------------------------------------------------
        # Make a student learner
        random.seed(config['NUM_TRAJS_GIVEN'])
        run_seed = random.sample(range(config['NUM_TRAJS_GIVEN']), 1)[0]
        student = make_student(run_seed, config)
        student.train(num_updates=config['NUM_STEPS_TRAIN'])
        
        # --------------------------------------------------------------------------------------------------------------
        # Test the model
        test_TICC_split = [TICC_return_cv[1][i] for i in ticc_test_idx]
        _, _, test_demos, _, _ = PartitionTICCSubTrajs(slice_te_agg_df, agg_feats, test_TICC_split, 
                                                    TICC_clusters=TICC_clusters)

        # Load the test trajectories
        test_trajs, test_returns = test_demos['trajs'], test_demos['returns']
        test_path = config['ENV_VOLUME_PATH'] + 'test_' + str(rp_idx) + '_trajs.npy'

        # Save the test trajs to file
        tmp_dict_array = np.array({'trajs': test_trajs, 'returns': test_returns})
        np.save(test_path, tmp_dict_array)

        # Assign the test data into clusters
        print('Testing: ')
        results, _, _, _ = student.general_test(test_path, avg_patterns=avg_patterns)
        repeat_results.append(results)

        with open(results_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([rp_idx] + results)

    avg_result = np.mean(np.array(repeat_results), axis=0)
    std_result = np.std(np.array(repeat_results), axis=0)
    agg_result = []

    for i in range(len(avg_result)):
        agg_result.append(f'{avg_result[i]:0.3f}({std_result[i]:0.2f})')

    with open(results_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['final'] + agg_result)