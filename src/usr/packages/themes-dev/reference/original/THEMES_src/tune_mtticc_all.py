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
from itertools import product
import json

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

    # ---------------------------------------------------------------------------------
    # Hyper parameter tuning arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('configs')
    targs = parser.parse_args()  

    with open(targs.configs,'r') as f:
        cstr = f.read()
    hargs = json.loads(cstr)
    print('Hyperparameters: \n\n',hargs,'\n\n')

    if not os.path.exists(hargs['log_path']):
        os.makedirs(hargs['log_path'])

    with open(hargs['log_path']+'params.json','w') as f:
        json.dump(hargs, f, indent=2)

    # -----------------------------------------------------------------------------------
    # Load the data
    concat_final_df = pd.read_csv('../../volume/lstm_filtered_dataframe.csv')

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
        tmp_agg_df['Action'] = np.max(np.array(tmp_agg_df[['a_2']]), axis=1)  # Md: Where is column 'a_2'?
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

    random.seed(len(norm_agg_split_df))

    final_train = random.sample(norm_agg_split_df, int(len(norm_agg_split_df) * (1-te_perc)))

    # print('Total Trajectories: {}'.format(len(norm_agg_split_df)))
    # print('Training Trajectories: {}'.format(len(final_train)))

    ## -------------------------------------------------------------------------------------------------------------------------    
    # Initialization: Apply MT-TICC to learn the sub-sequences
    min_bic = float('inf')
    outfile = open(hargs['log_path']+'result.txt',"w")
    
    MTTICC_path = '../../THEMESE_MTTICC_results' + '.npy'
    TICC_return_cv = TICCSplitTrajs(args['env_name'], final_train, sel_feats, time_col, TICC_train=True, fixed_window=hargs['window_size'],
                                    TICC_clusters=hargs['TICC_cluster'],
                                    lambda_parameter=hargs['lambda'], beta=hargs['beta'], num_proc=6,
                                    input_pattern='multiple', window_pattern='fixed', maxIters=50,
                                    save_TICC_results=False, TICC_cluster_path=MTTICC_path)
    bic = TICC_return_cv[3]
    outfile.write(str(bic))
    
    # param_grid = {
    #     'TICC_cluster' : [5,6,7,8,9,10,11,12,13,14], #10
    #     'window_size' : [2,3,4,5], #4
    #     'lambda' : [1e-3,1e-4,5e-4,1e-5,5e-5,1e-6,1e-7,1e-8], #8
    #     'beta' : [1,2,3,4,5,6,7,8,9,10] #10
    # }

    # param_combinations = list(product(*param_grid.values()))
    # optimal_param = param_combinations[0]

    # for params in param_combinations:
    #     params = dict(zip(param_grid.keys(), params))
    #     MTTICC_path = '../../THEMESE_MTTICC_results' + '.npy'
    #     TICC_return_cv = TICCSplitTrajs(args['env_name'], final_train, sel_feats, time_col, TICC_train=True, fixed_window=params['window_size'],
    #                                     TICC_clusters=params['TICC_cluster'],
    #                                     lambda_parameter=params['lambda'], beta=params['beta'], num_proc=6,
    #                                     input_pattern='multiple', window_pattern='fixed', maxIters=50,
    #                                     save_TICC_results=False, TICC_cluster_path=MTTICC_path)

    #     # Getting the BIC for tuning MTTICC
    #     bic = TICC_return_cv[3]
    #     if bic < min_bic:
    #         min_bic = bic
    #         optimal_param = params
    #         print('------------Got new minimum!!!---------------\n')
    #         outfile.write("------------Got new minimum!!!---------------")
    #     outfile.write("For-> K={}; w={}; lambda={}; beta={} ---> BIC:{}\n".format(params['TICC_cluster'], params['window_size'], params['lambda'], params['beta'], bic))

    # print("Optimal Parameters: ")
    # outfile.write("Optimal Parameters: \n")
    # print(optimal_param)
    # outfile.write("For-> K={}; w={}; lambda={}; beta={} ---> BIC:{}\n".format(optimal_param['TICC_cluster'], optimal_param['window_size'], optimal_param['lambda'], optimal_param['beta'], min_bic))
    # print("Min BIC: ")
    # print(min_bic)