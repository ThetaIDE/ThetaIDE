import sys 
sys.path.append('../../')
sys.path.append('../')

try:
    from __head__ import *
except:
    from .__head__ import *

from network import StudentNetwork
from student import (
    BaseStudent,
    EDMStudent ,
)
import random
import itertools

from student.evaluation_utils import *
#from env import CustomizeEnv
import CustomizeEnv

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score

import sys
sys.path.append('../')
sys.path.append('../../')

try:
    from __head__ import *
except:
    from .__head__ import *

from network import StudentNetwork
from student import (
    BaseStudent,
    EDMStudent ,
)
from student.evaluation_utils import *
#from env import CustomizeEnv
import copy
import random
import itertools

from scipy.spatial.distance import euclidean
from tslearn.metrics import gak
from fastdtw import fastdtw
#from baselines.DTW import KMedoids
from sklearn_extra.cluster import KMedoids

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import math
from math import e


# ---------------------------------------------------------------------------------------------------
## Make a student agent
# EDMStudent --> BaseStudent & CUDAAgent
# BaseStudent --> SerializableAgent --> TrainableAgent --> BaseAgent
# CUDAAgent --> TrainableAgent --> BaseAgent
def make_student(
        run_seed: int, config
) -> BaseStudent:
    # Define the environment
    env = config['env_maker']

    # Set the paths for roll-out trajectories and model
    trajs_path = config['ENV_VOLUME_PATH'] + '/trajs/'
    model_path = config['MODEL_PATH']

    # Dimension of states and actions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Set the parameters
    run_seed = run_seed
    batch_size = config['BATCH_SIZE']
    buffer_size_in_trajs = config['NUM_TRAJS_GIVEN'] # Number of trajectories

    # Get the path to load trajectories
    teacher = config['TEACHER_PATH'] # config['BASE_PATH'] + get_trajs_path(config, 'expert')

    # List of Gym environments
    gym_env = config['GYM_ENV']

    qvalue_function = StudentNetwork(
        in_dim=state_dim,
        out_dim=action_dim,
        width=config['MLP_WIDTHS'],
    )

    adam_alpha = config['ADAM_ALPHA']
    adam_betas = config['ADAM_BETAS']
    sgld_buffer_size = config['SGLD_BUFFER_SIZE']
    sgld_learn_rate = config['SGLD_LEARN_RATE']
    sgld_noise_coef = config['SGLD_NOISE_COEF']
    sgld_num_steps = config['SGLD_NUM_STEPS']
    sgld_reinit_freq = config['SGLD_REINIT_FREQ']
    sample_method = config['SAMPLE_BUFFER']

    model_learner = config['MODEL_LEARNER']
    mlp_widths = config['MLP_WIDTHS']

    return EDMStudent(
        env=env,
        trajs_path=trajs_path,
        model_path=model_path,
        run_seed=run_seed,
        batch_size=batch_size,
        buffer_size_in_trajs=buffer_size_in_trajs,
        teacher=teacher,
        qvalue_function=qvalue_function,
        adam_alpha=adam_alpha,
        adam_betas=adam_betas,
        sgld_buffer_size=sgld_buffer_size,
        sgld_learn_rate=sgld_learn_rate,
        sgld_noise_coef=sgld_noise_coef,
        sgld_num_steps=sgld_num_steps,
        sgld_reinit_freq=sgld_reinit_freq,
        gym_env=gym_env,
        sample_method=sample_method,
        model_learner=model_learner,
        mlp_widths = mlp_widths
    )

# Encapsulated EDM process
# Return a trained EDM student
def EncapsulateEDM(config, base_path, teacher_path, verbo=True):
    config['BASE_PATH'] = base_path
    config['TEACHER_PATH'] = teacher_path
    config['NUM_TRAJS_GIVEN'] = len(np.load(config['TEACHER_PATH'], allow_pickle=True).item()['trajs'])

    # ---------------------------------------------------------------------------------------------------
    # Set the parameters to transform the states
    config['STATE_TRANSFORM'] = {
        'MountainCar_v2': {'reform_n': 3, 'reform_mode': 'skipping'}}

    if config['ENV'] not in config['STATE_TRANSFORM'].keys():
        config['STATE_TRANSFORM'][config['ENV']] = {'reform_n': 1, 'reform_mode': 'none'}

    state_trans_para = config['STATE_TRANSFORM'][config['ENV']]

    # ---------------------------------------------------------------------------------------------------
    # Set up the environment (Gym/Customized)
    if config['ENV'] in config['GYM_ENV']:
        config['env_maker'] = gym.make(config['ENV'])
    elif config['ENV'] == 'MountainCar_v2':
        reform_n, reform_mode = state_trans_para['reform_n'], state_trans_para['reform_mode']
        config['env_maker'] = eval('CustomizeEnv.' + config['ENV'] +
                                   '(' + str(reform_n) + ", '" + reform_mode + "')")
    else:
        config['env_maker'] = eval('CustomizeEnv.' + config['ENV'] + '()')

    # ---------------------------------------------------------------------------------------------------
    # Make a student learner
    random.seed(config['NUM_TRAJS_GIVEN'])
    run_seed = random.sample(range(config['NUM_TRAJS_GIVEN']), 1)[0]
    student = make_student(run_seed, config)

    loss_list, loss_pi_list, loss_rho_list = student.train(num_updates=config['NUM_STEPS_TRAIN'])

    # if verbo:
    #     print('********CHECK LOSS*******')
    #     plt.plot(loss_list)
    #     plt.show()
        
    #     plt.plot(loss_pi_list)
    #     plt.show()
    #     plt.plot(loss_rho_list)
    #     plt.show()

    return student, config


# -------------------------------------------------------------------------------------------------------
# Randomly split a list into sub-lists with the equal size
def EqualPartition(list_in, n, init_seed=0):
    random.seed(len(list_in)+init_seed)
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


# Calculate the Boltzmann Policy based on Q-values
def BoltzmannPolicy(Q, a, beta=0.5):
    BQ = np.exp(beta * Q)
    return BQ[a]/sum(BQ)


# Calculate the probability of Zij
def CalculateZij(clus_trajs, student, config):
    prob_list, all_probs = [], []
    for traj_idx in range(len(clus_trajs)):
        traj_prob_list = []
        traj = clus_trajs[traj_idx]
        for sa_idx in range(len(traj)):
            sa = traj[sa_idx]
            s, a = sa[0], sa[1][0]
            if config['MODEL_LEARNER'] == 'MLP':
                Q = student.getmlpmodel().predict_proba(s.reshape(1, -1))[0]
            elif config['MODEL_LEARNER'] == 'EDM':
                Q = student.select_action(s)[2]

            tmp_prob = BoltzmannPolicy(Q, a, config['EMEDM_BETA'])
            traj_prob_list.append(tmp_prob)

        prob_list.append(np.prod(traj_prob_list))
        all_probs.append(traj_prob_list)

    return prob_list, all_probs


# Save sliced trajectories in each cluster to file
def SliceClusTrajSaveToFile(clus_teachers, pred_labs, nrCl, iteration, clus_base_path, zij=np.nan, decay_ratio=1):
    clus_trajs, clus_returns = clus_teachers['trajs'], clus_teachers['returns']

    for clus_idx in range(nrCl):
        # Get the trajs belonging to current cluster
        tmp_idx = [idx for idx in range(len(pred_labs)) if pred_labs[idx] == clus_idx]

        # Slice the data with a decay ratio based on zij
        if zij is not np.nan and decay_ratio < 1:
            tmp_zij = zij[:, clus_idx]
            sel_idx = tmp_zij.argsort()[-math.ceil(len(tmp_idx) * decay_ratio):][::-1]
        else:
            sel_idx = tmp_idx
        tmp_traj = [clus_trajs[idx] for idx in sel_idx]
        tmp_return = [clus_returns[idx] for idx in sel_idx]

        # Save the trajs to file
        tmp_dict_array = np.array({'trajs': tmp_traj, 'returns': tmp_return})
        tmp_path = clus_base_path + 'teachers' + '_Iter_' + str(iteration) + '_Clus' + str(clus_idx) + '.npy'
        np.save(tmp_path, tmp_dict_array)


# M-step for EM-EDM
def EM_EDM_Mstep(pred_labs, nrCl, iteration, config, base_path, clus_base_path):
    # Calculate the rho
    rho = np.array([pred_labs.count(i) / len(pred_labs) for i in range(nrCl)]).reshape(-1, 1)

    # Learn the policy from each cluster
    student_list = []
    for clus_idx in range(nrCl):
        teacher_path = clus_base_path + 'teachers_' + 'Iter_' + str(iteration) + '_Clus' + str(clus_idx) + '.npy'
        student, _ = EncapsulateEDM(config, base_path, teacher_path)
        student_list.append(student)

    return student_list, rho


# E-step for EM-EDM
def EM_EDM_Estep(clus_trajs, nrCl, student_list, rho, config):
    # Calculate the zij
    zij = np.zeros([len(clus_trajs), nrCl])
    all_probs = []
    for clus_idx in range(nrCl):
        tmp_zij, tmp_probs = CalculateZij(clus_trajs, student_list[clus_idx], config)
        zij[:, clus_idx] = tmp_zij * rho[clus_idx]
        all_probs.append(tmp_probs)

    # Normalize the zij
    norm_zij = zij / np.sum(zij, axis=1).reshape(-1, 1)

    # Reassign the labels
    pred_labs = [np.argmax(row) for row in norm_zij]

    # Collapse the cluster if the size is below a pre-defined threshold
    remove_clus, keep_clus_idx = False, []
    for clus_idx in range(nrCl):
        clus_size = pred_labs.count(clus_idx)
        if clus_size <= config['EMEDM_CLUSTER_THRES']:
            print('** A Cluster (size = {}) is removed. **'.format(clus_size))
            if (len(np.unique(pred_labs))) > 1:
                nrCl -= 1
            remove_clus = True
        else:
            keep_clus_idx.append(clus_idx)

    if remove_clus:
        norm_zij = norm_zij[:, keep_clus_idx]
        pred_labs = [np.argmax(row) for row in norm_zij]

    return pred_labs, norm_zij, all_probs, nrCl


# Calculate the LLH
def Calculate_EM_EDM_Cluster_LLH(pred_labs, all_probs, nrCl):
    LLH_list = []
    for clus_idx in range(nrCl):
        tmp_idx = [idx for idx in range(len(pred_labs)) if pred_labs[idx] == clus_idx]
        tmp_probs = [all_probs[clus_idx][idx] for idx in tmp_idx]
        tmp_LLH = sum([np.log(i) for i in list(itertools.chain.from_iterable(tmp_probs))])
        LLH_list.append(tmp_LLH)
    return LLH_list


# Warp up the overall EM-EDM process
def EMEDMWarped(clus_teachers, nrCl, pred_labs, config, decay_ratio=1, decay_expert=False):
    clus_trajs, clus_returns = clus_teachers['trajs'], clus_teachers['returns']
    base_path = config['BASE_PATH']
    clus_base_path = config['CLUS_BASE_PATH']

    rho = np.ones([nrCl, 1]) / nrCl
    curr_LLH, tmp_zij = -np.inf, np.nan

    LLH = []
    for iteration in range(config['EMEDM_ITERS']):
        print('*** Iteration: ', iteration)

        # Save expert trajectories from each cluster to files
        if decay_expert:
            SliceClusTrajSaveToFile(clus_teachers, pred_labs, nrCl, iteration, clus_base_path,
                                    tmp_zij, decay_ratio)
        else:
            SliceClusTrajSaveToFile(clus_teachers, pred_labs, nrCl, iteration, clus_base_path)

        # M-step
        student_list, rho = EM_EDM_Mstep(pred_labs, nrCl, iteration, config, base_path, clus_base_path)
        # E-step
        tmp_pred_labs, tmp_zij, tmp_all_probs, nrCl = EM_EDM_Estep(clus_trajs, nrCl, student_list, rho, config)

        # ----------------------------------------------------------------------------------------------
        # Check the convergence
        tmp_LLH_list = Calculate_EM_EDM_Cluster_LLH(tmp_pred_labs, tmp_all_probs, nrCl)
        if abs(curr_LLH - sum(tmp_LLH_list)) < config['EMEDM_LLH_THRES_CONV']:  # Converged
            print('Converged at {}-th iteration'.format(iteration))
            break
        elif len(np.unique(tmp_pred_labs)) == 1:  # Collapsed to single cluster
            print('Collapsed to 1 cluster at {}-th iteration'.format(iteration))
            break
        elif curr_LLH - sum(tmp_LLH_list) > config['EMEDM_LLH_THRES_DES']:  # LLH Decreases
            print('LLH starts decreasing at {}-th iteration'.format(iteration))
            break
        else:
            pred_labs, all_probs = tmp_pred_labs, tmp_all_probs
            curr_LLH = sum(tmp_LLH_list)
            LLH.append(curr_LLH)

    return student_list, rho, LLH, nrCl, tmp_pred_labs, tmp_zij


# -------------------------------------------------------------------------------------------------------
# Test the model
def PredictClusterForTest(test_model, test_trajs, nrCl, rho, config):
    # Calculate the zij
    zij = np.zeros([len(test_trajs), nrCl])
    all_probs = []
    for clus_idx in range(nrCl):
        tmp_zij, tmp_probs = CalculateZij(test_trajs, test_model[clus_idx], config)
        zij[:, clus_idx] = tmp_zij * rho[clus_idx]
        all_probs.append(tmp_probs)

    # Normalize the zij
    norm_zij = zij / np.sum(zij, axis=1).reshape(-1, 1)

    # Reassign the labels
    pred_labs = [np.argmax(row) for row in norm_zij]

    return pred_labs


# -------------------------------------------------------------------------------------------------------
# Get the dictionary for index of different labels
def GetUniqueLabelDict(all_label):
    unique_label = set(all_label)
    label_dict = {l: [idx for idx, element in enumerate(all_label) if element == l] for l in unique_label}
    return label_dict


# Check the accuracy of the clustering results
def CheckClusterPurity(pred_label, true_label, true_label_dict, verbo=False):
    # Get the indexes of each unique label
    pred_dict = GetUniqueLabelDict(pred_label)
    true_dict = GetUniqueLabelDict(true_label)

    # Purity for each predicted cluster
    # purity_list: purity for each cluster
    # clLab_list: ground-truth label for each cluster
    # pred_count: data point number in each cluster
    purity_list, clLab_list, pred_count = [], [], []
    pred_labs = sorted(pred_dict.keys())
    pred2true_dict = {}
    for tmp_lab in pred_labs:
        # Display the size of current predicted cluster
        if verbo: print('# of Predicted Cluster ' + str(tmp_lab) + ': ', pred_label.count(tmp_lab))

        # Get the ground-truth for the points in current cluster
        tmp_gt = [true_label[i] for i in pred_dict[tmp_lab]]
        cl, count = np.unique(tmp_gt, return_counts=True)
        cl_lab = max(tmp_gt, key=tmp_gt.count)
        clLab_list.append(cl_lab)
        pred2true_dict[tmp_lab] = cl_lab
        if verbo: print('Pred cluster: ', true_label_dict[cl_lab])

        # Find the ground-truth label with the maximal number as the label for current cluster
        if len(count) > 0:
            tmp_purity = np.max(count) / len(tmp_gt)
        else:
            tmp_purity = 0
        purity_list.append(tmp_purity)
        pred_count.append(len(tmp_gt))

        # Display the purity of current cluster
        if verbo: print('Purity: ', tmp_purity, '\n--------------------')

    # Overall purity
    # Check the number of point index in each cluster falling into different true labels
    # Then select the maximum one
    correct_num = 0
    for check_key in pred_dict.keys():
        correct_num += max([len(set(pred_dict[check_key]).intersection(set(true_dict[target_key])))
                            for target_key in true_dict.keys()])
    purity = correct_num / len(pred_label)
    print('Purity per cluster: ', purity_list)
    print('Overall purity: ', purity)

    # Purity for each ground-truth label
    true_labs = sorted(list(np.unique(clLab_list)))
    class_purity_list = []
    for lab in true_labs:
        # Find the cluster index with the lab
        tmp_idx = [idx for idx in range(len(clLab_list)) if clLab_list[idx] == lab]
        # tmp_purity: purity across the lab clusters, initialized as 0
        # tmp_point_num: total number of points across the lab clusters
        tmp_purity, tmp_point_num = 0, sum([pred_count[idx] for idx in tmp_idx])
        for idx in tmp_idx:
            tmp_purity += purity_list[idx] * pred_count[idx] / tmp_point_num
        class_purity_list.append(tmp_purity)
        if verbo: print('Purity of cluster {}: {}'.format(true_label_dict[int(lab)], tmp_purity))

    return [purity, class_purity_list, purity_list], pred2true_dict


def ModelEvaluate_orig(real_lab, pre_lab, labels_list, avg_patterns='weighted', verbo=False): # 'weighted'/'micro'/‘macro’
    conf_matrix = confusion_matrix(real_lab, pre_lab, labels=labels_list)
    acc = accuracy_score(real_lab, pre_lab)
    recall = recall_score(real_lab, pre_lab, labels=labels_list, average=avg_patterns)
    precision = precision_score(real_lab, pre_lab, labels=labels_list, average=avg_patterns)
    fscore = f1_score(real_lab, pre_lab, labels=labels_list, average=avg_patterns)

    if verbo:
        print('Performance Measurements:')
        print('Confusion matrix: \n', conf_matrix)
        print('Accuracy: ', acc)
        print('Recall: ', recall)
        print('Precision: ', precision)
        print('F-score: ', fscore)
    return conf_matrix, [acc, recall, precision, fscore]


# ----------------------------------------------------------------------------------------------------------
# Evaluate the clustering results by purity and other metrics
def ClusterEvaluation(pred_label, true_label, idx_ToCompare, true_label_dict, verbo=False):
    # Calculate the purity of clustering results
    all_pred_label = pred_label
    pred_label = [all_pred_label[idx] for idx in idx_ToCompare]
    comp_true_label = [true_label[idx] for idx in idx_ToCompare]
    purity_all, pred2true_dict = CheckClusterPurity(pred_label, comp_true_label, true_label_dict, verbo=verbo)
    if verbo:
        print('Purity: ', purity_all[0])

    # Calculate the metrics for the clustering results
    pred2true_label = [int(pred2true_dict[val]) for val in pred_label]
    true_label = [int(val) for val in comp_true_label]
    lab_list = np.unique(list(pred2true_dict.values())).tolist()
    metrics_lab = ModelEvaluate_orig(true_label, pred2true_label, lab_list)
    if verbo:
        print('fscore: ', metrics_lab[1][3])

    # Calculate the other metrics given the labels
    NMI_val = normalized_mutual_info_score(true_label, pred2true_label)
    ARI_val = adjusted_rand_score(true_label, pred2true_label)
    h_val = homogeneity_score(true_label, pred2true_label)
    c_val = completeness_score(true_label, pred2true_label)
    v_val = v_measure_score(true_label, pred2true_label)
    # metrics_lab[1].append(purity_all[0])
    metrics_lab[1].append(NMI_val)
    metrics_lab[1].append(ARI_val)
    metrics_lab[1].append(h_val)
    metrics_lab[1].append(c_val)
    metrics_lab[1].append(v_val)

    # Check the metrics for each cluster
    metrics_cluster = []
    clus_lab_list = np.arange(int(max(np.unique(list(pred2true_dict.values())) + 1).tolist()))
    for clusIdx in list(pred2true_dict.keys()):
        tmpIdx = [idx for idx, val in enumerate(pred_label) if val == clusIdx]
        tmp_pred = [pred2true_dict[clusIdx]] * len(tmpIdx)
        tmp_true = [true_label[idx] for idx in tmpIdx]
        tmp_cm = confusion_matrix(tmp_true, tmp_pred, labels=clus_lab_list)
        metrics_cluster.append(tmp_cm)

    return purity_all, metrics_lab, metrics_cluster, pred2true_dict, pred2true_label


# ----------------------------------------------------------------------------------------------------------
# Convert Demos to arrays
def ConvertDemos2Arrays(clus_trajs):
    mixed_demos = []
    for idx in range(len(clus_trajs)):
        tmp_demo = np.array([list(clus_trajs[idx][j][0]) for j in range(len(clus_trajs[idx]))])
        mixed_demos.append(tmp_demo)
    allData_array = np.array(mixed_demos)
    return allData_array


# Initialize the clusters by pair-wise distances among the demos
# dist_measure: 'gak' / 'dtw'
def InitClusByDistance(clus_trajs, n_clusters, dist_measure='dtw', max_iter=10):
    # Convert the training data
    allData_array = ConvertDemos2Arrays(clus_trajs)

    # Generate the DTW distance matrix
    dis_matrix = np.zeros((len(clus_trajs), len(clus_trajs)))

    # Calculate the pair-wise distance
    print('Calculating the pairwise distance ... ')
    for i in range(len(clus_trajs)):
        if i%20 == 0: print(i)
        for j in range(i + 1, len(clus_trajs)):
            seq_i = allData_array[i]
            seq_j = allData_array[j]
            if dist_measure == 'dtw':
                temp_distance, _ = fastdtw(seq_i, seq_j, dist=euclidean)
            elif dist_measure == 'gak':
                temp_distance = gak(seq_i, seq_j)
            dis_matrix[i,j] = temp_distance
            dis_matrix[j,i] = temp_distance

    kmedoids = KMedoids(n_clusters=n_clusters, dist_matrix=dis_matrix, max_iter=max_iter)
    clusterModel = kmedoids.fit(allData_array, verbose=False)
    tr_pred = clusterModel.predict(allData_array)[0]
    return clusterModel, tr_pred


# Initialize the clusters by DTW
def InitEMEDMClusters(clus_trajs, nrCl, sampling_num, init_mode='random', DTW_thres=60, max_iter=10, init_seed=0):
    # Initialization method
    if sampling_num < DTW_thres:
        dist_measure = 'dtw' # 'gak' / 'dtw'
    else:
        dist_measure = 'gak'

    # Initialize the clusters -- Randomly split data into clusters with equal size
    if init_mode == 'random':
        init_labs = np.zeros(len(clus_trajs))
        init_idx = EqualPartition(np.arange(len(clus_trajs)), nrCl, init_seed=init_seed)
        init_clus_size = int(len(init_labs)/nrCl)
        for clus_idx in range(nrCl):
            init_labs[init_idx[clus_idx]] = clus_idx
        pred_labs = list(init_labs)
    elif init_mode == 'dtw':
        clusterModel, _ = list(InitClusByDistance(clus_trajs, n_clusters=nrCl,
                                                  dist_measure=dist_measure, max_iter=max_iter))
        pred_labs = list(clusterModel.predict(ConvertDemos2Arrays(clus_trajs))[0])
    elif init_mode == 'kmeans':
        tmp_mean = [np.mean([clus_trajs[j][i][0] for i in range(len(clus_trajs[j]))], axis=0) for j in
                    range(len(clus_trajs))]
        kmeans = KMeans(n_clusters=nrCl, random_state=0).fit(tmp_mean)
        pred_labs = list(kmeans.labels_)
    return pred_labs

# ----------------------------------------------------------------------------------------------------------
# Split trajectories belonging to different clusters
def SplitTrajsByLabs(clus_data, nrCl, pre_lab):
    clus_trajs, clus_returns = clus_data['trajs'], clus_data['returns']
    clus_id = [[idx for idx, val in enumerate(pre_lab) if val == lab]
               for lab in range(nrCl)]
    clus_dict_array = []
    for clusIdx in range(nrCl):
        tmp_dict = {'trajs': [clus_trajs[idx] for idx in clus_id[clusIdx]],
                    'returns': [clus_returns[idx] for idx in clus_id[clusIdx]]}
        tmp_dict_array = np.array(tmp_dict)
        clus_dict_array.append(tmp_dict_array)

    return clus_dict_array


# Save the trajs to file
def SaveSplitTrajsToFile(base_split_path, clus_dict_array, nrClIdx_list, sampling_num, r_num=-1, mode='test'):
    clus_path_dict = {}
    for clusIdx in nrClIdx_list:
        tmp_path = (base_split_path + mode + '_Samp_' + str(sampling_num) +
                    '_Iter_' + str(r_num) + '_Clus_' + str(clusIdx) + '.npy')

        np.save(tmp_path, clus_dict_array[clusIdx])
        clus_path_dict[clusIdx] = tmp_path

    return clus_path_dict
