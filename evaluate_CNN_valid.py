import numpy as np
import glob
import time
import cPickle as pickle
from keras.models import Model, load_model
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_curve, auc

# NOTE: during this project I've changed my coding style
# and was too lazy to edit the old code to match the new style
# so please ignore any style related wierdness
# thanks for not being petty about unimportant shit

# ALSO NOTE: prints are for logging purposes

#%% helper functions


def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms):
    
    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype='bool')
    for row_ind in row_inds_spike_times_map.keys():
        for spike_time in row_inds_spike_times_map[row_ind]:
            bin_spikes_matrix[row_ind,spike_time] = 1.0
    
    return bin_spikes_matrix


def parse_sim_experiment_file(sim_experiment_file):
    
    print('-----------------------------------------------------------------')
    print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")
    loading_start_time = time.time()
    experiment_dict = pickle.load(open(sim_experiment_file, "rb" ))
    
    # gather params
    num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])
    num_segments    = len(experiment_dict['Params']['allSegmentsType'])
    sim_duration_ms = experiment_dict['Params']['totalSimDurationInSec'] * 1000
    num_ex_synapses  = num_segments
    num_inh_synapses = num_segments
    num_synapses = num_ex_synapses + num_inh_synapses
    
    # collect X, y_spike, y_soma
    X = np.zeros((num_synapses,sim_duration_ms,num_simulations), dtype='bool')
    y_spike = np.zeros((sim_duration_ms,num_simulations))
    y_soma  = np.zeros((sim_duration_ms,num_simulations))
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex  = dict2bin(sim_dict['exInputSpikeTimes'] , num_segments, sim_duration_ms)
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments, sim_duration_ms)
        X[:,:,k] = np.vstack((X_ex,X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        y_spike[spike_times,k] = 1.0
        y_soma[:,k] = sim_dict['somaVoltageLowRes']

    loading_duration_sec = time.time() - loading_start_time
    print('loading took %.3f seconds' %(loading_duration_sec))
    print('-----------------------------------------------------------------')

    return X, y_spike, y_soma


def parse_multiple_sim_experiment_files(sim_experiment_files):
    
    for k, sim_experiment_file in enumerate(sim_experiment_files):
        X_curr, y_spike_curr, y_soma_curr = parse_sim_experiment_file(sim_experiment_file)
        
        if k == 0:
            X       = X_curr
            y_spike = y_spike_curr
            y_soma  = y_soma_curr
        else:
            X       = np.dstack((X,X_curr))
            y_spike = np.hstack((y_spike,y_spike_curr))
            y_soma  = np.hstack((y_soma,y_soma_curr))

    return X, y_spike, y_soma


def calc_AUC_at_desired_FP(y_test, y_test_hat, desired_false_positive_rate=0.01):
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_test_hat.ravel())

    linear_spaced_FPR = np.linspace(0,1,num=20000)
    linear_spaced_TPR = np.interp(linear_spaced_FPR, fpr, tpr)
    
    desired_fp_ind = min(max(1, np.argmin(abs(linear_spaced_FPR - desired_false_positive_rate))), linear_spaced_TPR.shape[0] -1)
    
    return linear_spaced_TPR[:desired_fp_ind].mean()


def calc_TP_at_desired_FP(y_test, y_test_hat, desired_false_positive_rate=0.0025):
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_test_hat.ravel())
    
    desired_fp_ind = np.argmin(abs(fpr - desired_false_positive_rate))
    if desired_fp_ind == 0:
        desired_fp_ind = 1

    return tpr[desired_fp_ind]


def exctract_key_results(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat, desired_FP_list=[0.0025,0.0100]):
    
    # evaluate the model and save the results
    print('----------------------------------------------------------------------------------------')
    print('calculating key results...')
    
    evaluation_start_time = time.time()
    
    # store results in the hyper param dict and return it
    evaluations_results_dict = {}
    
    for desired_FP in desired_FP_list:
        TP_at_desired_FP  = calc_TP_at_desired_FP(y_spikes_GT, y_spikes_hat, desired_false_positive_rate=desired_FP)
        AUC_at_desired_FP = calc_AUC_at_desired_FP(y_spikes_GT, y_spikes_hat, desired_false_positive_rate=desired_FP)
        print('-----------------------------------')
        print('TP  at %.4f FP rate = %.4f' %(desired_FP, TP_at_desired_FP))
        print('AUC at %.4f FP rate = %.4f' %(desired_FP, AUC_at_desired_FP))
        TP_key_string = 'TP @ %.4f FP' %(desired_FP)
        evaluations_results_dict[TP_key_string] = TP_at_desired_FP
    
        AUC_key_string = 'AUC @ %.4f FP' %(desired_FP)
        evaluations_results_dict[AUC_key_string] = AUC_at_desired_FP
    
    print('--------------------------------------------------')
    fpr, tpr, thresholds = roc_curve(y_spikes_GT.ravel(), y_spikes_hat.ravel())
    AUC_score = auc(fpr, tpr)
    print('AUC = %.4f' %(AUC_score))
    print('--------------------------------------------------')
    
    soma_explained_variance_percent = 100.0 * explained_variance_score(y_soma_GT.ravel(),y_soma_hat.ravel())
    soma_RMSE = np.sqrt(MSE(y_soma_GT.ravel(),y_soma_hat.ravel()))
    soma_MAE  = MAE(y_soma_GT.ravel(),y_soma_hat.ravel())
    
    print('--------------------------------------------------')
    print('soma explained_variance percent = %.2f%s' %(soma_explained_variance_percent, '%'))
    print('soma RMSE = %.3f [mV]' %(soma_RMSE))
    print('soma MAE = %.3f [mV]' %(soma_MAE))
    print('--------------------------------------------------')
    
    evaluations_results_dict['AUC'] = AUC_score
    evaluations_results_dict['soma_explained_variance_percent'] = soma_explained_variance_percent
    evaluations_results_dict['soma_RMSE'] = soma_RMSE
    evaluations_results_dict['soma_MAE'] = soma_MAE
    
    evaluation_duration_min = (time.time() - evaluation_start_time) / 60
    print('finished evaluation. time took to evaluate results is %.2f minutes' %(evaluation_duration_min))
    print('----------------------------------------------------------------------------------------')
    
    return evaluations_results_dict


def filter_and_exctract_key_results(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat, desired_FP_list=[0.0025,0.0100],
                                    ignore_time_at_start_ms=500, num_spikes_per_sim=[0,24]):

    time_points_to_eval = np.arange(y_spikes_GT.shape[1]) >= ignore_time_at_start_ms
    simulations_to_eval = np.logical_and((y_spikes_GT.sum(axis=1) >= num_spikes_per_sim[0]),(y_spikes_GT.sum(axis=1) <= num_spikes_per_sim[1]))
    
    print('total amount of simualtions is %d' %(y_spikes_GT.shape[0]))
    print('percent of simulations kept = %.2f%s' %(100 * simulations_to_eval.mean(),'%'))
    
    y_spikes_GT_to_eval  = y_spikes_GT[simulations_to_eval,:][:,time_points_to_eval]
    y_spikes_hat_to_eval = y_spikes_hat[simulations_to_eval,:][:,time_points_to_eval]
    y_soma_GT_to_eval    = y_soma_GT[simulations_to_eval,:][:,time_points_to_eval]
    y_soma_hat_to_eval   = y_soma_hat[simulations_to_eval,:][:,time_points_to_eval]
    
    return exctract_key_results(y_spikes_GT_to_eval, y_spikes_hat_to_eval, y_soma_GT_to_eval, y_soma_hat_to_eval, desired_FP_list=desired_FP_list)


#%% evel scrip params

synapse_type = 'NMDA'
# synapse_type = 'AMPA'
# synapse_type = 'AMPA_SK'

if synapse_type == 'NMDA':
    train_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_NMDA_train/'
    valid_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_NMDA_valid/'
    test_data_dir  = '/Reseach/Single_Neuron_InOut/data/L5PC_NMDA_test/'
    models_dir     = '/Reseach/Single_Neuron_InOut/models/NMDA/'
    results_dir    = '/Reseach/Single_Neuron_InOut/models/NMDA/eval_results/'
    
elif synapse_type == 'AMPA':
    train_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_train/'
    valid_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_valid/'
    test_data_dir  = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_test/'
    models_dir     = '/Reseach/Single_Neuron_InOut/models/AMPA/'
    results_dir    = '/Reseach/Single_Neuron_InOut/models/AMPA/eval_results/'
    
elif synapse_type == 'AMPA_SK':
    train_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_SK_train/'
    valid_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_SK_valid/'
    test_data_dir  = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_SK_test/'
    models_dir     = '/Reseach/Single_Neuron_InOut/models/AMPA_SK/'
    results_dir    = '/Reseach/Single_Neuron_InOut/models/AMPA_SK/eval_results/'

print('-----------------------------------------------')
print('finding data')
print('-----------------------------------------------')

valid_files = glob.glob(valid_data_dir + '*_128_simulationRuns*_6_secDuration_*')
test_files  = glob.glob(test_data_dir  + '*_128_simulationRuns*_6_secDuration_*')

models_to_evalulate = glob.glob(models_dir + '*.h5')

# remove from models list all models that are already evaluated in a previous run
model_names_to_evaluate = [x.split('/')[-1].split('.')[0] for x in models_to_evalulate]
models_already_evaluated = glob.glob(results_dir + '*.pickle')
model_names_already_evaluated = [x.split('/')[-1].split('.')[0] for x in models_already_evaluated]
model_names_to_eval_short = list(set(model_names_to_evaluate).difference(set(model_names_already_evaluated)))
model_names_to_eval_full = [(models_dir + x + '.h5') for x in model_names_to_eval_short]
models_to_evalulate = model_names_to_eval_full

print('-----')
print(len(model_names_to_evaluate))
print(len(model_names_already_evaluated))
print(len(model_names_to_eval_full))
print('-----')

print('number of validation files is %d' %(len(valid_files)))
print('number of test files is %d' %(len(test_files)))
print('number of models to evaluate is %d' %(len(models_to_evalulate)))
print('-----------------------------------------------')

print('models that will be evaluated are:')
for k, curr_model_name in enumerate(models_to_evalulate):
    print('%d: %s' %(k + 1, curr_model_name))
print('-----------------------------------------------')

#%% load valid and test datasets

print('----------------------------------------------------------------------------------------')
print('loading testing files...')
test_file_loading_start_time = time.time()

v_threshold = -55

# load validation data
X_test, y_spike_test, y_soma_test = parse_multiple_sim_experiment_files(valid_files)
y_soma_test[y_soma_test > v_threshold] = v_threshold

test_file_loading_duration_min = (time.time() - test_file_loading_start_time) / 60
print('time took to load data is %.3f minutes' %(test_file_loading_duration_min))
print('----------------------------------------------------------------------------------------')

#%% loop through all models files, make prediction on valid set, evaluate perfomrance and store

for k, model_filename in enumerate(models_to_evalulate):

    print('------------------------------')
    print('starting evaluating model %d' %(k + 1))
    print('------------------------------')

    print('----------------------------------------------------------------------------------------')
    print('loading model "%s"' %(model_filename.split('/')[-1]))

    model_loading_start_time = time.time()

    temporal_conv_net = load_model(model_filename)
    temporal_conv_net.summary()
    
    input_window_size = temporal_conv_net.input_shape[1]

    # load pickle file
    model_metadata_filename = model_filename.split('.h5')[0] + '.pickle'
    model_metadata_dict = pickle.load(open(model_metadata_filename, "rb" ))
    
    architecture_dict = model_metadata_dict['architecture_dict']
    time_window_T = (np.array(architecture_dict['filter_sizes_per_layer']) - 1).sum() + 1
    overlap_size = min(max(time_window_T + 1, min(150, input_window_size - 50)), 250)

    print('overlap_size = %d' %(overlap_size))
    print('time_window_T = %d' %(time_window_T))

    model_loading_duration_min = (time.time() - model_loading_start_time) / 60
    print('time took to load model is %.3f minutes' %(model_loading_duration_min))
    print('----------------------------------------------------------------------------------------')


    # create spike predictions on test set
    print('----------------------------------------------------------------------------------------')
    print('predicting using model...')

    prediction_start_time = time.time()
    
    y_train_soma_bias = -67.7
    
    X_test_for_keras = np.transpose(X_test,axes=[2,1,0])
    y1_test_for_keras = y_spike_test.T[:,:,np.newaxis]
    y2_test_for_keras = y_soma_test.T[:,:,np.newaxis] - y_train_soma_bias
    
    y1_test_for_keras_hat = np.zeros(y1_test_for_keras.shape)
    y2_test_for_keras_hat = np.zeros(y2_test_for_keras.shape)
    
    num_test_splits = 2 + (X_test_for_keras.shape[1] - input_window_size) / (input_window_size - overlap_size)
    
    for k in range(num_test_splits):
        start_time_ind = k * (input_window_size - overlap_size)
        end_time_ind   = start_time_ind + input_window_size
        
        curr_X_test_for_keras = X_test_for_keras[:,start_time_ind:end_time_ind,:]
        
        if curr_X_test_for_keras.shape[1] < input_window_size:
            padding_size = input_window_size - curr_X_test_for_keras.shape[1]
            X_pad = np.zeros((curr_X_test_for_keras.shape[0],padding_size,curr_X_test_for_keras.shape[2]))
            curr_X_test_for_keras = np.hstack((curr_X_test_for_keras,X_pad))
            
        curr_y1_test_for_keras, curr_y2_test_for_keras, _ = temporal_conv_net.predict(curr_X_test_for_keras)
    
        if k == 0:
            y1_test_for_keras_hat[:,:end_time_ind,:] = curr_y1_test_for_keras
            y2_test_for_keras_hat[:,:end_time_ind,:] = curr_y2_test_for_keras
        elif k == (num_test_splits - 1):
            t0 = start_time_ind + overlap_size
            duration_to_fill = y1_test_for_keras_hat.shape[1] - t0
            y1_test_for_keras_hat[:,t0:,:] = curr_y1_test_for_keras[:,overlap_size:(overlap_size + duration_to_fill),:]
            y2_test_for_keras_hat[:,t0:,:] = curr_y2_test_for_keras[:,overlap_size:(overlap_size + duration_to_fill),:]
        else:
            t0 = start_time_ind + overlap_size
            y1_test_for_keras_hat[:,t0:end_time_ind,:] = curr_y1_test_for_keras[:,overlap_size:,:]
            y2_test_for_keras_hat[:,t0:end_time_ind,:] = curr_y2_test_for_keras[:,overlap_size:,:]
    
    # zero score the prediction and align it with the actual test
    s_dst = y2_test_for_keras.std()
    m_dst = y2_test_for_keras.mean()
    
    s_src = y2_test_for_keras_hat.std()
    m_src = y2_test_for_keras_hat.mean()
    
    y2_test_for_keras_hat = (y2_test_for_keras_hat - m_src) / s_src
    y2_test_for_keras_hat = s_dst * y2_test_for_keras_hat + m_dst
    
    y_test = y_spike_test
    y_test_hat = y1_test_for_keras_hat[:,:,0].T
    
    # convert to simple (num_sims X num_time_points) format
    y_spikes_GT  = y_test.T
    y_spikes_hat = y_test_hat.T
    y_soma_GT    = y2_test_for_keras[:,:,0]
    y_soma_hat   = y2_test_for_keras_hat[:,:,0]
    
    prediction_duration_min = (time.time() - prediction_start_time) / 60
    print('finished prediction. time took to predict is %.2f minutes' %(prediction_duration_min))
    print('----------------------------------------------------------------------------------------')
    

    # evaluate the model and save the results
    print('----------------------------------------------------------------------------------------')
    print('calculating and saving key results...')

    saving_start_time = time.time()

    desired_FP_list = [0.0001, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0050, 0.0100, 0.0200, 0.0300, 0.0400, 0.0500, 0.1000]
    evaluations_results_dict = {}

    # (1) we ignore the first 500 ms so that the simulation will "settle" and forget about any initial conditions
    # (2) we control for the number of spikes per simulation in order to make proper comparisons for different biophysical models (AMPA,NMDA,AMPA_SK, etc.)
    # this is due to the fact the the number of output spikes greatley changes the results and it's important to keep a tight control on it

    ignore_time_at_start_ms = 500
    num_spikes_per_sim = [0,18]
    filter_string = 'starting_at_%dms_spikes_in_[%d,%d]_range' %(ignore_time_at_start_ms, num_spikes_per_sim[0], num_spikes_per_sim[1])
    evaluations_results_dict[filter_string] = filter_and_exctract_key_results(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat,
                                                                              desired_FP_list=desired_FP_list,
                                                                              ignore_time_at_start_ms=ignore_time_at_start_ms,
                                                                              num_spikes_per_sim=num_spikes_per_sim)

    ignore_time_at_start_ms = 500
    num_spikes_per_sim = [1,24]
    filter_string = 'starting_at_%dms_spikes_in_[%d,%d]_range' %(ignore_time_at_start_ms, num_spikes_per_sim[0], num_spikes_per_sim[1])
    evaluations_results_dict[filter_string] = filter_and_exctract_key_results(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat,
                                                                              desired_FP_list=desired_FP_list,
                                                                              ignore_time_at_start_ms=ignore_time_at_start_ms,
                                                                              num_spikes_per_sim=num_spikes_per_sim)

    ignore_time_at_start_ms = 500
    num_spikes_per_sim = [0,24]
    filter_string = 'starting_at_%dms_spikes_in_[%d,%d]_range' %(ignore_time_at_start_ms, num_spikes_per_sim[0], num_spikes_per_sim[1])
    evaluations_results_dict[filter_string] = filter_and_exctract_key_results(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat,
                                                                              desired_FP_list=desired_FP_list,
                                                                              ignore_time_at_start_ms=ignore_time_at_start_ms,
                                                                              num_spikes_per_sim=num_spikes_per_sim)

    ignore_time_at_start_ms = 500
    num_spikes_per_sim = [0,30]
    filter_string = 'starting_at_%dms_spikes_in_[%d,%d]_range' %(ignore_time_at_start_ms, num_spikes_per_sim[0], num_spikes_per_sim[1])
    evaluations_results_dict[filter_string] = filter_and_exctract_key_results(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat,
                                                                              desired_FP_list=desired_FP_list,
                                                                              ignore_time_at_start_ms=ignore_time_at_start_ms,
                                                                              num_spikes_per_sim=num_spikes_per_sim)

    ignore_time_at_start_ms = 500
    num_spikes_per_sim = [0,90]
    filter_string = 'starting_at_%dms_spikes_in_[%d,%d]_range' %(ignore_time_at_start_ms, num_spikes_per_sim[0], num_spikes_per_sim[1])
    evaluations_results_dict[filter_string] = filter_and_exctract_key_results(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat,
                                                                              desired_FP_list=desired_FP_list,
                                                                              ignore_time_at_start_ms=ignore_time_at_start_ms,
                                                                              num_spikes_per_sim=num_spikes_per_sim)
    
    model_metadata_dict['evaluations_results_dict'] = evaluations_results_dict
    metadata_evaluation_filename = results_dir + model_metadata_filename.split('/')[-1]
    print('saving:   "%s"' %(metadata_evaluation_filename))
    pickle.dump(model_metadata_dict, open(metadata_evaluation_filename, "wb"), protocol=2)
    
    saving_duration_min = (time.time() - saving_start_time) / 60
    print('time took to evaluate and save results is %.3f minutes' %(saving_duration_min))
    print('----------------------------------------------------------------------------------------')
    
#%%

print('finihsed evaluation script')
