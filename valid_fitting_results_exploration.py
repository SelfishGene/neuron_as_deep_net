import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import pickle
from scipy.stats import norm

# NOTE: during this project I've changed my coding style
# and was too lazy to edit the old code to match the new style
# so please ignore any style related wierdness
# thanks for not being petty about unimportant shit

# ALSO NOTE: prints are for logging purposes

#%% look at validation evaluations results

folder_name = '/Reseach/Single_Neuron_InOut/models/'
list_of_evaluated_models = glob.glob(folder_name + '*/eval_results/*.pickle')


def extract_useful_info(learning_dict_filename):
    
    learning_dict = pickle.load(open(learning_dict_filename, "rb" ), encoding='latin1')
    data_dict         = learning_dict['data_dict']
    architecture_dict = learning_dict['architecture_dict']
    training_dict     = learning_dict['training_history_dict']
    results_dict      = learning_dict['evaluations_results_dict']['starting_at_500ms_spikes_in_[0,24]_range']

    # biophyisical model type
    biophyisical_model_type_1 = learning_dict_filename.split('/')[-1].split('_')[0]
    biophyisical_model_type_2 = data_dict['train_files'][0].split('/')[-2].split('_')[1]
    biophyisical_model_type_3 = data_dict['valid_files'][0].split('/')[-2].split('_')[1]
    biophyisical_model_type_4 = data_dict['test_files'][0].split('/')[-2].split('_')[1]

    assert(biophyisical_model_type_1 == biophyisical_model_type_2)
    assert(biophyisical_model_type_1 == biophyisical_model_type_3)
    assert(biophyisical_model_type_1 == biophyisical_model_type_4)

    # NN model type
    NN_model_type_1 = learning_dict_filename.split('/')[-1].split('_')[1]
    
    if NN_model_type_1 == 'SK':
        NN_model_type_1 = learning_dict_filename.split('/')[-1].split('_')[2]
        biophyisical_model_type_1 = biophyisical_model_type_1 + '_SK'
    
    # NN input time window
    NN_input_time_window_1 = int(learning_dict_filename.split('/')[-1].split('__')[1].split('x')[-1])
    
    useful_results_dict = {}
    useful_results_dict['biophysical_model_type'] = biophyisical_model_type_1
    useful_results_dict['NN_model_type']          = NN_model_type_1
    useful_results_dict['NN_depth']               = architecture_dict['network_depth']
    useful_results_dict['NN_width']               = np.array(architecture_dict['num_filters_per_layer']).mean().astype(int)
    useful_results_dict['NN_input_time_window']   = NN_input_time_window_1
    useful_results_dict['NN_num_train_samples']   = sum(training_dict['num_train_samples'])
    useful_results_dict['NN_unique_train_files']  = len(training_dict['train_files_histogram'][-1].keys())

    useful_results_dict['spikes AUC']                = results_dict['AUC']
    useful_results_dict['spikes D prime']            = np.sqrt(2) * norm.ppf(results_dict['AUC'])
    useful_results_dict['spikes TP @ 0.25% FP']      = results_dict['TP @ 0.0025 FP']
    useful_results_dict['spikes TP @ 0.1% FP']       = results_dict['TP @ 0.0010 FP']
    useful_results_dict['spikes AUC @ 1% FP']        = results_dict['AUC @ 0.0100 FP']
    useful_results_dict['soma RMSE']                 = results_dict['soma_RMSE']
    useful_results_dict['soma MAE']                  = results_dict['soma_MAE']
    useful_results_dict['soma explained variance %'] = results_dict['soma_explained_variance_percent']
    
    useful_results_dict['full model filename'] = learning_dict_filename.split('/')[-1].split('.')[0]
    
    return useful_results_dict


list_of_useful_results_dict = []
for k, learning_dict_filename in enumerate(list_of_evaluated_models):
    useful_results_dict = extract_useful_info(learning_dict_filename)
    list_of_useful_results_dict.append(useful_results_dict)

print('finished loading %d model results' %(len(list_of_useful_results_dict)))

#%% insert results into a pandas dataframe

num_rows = len(list_of_useful_results_dict)
cols = list_of_useful_results_dict[-1].keys()

# columns in the "right order"
cols = [
 'biophysical_model_type',
 'NN_depth',
 'NN_width',
 'NN_input_time_window',
 'NN_model_type',
 'spikes D prime',
 'spikes AUC',
 'spikes AUC @ 1% FP',
 'soma explained variance %',
 'soma RMSE',
 'soma MAE',
 'spikes TP @ 0.1% FP',
 'spikes TP @ 0.25% FP',
 'NN_num_train_samples',
 'NN_unique_train_files',
 'full model filename']

results_dataframe = pd.DataFrame(index=range(num_rows), columns=cols)

for k, useful_res_row in enumerate(list_of_useful_results_dict):
    for key, value in useful_res_row.items():
        results_dataframe.loc[k,key] = value
    
print('finished building dataframe')

#%% remove extreemly poor perfomers from analysis

results_dataframe = results_dataframe.loc[results_dataframe['spikes D prime'] >= 0.5,:].reset_index(drop=True)
results_dataframe = results_dataframe.loc[results_dataframe['spikes AUC'] >= 0.6,:].reset_index(drop=True)
results_dataframe = results_dataframe.loc[results_dataframe['soma explained variance %'] >= 60,:].reset_index(drop=True)

#%% filter number of training samples

results_dataframe = results_dataframe.loc[results_dataframe['NN_width'] <= 256,:].reset_index(drop=True)
results_dataframe = results_dataframe.loc[results_dataframe['NN_input_time_window'] <= 260,:].reset_index(drop=True)
results_dataframe = results_dataframe.loc[results_dataframe['NN_num_train_samples'] >= 50000,:].reset_index(drop=True)
results_dataframe = results_dataframe.loc[results_dataframe['NN_num_train_samples'] <= 5000000,:].reset_index(drop=True)

#%% show various graphs


def show_dataframe_scatters(results_dataframe, noise_level=1.0):
    
    AMPA_SK_rows = results_dataframe['biophysical_model_type'] == 'AMPA_SK'
    AMPA_rows = results_dataframe['biophysical_model_type'] == 'AMPA'
    NMDA_rows = results_dataframe['biophysical_model_type'] == 'NMDA'
    
    alpha = 0.6
    fontsize = 20
    plt.close('all')


    ### Depth
    AMPA_SK_depth = results_dataframe.loc[AMPA_SK_rows,'NN_depth']
    AMPA_SK_depth = AMPA_SK_depth + noise_level * 0.15 * np.random.randn(AMPA_SK_depth.shape[0])
    
    AMPA_depth = results_dataframe.loc[AMPA_rows,'NN_depth']
    AMPA_depth = AMPA_depth + noise_level * 0.15 * np.random.randn(AMPA_depth.shape[0])
    
    NMDA_depth = results_dataframe.loc[NMDA_rows,'NN_depth']
    NMDA_depth = NMDA_depth + noise_level * 0.15 * np.random.randn(NMDA_depth.shape[0])
    
    plt.figure(figsize=(18,16))
    plt.suptitle('network performace as function of depth', fontsize=fontsize)
    plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.94,hspace=0.15,wspace=0.15)
    
    plt.subplot(3,2,1)
    plt.scatter(AMPA_SK_depth, results_dataframe.loc[AMPA_SK_rows,'spikes D prime'], c='orange', alpha=alpha)
    plt.scatter(AMPA_depth, results_dataframe.loc[AMPA_rows,'spikes D prime'], c='m', alpha=alpha)
    plt.scatter(NMDA_depth, results_dataframe.loc[NMDA_rows,'spikes D prime'], c='b', alpha=alpha)
    plt.xlabel('network depth', fontsize=fontsize); plt.ylabel('D prime', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,3)
    plt.scatter(AMPA_SK_depth, results_dataframe.loc[AMPA_SK_rows,'spikes AUC'], c='orange', alpha=alpha)
    plt.scatter(AMPA_depth, results_dataframe.loc[AMPA_rows,'spikes AUC'], c='m', alpha=alpha)
    plt.scatter(NMDA_depth, results_dataframe.loc[NMDA_rows,'spikes AUC'], c='b', alpha=alpha)
    plt.xlabel('network depth', fontsize=fontsize); plt.ylabel('AUC', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,5)
    plt.scatter(AMPA_SK_depth, results_dataframe.loc[AMPA_SK_rows,'spikes TP @ 0.25% FP'], c='orange', alpha=alpha)
    plt.scatter(AMPA_depth, results_dataframe.loc[AMPA_rows,'spikes TP @ 0.25% FP'], c='m', alpha=alpha)
    plt.scatter(NMDA_depth, results_dataframe.loc[NMDA_rows,'spikes TP @ 0.25% FP'], c='b', alpha=alpha)
    plt.xlabel('network depth', fontsize=fontsize); plt.ylabel('TP @ 0.25% FP', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    
    plt.subplot(3,2,2)
    plt.scatter(AMPA_SK_depth, results_dataframe.loc[AMPA_SK_rows,'soma explained variance %'], c='orange', alpha=alpha)
    plt.scatter(AMPA_depth, results_dataframe.loc[AMPA_rows,'soma explained variance %'], c='m', alpha=alpha)
    plt.scatter(NMDA_depth, results_dataframe.loc[NMDA_rows,'soma explained variance %'], c='b', alpha=alpha)
    plt.xlabel('network depth', fontsize=fontsize); plt.ylabel('soma explained variance %', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,4)
    plt.scatter(AMPA_SK_depth, results_dataframe.loc[AMPA_SK_rows,'soma RMSE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_depth, results_dataframe.loc[AMPA_rows,'soma RMSE'], c='m', alpha=alpha)
    plt.scatter(NMDA_depth, results_dataframe.loc[NMDA_rows,'soma RMSE'], c='b', alpha=alpha)
    plt.xlabel('network depth', fontsize=fontsize); plt.ylabel('soma RMSE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,6)
    plt.scatter(AMPA_SK_depth, results_dataframe.loc[AMPA_SK_rows,'soma MAE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_depth, results_dataframe.loc[AMPA_rows,'soma MAE'], c='m', alpha=alpha)
    plt.scatter(NMDA_depth, results_dataframe.loc[NMDA_rows,'soma MAE'], c='b', alpha=alpha)
    plt.xlabel('network depth', fontsize=fontsize); plt.ylabel('soma MAE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)

    
    ### Width
    AMPA_SK_width = results_dataframe.loc[AMPA_SK_rows,'NN_width']
    AMPA_SK_width = AMPA_SK_width + noise_level * 2.0 * np.random.randn(AMPA_SK_width.shape[0])
    
    AMPA_width = results_dataframe.loc[AMPA_rows,'NN_width']
    AMPA_width = AMPA_width + noise_level * 2.0 * np.random.randn(AMPA_width.shape[0])

    NMDA_width = results_dataframe.loc[NMDA_rows,'NN_width']
    NMDA_width = NMDA_width + noise_level * 2.0 * np.random.randn(NMDA_width.shape[0])
    
    plt.figure(figsize=(18,16))
    plt.suptitle('network performace as function of width', fontsize=fontsize)
    plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.94,hspace=0.15,wspace=0.15)
    
    plt.subplot(3,2,1)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'spikes D prime'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'spikes D prime'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'spikes D prime'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('D prime', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,3)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'spikes AUC'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'spikes AUC'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'spikes AUC'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('AUC', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,5)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'spikes TP @ 0.25% FP'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'spikes TP @ 0.25% FP'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'spikes TP @ 0.25% FP'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('TP @ 0.25% FP', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    
    plt.subplot(3,2,2)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'soma explained variance %'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'soma explained variance %'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'soma explained variance %'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('soma explained variance %', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,4)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'soma RMSE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'soma RMSE'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'soma RMSE'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('soma RMSE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,6)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'soma MAE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'soma MAE'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'soma MAE'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('soma MAE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)


    ### Width (log scale)
    AMPA_SK_width = results_dataframe.loc[AMPA_SK_rows,'NN_width']
    AMPA_SK_width = AMPA_SK_width + noise_level * 0.5 * np.random.randn(AMPA_SK_width.shape[0])
    AMPA_SK_width[AMPA_SK_width <= 0] = 0.1

    AMPA_width = results_dataframe.loc[AMPA_rows,'NN_width']
    AMPA_width = AMPA_width + noise_level * 0.5 * np.random.randn(AMPA_width.shape[0])
    AMPA_width[AMPA_width <= 0] = 0.1

    NMDA_width = results_dataframe.loc[NMDA_rows,'NN_width']
    NMDA_width = NMDA_width + noise_level * 0.5 * np.random.randn(NMDA_width.shape[0])
    NMDA_width[NMDA_width <= 0] = 0.1
    
    plt.figure(figsize=(18,16))
    plt.suptitle('network performace as function of width', fontsize=fontsize)
    plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.94,hspace=0.15,wspace=0.15)
    
    plt.subplot(3,2,1)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'spikes D prime'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'spikes D prime'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'spikes D prime'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('D prime', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,3)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'spikes AUC'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'spikes AUC'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'spikes AUC'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('AUC', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,5)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'spikes TP @ 0.25% FP'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'spikes TP @ 0.25% FP'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'spikes TP @ 0.25% FP'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('TP @ 0.25% FP', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    
    plt.subplot(3,2,2)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'soma explained variance %'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'soma explained variance %'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'soma explained variance %'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('soma explained variance %', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,4)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'soma RMSE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'soma RMSE'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'soma RMSE'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('soma RMSE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,6)
    plt.scatter(AMPA_SK_width, results_dataframe.loc[AMPA_SK_rows,'soma MAE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_width, results_dataframe.loc[AMPA_rows,'soma MAE'], c='m', alpha=alpha)
    plt.scatter(NMDA_width, results_dataframe.loc[NMDA_rows,'soma MAE'], c='b', alpha=alpha)
    plt.xlabel('network width', fontsize=fontsize); plt.ylabel('soma MAE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
        
    
    ### Time window size
    AMPA_SK_T_size = results_dataframe.loc[AMPA_SK_rows,'NN_input_time_window']
    AMPA_SK_T_size = AMPA_SK_T_size + noise_level * 1.5 * np.random.randn(AMPA_SK_T_size.shape[0])
    
    AMPA_T_size = results_dataframe.loc[AMPA_rows,'NN_input_time_window']
    AMPA_T_size = AMPA_T_size + noise_level * 1.5 * np.random.randn(AMPA_T_size.shape[0])
    
    NMDA_T_size = results_dataframe.loc[NMDA_rows,'NN_input_time_window']
    NMDA_T_size = NMDA_T_size + noise_level * 1.5 * np.random.randn(NMDA_T_size.shape[0])
    
    plt.figure(figsize=(18,16))
    plt.suptitle('network performace as function of time window size', fontsize=fontsize)
    plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.94,hspace=0.15,wspace=0.15)
    
    plt.subplot(3,2,1)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'spikes D prime'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'spikes D prime'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'spikes D prime'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('D prime', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,3)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'spikes AUC'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'spikes AUC'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'spikes AUC'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('AUC', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,5)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'spikes TP @ 0.25% FP'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'spikes TP @ 0.25% FP'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'spikes TP @ 0.25% FP'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('TP @ 0.25% FP', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    
    plt.subplot(3,2,2)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'soma explained variance %'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'soma explained variance %'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'soma explained variance %'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('soma explained variance %', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,4)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'soma RMSE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'soma RMSE'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'soma RMSE'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('soma RMSE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,6)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'soma MAE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'soma MAE'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'soma MAE'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('soma MAE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    
    
    # Time window size (log scale)
    plt.figure(figsize=(18,16))
    plt.suptitle('network performace as function of time window size (log scale)', fontsize=fontsize)
    plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.94,hspace=0.15,wspace=0.15)
    
    plt.subplot(3,2,1)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'spikes D prime'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'spikes D prime'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'spikes D prime'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('D prime', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,3)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'spikes AUC'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'spikes AUC'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'spikes AUC'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('AUC', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,5)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'spikes TP @ 0.25% FP'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'spikes TP @ 0.25% FP'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'spikes TP @ 0.25% FP'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('TP @ 0.25% FP', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')

    plt.subplot(3,2,2)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'soma explained variance %'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'soma explained variance %'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'soma explained variance %'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('soma explained variance %', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,4)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'soma RMSE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'soma RMSE'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'soma RMSE'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('soma RMSE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,6)
    plt.scatter(AMPA_SK_T_size, results_dataframe.loc[AMPA_SK_rows,'soma MAE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_T_size, results_dataframe.loc[AMPA_rows,'soma MAE'], c='m', alpha=alpha)
    plt.scatter(NMDA_T_size, results_dataframe.loc[NMDA_rows,'soma MAE'], c='b', alpha=alpha)
    plt.xlabel('network time window size', fontsize=fontsize); plt.ylabel('soma MAE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')

    
    ### num samples
    AMPA_SK_num_trains_samples = results_dataframe.loc[AMPA_SK_rows,'NN_num_train_samples'] / 1000.0
    AMPA_num_trains_samples = results_dataframe.loc[AMPA_rows,'NN_num_train_samples'] / 1000.0
    NMDA_num_trains_samples = results_dataframe.loc[NMDA_rows,'NN_num_train_samples'] / 1000.0
    
    plt.figure(figsize=(18,16))
    plt.suptitle('network performace as function of num training samples', fontsize=fontsize)
    plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.94,hspace=0.15,wspace=0.15)
    
    plt.subplot(3,2,1)
    plt.scatter(AMPA_SK_num_trains_samples, results_dataframe.loc[AMPA_SK_rows,'spikes D prime'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_trains_samples, results_dataframe.loc[AMPA_rows,'spikes D prime'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_trains_samples, results_dataframe.loc[NMDA_rows,'spikes D prime'], c='b', alpha=alpha)
    plt.xlabel('network num train samples (thousands)', fontsize=fontsize); plt.ylabel('D prime', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,3)
    plt.scatter(AMPA_SK_num_trains_samples, results_dataframe.loc[AMPA_SK_rows,'spikes AUC'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_trains_samples, results_dataframe.loc[AMPA_rows,'spikes AUC'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_trains_samples, results_dataframe.loc[NMDA_rows,'spikes AUC'], c='b', alpha=alpha)
    plt.xlabel('network num train samples (thousands)', fontsize=fontsize); plt.ylabel('AUC', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,5)
    plt.scatter(AMPA_SK_num_trains_samples, results_dataframe.loc[AMPA_SK_rows,'spikes TP @ 0.25% FP'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_trains_samples, results_dataframe.loc[AMPA_rows,'spikes TP @ 0.25% FP'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_trains_samples, results_dataframe.loc[NMDA_rows,'spikes TP @ 0.25% FP'], c='b', alpha=alpha)
    plt.xlabel('network num train samples (thousands)', fontsize=fontsize); plt.ylabel('TP @ 0.25% FP', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')

    plt.subplot(3,2,2)
    plt.scatter(AMPA_SK_num_trains_samples, results_dataframe.loc[AMPA_SK_rows,'soma explained variance %'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_trains_samples, results_dataframe.loc[AMPA_rows,'soma explained variance %'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_trains_samples, results_dataframe.loc[NMDA_rows,'soma explained variance %'], c='b', alpha=alpha)
    plt.xlabel('network num train samples (thousands)', fontsize=fontsize); plt.ylabel('soma explained variance %', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,4)
    plt.scatter(AMPA_SK_num_trains_samples, results_dataframe.loc[AMPA_SK_rows,'soma RMSE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_trains_samples, results_dataframe.loc[AMPA_rows,'soma RMSE'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_trains_samples, results_dataframe.loc[NMDA_rows,'soma RMSE'], c='b', alpha=alpha)
    plt.xlabel('network num train samples (thousands)', fontsize=fontsize); plt.ylabel('soma RMSE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')
    plt.subplot(3,2,6)
    plt.scatter(AMPA_SK_num_trains_samples, results_dataframe.loc[AMPA_SK_rows,'soma MAE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_trains_samples, results_dataframe.loc[AMPA_rows,'soma MAE'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_trains_samples, results_dataframe.loc[NMDA_rows,'soma MAE'], c='b', alpha=alpha)
    plt.xlabel('network num train samples (thousands)', fontsize=fontsize); plt.ylabel('soma MAE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize); plt.xscale('log')


    ### num samples
    AMPA_SK_num_unique_train_files = results_dataframe.loc[AMPA_SK_rows,'NN_unique_train_files']
    AMPA_num_unique_train_files = results_dataframe.loc[AMPA_rows,'NN_unique_train_files']
    NMDA_num_unique_train_files = results_dataframe.loc[NMDA_rows,'NN_unique_train_files']
    
    plt.figure(figsize=(18,16))
    plt.suptitle('network performace as function of num unique training files', fontsize=fontsize)
    plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.94,hspace=0.15,wspace=0.15)
    
    plt.subplot(3,2,1)
    plt.scatter(AMPA_SK_num_unique_train_files, results_dataframe.loc[AMPA_SK_rows,'spikes D prime'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_unique_train_files, results_dataframe.loc[AMPA_rows,'spikes D prime'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_unique_train_files, results_dataframe.loc[NMDA_rows,'spikes D prime'], c='b', alpha=alpha)
    plt.xlabel('num unique training files', fontsize=fontsize); plt.ylabel('D prime', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,3)
    plt.scatter(AMPA_SK_num_unique_train_files, results_dataframe.loc[AMPA_SK_rows,'spikes AUC'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_unique_train_files, results_dataframe.loc[AMPA_rows,'spikes AUC'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_unique_train_files, results_dataframe.loc[NMDA_rows,'spikes AUC'], c='b', alpha=alpha)
    plt.xlabel('num unique training files', fontsize=fontsize); plt.ylabel('AUC', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,5)
    plt.scatter(AMPA_SK_num_unique_train_files, results_dataframe.loc[AMPA_SK_rows,'spikes TP @ 0.25% FP'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_unique_train_files, results_dataframe.loc[AMPA_rows,'spikes TP @ 0.25% FP'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_unique_train_files, results_dataframe.loc[NMDA_rows,'spikes TP @ 0.25% FP'], c='b', alpha=alpha)
    plt.xlabel('num unique training files', fontsize=fontsize); plt.ylabel('TP @ 0.25% FP', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    
    plt.subplot(3,2,2)
    plt.scatter(AMPA_SK_num_unique_train_files, results_dataframe.loc[AMPA_SK_rows,'soma explained variance %'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_unique_train_files, results_dataframe.loc[AMPA_rows,'soma explained variance %'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_unique_train_files, results_dataframe.loc[NMDA_rows,'soma explained variance %'], c='b', alpha=alpha)
    plt.xlabel('num unique training files', fontsize=fontsize); plt.ylabel('soma explained variance %', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,4)
    plt.scatter(AMPA_SK_num_unique_train_files, results_dataframe.loc[AMPA_SK_rows,'soma RMSE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_unique_train_files, results_dataframe.loc[AMPA_rows,'soma RMSE'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_unique_train_files, results_dataframe.loc[NMDA_rows,'soma RMSE'], c='b', alpha=alpha)
    plt.xlabel('num unique training files', fontsize=fontsize); plt.ylabel('soma RMSE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)
    plt.subplot(3,2,6)
    plt.scatter(AMPA_SK_num_unique_train_files, results_dataframe.loc[AMPA_SK_rows,'soma MAE'], c='orange', alpha=alpha)
    plt.scatter(AMPA_num_unique_train_files, results_dataframe.loc[AMPA_rows,'soma MAE'], c='m', alpha=alpha)
    plt.scatter(NMDA_num_unique_train_files, results_dataframe.loc[NMDA_rows,'soma MAE'], c='b', alpha=alpha)
    plt.xlabel('num unique training files', fontsize=fontsize); plt.ylabel('soma MAE', fontsize=fontsize)
    plt.legend(['AMPA_SK','AMPA','NMDA'], fontsize=fontsize)


show_dataframe_scatters(results_dataframe)

#%% create a compact dataset with only the best model per (D,W,T,type)

select_best_AUC = True  # can choose whether to select based on somatic voltage or spikes prediction

DWTt_columns = ['NN_depth','NN_width','NN_input_time_window','NN_model_type','biophysical_model_type']
unique_DWTt_options = results_dataframe.loc[:,DWTt_columns].drop_duplicates().reset_index(drop=True)

num_unique_options = unique_DWTt_options.shape[0]

# go over all unique options, extract them from the database, and store the best values for each metric
best_results_dataframe = pd.DataFrame(columns=results_dataframe.columns.tolist())
for k in range(num_unique_options):
    DVTt_option = unique_DWTt_options.loc[k,:]
    
    option_rows = np.all((results_dataframe[DWTt_columns] == DVTt_option), axis=1)
    results_subset_df = results_dataframe.loc[option_rows,:].reset_index(drop=True)
    
    # choose the best according to some criteria
    if select_best_AUC:
        best_row_ind = np.argmax(np.array(results_subset_df['spikes AUC']))
    else:
        best_row_ind = np.argmin(np.array(results_subset_df['soma RMSE']))
    best_results_dataframe = best_results_dataframe.append(results_subset_df.loc[best_row_ind,:])
    
best_results_dataframe = best_results_dataframe.reset_index(drop=True)

sorting_order = ['biophysical_model_type',
                 'NN_depth',
                 'NN_model_type',
                 'NN_width',
                 'NN_input_time_window',
                 'spikes D prime']

# organize best results dataframe for saving
best_results_dataframe = best_results_dataframe.sort_values(by=sorting_order)

output_filename = '/Reseach/Single_Neuron_InOut/models/best_models/best_results_valid_%d_models.csv' %(num_unique_options)
best_results_dataframe.to_csv(output_filename, index=False)

#%% filter the very extreeme results

best_results_dataframe = best_results_dataframe.loc[best_results_dataframe['NN_width'] <= 256,:].reset_index(drop=True)
best_results_dataframe = best_results_dataframe.loc[best_results_dataframe['NN_input_time_window'] >= 10,:].reset_index(drop=True)
best_results_dataframe = best_results_dataframe.loc[best_results_dataframe['NN_input_time_window'] <= 260,:].reset_index(drop=True)
best_results_dataframe = best_results_dataframe.loc[best_results_dataframe['NN_num_train_samples'] >= 100000,:].reset_index(drop=True)

#%% show all data as scatter

show_dataframe_scatters(best_results_dataframe, noise_level=0.0)

#%% add to best results dataframe several "artificial" rows

FCN_rows = best_results_dataframe.loc[best_results_dataframe.loc[:,'NN_model_type'] == 'FCN',:]
FCN_rows.loc[:,['NN_model_type']] = 'TCN'
FCN_rows = FCN_rows.reset_index(drop=True)
best_results_dataframe = pd.concat((best_results_dataframe,FCN_rows),axis=0).reset_index(drop=True)


def keep_duplicate_x_with_extreeme_y(x,y, use_min=False):
    x_array = np.array(x)
    y_array = np.array(y)
    
    new_x_vals = np.sort(np.unique(x_array))
    new_y_vals = []
    
    for x_val in new_x_vals:
        if use_min:
            new_y_vals.append(y_array[x_array == x_val].min())
        else:
            new_y_vals.append(y_array[x_array == x_val].max())
    new_y_vals = np.array(new_y_vals)
    
    return new_x_vals, new_y_vals


def keep_only_strictly_changing_values(x, y, inreasing=True):
    
    y_span = y.max() - y.min()
    
    if inreasing:
        is_changing = np.diff(y) > 0.02 * y_span
    else:
        is_changing = np.diff(y) < -0.02 * y_span
        
    is_changing = np.concatenate((np.zeros(1)==0, is_changing))
    is_changing[-1] = True
    
    inds_to_keep = np.nonzero(is_changing)[0]
    x_subset = x[inds_to_keep]
    y_subset = y[inds_to_keep]
    
    return x_subset, y_subset


#%% show accuracy vs complexity graphs (slice and dice in various ways)


### NN_depth
x_axis_name = 'NN_depth'
fontsize = 20
use_log_scale = False
type_columns = ['biophysical_model_type']
unique_type_options = best_results_dataframe.loc[:,type_columns].drop_duplicates().reset_index(drop=True)
num_unique_options = unique_type_options.shape[0]
max_cols = ['spikes D prime', 'spikes AUC @ 1% FP', 'spikes TP @ 0.1% FP',
            'soma explained variance %', 'spikes AUC', 'spikes TP @ 0.25% FP']
min_cols = ['soma MAE', 'soma RMSE']

# go over all unique options, extract them from the database, and store the best values for each metric
X_vs_depth_curve_dicts = {}
for k in range(num_unique_options):
    type_option = unique_type_options.loc[k,:]
    curve_key = type_option['biophysical_model_type']
    
    option_rows = np.all((best_results_dataframe[type_columns] == type_option), axis=1)
    results_subset_df = best_results_dataframe.loc[option_rows,:]

    sorted_results_subset_df = results_subset_df.sort_values(by=[x_axis_name])
    sorted_results_subset_df_max = sorted_results_subset_df.cummax()
    sorted_results_subset_df_min = sorted_results_subset_df.cummin()
    
    # max cols
    sorted_results_subset_df.loc[:,max_cols] = sorted_results_subset_df_max.loc[:,max_cols]
    # min cols
    sorted_results_subset_df.loc[:,min_cols] = sorted_results_subset_df_min.loc[:,min_cols]
        
    # assemble dict
    X_vs_depth_curve_dicts[curve_key] = sorted_results_subset_df
    
plt.close('all')
plt.figure(figsize=(17,12))
plt.suptitle('network performace as function of %s' %(x_axis_name), fontsize=fontsize)
plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.92,hspace=0.15,wspace=0.15)

plot_ind = 1

# positive
list_of_target_cols = ['spikes AUC', 'soma explained variance %','spikes D prime']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(2,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col])
        x,y = keep_only_strictly_changing_values(x, y, inreasing=True)
        
        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names, loc='lower right')
    plot_ind = plot_ind + 1
    
# negative curves
list_of_target_cols = ['soma RMSE']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(2,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col], use_min=True)
        x,y = keep_only_strictly_changing_values(x, y, inreasing=False)
        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names, loc='upper right')
    plot_ind = plot_ind + 1


### NN_width
x_axis_name = 'NN_width'
fontsize = 20
use_log_scale = False
type_columns = ['biophysical_model_type']
unique_type_options = best_results_dataframe.loc[:,type_columns].drop_duplicates().reset_index(drop=True)
num_unique_options = unique_type_options.shape[0]
max_cols = ['spikes D prime', 'spikes AUC @ 1% FP', 'spikes TP @ 0.1% FP',
            'soma explained variance %', 'spikes AUC', 'spikes TP @ 0.25% FP']
min_cols = ['soma MAE', 'soma RMSE']

# go over all unique options, extract them from the database, and store the best values for each metric
X_vs_depth_curve_dicts = {}
for k in range(num_unique_options):
    type_option = unique_type_options.loc[k,:]
    curve_key = type_option['biophysical_model_type']

    option_rows = np.all((best_results_dataframe[type_columns] == type_option), axis=1)
    results_subset_df = best_results_dataframe.loc[option_rows,:]

    sorted_results_subset_df = results_subset_df.sort_values(by=[x_axis_name])
    sorted_results_subset_df_max = sorted_results_subset_df.cummax()
    sorted_results_subset_df_min = sorted_results_subset_df.cummin()
    
    # max cols
    sorted_results_subset_df.loc[:,max_cols] = sorted_results_subset_df_max.loc[:,max_cols]
    # min cols
    sorted_results_subset_df.loc[:,min_cols] = sorted_results_subset_df_min.loc[:,min_cols]
        
    # assemble dict
    X_vs_depth_curve_dicts[curve_key] = sorted_results_subset_df
    
    
plt.figure(figsize=(17,12))
plt.suptitle('network performace as function of %s' %(x_axis_name), fontsize=fontsize)
plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.92,hspace=0.15,wspace=0.15)

plot_ind = 1

# positive
list_of_target_cols = ['spikes AUC', 'soma explained variance %','spikes D prime']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(2,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col])
        x,y = keep_only_strictly_changing_values(x, y, inreasing=True)
        
        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names, loc='lower right')
    plot_ind = plot_ind + 1
    
# negative curves
list_of_target_cols = ['soma RMSE']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(2,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col], use_min=True)
        x,y = keep_only_strictly_changing_values(x, y, inreasing=False)
        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names, loc='upper right')
    plot_ind = plot_ind + 1


### NN_input_time_window
x_axis_name = 'NN_input_time_window'
fontsize = 20
use_log_scale = False
type_columns = ['biophysical_model_type']
unique_type_options = best_results_dataframe.loc[:,type_columns].drop_duplicates().reset_index(drop=True)
num_unique_options = unique_type_options.shape[0]
max_cols = ['spikes D prime', 'spikes AUC @ 1% FP', 'spikes TP @ 0.1% FP',
            'soma explained variance %', 'spikes AUC', 'spikes TP @ 0.25% FP']
min_cols = ['soma MAE', 'soma RMSE']

# go over all unique options, extract them from the database, and store the best values for each metric
X_vs_depth_curve_dicts = {}
for k in range(num_unique_options):
    type_option = unique_type_options.loc[k,:]
    curve_key = type_option['biophysical_model_type']

    option_rows = np.all((best_results_dataframe[type_columns] == type_option), axis=1)
    results_subset_df = best_results_dataframe.loc[option_rows,:]

    sorted_results_subset_df = results_subset_df.sort_values(by=[x_axis_name])
    sorted_results_subset_df_max = sorted_results_subset_df.cummax()
    sorted_results_subset_df_min = sorted_results_subset_df.cummin()
    
    # max cols
    sorted_results_subset_df.loc[:,max_cols] = sorted_results_subset_df_max.loc[:,max_cols]
    # min cols
    sorted_results_subset_df.loc[:,min_cols] = sorted_results_subset_df_min.loc[:,min_cols]
        
    # assemble dict
    X_vs_depth_curve_dicts[curve_key] = sorted_results_subset_df
    
    
plt.figure(figsize=(17,12))
plt.suptitle('network performace as function of %s' %(x_axis_name), fontsize=fontsize)
plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.92,hspace=0.15,wspace=0.15)

plot_ind = 1

# positive
list_of_target_cols = ['spikes AUC', 'soma explained variance %','spikes D prime']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(2,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col])
        x,y = keep_only_strictly_changing_values(x, y, inreasing=True)

        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names, loc='lower right')
    plot_ind = plot_ind + 1

# negative curves
list_of_target_cols = ['soma RMSE']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(2,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col], use_min=True)
        x,y = keep_only_strictly_changing_values(x, y, inreasing=False)
        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names, loc='upper right')
    plot_ind = plot_ind + 1
    
#%% delve deeper
    
plt.close('all')

#%% show accuracy vs depth graph (condition also on architecture type (FCN/TCN))

x_axis_name = 'NN_depth'
fontsize = 20
use_log_scale = False
type_columns = ['biophysical_model_type','NN_model_type']
unique_type_options = best_results_dataframe.loc[:,type_columns].drop_duplicates().reset_index(drop=True)
num_unique_options = unique_type_options.shape[0]
max_cols = ['spikes D prime', 'spikes AUC @ 1% FP', 'spikes TP @ 0.1% FP',
            'soma explained variance %', 'spikes AUC', 'spikes TP @ 0.25% FP']
min_cols = ['soma MAE', 'soma RMSE']

# go over all unique options, extract them from the database, and store the best values for each metric
X_vs_depth_curve_dicts = {}
for k in range(num_unique_options):
    type_option = unique_type_options.loc[k,:]
    curve_key = type_option['biophysical_model_type'] + '_' + type_option['NN_model_type']

    print(curve_key)
    
    option_rows = np.all((best_results_dataframe[type_columns] == type_option), axis=1)
    results_subset_df = best_results_dataframe.loc[option_rows,:]

    sorted_results_subset_df = results_subset_df.sort_values(by=[x_axis_name])
    sorted_results_subset_df_max = sorted_results_subset_df.cummax()
    sorted_results_subset_df_min = sorted_results_subset_df.cummin()
    
    # max cols
    sorted_results_subset_df.loc[:,max_cols] = sorted_results_subset_df_max.loc[:,max_cols]
    # min cols
    sorted_results_subset_df.loc[:,min_cols] = sorted_results_subset_df_min.loc[:,min_cols]
        
    # assemble dict
    X_vs_depth_curve_dicts[curve_key] = sorted_results_subset_df
    
    
plt.figure(figsize=(17,17))
plt.suptitle('network performace as function of %s' %(x_axis_name), fontsize=fontsize)
plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.94,hspace=0.15,wspace=0.15)

plot_ind = 1

# positive
list_of_target_cols = ['spikes D prime', 'spikes AUC', 'spikes TP @ 0.25% FP',  'soma explained variance %']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(3,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col])
        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names)
    plot_ind = plot_ind + 1
    
# negative curves
list_of_target_cols = ['soma MAE', 'soma RMSE']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(3,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col], use_min=True)
        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names)
    plot_ind = plot_ind + 1


#%% show accuracy vs width graph

x_axis_name = 'NN_width'
fontsize = 20
use_log_scale = False
type_columns = ['biophysical_model_type','NN_model_type']
unique_type_options = best_results_dataframe.loc[:,type_columns].drop_duplicates().reset_index(drop=True)
num_unique_options = unique_type_options.shape[0]
max_cols = ['spikes D prime', 'spikes AUC @ 1% FP', 'spikes TP @ 0.1% FP',
            'soma explained variance %', 'spikes AUC', 'spikes TP @ 0.25% FP']
min_cols = ['soma MAE', 'soma RMSE']

# go over all unique options, extract them from the database, and store the best values for each metric
X_vs_depth_curve_dicts = {}
for k in range(num_unique_options):
    type_option = unique_type_options.loc[k,:]
    curve_key = type_option['biophysical_model_type'] + '_' + type_option['NN_model_type']

    print(curve_key)
    
    option_rows = np.all((best_results_dataframe[type_columns] == type_option), axis=1)
    results_subset_df = best_results_dataframe.loc[option_rows,:]

    sorted_results_subset_df = results_subset_df.sort_values(by=[x_axis_name])
    sorted_results_subset_df_max = sorted_results_subset_df.cummax()
    sorted_results_subset_df_min = sorted_results_subset_df.cummin()
    
    # max cols
    sorted_results_subset_df.loc[:,max_cols] = sorted_results_subset_df_max.loc[:,max_cols]
    # min cols
    sorted_results_subset_df.loc[:,min_cols] = sorted_results_subset_df_min.loc[:,min_cols]
        
    # assemble dict
    X_vs_depth_curve_dicts[curve_key] = sorted_results_subset_df
    
    
plt.figure(figsize=(17,17))
plt.suptitle('network performace as function of %s' %(x_axis_name), fontsize=fontsize)
plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.94,hspace=0.15,wspace=0.15)

plot_ind = 1

# positive
list_of_target_cols = ['spikes D prime', 'spikes AUC', 'spikes TP @ 0.25% FP',  'soma explained variance %']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(3,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col])
        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names)
    plot_ind = plot_ind + 1
    
# negative curves
list_of_target_cols = ['soma MAE', 'soma RMSE']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(3,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col], use_min=True)
        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names)
    plot_ind = plot_ind + 1


#%% show accuracy vs time window size graph

x_axis_name = 'NN_input_time_window'
fontsize = 20
use_log_scale = False
type_columns = ['biophysical_model_type','NN_model_type']
unique_type_options = best_results_dataframe.loc[:,type_columns].drop_duplicates().reset_index(drop=True)
num_unique_options = unique_type_options.shape[0]
max_cols = ['spikes D prime', 'spikes AUC @ 1% FP', 'spikes TP @ 0.1% FP',
            'soma explained variance %', 'spikes AUC', 'spikes TP @ 0.25% FP']
min_cols = ['soma MAE', 'soma RMSE']

# go over all unique options, extract them from the database, and store the best values for each metric
X_vs_depth_curve_dicts = {}
for k in range(num_unique_options):
    type_option = unique_type_options.loc[k,:]
    curve_key = type_option['biophysical_model_type'] + '_' + type_option['NN_model_type']

    print(curve_key)
    
    option_rows = np.all((best_results_dataframe[type_columns] == type_option), axis=1)
    results_subset_df = best_results_dataframe.loc[option_rows,:]

    sorted_results_subset_df = results_subset_df.sort_values(by=[x_axis_name])
    sorted_results_subset_df_max = sorted_results_subset_df.cummax()
    sorted_results_subset_df_min = sorted_results_subset_df.cummin()
    
    # max cols
    sorted_results_subset_df.loc[:,max_cols] = sorted_results_subset_df_max.loc[:,max_cols]
    # min cols
    sorted_results_subset_df.loc[:,min_cols] = sorted_results_subset_df_min.loc[:,min_cols]
        
    # assemble dict
    X_vs_depth_curve_dicts[curve_key] = sorted_results_subset_df
    
    
plt.figure(figsize=(17,17))
plt.suptitle('network performace as function of %s' %(x_axis_name), fontsize=fontsize)
plt.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.94,hspace=0.15,wspace=0.15)

plot_ind = 1

# positive
list_of_target_cols = ['spikes D prime', 'spikes AUC', 'spikes TP @ 0.25% FP',  'soma explained variance %']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(3,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col])
        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names)
    plot_ind = plot_ind + 1
    
#  negative curves
list_of_target_cols = ['soma MAE', 'soma RMSE']
for k, target_col in enumerate(list_of_target_cols):
    plt.subplot(3,2,plot_ind); plt.title('%s vs %s' %(target_col,x_axis_name))
    plt.xlabel(x_axis_name); plt.ylabel(target_col)
    
    list_of_curve_names = []
    for curve_key, X_vs_depth_df in X_vs_depth_curve_dicts.items():
        list_of_curve_names.append(curve_key)
        
        x,y = keep_duplicate_x_with_extreeme_y(X_vs_depth_df[x_axis_name], X_vs_depth_df[target_col], use_min=True)
        if use_log_scale:
            plt.semilogx(x,y)
        else:
            plt.plot(x,y)
        
    plt.legend(list_of_curve_names)
    plot_ind = plot_ind + 1

#%% create pairwise combinations scatter plots

type_columns = ['biophysical_model_type','NN_model_type']
unique_type_options = best_results_dataframe.loc[:,type_columns].drop_duplicates().reset_index(drop=True)
num_unique_options = unique_type_options.shape[0]

option_colors = ['red','blue','orange','green','purple','magenta']
option_colors = ['cyan','green','red','orange','blue','yellow']

label_handles = []
# go over all unique options, extract them from the database, and store the best values for each metric
plt.figure(figsize=(20,10))
for k in range(num_unique_options):
    type_option = unique_type_options.loc[k,:]
    curve_key = type_option['biophysical_model_type'] + '_' + type_option['NN_model_type']
    print(curve_key)
    
    option_rows = np.all((best_results_dataframe[type_columns] == type_option), axis=1)
    results_subset_df = best_results_dataframe.loc[option_rows,:]
    num_rows = results_subset_df.shape[0]

    plt.subplot(1,3,1); plt.title('width vs time window size')
    plt.scatter(results_subset_df['NN_input_time_window'] + 4.0 * np.random.randn(num_rows),
                results_subset_df['NN_width'] + 2.0 * np.random.randn(num_rows), c=option_colors[k], alpha=0.8)
    plt.xlabel('T'); plt.ylabel('W'); plt.xlim(0,200); plt.ylim(0,150)
    
    plt.subplot(1,3,2); plt.title('width vs depth')
    plt.scatter(results_subset_df['NN_depth'] + 0.2 * np.random.randn(num_rows),
                results_subset_df['NN_width'] + 2.0 * np.random.randn(num_rows), c=option_colors[k], alpha=0.8)
    plt.xlabel('D'); plt.ylabel('W'); plt.xlim(0,9); plt.ylim(0,150)

    plt.subplot(1,3,3); plt.title('depth vs time window size')
    plt.scatter(results_subset_df['NN_depth'] + 0.2 * np.random.randn(num_rows),
                results_subset_df['NN_input_time_window'] + 4.0 * np.random.randn(num_rows), c=option_colors[k], alpha=0.8)
    plt.xlabel('D'); plt.ylabel('T'); plt.xlim(0,9); plt.ylim(0,200)
    
    curr_patch = mpatches.Patch(color=option_colors[k], label=curve_key)
    label_handles.append(curr_patch)
    
plt.legend(handles=label_handles, loc='lower right')

#%%






