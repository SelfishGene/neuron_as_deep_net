import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pickle
from scipy.stats import norm

# NOTE: during this project I've changed my coding style
# and was too lazy to edit the old code to match the new style
# so please ignore any style related wierdness
# thanks for not being petty about unimportant shit

# ALSO NOTE: prints are for logging purposes

#%% open all evaluations pickles and create a compact csv file


def extract_useful_info(learning_dict_filename, req_results_key='starting_at_500ms_spikes_in_[0,24]_range'):
    
    learning_dict = pickle.load(open(learning_dict_filename, "rb" ), encoding='latin1')
    data_dict         = learning_dict['data_dict']
    architecture_dict = learning_dict['architecture_dict']
    training_dict     = learning_dict['training_history_dict']
    results_dict      = learning_dict['evaluations_results_dict'][req_results_key]

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
    
    try:
        results_dict_list = [x[req_results_key] for x in learning_dict['list_of_subset_eval_results_dict']]
        AUC_list                     = np.array([x['AUC'] for x in results_dict_list])
        soma_variance_explained_list = np.array([x['soma_explained_variance_percent'] for x in results_dict_list])
    
        useful_results_dict['spikes AUC mean of subsets']                = AUC_list.mean()
        useful_results_dict['spikes AUC std of subsets']                 = AUC_list.std()
        useful_results_dict['soma explained variance % mean of subsets'] = soma_variance_explained_list.mean()
        useful_results_dict['soma explained variance % std of subsets']  = soma_variance_explained_list.std()
    except:
        print('no subsets list')
        
    useful_results_dict['full model filename'] = learning_dict_filename.split('/')[-1].split('.')[0]
    
    return useful_results_dict


#%% look at evaluations results

models_folder = '/Reseach/Single_Neuron_InOut/models/best_models/'
list_of_evaluated_models = glob.glob(models_folder + '*/*_evaluation_test.pickle')

list_of_useful_results_dict = []
for k, learning_dict_filename in enumerate(list_of_evaluated_models):
    useful_results_dict = extract_useful_info(learning_dict_filename)
    list_of_useful_results_dict.append(useful_results_dict)

print('finished loading %d model results' %(len(list_of_useful_results_dict)))

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
 'spikes AUC std of subsets',
 'soma explained variance % std of subsets',
 'NN_num_train_samples',
 'NN_unique_train_files',
 'full model filename']

best_results_dataframe = pd.DataFrame(index=range(num_rows), columns=cols)

for k, useful_res_row in enumerate(list_of_useful_results_dict):
    for key, value in useful_res_row.items():
        if key in cols:
            best_results_dataframe.loc[k,key] = value
    
print('finished building dataframe')

#%% sort and save

sorting_order = ['biophysical_model_type',
                 'NN_depth',
                 'NN_model_type',
                 'NN_input_time_window',
                 'NN_width',
                 'spikes AUC']

# organize best results dataframe for saving
best_results_dataframe = best_results_dataframe.sort_values(by=sorting_order)

output_filename = '/Reseach/Single_Neuron_InOut/models/best_models/best_results_test_%d_models.csv' %(best_results_dataframe.shape[0])
best_results_dataframe.to_csv(output_filename, index=False)

#%% open csv file and display it

models_results_folder = '/Reseach/Single_Neuron_InOut/models/best_models/'
best_results_dataframe = pd.read_csv(models_results_folder + 'best_results_test_105_models.csv')

#%% filter the results for reasonable param ranges

print(best_results_dataframe.shape)

best_results_dataframe = best_results_dataframe.loc[best_results_dataframe['NN_input_time_window'] >= 10,:].reset_index(drop=True)
best_results_dataframe = best_results_dataframe.loc[best_results_dataframe['NN_input_time_window'] <= 260,:].reset_index(drop=True)

print(best_results_dataframe.shape)

#%% show various scatter plots of performance vs (width, depth, time, etc...)


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


# show all data as scatters
show_dataframe_scatters(best_results_dataframe, noise_level=0.0)

#%% add to best results dataframe several "artificial" rows that duplicate FCN as TCN

FCN_rows = best_results_dataframe.loc[best_results_dataframe.loc[:,'NN_model_type'] == 'FCN',:]
FCN_rows.loc[:,['NN_model_type']] = 'TCN'
FCN_rows = FCN_rows.reset_index(drop=True)
best_results_dataframe = pd.concat((best_results_dataframe,FCN_rows),axis=0).reset_index(drop=True)


def keep_duplicate_x_with_extreeme_y(x,y, e, use_min=False):
    x_array = np.array(x)
    y_array = np.array(y)
    e_array = np.array(e)
    
    new_x_vals = np.sort(np.unique(x_array))
    new_y_vals = []
    new_e_vals = []
    
    for x_val in new_x_vals:
        relevent_inds = x_array == x_val
        relevent_y_values = y_array[relevent_inds]
        relevent_e_values = e_array[relevent_inds]
        
        if use_min:
            argextreeme_within_relevent = relevent_y_values.argmin()
        else:
            argextreeme_within_relevent = relevent_y_values.argmax()
            
        y_value = relevent_y_values[argextreeme_within_relevent]
        e_value = relevent_e_values[argextreeme_within_relevent]
            
        new_y_vals.append(y_value)
        new_e_vals.append(e_value)

    new_y_vals = np.array(new_y_vals)
    new_e_vals = np.array(new_e_vals)
    
    return new_x_vals, new_y_vals, new_e_vals


def keep_only_strictly_changing_values(x, y, e, change_threshold=0.02, inreasing=True):
    
    y_span = y.max() - y.min()
    
    if inreasing:
        is_changing = np.diff(y) > change_threshold * y_span
    else:
        is_changing = np.diff(y) < -change_threshold * y_span
        
    is_changing = np.concatenate((np.zeros(1)==0, is_changing))
    is_changing[-1] = True
    
    inds_to_keep = np.nonzero(is_changing)[0]
    x_subset = x[inds_to_keep]
    y_subset = y[inds_to_keep]
    e_subset = e[inds_to_keep]
    
    return x_subset, y_subset, e_subset


#%% show accuracy vs complexity graphs (clean)

plt.close('all')

list_of_x_axis_col_names = ['NN_depth', 'NN_width', 'NN_input_time_window']
list_of_y_axis_col_names = ['spikes AUC', 'soma explained variance %']

type_columns = ['biophysical_model_type']
unique_type_options = best_results_dataframe.loc[:,type_columns].drop_duplicates().reset_index(drop=True)
num_unique_options = unique_type_options.shape[0]

legend_labeling = {}
legend_labeling['NMDA'] = 'NMDA synapses'
legend_labeling['AMPA'] = 'AMPA synapses'
legend_labeling['AMPA_SK'] = 'AMPA synapses w\o SK_E2'

change_threshold_dict = {}
change_threshold_dict['NN_depth'] = 0.01
change_threshold_dict['NN_width'] = 0.05
change_threshold_dict['NN_input_time_window'] = 0.03


fontsize = 27
xy_label_fontsize = 20
plt.close('all')
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(17,12))
fig.suptitle('best network performace as function of network complexity', fontsize=fontsize)
fig.subplots_adjust(left=0.06,right=0.97,bottom=0.06,top=0.92,hspace=0.15,wspace=0.15)

for fig_col_ind, x_axis_name in enumerate(list_of_x_axis_col_names):

    # go over all unique options, extract them from the database, and store the best values for each metric
    X_vs_depth_curve_dicts = {}
    for k in range(num_unique_options):
        type_option = unique_type_options.loc[k,:]
        curve_key = type_option['biophysical_model_type']
        
        option_rows = np.all((best_results_dataframe[type_columns] == type_option), axis=1)
        results_subset_df = best_results_dataframe.loc[option_rows,:]
    
        sorted_results_subset_df = results_subset_df.sort_values(by=[x_axis_name])
        sorted_results_subset_df_max = sorted_results_subset_df.cummax()
        
        # max cols
        sorted_results_subset_df.loc[:,list_of_y_axis_col_names] = sorted_results_subset_df_max.loc[:,list_of_y_axis_col_names]
            
        # assemble dict
        X_vs_depth_curve_dicts[curve_key] = sorted_results_subset_df
        
        
    # go over each accuracy meassure
    for fig_row_ind, y_axis_name in enumerate(list_of_y_axis_col_names):
        
        e_axis_name = y_axis_name + ' std of subsets'
        
        if fig_row_ind == 0:
            ax[fig_row_ind,fig_col_ind].set_ylim([0.979,0.9981])
            
        if fig_row_ind == 1:
            ax[fig_row_ind,fig_col_ind].set_xlabel(x_axis_name, fontsize=xy_label_fontsize)
            ax[fig_row_ind,fig_col_ind].set_ylim([90.9,98.3])

        if fig_col_ind == 0:
            ax[fig_row_ind,fig_col_ind].set_ylabel(y_axis_name, fontsize=xy_label_fontsize)
        
        list_of_curve_names = []
        for curve_key, y_vs_x_df in X_vs_depth_curve_dicts.items():
            list_of_curve_names.append(legend_labeling[curve_key])
            
            x,y,e = keep_duplicate_x_with_extreeme_y(y_vs_x_df[x_axis_name], y_vs_x_df[y_axis_name], y_vs_x_df[e_axis_name])
            x,y,e = keep_only_strictly_changing_values(x, y, e, change_threshold=change_threshold_dict[x_axis_name], inreasing=True)
            ax[fig_row_ind,fig_col_ind].errorbar(x,y, yerr=e)

        ax[fig_row_ind,fig_col_ind].legend(list_of_curve_names, loc='lower right')
        

#%%

