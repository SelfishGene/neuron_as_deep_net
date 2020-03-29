import os
import numpy as np
import pandas as pd
import shutil
import json
import glob
import time
from keras.models import Model, load_model

# NOTE: during this project I've changed my coding style
# and was too lazy to edit the old code to match the new style
# so please ignore any style related wierdness
# thanks for not being petty about unimportant shit

# ALSO NOTE: prints are for logging purposes

#%% copy models from models folder to best models folder according to the "best_results_valid_105_models.csv" file

models_dir      = '/Reseach/Single_Neuron_InOut/models/'
best_models_dir = '/Reseach/Single_Neuron_InOut/models/best_models/'

# load pandas dataframe and extract filenames of best models
best_models_df  = pd.read_csv(best_models_dir + 'best_results_valid_105_models.csv')
all_best_models = best_models_df['full model filename'].tolist()

# find all models
all_models = glob.glob(models_dir + '*/*.h5')
all_model_names = [x.split('/')[-1].split('.')[0] for x in all_models]

# get intersection of all existing models and requested best models
all_best_model_names_short = list(set(all_model_names).intersection(set(all_best_models)))

# determine full paths of all best models
all_best_models_names_full = []
for model_name_short in all_best_model_names_short:
    model_name_full = glob.glob(models_dir + '*/' + model_name_short + '.h5')[0]
    all_best_models_names_full.append(model_name_full)

models_to_resave = all_best_models_names_full

print('-----')
print(len(all_model_names))
print(len(all_best_model_names_short))
print(len(all_best_models_names_full))
print('-----')

print('number of all models in total %d' %(len(all_model_names)))
print('number of best models to resave is %d' %(len(models_to_resave)))
print('-----------------------------------------------')

print('models that will be evaluated are:')
for k, curr_model_name in enumerate(all_best_models_names_full):
    print('%d: %s' %(k + 1, curr_model_name))
print('-----------------------------------------------')

# loop over all models that need saving
for k, model_filename in enumerate(models_to_resave):

    print('------------------------------')
    print('starting re-saving of model %d' %(k + 1))
    print('------------------------------')
    
    saving_start_time = time.time()
    only_model_filename = model_filename.split('/')[-1]

    print('------------------------------------------------------------------------------------------------------------')
    print('loading model "%s"' %(only_model_filename))

    # load current model
    temporal_conv_net = load_model(model_filename)
    temporal_conv_net.summary()
    
    # create new folder for the model architecture in "best_models_dir"
    dir_name = best_models_dir + only_model_filename.split('__20')[0] + '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # inside new architecture folder save:
    model_prefix = only_model_filename.split('.')[0]
    
    # (*) architecture as json
    json_target_filename = dir_name + model_prefix + '_json.json'
    print('saving architecture json: \n        %s' %(json_target_filename))
    json_string = temporal_conv_net.to_json()
    with open(json_target_filename, 'w') as outfile:
        json.dump(json_string, outfile)

    # (*) weights as hdf5
    weights_target_filename = dir_name + model_prefix + '_weights.h5'
    print('saving weights: \n        %s' %(weights_target_filename))
    temporal_conv_net.save_weights(weights_target_filename)
    
    # (*) save the model directly as hdf5
    model_target_filename = dir_name + model_prefix + '_model.h5'
    temporal_conv_net.save(model_target_filename)
    
    # (*) original training pickle file
    training_pickle_source_filename = model_filename.split('.h5')[0] + '.pickle'
    training_pickle_target_filename = dir_name + model_prefix + '_training.pickle'
    print('copying training pickle file... \n from : %s \n to   : %s' %(training_pickle_source_filename, training_pickle_target_filename))
    shutil.copyfile(training_pickle_source_filename, training_pickle_target_filename)

    # (*) original evaluation pickle file
    try:
        evaluation_pickle_source_filename = glob.glob(models_dir + '*/' + model_prefix + '.pickle')[0]
        evaluation_pickle_target_filename = dir_name + model_prefix + '_evaluation.pickle'
        print('copying evaluation pickle file... \n from : %s \n to   : %s' %(evaluation_pickle_source_filename, evaluation_pickle_target_filename))
        shutil.copyfile(evaluation_pickle_source_filename, evaluation_pickle_target_filename)
    except:
        print('couldnt find evaluation pickle file')

    saving_duration_min = (time.time() - saving_start_time) / 60
    print('time took to re-save model is %.3f minutes' %(saving_duration_min))
    print('------------------------------------------------------------------------------------------------------------')
    
#%%

print('finihsed re-saving script')
