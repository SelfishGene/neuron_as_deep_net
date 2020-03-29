import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import glob
from skimage.transform import resize
import time
import pickle
import imageio
from scipy import signal
from keras.models import Model, load_model
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_curve, auc

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

save_figures = True
all_file_endings_to_use = ['.png', '.pdf', '.svg']

# NOTE: during this project I've changed my coding style
# and was too lazy to edit the old code to match the new style
# so please ignore any style related wierdness
# thanks for not being petty about unimportant shit

# ALSO NOTE: prints are for logging purposes

#%% helper functions


def bin2dict(bin_spikes_matrix):
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds,spike_times):
        if row_ind in row_inds_spike_times_map.keys():
            row_inds_spike_times_map[row_ind].append(syn_time)
        else:
            row_inds_spike_times_map[row_ind] = [syn_time]

    return row_inds_spike_times_map


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
    experiment_dict = pickle.load(open(sim_experiment_file, "rb" ), encoding='latin1')
    
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

    linear_spaced_FPR = np.linspace(0,1, num=20000)
    linear_spaced_TPR = np.interp(linear_spaced_FPR, fpr, tpr)
    
    desired_fp_ind = min(max(1, np.argmin(abs(linear_spaced_FPR - desired_false_positive_rate))), linear_spaced_TPR.shape[0] - 1)
    
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
    
    soma_explained_variance_percent = 100.0 * explained_variance_score(y_soma_GT.ravel(), y_soma_hat.ravel())
    soma_RMSE = np.sqrt(MSE(y_soma_GT.ravel(), y_soma_hat.ravel()))
    soma_MAE  = MAE(y_soma_GT.ravel(), y_soma_hat.ravel())
    
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

model_string = 'NMDA'
# model_string = 'AMPA'
# model_string = 'AMPA_SK'

# model_size = 'small'
model_size = 'large'

models_dir = '/Reseach/Single_Neuron_InOut/models/best_models/'
data_dir   = '/Reseach/Single_Neuron_InOut/data/'

if model_string == 'NMDA':
    valid_data_dir     = data_dir + 'L5PC_NMDA_valid/'
    test_data_dir      = data_dir + 'L5PC_NMDA_test/'
    output_figures_dir = '/Reseach/Single_Neuron_InOut/figures/NMDA/'
    
    if model_size == 'small':
        model_dir = models_dir + '/NMDA_FCN__DxWxT_1x128x43/'
        NN_illustration_filename = '/Reseach/Single_Neuron_InOut/figures/NN_Illustrations/FCN_1_layer.png'
    elif model_size == 'large':
        model_dir = models_dir + '/NMDA_TCN__DxWxT_7x128x153/'
        NN_illustration_filename = '/Reseach/Single_Neuron_InOut/figures/NN_Illustrations/TCN_7_layers.png'

elif model_string == 'AMPA':
    valid_data_dir     = data_dir + 'L5PC_AMPA_valid/'
    test_data_dir      = data_dir + 'L5PC_AMPA_test/'
    output_figures_dir = '/Reseach/Single_Neuron_InOut/figures/AMPA/'
    
    if model_size == 'small':
        model_dir = models_dir + '/AMPA_FCN__DxWxT_1x128x43/'
        NN_illustration_filename = '/Reseach/Single_Neuron_InOut/figures/NN_Illustrations/FCN_1_layer.png'
    elif model_size == 'large':
        model_dir = models_dir + '/AMPA_TCN__DxWxT_4x64x120/'
        NN_illustration_filename = '/Reseach/Single_Neuron_InOut/figures/NN_Illustrations/TCN_4_layers.png'

elif model_string == 'AMPA_SK':
    valid_data_dir     = data_dir + 'L5PC_AMPA_SK_valid/'
    test_data_dir      = data_dir + 'L5PC_AMPA_SK_test/'
    output_figures_dir = '/Reseach/Single_Neuron_InOut/figures/AMPA_SK/'

    if model_size == 'small':
        model_dir = models_dir + '/AMPA_SK_FCN__DxWxT_1x128x46/'
        NN_illustration_filename = '/Reseach/Single_Neuron_InOut/figures/NN_Illustrations/FCN_1_layer.png'
    elif model_size == 'large':
        model_dir = models_dir + '/AMPA_SK_TCN__DxWxT_4x64x120/'
        NN_illustration_filename = '/Reseach/Single_Neuron_InOut/figures/NN_Illustrations/TCN_4_layers.png'

print('-----------------------------------------------')
print('finding data and model')
print('-----------------------------------------------')

valid_files = sorted(glob.glob(valid_data_dir + '*_128_simulationRuns*_6_secDuration_*'))
test_files  = sorted(glob.glob(test_data_dir  + '*_128_simulationRuns*_6_secDuration_*'))

model_filename = glob.glob(model_dir + '*_model.h5')[0]
model_metadata_filename = glob.glob(model_dir + '*_training.pickle')[0]

print('model found          : "%s"' %(model_filename.split('/')[-1]))
print('model metadata found : "%s"' %(model_metadata_filename.split('/')[-1]))
print('number of validation files is %d' %(len(valid_files)))
print('number of test files is %d' %(len(test_files)))
print('-----------------------------------------------')

#%% load valid and test datasets

print('----------------------------------------------------------------------------------------')
print('loading testing files...')
test_file_loading_start_time = time.time()

v_threshold = -55

# load test data
X_test , y_spike_test , y_soma_test  = parse_multiple_sim_experiment_files(test_files)
y_soma_test[y_soma_test > v_threshold] = v_threshold

test_file_loading_duration_min = (time.time() - test_file_loading_start_time) / 60
print('time took to load data is %.3f minutes' %(test_file_loading_duration_min))
print('----------------------------------------------------------------------------------------')

#%% load morphology

morphology_folder = '/Reseach/Single_Neuron_InOut/morphology/'
morphology_filename = morphology_folder + 'morphology_dict.pickle'
morphology_dict = pickle.load(open(morphology_filename, "rb" ), encoding='latin1')

allSectionsLength                  = morphology_dict['all_sections_length']
allSections_DistFromSoma           = morphology_dict['all_sections_distance_from_soma']
allSegmentsLength                  = morphology_dict['all_segments_length']
allSegmentsType                    = morphology_dict['all_segments_type']
allSegments_DistFromSoma           = morphology_dict['all_segments_distance_from_soma']
allSegments_SectionDistFromSoma    = morphology_dict['all_segments_section_distance_from_soma']
allSegments_SectionInd             = morphology_dict['all_segments_section_index']
allSegments_seg_ind_within_sec_ind = morphology_dict['all_segments_segment_index_within_section_index']

all_basal_section_coords  = morphology_dict['all_basal_section_coords']
all_basal_segment_coords  = morphology_dict['all_basal_segment_coords']
all_apical_section_coords = morphology_dict['all_apical_section_coords']
all_apical_segment_coords = morphology_dict['all_apical_segment_coords']

# show some colored DVTs with morphology colored with same segment color and soma voltage at the bottom
list_of_basal_section_inds  = np.unique(sorted([x[0] for x in list(all_basal_segment_coords.keys())]))
list_of_apical_section_inds = np.unique(sorted([x[0] for x in list(all_apical_segment_coords.keys())]))

seg_ind_to_xyz_coords_map = {}
seg_ind_to_sec_ind_map = {}
for k in range(len(allSegmentsType)):
    curr_segment_ind = allSegments_seg_ind_within_sec_ind[k]
    if allSegmentsType[k] == 'basal':
        curr_section_ind = allSegments_SectionInd[k]
        seg_ind_to_xyz_coords_map[k] = all_basal_segment_coords[(curr_section_ind,curr_segment_ind)]
        seg_ind_to_sec_ind_map[k] = ('basal', curr_section_ind)
    elif allSegmentsType[k] == 'apical':
        curr_section_ind = allSegments_SectionInd[k] - len(list_of_basal_section_inds)
        seg_ind_to_xyz_coords_map[k] = all_apical_segment_coords[(curr_section_ind,curr_segment_ind)]
        seg_ind_to_sec_ind_map[k] = ('apical', curr_section_ind)
    else:
        print('error!')

# plot 3 color image of the morphology
plt.close('all')

num_segments = 639
basal_cutoff = 262
tuft_cutoff  = [366,559]

apical_color = 'g'
oblique_color = 'orange'
basal_color = 'm'

basal_syn_inds    = np.arange(basal_cutoff)
oblique_syn_inds  = np.hstack((np.arange(basal_cutoff,tuft_cutoff[0]),np.arange(tuft_cutoff[1],num_segments)))
tuft_syn_inds     = np.arange(tuft_cutoff[0],tuft_cutoff[1])

all_basal_section_inds   = np.unique([seg_ind_to_sec_ind_map[x][1] for x in basal_syn_inds])
all_oblique_section_inds = np.unique([seg_ind_to_sec_ind_map[x][1] for x in oblique_syn_inds])
all_tuft_section_inds    = np.unique([seg_ind_to_sec_ind_map[x][1] for x in tuft_syn_inds])

# remove overlaping sections if any
all_oblique_section_inds = np.array(list(set(all_oblique_section_inds) - set(all_tuft_section_inds)))

# collect all basal, oblique, tuft segments
width_mult_factor = 1.2

plt.figure(figsize=(9,15))

# basal segments
for key in basal_syn_inds:
    line_width = width_mult_factor * np.array(seg_ind_to_xyz_coords_map[key]['d']).mean()
    plt.plot(seg_ind_to_xyz_coords_map[key]['x'],seg_ind_to_xyz_coords_map[key]['y'],lw=line_width,color=basal_color)

# oblique segments
for key in oblique_syn_inds:
    line_width = width_mult_factor * np.array(seg_ind_to_xyz_coords_map[key]['d']).mean()
    plt.plot(seg_ind_to_xyz_coords_map[key]['x'],seg_ind_to_xyz_coords_map[key]['y'],lw=line_width,color=oblique_color)

# tuft segments
for key in tuft_syn_inds:
    line_width = width_mult_factor * np.array(seg_ind_to_xyz_coords_map[key]['d']).mean()
    plt.plot(seg_ind_to_xyz_coords_map[key]['x'],seg_ind_to_xyz_coords_map[key]['y'],lw=line_width,color=apical_color)

# add black soma
plt.scatter(x=46.0,y=15.8,s=180,c='k', zorder=100)
plt.xlim(-180,235)
plt.ylim(-210,1200)
plt.axis('off')

if save_figures:
    figure_name = '%s__morphology' %(model_dir.split('/')[-2])
    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            plt.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
        else:
            subfolder = '%s/' %(file_ending.split('.')[-1])
            plt.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

#%% load model
print('----------------------------------------------------------------------------------------')
print('loading model "%s"' %(model_filename.split('/')[-1]))

model_loading_start_time = time.time()

temporal_conv_net = load_model(model_filename)
temporal_conv_net.summary()

input_window_size = temporal_conv_net.input_shape[1]

# load metadata pickle file
model_metadata_dict = pickle.load(open(model_metadata_filename, "rb" ), encoding='latin1')

architecture_dict = model_metadata_dict['architecture_dict']
time_window_T = (np.array(architecture_dict['filter_sizes_per_layer']) - 1).sum() + 1
overlap_size = min(max(time_window_T + 1, min(150, input_window_size - 50)), 250)

print('overlap_size = %d' %(overlap_size))
print('time_window_T = %d' %(time_window_T))
print('input shape: %s' %(str(temporal_conv_net.get_input_shape_at(0))))

model_loading_duration_min = (time.time() - model_loading_start_time) / 60
print('time took to load model is %.3f minutes' %(model_loading_duration_min))
print('----------------------------------------------------------------------------------------')

#%% create spike predictions on test set

print('----------------------------------------------------------------------------------------')
print('predicting using model...')

prediction_start_time = time.time()

y_train_soma_bias = -67.7

X_test_for_TCN = np.transpose(X_test,axes=[2,1,0])
y1_test_for_TCN = y_spike_test.T[:,:,np.newaxis]
y2_test_for_TCN = y_soma_test.T[:,:,np.newaxis] - y_train_soma_bias

y1_test_for_TCN_hat = np.zeros(y1_test_for_TCN.shape)
y2_test_for_TCN_hat = np.zeros(y2_test_for_TCN.shape)

num_test_splits = int(2 + (X_test_for_TCN.shape[1] - input_window_size) / (input_window_size - overlap_size))

for k in range(num_test_splits):
    start_time_ind = k * (input_window_size - overlap_size)
    end_time_ind   = start_time_ind + input_window_size
    
    curr_X_test_for_TCN = X_test_for_TCN[:,start_time_ind:end_time_ind,:]
    
    if curr_X_test_for_TCN.shape[1] < input_window_size:
        padding_size = input_window_size - curr_X_test_for_TCN.shape[1]
        X_pad = np.zeros((curr_X_test_for_TCN.shape[0],padding_size,curr_X_test_for_TCN.shape[2]))
        curr_X_test_for_TCN = np.hstack((curr_X_test_for_TCN,X_pad))
        
    curr_y1_test_for_TCN, curr_y2_test_for_TCN, _ = temporal_conv_net.predict(curr_X_test_for_TCN)

    if k == 0:
        y1_test_for_TCN_hat[:,:end_time_ind,:] = curr_y1_test_for_TCN
        y2_test_for_TCN_hat[:,:end_time_ind,:] = curr_y2_test_for_TCN
    elif k == (num_test_splits - 1):
        t0 = start_time_ind + overlap_size
        duration_to_fill = y1_test_for_TCN_hat.shape[1] - t0
        y1_test_for_TCN_hat[:,t0:,:] = curr_y1_test_for_TCN[:,overlap_size:(overlap_size + duration_to_fill),:]
        y2_test_for_TCN_hat[:,t0:,:] = curr_y2_test_for_TCN[:,overlap_size:(overlap_size + duration_to_fill),:]
    else:
        t0 = start_time_ind + overlap_size
        y1_test_for_TCN_hat[:,t0:end_time_ind,:] = curr_y1_test_for_TCN[:,overlap_size:,:]
        y2_test_for_TCN_hat[:,t0:end_time_ind,:] = curr_y2_test_for_TCN[:,overlap_size:,:]

# zero score the prediction and align it with the actual test
s_dst = y2_test_for_TCN.std()
m_dst = y2_test_for_TCN.mean()

s_src = y2_test_for_TCN_hat.std()
m_src = y2_test_for_TCN_hat.mean()

y2_test_for_TCN_hat = (y2_test_for_TCN_hat - m_src) / s_src
y2_test_for_TCN_hat = s_dst * y2_test_for_TCN_hat + m_dst

# convert to simple (num_simulations, num_time_points) format
y_spikes_GT  = y1_test_for_TCN[:,:,0]
y_spikes_hat = y1_test_for_TCN_hat[:,:,0]
y_soma_GT    = y2_test_for_TCN[:,:,0]
y_soma_hat   = y2_test_for_TCN_hat[:,:,0]

prediction_duration_min = (time.time() - prediction_start_time) / 60
print('finished prediction. time took to predict is %.2f minutes' %(prediction_duration_min))
print('----------------------------------------------------------------------------------------')

#%% evaluate the model and save the results

print('----------------------------------------------------------------------------------------')
print('calculating key accuracy results...')

saving_start_time = time.time()

desired_FP_list = [0.0001, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0050, 0.0100, 0.0200, 0.0300, 0.0400, 0.0500, 0.1000]
evaluations_results_dict = {}

ignore_time_at_start_ms = 500
num_spikes_per_sim = [0,18]
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

model_metadata_dict['evaluations_results_dict'] = evaluations_results_dict

print('---------------------------')
print('main results:')
print('---------------------------')
print('TP @ 0.0025 FP = %.3f' %(evaluations_results_dict['starting_at_500ms_spikes_in_[0,24]_range']['TP @ 0.0025 FP']))
print('spikes AUC = %.4f' %(evaluations_results_dict['starting_at_500ms_spikes_in_[0,24]_range']['AUC']))
print('soma explained var = %.2f%s' %(evaluations_results_dict['starting_at_500ms_spikes_in_[0,24]_range']['soma_explained_variance_percent'],'%'))
print('soma RMSE = %.3f [mV]' %(evaluations_results_dict['starting_at_500ms_spikes_in_[0,24]_range']['soma_RMSE']))
print('soma MAE = %.3f [mV]' %(evaluations_results_dict['starting_at_500ms_spikes_in_[0,24]_range']['soma_MAE']))
print('---------------------------')

saving_duration_min = (time.time() - saving_start_time) / 60
print('time took to calculate key prediction accuracy results is %.3f minutes' %(saving_duration_min))
print('----------------------------------------------------------------------------------------')
    
##%% plot the evaluation figures:
# (1) ROC curve of binary prediction
# (2) cross correlation between prediction and GT (illustrating the temporal accuracy of the prediction)
# (3) voltage prediction scatter plot

plt.close('all')

ignore_time_at_start_ms = 500
num_spikes_per_sim = [0,24]

xytick_labels_fontsize = 18
title_fontsize = 29
xylabels_fontsize = 22
legend_fontsize = 18

fig = plt.figure(figsize=(10,11))

time_points_to_eval = np.arange(y_spikes_GT.shape[1]) >= ignore_time_at_start_ms
simulations_to_eval = np.logical_and((y_spikes_GT.sum(axis=1) >= num_spikes_per_sim[0]),(y_spikes_GT.sum(axis=1) <= num_spikes_per_sim[1]))

print('total amount of simualtions is %d' %(y_spikes_GT.shape[0]))
print('percent of simulations kept = %.2f%s' %(100 * simulations_to_eval.mean(),'%'))

y_spikes_GT_to_eval  = y_spikes_GT[simulations_to_eval,:][:,time_points_to_eval]
y_spikes_hat_to_eval = y_spikes_hat[simulations_to_eval,:][:,time_points_to_eval]
y_soma_GT_to_eval    = y_soma_GT[simulations_to_eval,:][:,time_points_to_eval]
y_soma_hat_to_eval   = y_soma_hat[simulations_to_eval,:][:,time_points_to_eval]

# ROC curve
desired_false_positive_rate = 0.002

fpr, tpr, thresholds = roc_curve(y_spikes_GT_to_eval.ravel(), y_spikes_hat_to_eval.ravel())

desired_fp_ind = np.argmin(abs(fpr - desired_false_positive_rate))
if desired_fp_ind == 0:
    desired_fp_ind = 1
actual_false_positive_rate = fpr[desired_fp_ind]

AUC_score = auc(fpr, tpr)

print('AUC = %.4f' %(AUC_score))
print('at %.4f FP rate, TP = %.4f' %(actual_false_positive_rate, tpr[desired_fp_ind]))

# cross correlation
half_time_window_size_ms = 50

desired_threshold = thresholds[desired_fp_ind]
ground_truth_output_spikes = y_spikes_GT_to_eval
predicted_output_spikes    = y_spikes_hat_to_eval > desired_threshold
num_test_traces            = y_spikes_GT_to_eval.shape[0]

zero_padding_matrix = np.zeros((num_test_traces,half_time_window_size_ms))
predicted_output_spikes_padded    = np.hstack((zero_padding_matrix,predicted_output_spikes,zero_padding_matrix))
ground_truth_output_spikes_padded = np.hstack((zero_padding_matrix,ground_truth_output_spikes,zero_padding_matrix))

recall_curve = np.zeros(1 + 2 * half_time_window_size_ms)
trace_inds, spike_inds = np.nonzero(ground_truth_output_spikes_padded)
for trace_ind, spike_ind in zip(trace_inds,spike_inds):
    recall_curve += predicted_output_spikes_padded[trace_ind,spike_ind - half_time_window_size_ms:1 + spike_ind + half_time_window_size_ms]
recall_curve /= recall_curve.sum()

filter_cross_corr = True
if filter_cross_corr:
    cc_filter_size = 2
    recall_curve_filtered = signal.convolve(recall_curve, (1.0 / cc_filter_size) * np.ones(cc_filter_size), mode='same')
    recall_curve = 0.5 * recall_curve + 0.5 * recall_curve_filtered

time_axis_cc = np.arange(-half_time_window_size_ms, half_time_window_size_ms + 1)

# voltage scatter plot
num_datapoints_in_scatter = 20000

selected_datapoints = np.random.choice(range(len(y_soma_GT_to_eval.ravel())),size=num_datapoints_in_scatter,replace=False)
selected_GT = y_soma_GT_to_eval.ravel()[selected_datapoints] + 0.02 * np.random.randn(num_datapoints_in_scatter) + y_train_soma_bias
selected_pred = y_soma_hat_to_eval.ravel()[selected_datapoints] + y_train_soma_bias

soma_explained_variance_percent = 100.0 * explained_variance_score(y_soma_GT_to_eval.ravel(), y_soma_hat_to_eval.ravel())
soma_RMSE = np.sqrt(MSE(y_soma_GT_to_eval.ravel(), y_soma_hat_to_eval.ravel()))
soma_MAE  = MAE(y_soma_GT_to_eval.ravel(), y_soma_hat_to_eval.ravel())

print('soma voltage prediction explained variance = %.2f%s' %(soma_explained_variance_percent,'%'))


gs2 = gridspec.GridSpec(5,2)
gs2.update(left=0.15, right=0.85, bottom=0.15, top=0.88, wspace=0.58, hspace=1.1)
a33_left  = plt.subplot(gs2[:2,0])
a33_right = plt.subplot(gs2[:2,1])
ax34      = plt.subplot(gs2[2:,:])

# ROC curve
a33_left.plot(fpr, tpr, c='k')
a33_left.set_xlabel('False alarm rate', fontsize=xylabels_fontsize)
a33_left.set_ylabel('Hit rate', fontsize=xylabels_fontsize)
a33_left.set_ylim(0,1.05)
a33_left.set_xlim(-0.03,1)

a33_left.spines['top'].set_visible(False)
a33_left.spines['right'].set_visible(False)

for tick_label in (a33_left.get_xticklabels() + a33_left.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

a33_left.set_xticks([0.0,0.5,1.0])
a33_left.set_yticks([0.0,0.5,1.0])

left, bottom, width, height = [0.264, 0.68, 0.14, 0.15]
a33_left_inset = fig.add_axes([left, bottom, width, height])
a33_left_inset.plot(fpr, tpr, c='k')
a33_left_inset.set_ylim(0,1.05)
a33_left_inset.set_xlim(-0.001,0.05)
a33_left_inset.spines['top'].set_visible(False)
a33_left_inset.spines['right'].set_visible(False)

a33_left_inset.scatter(actual_false_positive_rate, tpr[desired_fp_ind + 1], c='r', s=100)

## cross correlation curve ( P( predicted spikes | ground truth==spike) )
max_firing_rate = 10 * int(max(1000 * recall_curve) / 10)
midpoint_firing_rate = 5 * int(max_firing_rate / 10)
a33_right.set_yticks([0, midpoint_firing_rate,max_firing_rate])

a33_right.plot(time_axis_cc, 1000 * recall_curve, c='k')
a33_right.set_ylim(0, 1.05 * 1000 * recall_curve.max())
a33_right.set_xlabel('$\Delta t$ (ms)', fontsize=xylabels_fontsize)
a33_right.set_ylabel('spike rate (Hz)', fontsize=xylabels_fontsize)
a33_right.set_xticks([-50,0,50])
a33_right.spines['top'].set_visible(False)
a33_right.spines['right'].set_visible(False)

for tick_label in (a33_right.get_xticklabels() + a33_right.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

# voltage scatter plot
soma_voltage_lims = np.round([np.percentile(selected_GT,0.2),np.percentile(selected_GT,99.8)]).astype(int)
soma_voltage_lims = np.round([np.percentile(selected_GT,0.2),-56]).astype(int)
voltage_granularity = 6
voltage_setpoint = -57
voltage_axis = np.arange(soma_voltage_lims[0],soma_voltage_lims[1])
voltage_ticks_to_show = np.unique(((voltage_axis - voltage_setpoint) / voltage_granularity).astype(int) * voltage_granularity + voltage_setpoint)
voltage_ticks_to_show = voltage_ticks_to_show[np.logical_and(voltage_ticks_to_show >= soma_voltage_lims[0],
                                                             voltage_ticks_to_show <= soma_voltage_lims[1])]
ax34.set_xticks(voltage_ticks_to_show)
ax34.set_yticks(voltage_ticks_to_show)

ax34.scatter(selected_GT,selected_pred, s=1.0, alpha=0.8)
ax34.set_xlabel('L5PC (%s) (mV)' %(model_string), fontsize=xylabels_fontsize)
ax34.set_ylabel('ANN (mV)', fontsize=xylabels_fontsize)
ax34.set_xlim(soma_voltage_lims[0],soma_voltage_lims[1])
ax34.set_ylim(soma_voltage_lims[0],soma_voltage_lims[1])

ax34.plot([-90,-50],[-90,-50], ls='-', c='k')

ax34.spines['top'].set_visible(False)
ax34.spines['right'].set_visible(False)

for tick_label in (ax34.get_xticklabels() + ax34.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

if save_figures:
    figure_name = '%s__model_evaluation' %(model_dir.split('/')[-2])
    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
        else:
            subfolder = '%s/' %(file_ending.split('.')[-1])
            fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

#%% show prediction trace

for k in range(30):
    num_spikes_per_simulation = y1_test_for_TCN.sum(axis=1)[:,0]
    possible_presentable_candidates = np.nonzero(np.logical_and(num_spikes_per_simulation >= 3, num_spikes_per_simulation <= 10))[0]
    selected_trace = np.random.choice(possible_presentable_candidates)
    
    zoomin_fraction = [0.25 + 0.23 * np.random.rand(), 0.52 + 0.23 * np.random.rand()]
    
    # selected_trace = 122
    # zoomin_fraction = [0.35,0.61]
    
    print('selected trace = %d' %(selected_trace))
    print('zoomin_fraction = %s' %(zoomin_fraction))
    print('at %.4f FP rate, TP = %.4f' %(actual_false_positive_rate, tpr[desired_fp_ind]))
    
    spike_trace_GT   = y1_test_for_TCN[selected_trace,:,0]
    spike_trace_pred = y1_test_for_TCN_hat[selected_trace,:,0] > desired_threshold
    
    output_spike_times_in_ms_GT   = np.nonzero(spike_trace_GT)[0]
    output_spike_times_in_ms_pred = np.nonzero(spike_trace_pred)[0]
    
    soma_voltage_trace_GT   = y2_test_for_TCN[selected_trace,:,0] + y_train_soma_bias
    soma_voltage_trace_pred = y2_test_for_TCN_hat[selected_trace,:,0] + y_train_soma_bias
    
    soma_voltage_trace_GT[output_spike_times_in_ms_GT] = 40
    soma_voltage_trace_pred[output_spike_times_in_ms_pred] = 40
        
    sim_duration_ms = spike_trace_GT.shape[0]
    
    # show raster plot and cell output
    time_in_sec = np.arange(sim_duration_ms) / 1000.0
    sim_duration_ms = spike_trace_GT.shape[0]
    sim_duration_sec = int(sim_duration_ms / 1000.0)

    xytick_labels_fontsize = 16
    title_fontsize = 26
    xylabels_fontsize = 19
    legend_fontsize = 15
    
    plt.close('all')
    fig = plt.figure(figsize=(17,8))
    
    gs1 = gridspec.GridSpec(2,1)
    gs1.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.01)
    
    ax11 = plt.subplot(gs1[0,0])
    ax12 = plt.subplot(gs1[1,0])
    ax11.axis('off')
    ax12.axis('off')
    
    ax11.plot(time_in_sec,soma_voltage_trace_GT,c='c')
    ax11.plot(time_in_sec,soma_voltage_trace_pred,c='m',linestyle=':')
    ax11.set_xlim(0,sim_duration_sec)
    ax11.set_ylabel('$V_m$ (mV)', fontsize=xylabels_fontsize)
    
    for tick_label in (ax11.get_xticklabels() + ax11.get_yticklabels()):
        tick_label.set_fontsize(xytick_labels_fontsize)
    
    zoomout_scalebar_xloc = 0.95 * sim_duration_sec
    zoomin_xlims = [zoomin_fraction[0] * sim_duration_sec, zoomin_fraction[1] * sim_duration_sec]
    zoomin_dur_sec = zoomin_xlims[1] - zoomin_xlims[0]
    zoomin_time_in_sec = np.logical_and(time_in_sec >= zoomin_xlims[0], time_in_sec <= zoomin_xlims[1])
    zoomin_ylims = [soma_voltage_trace_GT[zoomin_time_in_sec].min() - 2.5, -52]
    zoomin_scalebar_xloc = zoomin_xlims[1] - 0.05 * zoomin_dur_sec
    
    ax12.plot(time_in_sec,soma_voltage_trace_GT,c='c')
    ax12.plot(time_in_sec,soma_voltage_trace_pred,c='m',linestyle=':')
    ax12.set_xlim(zoomin_xlims[0],zoomin_xlims[1])
    ax12.set_ylim(zoomin_ylims[0],zoomin_ylims[1])
    ax12.set_ylabel('$V_m$ (mV)', fontsize=xylabels_fontsize)
    ax12.set_xlabel('time (sec)', fontsize=xylabels_fontsize)
    
    for tick_label in (ax12.get_xticklabels() + ax12.get_yticklabels()):
        tick_label.set_fontsize(xytick_labels_fontsize)

    # add scale bar to top plot
    scalebar_loc = np.array([zoomout_scalebar_xloc,-25])
    scalebar_size_x = 0.6
    scalebar_str_x = '600 ms'
    scalebar_size_y = 40
    scalebar_str_y = '40 mV'
    
    x = [scalebar_loc[0], scalebar_loc[0] - scalebar_size_x]
    y = [scalebar_loc[1], scalebar_loc[1]]
    ax11.plot(x,y,lw=2,c='k')
    ax11.text(scalebar_loc[0] - 0.05 * scalebar_size_x, scalebar_loc[1] - 0.15 * scalebar_size_y,
              scalebar_str_x, color='k', fontsize=15, ha='right', va='top', rotation='horizontal')
    
    x = [scalebar_loc[0], scalebar_loc[0]]
    y = [scalebar_loc[1], scalebar_loc[1] + scalebar_size_y]
    ax11.plot(x,y,lw=2,c='k')
    ax11.text(scalebar_loc[0] + 0.1 * scalebar_size_x, scalebar_loc[1] + 0.6 * scalebar_size_y,
              scalebar_str_y, color='k', fontsize=15, ha='left', va='top', rotation='horizontal')

    # add dashed rectangle
    rect_w = zoomin_xlims[1] - zoomin_xlims[0]
    rect_h = zoomin_ylims[1] - zoomin_ylims[0]
    rect_bl_x = zoomin_xlims[0]
    rect_bl_y = zoomin_ylims[0]
    dashed_rectangle = mpatches.Rectangle((rect_bl_x,rect_bl_y),rect_w,rect_h,linewidth=2,edgecolor='k',linestyle='--',facecolor='none')
    
    ax11.add_patch(dashed_rectangle)

    # add scalebar to bottom plot
    scalebar_loc = np.array([zoomin_scalebar_xloc,-60])
    scalebar_size_x = 0.06
    scalebar_str_x = '60 ms'
    scalebar_size_y = 5
    scalebar_str_y = '5 mV'
    
    x = [scalebar_loc[0], scalebar_loc[0] - scalebar_size_x]
    y = [scalebar_loc[1], scalebar_loc[1]]
    ax12.plot(x,y,lw=2,c='k')
    ax12.text(scalebar_loc[0] - 0.15 * scalebar_size_x, scalebar_loc[1] - 0.15 * scalebar_size_y,
              scalebar_str_x, color='k', fontsize=15, ha='right', va='top', rotation='horizontal')
    
    x = [scalebar_loc[0], scalebar_loc[0]]
    y = [scalebar_loc[1], scalebar_loc[1] + scalebar_size_y]
    ax12.plot(x,y,lw=2,c='k')
    ax12.text(scalebar_loc[0] + 0.1 * scalebar_size_x, scalebar_loc[1] + 0.6 * scalebar_size_y,
              scalebar_str_y, color='k', fontsize=15, ha='left', va='top', rotation='horizontal')
    
    if save_figures:
        figure_name = '%s__single_prediction_trace_%d' %(model_dir.split('/')[-2], selected_trace)
        for file_ending in all_file_endings_to_use:
            if file_ending == '.png':
                fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
            else:
                subfolder = '%s/' %(file_ending.split('.')[-1])
                fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')


#%% show several prediction traces

num_subplots = 5

xytick_labels_fontsize = 16
title_fontsize = 26
xylabels_fontsize = 19
legend_fontsize = 15

num_spikes_per_simulation = y1_test_for_TCN.sum(axis=1)[:,0]
possible_presentable_candidates = np.nonzero(np.logical_and(num_spikes_per_simulation >= 3, num_spikes_per_simulation <= 15))[0]
selected_traces = np.random.choice(possible_presentable_candidates, size=num_subplots)

plt.close('all')
fig, ax = plt.subplots(nrows=num_subplots, ncols=1, figsize=(20,30))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.01)
for k, selected_trace in enumerate(selected_traces):
    
    spike_trace_GT   = y1_test_for_TCN[selected_trace,:,0]
    spike_trace_pred = y1_test_for_TCN_hat[selected_trace,:,0] > desired_threshold
    
    output_spike_times_in_ms_GT   = np.nonzero(spike_trace_GT)[0]
    output_spike_times_in_ms_pred = np.nonzero(spike_trace_pred)[0]
    
    soma_voltage_trace_GT   = y2_test_for_TCN[selected_trace,:,0] + y_train_soma_bias
    soma_voltage_trace_pred = y2_test_for_TCN_hat[selected_trace,:,0] + y_train_soma_bias
    
    soma_voltage_trace_GT[output_spike_times_in_ms_GT] = 40
    soma_voltage_trace_pred[output_spike_times_in_ms_pred] = 40
        
    ax[k].axis('off')
    ax[k].plot(time_in_sec,soma_voltage_trace_GT,c='c')
    ax[k].plot(time_in_sec,soma_voltage_trace_pred,c='m',linestyle=':')
    ax[k].set_xlim(0.02,sim_duration_sec)
    ax[k].set_ylabel('$V_m$ (mV)', fontsize=xylabels_fontsize)
    for tick_label in (ax11.get_xticklabels() + ax11.get_yticklabels()):
        tick_label.set_fontsize(xytick_labels_fontsize)
    
    if k == int(num_subplots / 2):
        # add scale bar to top plot
        scalebar_loc = np.array([zoomout_scalebar_xloc,-25])
        scalebar_size_x = 0.6
        scalebar_str_x = '600 ms'
        scalebar_size_y = 40
        scalebar_str_y = '40 mV'
        
        x = [scalebar_loc[0], scalebar_loc[0] - scalebar_size_x]
        y = [scalebar_loc[1], scalebar_loc[1]]
        ax[k].plot(x,y,lw=2,c='k')
        ax[k].text(scalebar_loc[0] - 0.05 * scalebar_size_x, scalebar_loc[1] - 0.15 * scalebar_size_y,
                   scalebar_str_x, color='k', fontsize=15, ha='right', va='top', rotation='horizontal')
        
        x = [scalebar_loc[0], scalebar_loc[0]]
        y = [scalebar_loc[1], scalebar_loc[1] + scalebar_size_y]
        ax[k].plot(x,y,lw=2,c='k')
        ax[k].text(scalebar_loc[0] + 0.1 * scalebar_size_x, scalebar_loc[1] + 0.6 * scalebar_size_y,
                   scalebar_str_y, color='k', fontsize=15, ha='left', va='top', rotation='horizontal')

if save_figures:
    figure_name = '%s__multiple_prediction_traces_%d' %(model_dir.split('/')[-2], np.random.randint(10))
    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
        else:
            subfolder = '%s/' %(file_ending.split('.')[-1])
            fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

#%% show several first layer weights
    
# show all first layer learned weights

plt.close('all')

first_layer_weights = temporal_conv_net.get_weights()[0]
time_span, _, num_filters = first_layer_weights.shape

ylims = np.array([-1.01,1.01]) * max(abs(first_layer_weights.max()),abs(first_layer_weights.min()))

if time_span <= 50:
    max_num_plots_per_figure = 32
elif time_span <= 100:
    max_num_plots_per_figure = 24
else:
    max_num_plots_per_figure = 16
    
total_num_figures = int(np.ceil(num_filters / float(max_num_plots_per_figure)))

for fig_ind in range(total_num_figures):
    start_filter_to_show = fig_ind * max_num_plots_per_figure
    end_filter_to_show   = min(num_filters, start_filter_to_show + max_num_plots_per_figure)

    filters_to_show = list(range(start_filter_to_show,end_filter_to_show))

    plt.figure(figsize=(34,17))
    for k, filter_ind in enumerate(filters_to_show):
        plt.subplot(1,len(filters_to_show),k + 1); plt.title('filter %d' %(filter_ind))
        plt.imshow(first_layer_weights[:,:,filter_ind].T,cmap='jet')
        # plt.clim(vmin=ylims[0],vmax=ylims[1])
        plt.axis('off')
    plt.tight_layout()


#%% show selected filter in depth and temporal profile as well
    
plt.close('all')

### NMDA 1x128x43
#interesting_filters = [4,14,17,27,29,44,56,59,60,65,84]

### NMDA 7x128x153
# interesting_filters = [8,17,25,37,52,54,59,66,71,89,91,93,96,114]

### AMPA 1x128x43
interesting_filters = [7,13,16,34,38,53,56,65,69,76,83,99,105,116,120,57,59,66,79,95]

### AMPA 4x64x120
# interesting_filters = [6,11,25,32,43,44,55,57]

### AMPA_SK 1x128x46
# interesting_filters = [0,5,13,27,40,46,49,63,78,107,97]

### AMPA_SK 4x64x120
# interesting_filters = [1,3,16,26,32,40,59,62,63]

selected_filter_ind = np.random.choice(interesting_filters)

filter_size = 2

first_layer_weights = np.flip(temporal_conv_net.get_weights()[0], axis=0)
time_span, _, num_filters = first_layer_weights.shape

weight_granularity = 0.06
time_granularity = 20

max_time_to_show = 40

use_filtered = True
if use_filtered:
    first_layer_weights_filtered = signal.convolve(first_layer_weights, (1.0 / filter_size) * np.ones((filter_size,1,1)), mode='valid')
    first_layer_weights = first_layer_weights_filtered

if first_layer_weights.shape[0] >= max_time_to_show:
    first_layer_weights = first_layer_weights[:max_time_to_show]

# invert if needed
exc_sum = first_layer_weights[:12,:num_segments,selected_filter_ind].sum()
inh_sum = first_layer_weights[:12,num_segments:,selected_filter_ind].sum()
exc_minus_inh = exc_sum - inh_sum

if exc_minus_inh < 0:
    first_layer_weights = -first_layer_weights

upper_limit = max(np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),99.95),np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),0.05))
ylims = np.array([-1.06,1.06]) * upper_limit

xlims = [-5 * int(first_layer_weights.shape[0] / 5),0]

num_segments = 639
basal_cutoff = 262
tuft_cutoff  = [366,559]

ex_basal_syn_inds    = np.arange(basal_cutoff)
ex_oblique_syn_inds  = np.hstack((np.arange(basal_cutoff,tuft_cutoff[0]),np.arange(tuft_cutoff[1],num_segments)))
ex_tuft_syn_inds     = np.arange(tuft_cutoff[0],tuft_cutoff[1])
inh_basal_syn_inds   = num_segments + ex_basal_syn_inds
inh_oblique_syn_inds = num_segments + ex_oblique_syn_inds
inh_tuft_syn_inds    = num_segments + ex_tuft_syn_inds

basal_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_basal_syn_inds,selected_filter_ind].T)
basal_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_basal_syn_inds,selected_filter_ind].T)
basal_weights_example_filter     = np.concatenate((basal_weights_example_filter_ex,basal_weights_example_filter_inh),axis=0)
oblique_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_oblique_syn_inds,selected_filter_ind].T)
oblique_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_oblique_syn_inds,selected_filter_ind].T)
oblique_weights_example_filter     = np.concatenate((oblique_weights_example_filter_ex, oblique_weights_example_filter_inh),axis=0)
tuft_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_tuft_syn_inds,selected_filter_ind].T)
tuft_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_tuft_syn_inds,selected_filter_ind].T)
tuft_weights_example_filter     = np.concatenate((tuft_weights_example_filter_ex,tuft_weights_example_filter_inh),axis=0)

time_axis = -np.arange(first_layer_weights.shape[0])

#%% create nice figure

ex_basal_color    = 'red'
ex_oblique_color  = 'darkorange'
ex_tuft_color     = 'yellow'
inh_basal_color   = 'darkblue'
inh_oblique_color = 'blue'
inh_tuft_color    = 'skyblue'

cmap = plt.cm.coolwarm

custom_lines = [Line2D([0], [0], color=ex_basal_color, lw=4),
                Line2D([0], [0], color=ex_oblique_color, lw=4),
                Line2D([0], [0], color=ex_tuft_color, lw=4),
                Line2D([0], [0], color=inh_basal_color, lw=4),
                Line2D([0], [0], color=inh_oblique_color, lw=4),
                Line2D([0], [0], color=inh_tuft_color, lw=4)]

all_traces_alpha = 0.08
mean_linewidth = 4.0

figure_xlims = xlims
figure_xlims[0] = max(-40, figure_xlims[0])

ex_basal_syn_inds    = np.arange(basal_cutoff)
ex_oblique_syn_inds  = np.hstack((np.arange(basal_cutoff,tuft_cutoff[0]),np.arange(tuft_cutoff[1],num_segments)))
ex_tuft_syn_inds     = np.arange(tuft_cutoff[0],tuft_cutoff[1])
inh_basal_syn_inds   = num_segments + ex_basal_syn_inds
inh_oblique_syn_inds = num_segments + ex_oblique_syn_inds
inh_tuft_syn_inds    = num_segments + ex_tuft_syn_inds

basal_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_basal_syn_inds,selected_filter_ind].T)
basal_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_basal_syn_inds,selected_filter_ind].T)
basal_weights_example_filter     = np.concatenate((basal_weights_example_filter_ex,basal_weights_example_filter_inh),axis=0)
oblique_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_oblique_syn_inds,selected_filter_ind].T)
oblique_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_oblique_syn_inds,selected_filter_ind].T)
oblique_weights_example_filter     = np.concatenate((oblique_weights_example_filter_ex, oblique_weights_example_filter_inh),axis=0)
tuft_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_tuft_syn_inds,selected_filter_ind].T)
tuft_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_tuft_syn_inds,selected_filter_ind].T)
tuft_weights_example_filter     = np.concatenate((tuft_weights_example_filter_ex,tuft_weights_example_filter_inh),axis=0)

combined_filter = np.concatenate((basal_weights_example_filter_ex,oblique_weights_example_filter_ex,tuft_weights_example_filter_ex,
                                  basal_weights_example_filter_inh,oblique_weights_example_filter_inh,tuft_weights_example_filter_inh),axis=0)

# draw 2 x 3 (basal,oblique,tuft) matrix
ex_basal_syn_inds    = np.arange(basal_cutoff)
ex_oblique_syn_inds  = np.hstack((np.arange(basal_cutoff,tuft_cutoff[0]),np.arange(tuft_cutoff[1],num_segments)))
ex_tuft_syn_inds     = np.arange(tuft_cutoff[0],tuft_cutoff[1])
inh_basal_syn_inds   = num_segments + ex_basal_syn_inds
inh_oblique_syn_inds = num_segments + ex_oblique_syn_inds
inh_tuft_syn_inds    = num_segments + ex_tuft_syn_inds

basal_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_basal_syn_inds,selected_filter_ind].T)
basal_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_basal_syn_inds,selected_filter_ind].T)
basal_weights_example_filter     = np.concatenate((basal_weights_example_filter_ex,basal_weights_example_filter_inh),axis=0)
oblique_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_oblique_syn_inds,selected_filter_ind].T)
oblique_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_oblique_syn_inds,selected_filter_ind].T)
oblique_weights_example_filter     = np.concatenate((oblique_weights_example_filter_ex, oblique_weights_example_filter_inh),axis=0)
tuft_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_tuft_syn_inds,selected_filter_ind].T)
tuft_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_tuft_syn_inds,selected_filter_ind].T)
tuft_weights_example_filter     = np.concatenate((tuft_weights_example_filter_ex,tuft_weights_example_filter_inh),axis=0)

time_axis = -np.arange(first_layer_weights.shape[0])
upper_limit = max(np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),99.8),np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),0.2))
weights_ylims = np.array([-1.08,1.08]) * upper_limit
weight_ticks_lims = (np.array(weights_ylims) / weight_granularity).astype(int) * weight_granularity

xytick_labels_fontsize = 27
title_fontsize = 37
xylabels_fontsize = 37
legend_fontsize = 16
all_traces_alpha = 0.08
mean_linewidth = 4.0

fig = plt.figure(figsize=(19,19))

gs1 = gridspec.GridSpec(1,3)
gs1.update(left=0.10, right=0.97, bottom=0.30, top=0.98, wspace=0.14, hspace=0.03)

gs2 = gridspec.GridSpec(1,3)
gs2.update(left=0.10, right=0.97, bottom=0.06, top=0.28, wspace=0.11, hspace=0.03)

ax00 = plt.subplot(gs1[0,0])
ax10 = plt.subplot(gs2[0,0])

ax01 = plt.subplot(gs1[0,1])
ax11 = plt.subplot(gs2[0,1])

ax02 = plt.subplot(gs1[0,2])
ax12 = plt.subplot(gs2[0,2])

ax00.axis('off')
ax01.axis('off')
ax02.axis('off')

# basal
weights_images = ax00.imshow(resize(basal_weights_example_filter, (combined_filter.shape[0], 200)),
                             cmap='jet', vmin=weights_ylims[0],vmax=weights_ylims[1], aspect='auto')
ax00.set_xticks([])
ax00.set_ylabel('Synaptic index', fontsize=xylabels_fontsize)
for ytick_label in ax00.get_yticklabels():
    ytick_label.set_fontsize(xytick_labels_fontsize)

ax_colorbar = inset_axes(ax00, width="67%", height="6%", loc=2)
cbar = plt.colorbar(weights_images, cax=ax_colorbar, orientation="horizontal", ticks=[weight_ticks_lims[0], 0, weight_ticks_lims[1]])
ax_colorbar.xaxis.set_ticks_position("bottom")
cbar.ax.tick_params(labelsize=xytick_labels_fontsize)
ax00.text(10, 132, 'Weight (A.U)', color='k', fontsize=title_fontsize, ha='left', va='top', rotation='horizontal')

ax10.plot(time_axis, np.fliplr(basal_weights_example_filter_ex).T , c='r', alpha=all_traces_alpha)
ax10.plot(time_axis, np.mean(np.fliplr(basal_weights_example_filter_ex).T, axis=1) , c='r', lw=mean_linewidth)
ax10.plot(time_axis, np.fliplr(basal_weights_example_filter_inh).T, c='b', alpha=all_traces_alpha)
ax10.plot(time_axis, np.mean(np.fliplr(basal_weights_example_filter_inh).T, axis=1) , c='b', lw=mean_linewidth)

ax10.set_xlim(time_axis.min(),time_axis.max())
ax10.set_xlabel('Time before $t_0$ (ms)', fontsize=xylabels_fontsize)
ax10.set_ylim(weights_ylims[0],weights_ylims[1])
ax10.set_ylabel('Weight (A.U)', fontsize=xylabels_fontsize)

time_ticks_to_show = np.unique((np.array(time_axis) / time_granularity).astype(int) * time_granularity)
ax10.set_xticks(time_ticks_to_show)

weights_axis = np.linspace(weights_ylims[0],weights_ylims[1],10)
weight_ticks_to_show = np.unique((np.array(weights_axis) / weight_granularity).astype(int) * weight_granularity)
ax10.set_yticks(weight_ticks_to_show)

ax10.spines['top'].set_visible(False)
ax10.spines['right'].set_visible(False)

for ytick_label in ax10.get_yticklabels():
    ytick_label.set_fontsize(xytick_labels_fontsize)
for xtick_label in ax10.get_xticklabels():
    xtick_label.set_fontsize(xytick_labels_fontsize)

# place a text box near the traces
#ax10.text(-25, 0.25, 'Exc', color='r', fontsize=20, verticalalignment='bottom')
#ax10.text(-25, -0.3, 'Inh', color='b', fontsize=20, verticalalignment='top')

# oblique
weights_images = ax01.imshow(resize(oblique_weights_example_filter, (combined_filter.shape[0], 200)),
                             cmap='jet', vmin=weights_ylims[0],vmax=weights_ylims[1], aspect='auto')
ax01.set_xticks([])
ax01.set_ylabel('Synaptic index', fontsize=xylabels_fontsize)
for ytick_label in ax01.get_yticklabels():
    ytick_label.set_fontsize(xytick_labels_fontsize)

ax11.plot(time_axis, np.fliplr(oblique_weights_example_filter_ex).T , c='r', alpha=all_traces_alpha)
ax11.plot(time_axis, np.mean(np.fliplr(oblique_weights_example_filter_ex).T, axis=1) , c='r', lw=mean_linewidth)
ax11.plot(time_axis, np.fliplr(oblique_weights_example_filter_inh).T, c='b', alpha=all_traces_alpha)
ax11.plot(time_axis, np.mean(np.fliplr(oblique_weights_example_filter_inh).T, axis=1) , c='b', lw=mean_linewidth)

ax11.set_xlim(time_axis.min(),time_axis.max())
ax11.set_xlabel('Time before $t_0$ (ms)', fontsize=xylabels_fontsize)
ax11.set_ylim(weights_ylims[0],weights_ylims[1])

time_ticks_to_show = np.unique((np.array(time_axis) / time_granularity).astype(int) * time_granularity)
ax11.set_xticks(time_ticks_to_show)

ax11.spines['top'].set_visible(False)
ax11.spines['right'].set_visible(False)
ax11.spines['left'].set_visible(False)

ax11.set_yticks([])
for ytick_label in ax11.get_yticklabels():
    ytick_label.set_fontsize(xytick_labels_fontsize)
for xtick_label in ax11.get_xticklabels():
    xtick_label.set_fontsize(xytick_labels_fontsize)

# tuft
weights_images = ax02.imshow(resize(tuft_weights_example_filter, (combined_filter.shape[0], 200)),
                             cmap='jet', vmin=weights_ylims[0],vmax=weights_ylims[1], aspect='auto')
ax02.set_xticks([])
ax02.set_ylabel('Synaptic index', fontsize=xylabels_fontsize)
for ytick_label in ax02.get_yticklabels():
    ytick_label.set_fontsize(xytick_labels_fontsize)

ax12.plot(time_axis, np.fliplr(tuft_weights_example_filter_ex).T , c='r', alpha=all_traces_alpha)
ax12.plot(time_axis, np.mean(np.fliplr(tuft_weights_example_filter_ex).T, axis=1) , c='r', lw=mean_linewidth)
ax12.plot(time_axis, np.fliplr(tuft_weights_example_filter_inh).T, c='b', alpha=all_traces_alpha)
ax12.plot(time_axis, np.mean(np.fliplr(tuft_weights_example_filter_inh).T, axis=1) , c='b', lw=mean_linewidth)

ax12.set_xlim(time_axis.min(),time_axis.max())
ax12.set_ylim(weights_ylims[0],weights_ylims[1])
ax12.set_xlabel('Time before $t_0$ (ms)', fontsize=xylabels_fontsize)
ax12.set_yticks([])

time_ticks_to_show = np.unique((np.array(time_axis) / time_granularity).astype(int) * time_granularity)
ax12.set_xticks(time_ticks_to_show)

ax12.spines['top'].set_visible(False)
ax12.spines['right'].set_visible(False)
ax12.spines['left'].set_visible(False)

for ytick_label in ax12.get_yticklabels():
    ytick_label.set_fontsize(xytick_labels_fontsize)
for xtick_label in ax12.get_xticklabels():
    xtick_label.set_fontsize(xytick_labels_fontsize)

if save_figures:
    figure_name = '%s__first_layer_weights_filter_ind_%d' %(model_dir.split('/')[-2], selected_filter_ind)
    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
        else:
            subfolder = '%s/' %(file_ending.split('.')[-1])
            fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

#%% full combined figure (version 2)

# content params
possible_presentable_candidates = np.nonzero(np.logical_and(num_spikes_per_simulation >= 4, num_spikes_per_simulation <= 15))[0]
selected_trace  = np.random.choice(possible_presentable_candidates)
zoomin_fraction = [0.23 + 0.24 * np.random.rand(), 0.53 + 0.24 * np.random.rand()]

### AMPA_SK 1x128x46
# selected_trace  = 315
# zoomin_fraction = [0.25,0.51]

# selected_trace  = 419
# zoomin_fraction = [0.32,0.545]

# interesting_filters = [4,14,17,27,29,44,56,59,60,65,84]
# selected_filter_inds = np.random.choice(interesting_filters, size=3)
# selected_filter_inds = [78,107,27]


### AMPA_SK 4x64x120
# selected_trace  = 211
# zoomin_fraction = [0.26,0.52]

# interesting_filters = [4,14,17,27,29,44,56,59,60,65,84]
# selected_filter_inds = np.random.choice(interesting_filters, size=3)
# selected_filter_inds = [16,32,59]


### AMPA 1x128x43
# selected_trace  = 1198
# zoomin_fraction = [0.45,0.73]

# selected_trace  = 1123
# zoomin_fraction = [0.37,0.62]

# interesting_filters = [4,14,17,27,29,44,56,59,60,65,84,120]
# selected_filter_inds = np.random.choice(interesting_filters, size=3)
# selected_filter_inds = [34,13,116]
# selected_filter_inds = [34,16,120]


### AMPA 4x64x120
# selected_trace  = 103
# zoomin_fraction = [0.45,0.73]

# interesting_filters = [4,14,17,27,29,44,56,59,60,65,84]
# selected_filter_inds = np.random.choice(interesting_filters, size=3)
# selected_filter_inds = [25,32,57]


### NMDA 1x128x43
#selected_trace  = 140
#zoomin_fraction = [0.48,0.79]

#interesting_filters = [4,14,17,27,29,44,56,59,60,65,84]
#selected_filter_inds = np.random.choice(interesting_filters, size=3)
#selected_filter_inds = [4,65,14]


### NMDA 7x128x153
# selected_trace  = 128
# zoomin_fraction = [0.61,0.85]

selected_trace  = 1313
zoomin_fraction = [0.34,0.65]

# selected_trace  = 564
# zoomin_fraction = [0.31,0.55]

# interesting_filters = [8,17,25,37,52,54,59,66,71,89,91,93,96,114]
# selected_filter_inds = np.random.choice(interesting_filters, size=3)
selected_filter_inds = [91,93,114]


use_filtered = True
filter_size = 3

# figure params
xytick_labels_fontsize = 15
title_fontsize = 26
xylabels_fontsize = 22
legend_fontsize = 15
all_traces_alpha = 0.08
mean_linewidth = 4.0

# figure layout
plt.close('all')
fig = plt.figure(figsize=(26,20))

gs_top_left = gridspec.GridSpec(nrows=1,ncols=1)
gs_top_left.update(left=0.04, right=0.20, bottom=0.45, top=0.95, wspace=0.5, hspace=0.01)
gs_top_middle = gridspec.GridSpec(nrows=7,ncols=1)
gs_top_middle.update(left=0.22, right=0.59, bottom=0.45, top=0.95, wspace=0.5, hspace=0.01)
gs_top_right = gridspec.GridSpec(nrows=5,ncols=2)
gs_top_right.update(left=0.65, right=0.97, bottom=0.47, top=0.95, wspace=0.3, hspace=0.5)

gs_bottom_left = gridspec.GridSpec(nrows=3,ncols=3)
gs_bottom_left.update(left=0.09, right=0.35, bottom=0.05, top=0.39, wspace=0.15, hspace=0.07)
gs_bottom_middle = gridspec.GridSpec(nrows=3,ncols=3)
gs_bottom_middle.update(left=0.40, right=0.66, bottom=0.05, top=0.39, wspace=0.15, hspace=0.07)
gs_bottom_right = gridspec.GridSpec(nrows=3,ncols=3)
gs_bottom_right.update(left=0.71, right=0.97, bottom=0.05, top=0.39, wspace=0.15, hspace=0.07)

# top
ax_morphology      = plt.subplot(gs_top_left[:,:])
ax_nn_illustration = plt.subplot(gs_top_middle[:3,:])
ax_trace_full      = plt.subplot(gs_top_middle[3:5,:])
ax_trace_zoomin    = plt.subplot(gs_top_middle[5:,:])
ax_roc        = plt.subplot(gs_top_right[:2,0])
ax_cross_corr = plt.subplot(gs_top_right[:2,1])
ax_scatter    = plt.subplot(gs_top_right[2:,:])

# bottom
ax_weights_left_basal_heatmap    = plt.subplot(gs_bottom_left[:2,0])
ax_weights_left_oblique_heatmap  = plt.subplot(gs_bottom_left[:2,1])
ax_weights_left_apical_heatmap   = plt.subplot(gs_bottom_left[:2,2])
ax_weights_left_basal_temporal   = plt.subplot(gs_bottom_left[2,0])
ax_weights_left_oblique_temporal = plt.subplot(gs_bottom_left[2,1])
ax_weights_left_apical_temporal  = plt.subplot(gs_bottom_left[2,2])

ax_weights_middle_basal_heatmap    = plt.subplot(gs_bottom_middle[:2,0])
ax_weights_middle_oblique_heatmap  = plt.subplot(gs_bottom_middle[:2,1])
ax_weights_middle_apical_heatmap   = plt.subplot(gs_bottom_middle[:2,2])
ax_weights_middle_basal_temporal   = plt.subplot(gs_bottom_middle[2,0])
ax_weights_middle_oblique_temporal = plt.subplot(gs_bottom_middle[2,1])
ax_weights_middle_apical_temporal  = plt.subplot(gs_bottom_middle[2,2])

ax_weights_right_basal_heatmap    = plt.subplot(gs_bottom_right[:2,0])
ax_weights_right_oblique_heatmap  = plt.subplot(gs_bottom_right[:2,1])
ax_weights_right_apical_heatmap   = plt.subplot(gs_bottom_right[:2,2])
ax_weights_right_basal_temporal   = plt.subplot(gs_bottom_right[2,0])
ax_weights_right_oblique_temporal = plt.subplot(gs_bottom_right[2,1])
ax_weights_right_apical_temporal  = plt.subplot(gs_bottom_right[2,2])
    
################################################
# set morphology
################################################

width_mult_factor = 1.2
apical_color = 'g'
oblique_color = 'orange'
basal_color = 'm'

# basal segments
for key in basal_syn_inds:
    line_width = width_mult_factor * np.array(seg_ind_to_xyz_coords_map[key]['d']).mean()
    ax_morphology.plot(seg_ind_to_xyz_coords_map[key]['x'],seg_ind_to_xyz_coords_map[key]['y'],lw=line_width,color=basal_color)

# oblique segments
for key in oblique_syn_inds:
    line_width = width_mult_factor * np.array(seg_ind_to_xyz_coords_map[key]['d']).mean()
    ax_morphology.plot(seg_ind_to_xyz_coords_map[key]['x'],seg_ind_to_xyz_coords_map[key]['y'],lw=line_width,color=oblique_color)

# tuft segments
for key in tuft_syn_inds:
    line_width = width_mult_factor * np.array(seg_ind_to_xyz_coords_map[key]['d']).mean()
    ax_morphology.plot(seg_ind_to_xyz_coords_map[key]['x'],seg_ind_to_xyz_coords_map[key]['y'],lw=line_width,color=apical_color)

# add black soma
ax_morphology.scatter(x=46.0,y=15.8,s=180,c='k', zorder=100)
ax_morphology.set_xlim(-180,235)
ax_morphology.set_ylim(-210,1200)
ax_morphology.set_axis_off()


################################################
# set illustration
################################################

ax_nn_illustration.set_axis_off()
ax_nn_illustration.imshow(imageio.imread(NN_illustration_filename))

################################################
# set traces
################################################

spike_trace_GT   = y1_test_for_TCN[selected_trace,:,0]
spike_trace_pred = y1_test_for_TCN_hat[selected_trace,:,0] > desired_threshold

output_spike_times_in_ms_GT   = np.nonzero(spike_trace_GT)[0]
output_spike_times_in_ms_pred = np.nonzero(spike_trace_pred)[0]

soma_voltage_trace_GT   = y2_test_for_TCN[selected_trace,:,0] + y_train_soma_bias
soma_voltage_trace_pred = y2_test_for_TCN_hat[selected_trace,:,0] + y_train_soma_bias

soma_voltage_trace_GT[output_spike_times_in_ms_GT] = 37
soma_voltage_trace_pred[output_spike_times_in_ms_pred] = 37


ax_trace_full.set_axis_off()
ax_trace_zoomin.set_axis_off()

ax_trace_full.plot(time_in_sec,soma_voltage_trace_GT,c='c')
ax_trace_full.plot(time_in_sec,soma_voltage_trace_pred,c='m',linestyle=':')
ax_trace_full.set_xlim(0.05,sim_duration_sec)
ax_trace_full.set_ylabel('$V_m$ (mV)', fontsize=xylabels_fontsize)

for tick_label in (ax_trace_full.get_xticklabels() + ax_trace_full.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

zoomout_scalebar_xloc = 0.95 * sim_duration_sec

zoomin_xlims = [zoomin_fraction[0] * sim_duration_sec, zoomin_fraction[1] * sim_duration_sec]
zoomin_dur_sec = zoomin_xlims[1] - zoomin_xlims[0]
zoomin_time_in_sec = np.logical_and(time_in_sec >= zoomin_xlims[0], time_in_sec <= zoomin_xlims[1])
zoomin_ylims = [soma_voltage_trace_GT[zoomin_time_in_sec].min() -2.5, -52]
zoomin_scalebar_xloc = zoomin_xlims[1] - 0.05 * zoomin_dur_sec

ax_trace_zoomin.plot(time_in_sec,soma_voltage_trace_GT,c='c')
ax_trace_zoomin.plot(time_in_sec,soma_voltage_trace_pred,c='m',linestyle=':')
ax_trace_zoomin.set_xlim(zoomin_xlims[0],zoomin_xlims[1])
ax_trace_zoomin.set_ylim(zoomin_ylims[0],zoomin_ylims[1])
ax_trace_zoomin.set_ylabel('$V_m$ (mV)', fontsize=xylabels_fontsize)
ax_trace_zoomin.set_xlabel('time (sec)', fontsize=xylabels_fontsize)

for tick_label in (ax_trace_zoomin.get_xticklabels() + ax_trace_zoomin.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)


# add scale bar to top plot
scalebar_loc = np.array([zoomout_scalebar_xloc,-25])
scalebar_size_x = 0.6
scalebar_str_x = '600 ms'
scalebar_size_y = 40
scalebar_str_y = '40 mV'

x = [scalebar_loc[0], scalebar_loc[0] - scalebar_size_x]
y = [scalebar_loc[1], scalebar_loc[1]]
ax_trace_full.plot(x,y,lw=2,c='k')
ax_trace_full.text(scalebar_loc[0] - 0.05 * scalebar_size_x, scalebar_loc[1] - 0.15 * scalebar_size_y,
                   scalebar_str_x, color='k', fontsize=15, ha='right', va='top', rotation='horizontal')

x = [scalebar_loc[0], scalebar_loc[0]]
y = [scalebar_loc[1], scalebar_loc[1] + scalebar_size_y]
ax_trace_full.plot(x,y,lw=2,c='k')
ax_trace_full.text(scalebar_loc[0] + 0.1 * scalebar_size_x, scalebar_loc[1] + 0.6 * scalebar_size_y,
                   scalebar_str_y, color='k', fontsize=15, ha='left', va='top', rotation='horizontal')


# add dashed rectangle
rect_w = zoomin_xlims[1] - zoomin_xlims[0]
rect_h = zoomin_ylims[1] - zoomin_ylims[0]
rect_bl_x = zoomin_xlims[0]
rect_bl_y = zoomin_ylims[0]
dashed_rectangle = mpatches.Rectangle((rect_bl_x,rect_bl_y),rect_w,rect_h,linewidth=2,edgecolor='k',linestyle='--',facecolor='none')
ax_trace_full.add_patch(dashed_rectangle)

# add scalebar to bottom plot
scalebar_loc = np.array([zoomin_scalebar_xloc,-60])
scalebar_size_x = 0.06
scalebar_str_x = '60 ms'
scalebar_size_y = 5
scalebar_str_y = '5 mV'

x = [scalebar_loc[0], scalebar_loc[0] - scalebar_size_x]
y = [scalebar_loc[1], scalebar_loc[1]]
ax_trace_zoomin.plot(x,y,lw=2,c='k')
ax_trace_zoomin.text(scalebar_loc[0] - 0.15 * scalebar_size_x, scalebar_loc[1] - 0.15 * scalebar_size_y,
                     scalebar_str_x, color='k', fontsize=15, ha='right', va='top', rotation='horizontal')

x = [scalebar_loc[0], scalebar_loc[0]]
y = [scalebar_loc[1], scalebar_loc[1] + scalebar_size_y]
ax_trace_zoomin.plot(x,y,lw=2,c='k')
ax_trace_zoomin.text(scalebar_loc[0] + 0.1 * scalebar_size_x, scalebar_loc[1] + 0.6 * scalebar_size_y,
                     scalebar_str_y, color='k', fontsize=15, ha='left', va='top', rotation='horizontal')

################################################
# set evaluation plots
################################################

# ROC curve
ax_roc.plot(fpr, tpr, c='k')
ax_roc.set_xlabel('False alarm rate', fontsize=xylabels_fontsize)
ax_roc.set_ylabel('Hit rate', fontsize=xylabels_fontsize)
ax_roc.set_ylim(0,1.05)
ax_roc.set_xlim(-0.03,1)
ax_roc.spines['top'].set_visible(False)
ax_roc.spines['right'].set_visible(False)
for tick_label in (ax_roc.get_xticklabels() + ax_roc.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)
ax_roc.set_xticks([0.0,0.5,1.0])
ax_roc.set_yticks([0.0,0.5,1.0])

# ROC inset plot
left, bottom, width, height = [0.70, 0.80, 0.075, 0.12]
ax_roc_inset = fig.add_axes([left, bottom, width, height])
ax_roc_inset.plot(fpr, tpr, c='k')
ax_roc_inset.set_ylim(0,1.05)
ax_roc_inset.set_xlim(-0.001,0.045)
ax_roc_inset.spines['top'].set_visible(False)
ax_roc_inset.spines['right'].set_visible(False)
ax_roc_inset.scatter(actual_false_positive_rate, tpr[desired_fp_ind + 1], c='r', s=100)
for tick_label in (ax_roc_inset.get_xticklabels() + ax_roc_inset.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize - 6)

# cross correlation curve
max_firing_rate = 10 * int(max(1000 * recall_curve) / 10)
midpoint_firing_rate = 5 * int(max_firing_rate / 10)
ax_cross_corr.set_yticks([0, midpoint_firing_rate,max_firing_rate])
ax_cross_corr.plot(time_axis_cc, 1000 * recall_curve, c='k')
ax_cross_corr.set_ylim(0, 1.05 * 1000 * recall_curve.max())
ax_cross_corr.set_xlabel('$\Delta t$ (ms)', fontsize=xylabels_fontsize)
ax_cross_corr.set_ylabel('spike rate (Hz)', fontsize=xylabels_fontsize)
ax_cross_corr.set_xticks([-50,0,50])
ax_cross_corr.spines['top'].set_visible(False)
ax_cross_corr.spines['right'].set_visible(False)
for tick_label in (ax_cross_corr.get_xticklabels() + ax_cross_corr.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

# voltage scatter plot
soma_voltage_lims = np.round([np.percentile(selected_GT,0.2),np.percentile(selected_GT,99.8)]).astype(int)
soma_voltage_lims = np.round([np.percentile(selected_GT,0.2),-56]).astype(int)
voltage_granularity = 6
voltage_setpoint = -57
voltage_axis = np.arange(soma_voltage_lims[0],soma_voltage_lims[1])
voltage_ticks_to_show = np.unique(((voltage_axis - voltage_setpoint) / voltage_granularity).astype(int) * voltage_granularity + voltage_setpoint)
voltage_ticks_to_show = voltage_ticks_to_show[np.logical_and(voltage_ticks_to_show >= soma_voltage_lims[0],
                                                             voltage_ticks_to_show <= soma_voltage_lims[1])]
ax_scatter.set_xticks(voltage_ticks_to_show)
ax_scatter.set_yticks(voltage_ticks_to_show)

ax_scatter.scatter(selected_GT,selected_pred, s=3.0, alpha=0.7)
ax_scatter.set_xlabel('L5PC (%s) (mV)' %(model_string), fontsize=xylabels_fontsize)
ax_scatter.set_ylabel('ANN (mV)', fontsize=xylabels_fontsize)
ax_scatter.set_xlim(soma_voltage_lims[0],soma_voltage_lims[1])
ax_scatter.set_ylim(soma_voltage_lims[0],soma_voltage_lims[1])
ax_scatter.plot([-90,-50],[-90,-50], ls='-', c='k')
ax_scatter.spines['top'].set_visible(False)
ax_scatter.spines['right'].set_visible(False)

for tick_label in (ax_scatter.get_xticklabels() + ax_scatter.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)


################################################
# set first layer weights plots
################################################

def draw_weights(first_layer_weights, selected_filter_ind, set_ylabel, ax00,ax10, ax01,ax11, ax02,ax12):

    time_span, _, num_filters = first_layer_weights.shape
    
    weight_granularity = 0.06
    time_granularity = 20
    max_time_to_show = 40
    
    if use_filtered:
        first_layer_weights_filtered = signal.convolve(first_layer_weights, (1.0 / filter_size) * np.ones((filter_size,1,1)), mode='valid')
        first_layer_weights = first_layer_weights_filtered
    
    if first_layer_weights.shape[0] >= max_time_to_show:
        first_layer_weights = first_layer_weights[:max_time_to_show]
    
    num_segments = 639
    basal_cutoff = 262
    tuft_cutoff  = [366,559]

    # invert if needed
    exc_sum = first_layer_weights[:12,:num_segments,selected_filter_ind].sum()
    inh_sum = first_layer_weights[:12,num_segments:,selected_filter_ind].sum()
    exc_minus_inh = exc_sum - inh_sum
    
    if exc_minus_inh < 0:
        first_layer_weights = -first_layer_weights
    
    upper_limit = max(np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),99.95),np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),0.05))
    xlims = [-5 * int(first_layer_weights.shape[0] / 5), 0]
    
    ex_basal_syn_inds    = np.arange(basal_cutoff)
    ex_oblique_syn_inds  = np.hstack((np.arange(basal_cutoff,tuft_cutoff[0]),np.arange(tuft_cutoff[1],num_segments)))
    ex_tuft_syn_inds     = np.arange(tuft_cutoff[0],tuft_cutoff[1])
    inh_basal_syn_inds   = num_segments + ex_basal_syn_inds
    inh_oblique_syn_inds = num_segments + ex_oblique_syn_inds
    inh_tuft_syn_inds    = num_segments + ex_tuft_syn_inds
    
    basal_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter     = np.concatenate((basal_weights_example_filter_ex,basal_weights_example_filter_inh),axis=0)
    oblique_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter     = np.concatenate((oblique_weights_example_filter_ex, oblique_weights_example_filter_inh),axis=0)
    tuft_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter     = np.concatenate((tuft_weights_example_filter_ex,tuft_weights_example_filter_inh),axis=0)
    
    time_axis = -np.arange(first_layer_weights.shape[0])
    
    ##%% create nice figure
    figure_xlims = xlims
    figure_xlims[0] = max(-40, figure_xlims[0])
    
    ex_basal_syn_inds    = np.arange(basal_cutoff)
    ex_oblique_syn_inds  = np.hstack((np.arange(basal_cutoff,tuft_cutoff[0]),np.arange(tuft_cutoff[1],num_segments)))
    ex_tuft_syn_inds     = np.arange(tuft_cutoff[0],tuft_cutoff[1])
    inh_basal_syn_inds   = num_segments + ex_basal_syn_inds
    inh_oblique_syn_inds = num_segments + ex_oblique_syn_inds
    inh_tuft_syn_inds    = num_segments + ex_tuft_syn_inds
    
    basal_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter     = np.concatenate((basal_weights_example_filter_ex,basal_weights_example_filter_inh),axis=0)
    oblique_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter     = np.concatenate((oblique_weights_example_filter_ex, oblique_weights_example_filter_inh),axis=0)
    tuft_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter     = np.concatenate((tuft_weights_example_filter_ex,tuft_weights_example_filter_inh),axis=0)
    
    combined_filter = np.concatenate((basal_weights_example_filter_ex,oblique_weights_example_filter_ex,tuft_weights_example_filter_ex,
                                      basal_weights_example_filter_inh,oblique_weights_example_filter_inh,tuft_weights_example_filter_inh),axis=0)
    
    ##%% draw 2 x 3 (basal,oblique,tuft) matrix
    ex_basal_syn_inds    = np.arange(basal_cutoff)
    ex_oblique_syn_inds  = np.hstack((np.arange(basal_cutoff,tuft_cutoff[0]),np.arange(tuft_cutoff[1],num_segments)))
    ex_tuft_syn_inds     = np.arange(tuft_cutoff[0],tuft_cutoff[1])
    inh_basal_syn_inds   = num_segments + ex_basal_syn_inds
    inh_oblique_syn_inds = num_segments + ex_oblique_syn_inds
    inh_tuft_syn_inds    = num_segments + ex_tuft_syn_inds
    
    basal_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter     = np.concatenate((basal_weights_example_filter_ex,basal_weights_example_filter_inh),axis=0)
    oblique_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter     = np.concatenate((oblique_weights_example_filter_ex, oblique_weights_example_filter_inh),axis=0)
    tuft_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter     = np.concatenate((tuft_weights_example_filter_ex,tuft_weights_example_filter_inh),axis=0)
    
    time_axis = -np.arange(first_layer_weights.shape[0])
    
    upper_limit = max(np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),99.8),np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),0.2))
    weights_ylims = np.array([-1.08,1.08]) * upper_limit
    
    weight_ticks_lims = (np.array(weights_ylims) / weight_granularity).astype(int) * weight_granularity
    
    ax00.axis('off')
    ax01.axis('off')
    ax02.axis('off')
    
    # basal
    weights_images = ax00.imshow(resize(basal_weights_example_filter, (combined_filter.shape[0], 200)),
                                 cmap='jet', vmin=weights_ylims[0],vmax=weights_ylims[1], aspect='auto')
    ax00.set_xticks([])
    ax00.set_ylabel('Synaptic index', fontsize=xylabels_fontsize)
    for ytick_label in ax00.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    
    ax_colorbar = inset_axes(ax00, width="67%", height="6%", loc=2)
    cbar = plt.colorbar(weights_images, cax=ax_colorbar, orientation="horizontal", ticks=[weight_ticks_lims[0], 0, weight_ticks_lims[1]])
    ax_colorbar.xaxis.set_ticks_position("bottom")
    cbar.ax.tick_params(labelsize=xytick_labels_fontsize - 2)
    ax00.text(10, 190, 'Weight (A.U)', color='k', fontsize=xytick_labels_fontsize + 1, ha='left', va='top', rotation='horizontal')
    
    ax10.plot(time_axis, np.fliplr(basal_weights_example_filter_ex).T , c='r', alpha=all_traces_alpha)
    ax10.plot(time_axis, np.mean(np.fliplr(basal_weights_example_filter_ex).T, axis=1) , c='r', lw=mean_linewidth)
    ax10.plot(time_axis, np.fliplr(basal_weights_example_filter_inh).T, c='b', alpha=all_traces_alpha)
    ax10.plot(time_axis, np.mean(np.fliplr(basal_weights_example_filter_inh).T, axis=1) , c='b', lw=mean_linewidth)
    
    ax10.set_xlim(time_axis.min(),time_axis.max())
    ax10.set_ylim(weights_ylims[0],weights_ylims[1])
    if set_ylabel:
        ax10.set_ylabel('Weight (A.U)', fontsize=xylabels_fontsize)
    
    time_ticks_to_show = np.unique((np.array(time_axis) / time_granularity).astype(int) * time_granularity)
    ax10.set_xticks(time_ticks_to_show)
    
    weights_axis = np.linspace(weights_ylims[0], weights_ylims[1], 10)
    weight_ticks_to_show = np.unique((np.array(weights_axis) / weight_granularity).astype(int) * weight_granularity)
    ax10.set_yticks(weight_ticks_to_show)
    
    ax10.spines['top'].set_visible(False)
    ax10.spines['right'].set_visible(False)
    
    for ytick_label in ax10.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    for xtick_label in ax10.get_xticklabels():
        xtick_label.set_fontsize(xytick_labels_fontsize)

    # oblique
    weights_images = ax01.imshow(resize(oblique_weights_example_filter, (combined_filter.shape[0], 200)),
                                 cmap='jet', vmin=weights_ylims[0],vmax=weights_ylims[1], aspect='auto')
    ax01.set_xticks([])
    ax01.set_ylabel('Synaptic index', fontsize=xylabels_fontsize)
    for ytick_label in ax01.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    
    ax11.plot(time_axis, np.fliplr(oblique_weights_example_filter_ex).T , c='r', alpha=all_traces_alpha)
    ax11.plot(time_axis, np.mean(np.fliplr(oblique_weights_example_filter_ex).T, axis=1) , c='r', lw=mean_linewidth)
    ax11.plot(time_axis, np.fliplr(oblique_weights_example_filter_inh).T, c='b', alpha=all_traces_alpha)
    ax11.plot(time_axis, np.mean(np.fliplr(oblique_weights_example_filter_inh).T, axis=1) , c='b', lw=mean_linewidth)
    
    ax11.set_xlim(time_axis.min(),time_axis.max())
    ax11.set_xlabel('Time before $t_0$ (ms)', fontsize=xylabels_fontsize)
    ax11.set_ylim(weights_ylims[0],weights_ylims[1])
    
    time_ticks_to_show = np.unique((np.array(time_axis) / time_granularity).astype(int) * time_granularity)
    ax11.set_xticks(time_ticks_to_show)
    
    ax11.spines['top'].set_visible(False)
    ax11.spines['right'].set_visible(False)
    ax11.spines['left'].set_visible(False)
    
    ax11.set_yticks([])
    for ytick_label in ax11.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    for xtick_label in ax11.get_xticklabels():
        xtick_label.set_fontsize(xytick_labels_fontsize)
        
    # tuft
    weights_images = ax02.imshow(resize(tuft_weights_example_filter, (combined_filter.shape[0], 200)),
                                 cmap='jet', vmin=weights_ylims[0],vmax=weights_ylims[1], aspect='auto')
    ax02.set_xticks([])
    ax02.set_ylabel('Synaptic index', fontsize=xylabels_fontsize)
    for ytick_label in ax02.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    
    ax12.plot(time_axis, np.fliplr(tuft_weights_example_filter_ex).T , c='r', alpha=all_traces_alpha)
    ax12.plot(time_axis, np.mean(np.fliplr(tuft_weights_example_filter_ex).T, axis=1) , c='r', lw=mean_linewidth)
    ax12.plot(time_axis, np.fliplr(tuft_weights_example_filter_inh).T, c='b', alpha=all_traces_alpha)
    ax12.plot(time_axis, np.mean(np.fliplr(tuft_weights_example_filter_inh).T, axis=1) , c='b', lw=mean_linewidth)
    
    ax12.set_xlim(time_axis.min(),time_axis.max())
    ax12.set_ylim(weights_ylims[0],weights_ylims[1])
    ax12.set_yticks([])
    
    time_ticks_to_show = np.unique((np.array(time_axis) / time_granularity).astype(int) * time_granularity)
    ax12.set_xticks(time_ticks_to_show)
    
    ax12.spines['top'].set_visible(False)
    ax12.spines['right'].set_visible(False)
    ax12.spines['left'].set_visible(False)
    
    for ytick_label in ax12.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    for xtick_label in ax12.get_xticklabels():
        xtick_label.set_fontsize(xytick_labels_fontsize)


draw_weights(first_layer_weights, selected_filter_inds[0], True,
             ax_weights_left_basal_heatmap,   ax_weights_left_basal_temporal,
             ax_weights_left_oblique_heatmap, ax_weights_left_oblique_temporal,
             ax_weights_left_apical_heatmap,  ax_weights_left_apical_temporal)


draw_weights(first_layer_weights, selected_filter_inds[1], False,
             ax_weights_middle_basal_heatmap,   ax_weights_middle_basal_temporal,
             ax_weights_middle_oblique_heatmap, ax_weights_middle_oblique_temporal,
             ax_weights_middle_apical_heatmap,  ax_weights_middle_apical_temporal)


draw_weights(first_layer_weights, selected_filter_inds[2], False,
             ax_weights_right_basal_heatmap,   ax_weights_right_basal_temporal,
             ax_weights_right_oblique_heatmap, ax_weights_right_oblique_temporal,
             ax_weights_right_apical_heatmap, ax_weights_right_apical_temporal)


# save figure
if save_figures:
    figure_name = '%s__full_combined_figure_v2_%d' %(model_dir.split('/')[-2], np.random.randint(50))
    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
        else:
            subfolder = '%s/' %(file_ending.split('.')[-1])
            fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

            
#%% full combined figure (version 3)

# content params
possible_presentable_candidates = np.nonzero(np.logical_and(num_spikes_per_simulation >= 4, num_spikes_per_simulation <= 15))[0]
selected_trace  = np.random.choice(possible_presentable_candidates)
zoomin_fraction = [0.23 + 0.24 * np.random.rand(), 0.53 + 0.24 * np.random.rand()]

### AMPA_SK 1x128x46
# selected_trace  = 315
# zoomin_fraction = [0.25,0.51]

# selected_trace  = 419
# zoomin_fraction = [0.32,0.545]

# interesting_filters = [4,14,17,27,29,44,56,59,60,65,84]
# selected_filter_inds = np.random.choice(interesting_filters, size=3)
# selected_filter_inds = [78,107,27]


### AMPA_SK 4x64x120
# selected_trace  = 211
# zoomin_fraction = [0.26,0.52]

# interesting_filters = [4,14,17,27,29,44,56,59,60,65,84]
# selected_filter_inds = np.random.choice(interesting_filters, size=3)
# selected_filter_inds = [16,32,59]


### AMPA 1x128x43
# selected_trace  = 1198
# zoomin_fraction = [0.45,0.73]

# selected_trace  = 1123
# zoomin_fraction = [0.37,0.62]

# interesting_filters = [4,14,17,27,29,44,56,59,60,65,84,120]
# selected_filter_inds = np.random.choice(interesting_filters, size=3)
# selected_filter_inds = [34,13,116]
# selected_filter_inds = [34,16,120]


### AMPA 4x64x120
# selected_trace  = 103
# zoomin_fraction = [0.45,0.73]

# interesting_filters = [4,14,17,27,29,44,56,59,60,65,84]
# selected_filter_inds = np.random.choice(interesting_filters, size=3)
# selected_filter_inds = [25,32,57]


### NMDA 1x128x43
#selected_trace  = 140
#zoomin_fraction = [0.48,0.79]

#interesting_filters = [4,14,17,27,29,44,56,59,60,65,84]
#selected_filter_inds = np.random.choice(interesting_filters, size=3)
#selected_filter_inds = [4,65,14]


### NMDA 7x128x153
# selected_trace  = 128
# zoomin_fraction = [0.61,0.85]

selected_trace  = 1313
zoomin_fraction = [0.34,0.65]

# selected_trace  = 564
# zoomin_fraction = [0.31,0.55]

# interesting_filters = [8,17,25,37,52,54,59,66,71,89,91,93,96,114]
# selected_filter_inds = np.random.choice(interesting_filters, size=3)
selected_filter_inds = [91,93,114]


use_filtered = True
filter_size = 3

# figure params
xytick_labels_fontsize = 18
title_fontsize = 26
xylabels_fontsize = 23
legend_fontsize = 15
all_traces_alpha = 0.08
mean_linewidth = 4.0

# figure layout
plt.close('all')
fig = plt.figure(figsize=(35,15))
gs_top_left = gridspec.GridSpec(nrows=1,ncols=1)
gs_top_left.update(left=0.01, right=0.17, bottom=0.05, top=0.95, wspace=0.5, hspace=0.01)
gs_top_middle = gridspec.GridSpec(nrows=7,ncols=1)
gs_top_middle.update(left=0.185, right=0.465, bottom=0.05, top=0.95, wspace=0.5, hspace=0.01)
gs_top_right = gridspec.GridSpec(nrows=3,ncols=1)
gs_top_right.update(left=0.515, right=0.63, bottom=0.09, top=0.95, wspace=0.3, hspace=0.22)

gs_top_right_2 = gridspec.GridSpec(nrows=3,ncols=3)
gs_top_right_2.update(left=0.69, right=0.99, bottom=0.09, top=0.89, wspace=0.15, hspace=0.07)

# morphology, illustratrion, trace and performance evaluation
ax_morphology      = plt.subplot(gs_top_left[:,:])
ax_nn_illustration = plt.subplot(gs_top_middle[:3,:])
ax_trace_full      = plt.subplot(gs_top_middle[3:5,:])
ax_trace_zoomin    = plt.subplot(gs_top_middle[5:,:])
ax_roc        = plt.subplot(gs_top_right[0,0])
ax_cross_corr = plt.subplot(gs_top_right[1,0])
ax_scatter    = plt.subplot(gs_top_right[2,0])

# weights plot
ax_weights_left_basal_heatmap    = plt.subplot(gs_top_right_2[:2,0])
ax_weights_left_oblique_heatmap  = plt.subplot(gs_top_right_2[:2,1])
ax_weights_left_apical_heatmap   = plt.subplot(gs_top_right_2[:2,2])
ax_weights_left_basal_temporal   = plt.subplot(gs_top_right_2[2,0])
ax_weights_left_oblique_temporal = plt.subplot(gs_top_right_2[2,1])
ax_weights_left_apical_temporal  = plt.subplot(gs_top_right_2[2,2])

    
################################################
# set morphology
################################################

width_mult_factor = 1.2
apical_color = 'g'
oblique_color = 'orange'
basal_color = 'm'

# basal segments
for key in basal_syn_inds:
    line_width = width_mult_factor * np.array(seg_ind_to_xyz_coords_map[key]['d']).mean()
    ax_morphology.plot(seg_ind_to_xyz_coords_map[key]['x'],seg_ind_to_xyz_coords_map[key]['y'],lw=line_width,color=basal_color)

# oblique segments
for key in oblique_syn_inds:
    line_width = width_mult_factor * np.array(seg_ind_to_xyz_coords_map[key]['d']).mean()
    ax_morphology.plot(seg_ind_to_xyz_coords_map[key]['x'],seg_ind_to_xyz_coords_map[key]['y'],lw=line_width,color=oblique_color)

# tuft segments
for key in tuft_syn_inds:
    line_width = width_mult_factor * np.array(seg_ind_to_xyz_coords_map[key]['d']).mean()
    ax_morphology.plot(seg_ind_to_xyz_coords_map[key]['x'],seg_ind_to_xyz_coords_map[key]['y'],lw=line_width,color=apical_color)

# add black soma
ax_morphology.scatter(x=46.0,y=15.8,s=180,c='k', zorder=100)
ax_morphology.set_xlim(-180,235)
ax_morphology.set_ylim(-210,1200)
ax_morphology.set_axis_off()


################################################
# set illustration
################################################

ax_nn_illustration.set_axis_off()
ax_nn_illustration.imshow(imageio.imread(NN_illustration_filename))

################################################
# set traces
################################################

spike_trace_GT   = y1_test_for_TCN[selected_trace,:,0]
spike_trace_pred = y1_test_for_TCN_hat[selected_trace,:,0] > desired_threshold

output_spike_times_in_ms_GT   = np.nonzero(spike_trace_GT)[0]
output_spike_times_in_ms_pred = np.nonzero(spike_trace_pred)[0]

soma_voltage_trace_GT   = y2_test_for_TCN[selected_trace,:,0] + y_train_soma_bias
soma_voltage_trace_pred = y2_test_for_TCN_hat[selected_trace,:,0] + y_train_soma_bias

soma_voltage_trace_GT[output_spike_times_in_ms_GT] = 37
soma_voltage_trace_pred[output_spike_times_in_ms_pred] = 37


ax_trace_full.set_axis_off()
ax_trace_zoomin.set_axis_off()

ax_trace_full.plot(time_in_sec,soma_voltage_trace_GT,c='c')
ax_trace_full.plot(time_in_sec,soma_voltage_trace_pred,c='m',linestyle=':')
ax_trace_full.set_xlim(0.05,sim_duration_sec)
ax_trace_full.set_ylabel('$V_m$ (mV)', fontsize=xylabels_fontsize)

for tick_label in (ax_trace_full.get_xticklabels() + ax_trace_full.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

zoomout_scalebar_xloc = 0.95 * sim_duration_sec

zoomin_xlims = [zoomin_fraction[0] * sim_duration_sec, zoomin_fraction[1] * sim_duration_sec]
zoomin_dur_sec = zoomin_xlims[1] - zoomin_xlims[0]
zoomin_time_in_sec = np.logical_and(time_in_sec >= zoomin_xlims[0], time_in_sec <= zoomin_xlims[1])
zoomin_ylims = [soma_voltage_trace_GT[zoomin_time_in_sec].min() -2.5, -52]
zoomin_scalebar_xloc = zoomin_xlims[1] - 0.05 * zoomin_dur_sec

ax_trace_zoomin.plot(time_in_sec,soma_voltage_trace_GT,c='c')
ax_trace_zoomin.plot(time_in_sec,soma_voltage_trace_pred,c='m',linestyle=':')
ax_trace_zoomin.set_xlim(zoomin_xlims[0],zoomin_xlims[1])
ax_trace_zoomin.set_ylim(zoomin_ylims[0],zoomin_ylims[1])
ax_trace_zoomin.set_ylabel('$V_m$ (mV)', fontsize=xylabels_fontsize)
ax_trace_zoomin.set_xlabel('time (sec)', fontsize=xylabels_fontsize)

for tick_label in (ax_trace_zoomin.get_xticklabels() + ax_trace_zoomin.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)


# add scale bar to top plot
scalebar_loc = np.array([zoomout_scalebar_xloc,-25])
scalebar_size_x = 0.6
scalebar_str_x = '600 ms'
scalebar_size_y = 40
scalebar_str_y = '40 mV'

x = [scalebar_loc[0], scalebar_loc[0] - scalebar_size_x]
y = [scalebar_loc[1], scalebar_loc[1]]
ax_trace_full.plot(x,y,lw=2,c='k')
ax_trace_full.text(scalebar_loc[0] - 0.05 * scalebar_size_x, scalebar_loc[1] - 0.15 * scalebar_size_y,
                   scalebar_str_x, color='k', fontsize=15, ha='right', va='top', rotation='horizontal')

x = [scalebar_loc[0], scalebar_loc[0]]
y = [scalebar_loc[1], scalebar_loc[1] + scalebar_size_y]
ax_trace_full.plot(x,y,lw=2,c='k')
ax_trace_full.text(scalebar_loc[0] + 0.1 * scalebar_size_x, scalebar_loc[1] + 0.6 * scalebar_size_y,
                   scalebar_str_y, color='k', fontsize=15, ha='left', va='top', rotation='horizontal')

# add dashed rectangle
rect_w = zoomin_xlims[1] - zoomin_xlims[0]
rect_h = zoomin_ylims[1] - zoomin_ylims[0]
rect_bl_x = zoomin_xlims[0]
rect_bl_y = zoomin_ylims[0]
dashed_rectangle = mpatches.Rectangle((rect_bl_x,rect_bl_y),rect_w,rect_h,linewidth=2,edgecolor='k',linestyle='--',facecolor='none')

ax_trace_full.add_patch(dashed_rectangle)

# add scalebar to bottom plot
scalebar_loc = np.array([zoomin_scalebar_xloc,-60])
scalebar_size_x = 0.06
scalebar_str_x = '60 ms'
scalebar_size_y = 5
scalebar_str_y = '5 mV'

x = [scalebar_loc[0], scalebar_loc[0] - scalebar_size_x]
y = [scalebar_loc[1], scalebar_loc[1]]
ax_trace_zoomin.plot(x,y,lw=2,c='k')
ax_trace_zoomin.text(scalebar_loc[0] - 0.15 * scalebar_size_x, scalebar_loc[1] - 0.15 * scalebar_size_y,
                     scalebar_str_x, color='k', fontsize=15, ha='right', va='top', rotation='horizontal')

x = [scalebar_loc[0], scalebar_loc[0]]
y = [scalebar_loc[1], scalebar_loc[1] + scalebar_size_y]
ax_trace_zoomin.plot(x,y,lw=2,c='k')
ax_trace_zoomin.text(scalebar_loc[0] + 0.1 * scalebar_size_x, scalebar_loc[1] + 0.6 * scalebar_size_y,
                     scalebar_str_y, color='k', fontsize=15, ha='left', va='top', rotation='horizontal')

################################################
# set evaluation plots
################################################

# ROC curve
ax_roc.plot(fpr, tpr, c='k')
ax_roc.set_xlabel('False alarm rate', fontsize=xylabels_fontsize)
ax_roc.set_ylabel('Hit rate', fontsize=xylabels_fontsize)
ax_roc.set_ylim(0,1.05)
ax_roc.set_xlim(-0.03,1)
ax_roc.spines['top'].set_visible(False)
ax_roc.spines['right'].set_visible(False)
for tick_label in (ax_roc.get_xticklabels() + ax_roc.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)
ax_roc.set_xticks([0.0,0.5,1.0])
ax_roc.set_yticks([0.0,0.5,1.0])

# ROC inset plot
left, bottom, width, height = [0.555, 0.75, 0.065, 0.15]
ax_roc_inset = fig.add_axes([left, bottom, width, height])
ax_roc_inset.plot(fpr, tpr, c='k')
ax_roc_inset.set_ylim(0,1.05)
ax_roc_inset.set_xlim(-0.001,0.045)
ax_roc_inset.spines['top'].set_visible(False)
ax_roc_inset.spines['right'].set_visible(False)
ax_roc_inset.scatter(actual_false_positive_rate, tpr[desired_fp_ind + 1], c='r', s=100)
for tick_label in (ax_roc_inset.get_xticklabels() + ax_roc_inset.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize - 6)

# cross correlation curve
max_firing_rate = 10 * int(max(1000 * recall_curve) / 10)
midpoint_firing_rate = 5 * int(max_firing_rate / 10)
ax_cross_corr.set_yticks([0, midpoint_firing_rate,max_firing_rate])
ax_cross_corr.plot(time_axis_cc, 1000 * recall_curve, c='k')
ax_cross_corr.set_ylim(0,1.05 * 1000 * recall_curve.max())
ax_cross_corr.set_xlabel('$\Delta t$ (ms)', fontsize=xylabels_fontsize)
ax_cross_corr.set_ylabel('spike rate (Hz)', fontsize=xylabels_fontsize)
ax_cross_corr.set_xticks([-50,0,50])
ax_cross_corr.spines['top'].set_visible(False)
ax_cross_corr.spines['right'].set_visible(False)
for tick_label in (ax_cross_corr.get_xticklabels() + ax_cross_corr.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)


# voltage scatter plot
soma_voltage_lims = np.round([np.percentile(selected_GT,0.2),np.percentile(selected_GT,99.8)]).astype(int)
soma_voltage_lims = np.round([np.percentile(selected_GT,0.2),-56]).astype(int)
voltage_granularity = 6
voltage_setpoint = -57
voltage_axis = np.arange(soma_voltage_lims[0],soma_voltage_lims[1])
voltage_ticks_to_show = np.unique(((voltage_axis - voltage_setpoint) / voltage_granularity).astype(int) * voltage_granularity + voltage_setpoint)
voltage_ticks_to_show = voltage_ticks_to_show[np.logical_and(voltage_ticks_to_show >= soma_voltage_lims[0],
                                                             voltage_ticks_to_show <= soma_voltage_lims[1])]
ax_scatter.set_xticks(voltage_ticks_to_show)
ax_scatter.set_yticks(voltage_ticks_to_show)

ax_scatter.scatter(selected_GT,selected_pred, s=3.0, alpha=0.7)
ax_scatter.set_xlabel('L5PC (%s) (mV)' %(model_string), fontsize=xylabels_fontsize)
ax_scatter.set_ylabel('ANN (mV)', fontsize=xylabels_fontsize)
ax_scatter.set_xlim(soma_voltage_lims[0],soma_voltage_lims[1])
ax_scatter.set_ylim(soma_voltage_lims[0],soma_voltage_lims[1])
ax_scatter.plot([-90,-50],[-90,-50], ls='-', c='k')
ax_scatter.spines['top'].set_visible(False)
ax_scatter.spines['right'].set_visible(False)
for tick_label in (ax_scatter.get_xticklabels() + ax_scatter.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

################################################
# set first layer weights plots
################################################


def draw_weights(first_layer_weights, selected_filter_ind, set_ylabel, ax00,ax10, ax01,ax11, ax02,ax12):

    time_span, _, num_filters = first_layer_weights.shape
    
    weight_granularity = 0.06
    time_granularity = 20
    max_time_to_show = 40
    
    if use_filtered:
        first_layer_weights_filtered = signal.convolve(first_layer_weights, (1.0 / filter_size) * np.ones((filter_size,1,1)), mode='valid')
        first_layer_weights = first_layer_weights_filtered
    
    if first_layer_weights.shape[0] >= max_time_to_show:
        first_layer_weights = first_layer_weights[:max_time_to_show]
    
    num_segments = 639
    basal_cutoff = 262
    tuft_cutoff  = [366,559]

    # invert if needed
    exc_sum = first_layer_weights[:12,:num_segments,selected_filter_ind].sum()
    inh_sum = first_layer_weights[:12,num_segments:,selected_filter_ind].sum()
    exc_minus_inh = exc_sum - inh_sum
    
    if exc_minus_inh < 0:
        first_layer_weights = -first_layer_weights
    
    upper_limit = max(np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),99.95),np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),0.05))
    xlims = [-5 * int(first_layer_weights.shape[0] / 5),0]
    
    ex_basal_syn_inds    = np.arange(basal_cutoff)
    ex_oblique_syn_inds  = np.hstack((np.arange(basal_cutoff,tuft_cutoff[0]),np.arange(tuft_cutoff[1],num_segments)))
    ex_tuft_syn_inds     = np.arange(tuft_cutoff[0],tuft_cutoff[1])
    inh_basal_syn_inds   = num_segments + ex_basal_syn_inds
    inh_oblique_syn_inds = num_segments + ex_oblique_syn_inds
    inh_tuft_syn_inds    = num_segments + ex_tuft_syn_inds
    
    basal_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter     = np.concatenate((basal_weights_example_filter_ex,basal_weights_example_filter_inh),axis=0)
    oblique_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter     = np.concatenate((oblique_weights_example_filter_ex, oblique_weights_example_filter_inh),axis=0)
    tuft_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter     = np.concatenate((tuft_weights_example_filter_ex,tuft_weights_example_filter_inh),axis=0)
    
    time_axis = -np.arange(first_layer_weights.shape[0])
    
    ##%% create nice figure
    figure_xlims = xlims
    figure_xlims[0] = max(-40, figure_xlims[0])
    
    ex_basal_syn_inds    = np.arange(basal_cutoff)
    ex_oblique_syn_inds  = np.hstack((np.arange(basal_cutoff,tuft_cutoff[0]),np.arange(tuft_cutoff[1],num_segments)))
    ex_tuft_syn_inds     = np.arange(tuft_cutoff[0],tuft_cutoff[1])
    inh_basal_syn_inds   = num_segments + ex_basal_syn_inds
    inh_oblique_syn_inds = num_segments + ex_oblique_syn_inds
    inh_tuft_syn_inds    = num_segments + ex_tuft_syn_inds
    
    basal_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter     = np.concatenate((basal_weights_example_filter_ex,basal_weights_example_filter_inh),axis=0)
    oblique_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter     = np.concatenate((oblique_weights_example_filter_ex, oblique_weights_example_filter_inh),axis=0)
    tuft_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter     = np.concatenate((tuft_weights_example_filter_ex,tuft_weights_example_filter_inh),axis=0)
    
    combined_filter = np.concatenate((basal_weights_example_filter_ex,oblique_weights_example_filter_ex,tuft_weights_example_filter_ex,
                                      basal_weights_example_filter_inh,oblique_weights_example_filter_inh,tuft_weights_example_filter_inh),axis=0)
    
    ##%% draw 2 x 3 (basal,oblique,tuft) matrix
    ex_basal_syn_inds    = np.arange(basal_cutoff)
    ex_oblique_syn_inds  = np.hstack((np.arange(basal_cutoff,tuft_cutoff[0]),np.arange(tuft_cutoff[1],num_segments)))
    ex_tuft_syn_inds     = np.arange(tuft_cutoff[0],tuft_cutoff[1])
    inh_basal_syn_inds   = num_segments + ex_basal_syn_inds
    inh_oblique_syn_inds = num_segments + ex_oblique_syn_inds
    inh_tuft_syn_inds    = num_segments + ex_tuft_syn_inds
    
    basal_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_basal_syn_inds,selected_filter_ind].T)
    basal_weights_example_filter     = np.concatenate((basal_weights_example_filter_ex,basal_weights_example_filter_inh),axis=0)
    oblique_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_oblique_syn_inds,selected_filter_ind].T)
    oblique_weights_example_filter     = np.concatenate((oblique_weights_example_filter_ex, oblique_weights_example_filter_inh),axis=0)
    tuft_weights_example_filter_ex  = np.fliplr(first_layer_weights[:,ex_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter_inh = np.fliplr(first_layer_weights[:,inh_tuft_syn_inds,selected_filter_ind].T)
    tuft_weights_example_filter     = np.concatenate((tuft_weights_example_filter_ex,tuft_weights_example_filter_inh),axis=0)
    
    time_axis = -np.arange(first_layer_weights.shape[0])
    
    upper_limit = max(np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),99.8),np.percentile(abs(first_layer_weights[:,:,selected_filter_ind]),0.2))
    weights_ylims = np.array([-1.08,1.08]) * upper_limit
    
    weight_ticks_lims = (np.array(weights_ylims) / weight_granularity).astype(int) * weight_granularity
    
    ax00.axis('off')
    ax01.axis('off')
    ax02.axis('off')
    
    # basal
    weights_images = ax00.imshow(resize(basal_weights_example_filter, (combined_filter.shape[0], 200)),
                                 cmap='jet', vmin=weights_ylims[0],vmax=weights_ylims[1], aspect='auto')
    ax00.set_xticks([])
    ax00.set_ylabel('Synaptic index', fontsize=xylabels_fontsize)
    for ytick_label in ax00.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    
    ax_colorbar = inset_axes(ax00, width="67%", height="6%", loc=2)
    cbar = plt.colorbar(weights_images, cax=ax_colorbar, orientation="horizontal", ticks=[weight_ticks_lims[0], 0, weight_ticks_lims[1]])
    ax_colorbar.xaxis.set_ticks_position("bottom")
    cbar.ax.tick_params(labelsize=xytick_labels_fontsize - 2)
    ax00.text(10, 190, 'Weight (A.U)', color='k', fontsize=xytick_labels_fontsize + 1, ha='left', va='top', rotation='horizontal')
    
    ax10.plot(time_axis, np.fliplr(basal_weights_example_filter_ex).T , c='r', alpha=all_traces_alpha)
    ax10.plot(time_axis, np.mean(np.fliplr(basal_weights_example_filter_ex).T, axis=1) , c='r', lw=mean_linewidth)
    ax10.plot(time_axis, np.fliplr(basal_weights_example_filter_inh).T, c='b', alpha=all_traces_alpha)
    ax10.plot(time_axis, np.mean(np.fliplr(basal_weights_example_filter_inh).T, axis=1) , c='b', lw=mean_linewidth)
    
    ax10.set_xlim(time_axis.min(),time_axis.max())
    ax10.set_ylim(weights_ylims[0],weights_ylims[1])
    if set_ylabel:
        ax10.set_ylabel('Weight (A.U)', fontsize=xylabels_fontsize)
    
    time_ticks_to_show = np.unique((np.array(time_axis) / time_granularity).astype(int) * time_granularity)
    ax10.set_xticks(time_ticks_to_show)
    
    weights_axis = np.linspace(weights_ylims[0],weights_ylims[1],10)
    weight_ticks_to_show = np.unique((np.array(weights_axis) / weight_granularity).astype(int) * weight_granularity)
    ax10.set_yticks(weight_ticks_to_show)

    ax10.spines['top'].set_visible(False)
    ax10.spines['right'].set_visible(False)
    
    for ytick_label in ax10.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    for xtick_label in ax10.get_xticklabels():
        xtick_label.set_fontsize(xytick_labels_fontsize)

    # oblique
    weights_images = ax01.imshow(resize(oblique_weights_example_filter, (combined_filter.shape[0], 200)),
                                 cmap='jet', vmin=weights_ylims[0],vmax=weights_ylims[1], aspect='auto')
    ax01.set_xticks([])
    ax01.set_ylabel('Synaptic index', fontsize=xylabels_fontsize)
    for ytick_label in ax01.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    
    ax11.plot(time_axis, np.fliplr(oblique_weights_example_filter_ex).T , c='r', alpha=all_traces_alpha)
    ax11.plot(time_axis, np.mean(np.fliplr(oblique_weights_example_filter_ex).T, axis=1) , c='r', lw=mean_linewidth)
    ax11.plot(time_axis, np.fliplr(oblique_weights_example_filter_inh).T, c='b', alpha=all_traces_alpha)
    ax11.plot(time_axis, np.mean(np.fliplr(oblique_weights_example_filter_inh).T, axis=1) , c='b', lw=mean_linewidth)
    
    ax11.set_xlim(time_axis.min(),time_axis.max())
    ax11.set_xlabel('Time before $t_0$ (ms)', fontsize=xylabels_fontsize)
    ax11.set_ylim(weights_ylims[0],weights_ylims[1])
    
    time_ticks_to_show = np.unique((np.array(time_axis) / time_granularity).astype(int) * time_granularity)
    ax11.set_xticks(time_ticks_to_show)
    
    ax11.spines['top'].set_visible(False)
    ax11.spines['right'].set_visible(False)
    ax11.spines['left'].set_visible(False)

    ax11.set_yticks([])
    for ytick_label in ax11.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    for xtick_label in ax11.get_xticklabels():
        xtick_label.set_fontsize(xytick_labels_fontsize)
        
    # tuft
    weights_images = ax02.imshow(resize(tuft_weights_example_filter, (combined_filter.shape[0], 200)),
                                 cmap='jet', vmin=weights_ylims[0],vmax=weights_ylims[1], aspect='auto')
    ax02.set_xticks([])
    ax02.set_ylabel('Synaptic index', fontsize=xylabels_fontsize)
    for ytick_label in ax02.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    
    ax12.plot(time_axis, np.fliplr(tuft_weights_example_filter_ex).T , c='r', alpha=all_traces_alpha)
    ax12.plot(time_axis, np.mean(np.fliplr(tuft_weights_example_filter_ex).T, axis=1) , c='r', lw=mean_linewidth)
    ax12.plot(time_axis, np.fliplr(tuft_weights_example_filter_inh).T, c='b', alpha=all_traces_alpha)
    ax12.plot(time_axis, np.mean(np.fliplr(tuft_weights_example_filter_inh).T, axis=1) , c='b', lw=mean_linewidth)
    
    ax12.set_xlim(time_axis.min(),time_axis.max())
    ax12.set_ylim(weights_ylims[0],weights_ylims[1])
    ax12.set_yticks([])
    
    time_ticks_to_show = np.unique((np.array(time_axis) / time_granularity).astype(int) * time_granularity)
    ax12.set_xticks(time_ticks_to_show)
    
    ax12.spines['top'].set_visible(False)
    ax12.spines['right'].set_visible(False)
    ax12.spines['left'].set_visible(False)
    
    for ytick_label in ax12.get_yticklabels():
        ytick_label.set_fontsize(xytick_labels_fontsize)
    for xtick_label in ax12.get_xticklabels():
        xtick_label.set_fontsize(xytick_labels_fontsize)


draw_weights(first_layer_weights, selected_filter_ind, True,
             ax_weights_left_basal_heatmap,   ax_weights_left_basal_temporal,
             ax_weights_left_oblique_heatmap, ax_weights_left_oblique_temporal,
             ax_weights_left_apical_heatmap,  ax_weights_left_apical_temporal)

# save figure
if save_figures:
    figure_name = '%s__full_combined_figure_v3_%d' %(model_dir.split('/')[-2], np.random.randint(50))
    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
        else:
            subfolder = '%s/' %(file_ending.split('.')[-1])
            fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

