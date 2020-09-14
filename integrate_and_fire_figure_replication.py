import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import signal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_curve, auc
from keras.models import Model
from keras.optimizers import Nadam
from keras.layers import Input, Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras import initializers

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

save_figures = False
# save_figures = True
all_file_endings_to_use = ['.png', '.pdf', '.svg']

# NOTE: during this project I've changed my coding style
# and was too lazy to edit the old code to match the new style
# so please ignore any style related wierdness
# thanks for not being petty about unimportant shit

output_figures_dir = '/Reseach/Single_Neuron_InOut/figures/IF/'

#%% some helper functions


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


def generate_input_spike_trains_for_simulation(sim_duration_ms=6000, num_exc_segments=80, num_inh_segments=20,
                                               num_exc_spikes_per_100ms_range=[0,100], num_exc_inh_spike_diff_per_100ms_range=[-100,100]):

    # randomly sample inst rate (with some uniform noise) smoothing sigma
    keep_inst_rate_const_for_ms = inst_rate_sampling_time_interval_options_ms[np.random.randint(len(inst_rate_sampling_time_interval_options_ms))]
    keep_inst_rate_const_for_ms += int(2 * inst_rate_sampling_time_interval_jitter_range * np.random.rand() - inst_rate_sampling_time_interval_jitter_range)
    
    # randomly sample smoothing sigma (with some uniform noise)
    temporal_inst_rate_smoothing_sigma = temporal_inst_rate_smoothing_sigma_options_ms[np.random.randint(len(temporal_inst_rate_smoothing_sigma_options_ms))]
    temporal_inst_rate_smoothing_sigma += int(2 * temporal_inst_rate_smoothing_sigma_jitter_range * np.random.rand() - temporal_inst_rate_smoothing_sigma_jitter_range)
    
    num_inst_rate_samples = int(np.ceil(float(sim_duration_ms) / keep_inst_rate_const_for_ms))
    
    # create the coarse inst rates with units of "total spikes per tree per 100 ms"
    num_ex_spikes_per_100ms   = np.random.uniform(low=num_exc_spikes_per_100ms_range[0], high=num_exc_spikes_per_100ms_range[1], size=(1, num_inst_rate_samples))
    num_inh_spikes_low_range  = np.maximum(0, num_ex_spikes_per_100ms + num_exc_inh_spike_diff_per_100ms_range[0])
    num_inh_spikes_high_range = num_ex_spikes_per_100ms + num_exc_inh_spike_diff_per_100ms_range[1]
    num_inh_spikes_per_100ms  = np.random.uniform(low=num_inh_spikes_low_range, high=num_inh_spikes_high_range, size=(1, num_inst_rate_samples))
    num_inh_spikes_per_100ms[num_inh_spikes_per_100ms < 0] = 0.0001

    # convert to units of "per_1um_per_1ms"
    ex_bas_spike_rate_per_1um_per_1ms   = num_ex_spikes_per_100ms   / (num_exc_segments  * 100.0)
    inh_bas_spike_rate_per_1um_per_1ms  = num_inh_spikes_per_100ms  / (num_inh_segments  * 100.0)

    # kron by space (uniform distribution across branches per tree)
    ex_spike_rate_per_seg_per_1ms   = np.kron(ex_bas_spike_rate_per_1um_per_1ms  , np.ones((num_exc_segments,1)))
    inh_spike_rate_per_seg_per_1ms  = np.kron(inh_bas_spike_rate_per_1um_per_1ms , np.ones((num_inh_segments,1)))

    # add some spatial multiplicative randomness (that will be added to the sampling noise)
    ex_spike_rate_per_seg_per_1ms  = np.random.uniform(low=0.5, high=1.5, size=ex_spike_rate_per_seg_per_1ms.shape ) * ex_spike_rate_per_seg_per_1ms
    inh_spike_rate_per_seg_per_1ms = np.random.uniform(low=0.5, high=1.5, size=inh_spike_rate_per_seg_per_1ms.shape) * inh_spike_rate_per_seg_per_1ms

    # kron by time (crop if there are leftovers in the end) to fill up the time to 1ms time bins
    ex_spike_rate_per_seg_per_1ms  = np.kron(ex_spike_rate_per_seg_per_1ms , np.ones((1, keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]
    inh_spike_rate_per_seg_per_1ms = np.kron(inh_spike_rate_per_seg_per_1ms, np.ones((1, keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]
    
    # filter the inst rates according to smoothing sigma
    smoothing_window = signal.gaussian(1.0 + 7 * temporal_inst_rate_smoothing_sigma, std=temporal_inst_rate_smoothing_sigma)[np.newaxis,:]
    smoothing_window /= smoothing_window.sum()
    seg_inst_rate_ex_smoothed  = signal.convolve(ex_spike_rate_per_seg_per_1ms,  smoothing_window, mode='same')
    seg_inst_rate_inh_smoothed = signal.convolve(inh_spike_rate_per_seg_per_1ms, smoothing_window, mode='same')
    
    # sample the instantanous spike prob and then sample the actual spikes
    ex_inst_spike_prob = np.random.exponential(scale=seg_inst_rate_ex_smoothed)
    ex_spikes_bin      = np.random.rand(ex_inst_spike_prob.shape[0], ex_inst_spike_prob.shape[1]) < ex_inst_spike_prob

    inh_inst_spike_prob = np.random.exponential(scale=seg_inst_rate_inh_smoothed)
    inh_spikes_bin      = np.random.rand(inh_inst_spike_prob.shape[0], inh_inst_spike_prob.shape[1]) < inh_inst_spike_prob

    all_spikes_bin = np.vstack((ex_spikes_bin, inh_spikes_bin))

    return all_spikes_bin


def simulate_integrate_and_fire_cell(presynaptic_input_spikes, synaptic_weights, membrane_time_const=20, v_reset=-95, v_threshold=-50, current_to_voltage_mult_factor=5):
    temporal_filter_length = int(7 * membrane_time_const) + 1
    syn_filter = signal.exponential(M=temporal_filter_length,center=0,tau=membrane_time_const,sym=False)[np.newaxis,:]
    syn_local_currents = signal.convolve(presynaptic_input_spikes, syn_filter, mode='full')[:,:presynaptic_input_spikes.shape[1]]
    soma_current       = signal.convolve(syn_local_currents, np.flipud(synaptic_weights), mode='valid')
    
    # make simulations
    soma_voltage = v_reset + current_to_voltage_mult_factor * soma_current.ravel()
    output_spike_times_in_ms = []
    for t in range(len(soma_voltage)):
        if (soma_voltage[t] > v_threshold) and ((t + 1) < len(soma_voltage)):
            t_start = t + 1
            t_end = min(len(soma_voltage), t_start + temporal_filter_length)
            soma_voltage[t_start:t_end] -= (soma_voltage[t + 1] - v_reset) * syn_filter.ravel()[:(t_end - t_start)]
            output_spike_times_in_ms.append(t)

    return soma_voltage, output_spike_times_in_ms


def generate_multiple_simulations(input_generation_func, cell_simulation_func, num_simulations):
    
    num_synapses, sim_duration_ms = input_generation_func().shape
    X = np.zeros(((num_synapses, sim_duration_ms, num_simulations)), dtype=np.bool)
    y_spikes = np.zeros(((sim_duration_ms, num_simulations)), dtype=np.bool)
    y_soma   = np.zeros(((sim_duration_ms, num_simulations)), dtype=np.float32)
    for sim_ind in range(num_simulations):
        presynaptic_input_spikes = input_generation_func()
        soma_voltage, output_spike_times_in_ms = cell_simulation_func(presynaptic_input_spikes)
        
        X[:,:,sim_ind] = presynaptic_input_spikes
        y_spikes[output_spike_times_in_ms,sim_ind] = 1.0
        y_soma[:,sim_ind] = soma_voltage

    return X, y_spikes, y_soma


#%% collect a large dataset of {input,output} "recordings" from an Integrate and Fire (I&F) simulation

random_seed = 1234
np.random.seed(random_seed)

# simulation params
num_ex_synapses  = 80
num_inh_synapses = 20
num_synapses     = num_ex_synapses + num_inh_synapses

v_reset     = -75
v_threshold = -55
current_to_voltage_mult_factor = 2
membrane_time_const = 20

# create synaptic weights vector "w"
synaptic_weights = np.ones((num_synapses, 1))
exc_inds  = range(num_ex_synapses)
inh_inds = list(set(range(num_synapses)) - set(exc_inds))
synaptic_weights[exc_inds] *=  1.0
synaptic_weights[inh_inds] *= -1.0

sim_duration_ms  = 6000
sim_duration_sec = sim_duration_ms / 1000.0

inst_rate_sampling_time_interval_options_ms   = [25,30,35,40,50,60,70,80,90,100]
temporal_inst_rate_smoothing_sigma_options_ms = [40,60,80,100]

inst_rate_sampling_time_interval_jitter_range   = 20
temporal_inst_rate_smoothing_sigma_jitter_range = 20

num_exc_spikes_per_100ms_range = [0, 50]
num_exc_inh_spike_diff_per_100ms_range = [-50, -15]

num_simulations_train = 12000
num_simulations_test  = 1000

dataset_generation_start_time = time.time()

input_generation_func = lambda  : generate_input_spike_trains_for_simulation(sim_duration_ms=sim_duration_ms,
                                                                             num_exc_segments=num_ex_synapses, num_inh_segments=num_inh_synapses,
                                                                             num_exc_spikes_per_100ms_range=num_exc_spikes_per_100ms_range,
                                                                             num_exc_inh_spike_diff_per_100ms_range=num_exc_inh_spike_diff_per_100ms_range)
cell_simulation_func  = lambda x: simulate_integrate_and_fire_cell(x, synaptic_weights, membrane_time_const=membrane_time_const,
                                                                   v_reset=v_reset, v_threshold=v_threshold, current_to_voltage_mult_factor=current_to_voltage_mult_factor)

X_train, y_spike_train, y_soma_train = generate_multiple_simulations(input_generation_func, cell_simulation_func, num_simulations_train)
X_test , y_spike_test , y_soma_test  = generate_multiple_simulations(input_generation_func, cell_simulation_func, num_simulations_test )

y_soma_train[y_soma_train > v_threshold] = v_threshold + 0.1
y_soma_test[y_soma_test   > v_threshold] = v_threshold + 0.1

dataset_generation_duration_sec = time.time() - dataset_generation_start_time
print('dataset generation took %.2f minutes' %(dataset_generation_duration_sec / 60))
print('each simulation took %.3f seconds to generate' %(dataset_generation_duration_sec / (num_simulations_train + num_simulations_test)))

#%% plot some validatory plots

plt.close('all')
# verify that the generated X's and y's are fine

# input raster plots

num_ms_raster = 2000

plt.figure(figsize=(20,10))
plt.subplots_adjust(left=0.03,right=0.97,top=0.97,bottom=0.03,hspace=0.2)
plt.subplot(6,1,1); plt.spy(X_train[:,:num_ms_raster,np.random.randint(num_simulations_train)], markersize=3); plt.title('train sample'); plt.axis('off')
plt.subplot(6,1,2); plt.spy(X_train[:,:num_ms_raster,np.random.randint(num_simulations_train)], markersize=3); plt.title('train sample'); plt.axis('off')
plt.subplot(6,1,3); plt.spy(X_train[:,:num_ms_raster,np.random.randint(num_simulations_train)], markersize=3); plt.title('train sample'); plt.axis('off')
plt.ylabel('synaptic index')
plt.subplot(6,1,4); plt.spy(X_test[:,:num_ms_raster ,np.random.randint(num_simulations_test) ], markersize=3); plt.title('test sample'); plt.axis('off')
plt.subplot(6,1,5); plt.spy(X_test[:,:num_ms_raster ,np.random.randint(num_simulations_test) ], markersize=3); plt.title('test sample'); plt.axis('off')
plt.subplot(6,1,6); plt.spy(X_test[:,:num_ms_raster ,np.random.randint(num_simulations_test) ], markersize=3); plt.title('test sample'); plt.axis('off')
plt.xlabel('time [ms]')

# binary spikes
plt.figure(figsize=(20,13))
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.32)
plt.subplot(6,1,1); plt.plot(y_spike_train[:,np.random.randint(num_simulations_train)], markersize=3); plt.title('train sample')
plt.subplot(6,1,2); plt.plot(y_spike_train[:,np.random.randint(num_simulations_train)], markersize=3); plt.title('train sample')
plt.subplot(6,1,3); plt.plot(y_spike_train[:,np.random.randint(num_simulations_train)], markersize=3); plt.title('train sample')
plt.subplot(6,1,4); plt.plot(y_spike_test[: ,np.random.randint(num_simulations_test) ], markersize=3); plt.title('test sample')
plt.subplot(6,1,5); plt.plot(y_spike_test[: ,np.random.randint(num_simulations_test) ], markersize=3); plt.title('test sample')
plt.subplot(6,1,6); plt.plot(y_spike_test[: ,np.random.randint(num_simulations_test) ], markersize=3); plt.title('test sample')
plt.xlabel('time [ms]')

# somatic voltage
plt.figure(figsize=(20,13))
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.32)
plt.subplot(6,1,1); plt.plot(y_soma_train[:,np.random.randint(num_simulations_train)], markersize=3); plt.title('train sample')
plt.subplot(6,1,2); plt.plot(y_soma_train[:,np.random.randint(num_simulations_train)], markersize=3); plt.title('train sample')
plt.subplot(6,1,3); plt.plot(y_soma_train[:,np.random.randint(num_simulations_train)], markersize=3); plt.title('train sample')
plt.ylabel('voltage [mV]')
plt.subplot(6,1,4); plt.plot(y_soma_test[: ,np.random.randint(num_simulations_test) ], markersize=3); plt.title('test sample')
plt.subplot(6,1,5); plt.plot(y_soma_test[: ,np.random.randint(num_simulations_test) ], markersize=3); plt.title('test sample')
plt.subplot(6,1,6); plt.plot(y_soma_test[: ,np.random.randint(num_simulations_test) ], markersize=3); plt.title('test sample')
plt.xlabel('time [ms]')

# calculate ISI CV
ISIs_train = []
sim_inds, spike_times = np.nonzero(y_spike_train.T)
for curr_sim_ind in np.unique(sim_inds):
    curr_ISIs = np.diff(spike_times[sim_inds == curr_sim_ind])
    ISIs_train += list(curr_ISIs)
ISI_CV_train = np.array(ISIs_train).std() / np.array(ISIs_train).mean()

# plot ISI distribution and make sure it's fine
plt.figure(figsize=(30,15))
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.12)
plt.subplot(2,1,1); plt.hist(ISIs_train, bins=range(0,2000,5))
plt.title('Inter spike interval (ISI) distribution')

plt.subplot(2,1,2); plt.hist(y_soma_train.ravel(), bins=150)
plt.title('Soma voltage distribution')

input_exc_inst_rate = X_train[:num_ex_synapses,:,:].mean() * 1000
input_inh_inst_rate = X_train[num_ex_synapses:,:,:].mean() * 1000

# summerize key statstics
print('-------------------------------------------')
print('train exc input firing rate = %.3f [Hz]' %(input_exc_inst_rate))
print('train inh input firing rate = %.3f [Hz]' %(input_inh_inst_rate))
print('-------------------------------------------')
print('train output firing rate = %.3f [Hz]' %(y_spike_train.mean() * 1000))
print('test  output firing rate = %.3f [Hz]' %(y_spike_test.mean() * 1000))
print('-------------------------------------------')
print('train output ISI Coefficient of Variation = %.3f' %(ISI_CV_train))
print('-------------------------------------------')


#%% helper function to create a temporally convolutional network


def create_temporaly_convolutional_model(max_input_window_size, num_segments, num_syn_types, filter_sizes_per_layer, num_filters_per_layer,
                                         activation_function_per_layer, l2_regularization_per_layer,
                                         strides_per_layer, dilation_rates_per_layer, initializer_per_layer):
    
    # define input and flatten it
    binary_input_mat = Input(shape=(max_input_window_size, num_segments * num_syn_types), name='input_layer')
        
    for k in range(len(filter_sizes_per_layer)):
        num_filters   = num_filters_per_layer[k]
        filter_size   = filter_sizes_per_layer[k]
        activation    = activation_function_per_layer[k]
        l2_reg        = l2_regularization_per_layer[k]
        stride        = strides_per_layer[k]
        dilation_rate = dilation_rates_per_layer[k]
        initializer   = initializer_per_layer[k]
        
        initializer = initializers.TruncatedNormal(stddev=initializer)
        first_layer_bias_initializer = initializers.Constant(value=0.1)

        if k == 0:
            x = Conv1D(num_filters, filter_size, activation=activation, bias_initializer=first_layer_bias_initializer, kernel_initializer=initializer,
                       kernel_regularizer=l2(l2_reg), strides=stride, dilation_rate=dilation_rate, padding='causal', name='layer_%d' %(k + 1))(binary_input_mat)
        else:
            x = Conv1D(num_filters, filter_size, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(l2_reg),
                       strides=stride, dilation_rate=dilation_rate, padding='causal', name='layer_%d' %(k + 1))(x)
        #x = BatchNormalization(name='layer_%d_BN'%(k+1))(x)

    output_spike_init_weights = initializers.TruncatedNormal(stddev=0.05)
    output_spike_init_bias    = initializers.Constant(value=-2.5)
    output_soma_init  = initializers.TruncatedNormal(stddev=0.05)

    output_spike_predictions = Conv1D(1, 1, activation='sigmoid', kernel_initializer=output_spike_init_weights, bias_initializer=output_spike_init_bias,
                                                                  kernel_regularizer=l2(1e-8), padding='causal', name='spikes')(x)
    output_soma_voltage_pred = Conv1D(1, 1, activation='linear' , kernel_initializer=output_soma_init, kernel_regularizer=l2(1e-8), padding='causal', name='soma')(x)

    temporaly_convolutional_network_model = Model(inputs=binary_input_mat, outputs=[output_spike_predictions, output_soma_voltage_pred])

    optimizer_to_use = Nadam(lr=0.0003)
    temporaly_convolutional_network_model.compile(optimizer=optimizer_to_use, loss=['binary_crossentropy','mse'], loss_weights=[1.0, 0.003])
    temporaly_convolutional_network_model.summary()

    return temporaly_convolutional_network_model


#%% define network architecture

max_input_window_size = 500
num_segments  = 100
num_syn_types = 1

network_name = '1_layer_TCN'

network_depth = 1
filter_sizes_per_layer        = [80] * network_depth
num_filters_per_layer         = [1] * network_depth
initializer_per_layer         = [0.25] * network_depth
activation_function_per_layer = ['linear'] * network_depth
l2_regularization_per_layer   = [1e-8] * network_depth
strides_per_layer             = [1] * network_depth
dilation_rates_per_layer      = [1] * network_depth

# define model
temporal_conv_net = create_temporaly_convolutional_model(max_input_window_size, num_segments, num_syn_types, filter_sizes_per_layer, num_filters_per_layer,
                                                         activation_function_per_layer, l2_regularization_per_layer,
                                                         strides_per_layer, dilation_rates_per_layer, initializer_per_layer)

# prepare data for training
X_train_for_TCN  = np.transpose(X_train,axes=[2,1,0])
y1_train_for_TCN = y_spike_train.T[:,:,np.newaxis]
y2_train_for_TCN = y_soma_train.T[:,:,np.newaxis] - y_soma_train.mean()


#%% train model

num_train_subsets = 100
num_epochs_per_subset = 1
batch_size = 16
val_split_ratio = 0.05

num_iter_per_epoch = int(((1 - val_split_ratio) * X_train_for_TCN.shape[0]) / batch_size) + 1

num_iterations    = [0]
train_spikes_loss = [np.nan]
valid_spikes_loss = [np.nan]
train_soma_loss   = [np.nan]
valid_soma_loss   = [np.nan]

for k in range(num_train_subsets):
    start_time_ind = np.random.randint(X_train_for_TCN.shape[1] - max_input_window_size - 1)
    end_time_ind   = start_time_ind + max_input_window_size
    
    print('%d: selected timepoint range = [%d, %d]' %(k + 1, start_time_ind, end_time_ind))
    
    history = temporal_conv_net.fit(X_train_for_TCN[:,start_time_ind:end_time_ind,:],
                                    [y1_train_for_TCN[:,start_time_ind:end_time_ind,:], y2_train_for_TCN[:,start_time_ind:end_time_ind,:]],
                                    epochs=num_epochs_per_subset, batch_size=batch_size, validation_split=val_split_ratio)

    num_iterations.append(num_iterations[-1] + num_iter_per_epoch)
    train_spikes_loss.append(history.history['spikes_loss'][0])
    train_soma_loss.append(history.history['soma_loss'][0])
    valid_spikes_loss.append(history.history['val_spikes_loss'][0])
    valid_soma_loss.append(history.history['val_soma_loss'][0])

#%% show learning curves

plt.close('all')

plt.figure(figsize=(15,10))
plt.subplots_adjust(left=0.08,right=0.95,top=0.95,bottom=0.06,hspace=0.3)

plt.subplot(4,1,1); plt.title('spikes loss')
plt.plot(np.array(num_iterations).ravel(), np.array(train_spikes_loss).ravel())
plt.plot(np.array(num_iterations).ravel(), np.array(valid_spikes_loss).ravel())
plt.legend(['train', 'valid'])
plt.ylabel('log loss')

plt.subplot(4,1,2); plt.title('soma loss')
plt.plot(np.array(num_iterations).ravel(), np.array(train_soma_loss).ravel())
plt.plot(np.array(num_iterations).ravel(), np.array(valid_soma_loss).ravel())
plt.legend(['train', 'valid'])
plt.ylabel('MSE')

plt.subplot(4,1,3); plt.title('spikes loss')
plt.semilogy(np.array(num_iterations).ravel(), np.array(train_spikes_loss).ravel())
plt.semilogy(np.array(num_iterations).ravel(), np.array(valid_spikes_loss).ravel())
plt.legend(['train', 'valid'])
plt.ylabel('log loss')

plt.subplot(4,1,4); plt.title('soma loss')
plt.semilogy(np.array(num_iterations).ravel(), np.array(train_soma_loss).ravel())
plt.semilogy(np.array(num_iterations).ravel(), np.array(valid_soma_loss).ravel())
plt.legend(['train', 'valid'])
plt.ylabel('MSE')
plt.xlabel('num train steps')

#%% show first layer weights with temporal cross section below

xytick_labels_fontsize = 16
title_fontsize = 30
xylabels_fontsize = 25
legend_fontsize = 26

first_layer_weights = temporal_conv_net.get_weights()[0][:,:,0].T

# correct positivity for presentation if necessary
is_excitation_negative = first_layer_weights[:num_ex_synapses,-20:].sum() < 0
if is_excitation_negative:
    first_layer_weights = -first_layer_weights

exc_max_avg_w_value = first_layer_weights[:num_ex_synapses,:].mean(axis=0).max()
inh_min_avg_w_value = first_layer_weights[num_ex_synapses:,:].mean(axis=0).min()

# make sure the range is symmetric for visualization purposes
vmin_max_range = [1.05 * inh_min_avg_w_value, -1.05 * inh_min_avg_w_value]

plt.close('all')
fig = plt.figure(figsize=(9,17))
gs = gridspec.GridSpec(3, 1)
gs.update(left=0.15, right=0.85, bottom=0.08, top=0.95, hspace=0.08)
ax1 = plt.subplot(gs[:2,0])
ax2 = plt.subplot(gs[2,0])

ax1.set_title('layer 1 spatio-temporal filter', fontsize=title_fontsize)
ax1.imshow(first_layer_weights,cmap='jet', vmin=vmin_max_range[0], vmax=vmin_max_range[1])
ax1.set_xticks([])
ax1.set_ylabel('syn index', fontsize=xylabels_fontsize)

for ytick_label in ax1.get_yticklabels():
    ytick_label.set_fontsize(xytick_labels_fontsize)

time_axis_weights = -np.arange(first_layer_weights.shape[1])
ax2.set_title('temporal cross sections', fontsize=title_fontsize)
ax2.plot(time_axis_weights, np.flipud(first_layer_weights[:num_ex_synapses,:].T),c='r')
ax2.plot(time_axis_weights, np.flipud(first_layer_weights[num_ex_synapses:,:].T),c='b')
ex_synapses_patch = mpatches.Patch(color='red', label='exc syn')
inh_synapses_patch = mpatches.Patch(color='blue', label='inh syn')
ax2.legend(handles=[ex_synapses_patch, inh_synapses_patch], fontsize=legend_fontsize, loc='upper left')
ax2.set_xlim(time_axis_weights.min(),time_axis_weights.max())
ax2.set_xlabel('time before prediction moment [ms]', fontsize=xylabels_fontsize)
ax2.set_ylabel('weight', fontsize=xylabels_fontsize)
ax2.set_ylim(vmin_max_range[0], vmin_max_range[1])

for ytick_label in ax2.get_yticklabels():
    ytick_label.set_fontsize(xytick_labels_fontsize)
for xtick_label in ax2.get_xticklabels():
    xtick_label.set_fontsize(xytick_labels_fontsize)

if save_figures:
    figure_name = 'figure1D learned_weights %d' %(np.random.randint(50))

    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
        else:
            subfolder = '%s/' %(file_ending.split('.')[-1])
            fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

#%% create spike predictions on test set from the fitted network output

prediction_start_time = time.time()

max_input_window_size

overlap_size = 120

X_test_for_TCN = np.transpose(X_test,axes=[2,1,0])
y1_test_for_TCN = y_spike_test.T[:,:,np.newaxis]
y2_test_for_TCN = y_soma_test.T[:,:,np.newaxis] - y_soma_train.mean()

y1_test_for_TCN_hat = np.zeros(y1_test_for_TCN.shape)
y2_test_for_TCN_hat = np.zeros(y2_test_for_TCN.shape)

num_test_splits = int(2 + (X_test_for_TCN.shape[1] - max_input_window_size) / (max_input_window_size - overlap_size))

for k in range(num_test_splits):
    start_time_ind = k * (max_input_window_size - overlap_size)
    end_time_ind   = start_time_ind + max_input_window_size
    
    curr_X_test_for_TCN = X_test_for_TCN[:,start_time_ind:end_time_ind,:]
    
    if curr_X_test_for_TCN.shape[1] < max_input_window_size:
        padding_size = max_input_window_size - curr_X_test_for_TCN.shape[1]
        X_pad = np.zeros((curr_X_test_for_TCN.shape[0],padding_size,curr_X_test_for_TCN.shape[2]))
        curr_X_test_for_TCN = np.hstack((curr_X_test_for_TCN,X_pad))
    curr_y1_test_for_TCN, curr_y2_test_for_TCN = temporal_conv_net.predict(curr_X_test_for_TCN)

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

prediction_duration_min = (time.time() - prediction_start_time) / 60
print('time took to predict is %.3f minutes' %(prediction_duration_min))


#%% show main evaluation metrics

plt.close('all')

xytick_labels_fontsize = 16
title_fontsize = 30
xylabels_fontsize = 25
legend_fontsize = 26

fig = plt.figure(figsize=(11,17))
gs = gridspec.GridSpec(3,1)
gs.update(left=0.12, right=0.95, bottom=0.05, top=0.92, hspace=0.6)
ax0 = plt.subplot(gs[0,0])
ax1 = plt.subplot(gs[1,0])
ax2 = plt.subplot(gs[2,0])

y_test = y_spike_test
y_test_hat = y1_test_for_TCN_hat[:,:,0].T

## plot histograms of prediction given ground truth
ax0.hist(y_test_hat[y_test == True ], bins=np.linspace(0,1,50), color='g', alpha=0.8, normed=True)
ax0.hist(y_test_hat[y_test == False], bins=np.linspace(0,1,50), color='b', alpha=0.8, normed=True)
ax0.set_title('spike probability prediction histograms', fontsize=title_fontsize)
ax0.set_xlabel('predicted spike probability', fontsize=xylabels_fontsize)
ax0.set_ylabel('density', fontsize=xylabels_fontsize)
ax0.legend(['P(prediction|spike)','P(prediction|no spike)'], fontsize=legend_fontsize)

for tick_label in (ax0.get_xticklabels() + ax0.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

## plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_test_hat.ravel())
AUC_score = auc(fpr, tpr)

ax1.plot(fpr, tpr)
ax1.set_title('ROC curve (AUC = %.3f)' %(AUC_score), fontsize=title_fontsize)
ax1.set_xlabel('False Positive Rate', fontsize=xylabels_fontsize)
ax1.set_ylabel('True Positive Rate', fontsize=xylabels_fontsize)
ax1.set_ylim(0,1.05)
ax1.set_xlim(-0.03,1)

for tick_label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

desired_false_positive_rate = 0.002
desired_fp_ind = np.argmin(abs(fpr - desired_false_positive_rate))
if desired_fp_ind == 0:
    desired_fp_ind = 1
actual_false_positive_rate = fpr[desired_fp_ind]
print('at %.4f FP rate, TP = %.4f' %(actual_false_positive_rate, tpr[desired_fp_ind]))

# organize the data for ploting
desired_threshold = thresholds[desired_fp_ind]
ground_truth_output_spikes = y_test.T
predicted_output_spikes    = y_test_hat.T > desired_threshold
num_test_traces            = X_test.shape[2]

## plot the cross correlation between the spike trains

half_time_window_size_ms = 50
delta_time = 10

inds_inside_delta = range(half_time_window_size_ms - delta_time, half_time_window_size_ms + 1 + delta_time)

# pad both spike train predictions with zeros from both sides
zero_padding_matrix = np.zeros((num_test_traces,half_time_window_size_ms))
predicted_output_spikes_padded    = np.hstack((zero_padding_matrix,predicted_output_spikes,zero_padding_matrix))
ground_truth_output_spikes_padded = np.hstack((zero_padding_matrix,ground_truth_output_spikes,zero_padding_matrix))

# calculate recall curve: P(predicted spikes|ground truth=spike)
recall_curve = np.zeros(1 + 2 * half_time_window_size_ms)
trace_inds, spike_inds = np.nonzero(ground_truth_output_spikes_padded)
for trace_ind, spike_ind in zip(trace_inds,spike_inds):
    recall_curve += predicted_output_spikes_padded[trace_ind,(spike_ind - half_time_window_size_ms):(1 + spike_ind + half_time_window_size_ms)]
recall_curve /= recall_curve.sum()
recall = recall_curve[inds_inside_delta].sum()

time_axis_cc = np.arange(-half_time_window_size_ms, half_time_window_size_ms + 1)

time_in_delta = time_axis_cc[inds_inside_delta]
recall_in_delta = recall_curve[inds_inside_delta]
recall_patch = mpatches.Patch(color='b', label='area = %.2f' %(recall))

ax2.set_title('$P(Prediction | GroundTruth = 1)$', fontsize=title_fontsize)
ax2.plot(time_axis_cc, recall_curve, c='k')
ax2.fill_between(time_in_delta, recall_in_delta, 0, facecolor='b', alpha=0.8)
ax2.legend(handles=[recall_patch],fontsize=legend_fontsize)
ax2.vlines([time_in_delta[0],time_in_delta[-1]], [0,0], [recall_in_delta[0],recall_in_delta[-1]], colors='k', linewidths=3.3)
ax2.set_ylim(0, 1.05 * recall_curve.max())
ax2.set_xlabel('$\Delta t$ [ms]', fontsize=xylabels_fontsize)
ax2.set_ylabel('density', fontsize=xylabels_fontsize)

for tick_label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

if save_figures:
    figure_name = 'figure1B spike predictions %d' %(np.random.randint(50))

    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
        else:
            subfolder = '%s/' %(file_ending.split('.')[-1])
            fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

## plot voltage trace scatter plot

fig = plt.figure(figsize=(12,15))
gs = gridspec.GridSpec(3, 1)
gs.update(left=0.12, right=0.95, bottom=0.05, top=0.92, hspace=0.37)
ax0 = plt.subplot(gs[:2,0])
ax1 = plt.subplot(gs[2,0])

xytick_labels_fontsize = 16
title_fontsize = 25
xylabels_fontsize = 22
legend_fontsize = 26

num_datapoints_in_scatter = 60000
mean_soma_voltage = y_soma_train.mean()
selected_datapoints = np.random.choice(range(len(y2_test_for_TCN.ravel())),size=num_datapoints_in_scatter,replace=False)
selected_GT = y2_test_for_TCN.ravel()[selected_datapoints] + 0.02 * np.random.randn(num_datapoints_in_scatter) + mean_soma_voltage
selected_pred = y2_test_for_TCN_hat.ravel()[selected_datapoints] + mean_soma_voltage

soma_explained_variance_percent = 100.0 * explained_variance_score(y2_test_for_TCN.ravel(),y2_test_for_TCN_hat.ravel())
soma_RMSE = np.sqrt(MSE(y2_test_for_TCN.ravel(),y2_test_for_TCN_hat.ravel()))
soma_MAE  = MAE(y2_test_for_TCN.ravel(),y2_test_for_TCN_hat.ravel())


ax0.scatter(selected_GT,selected_pred, s=1.5, alpha=0.8)
ax0.set_title('soma voltage prediction. explained variance = %.2f%s' %(soma_explained_variance_percent,'%'), fontsize=title_fontsize)
ax0.set_xlabel('ground truth soma voltage [mV]', fontsize=xylabels_fontsize)
ax0.set_ylabel('predicted soma voltage [mV]', fontsize=xylabels_fontsize)
soma_voltage_lims = np.round([np.percentile(selected_pred,0.2),np.percentile(selected_pred,99.8)]).astype(int)
voltage_granularity = 5
voltage_setpoint = -56
voltage_axis = np.arange(soma_voltage_lims[0],soma_voltage_lims[1])
voltage_ticks_to_show = np.unique(((voltage_axis - voltage_setpoint) / voltage_granularity).astype(int) * voltage_granularity + voltage_setpoint)
voltage_ticks_to_show = voltage_ticks_to_show[np.logical_and(voltage_ticks_to_show >= soma_voltage_lims[0],
                                                             voltage_ticks_to_show <= soma_voltage_lims[1])]
ax0.set_xticks(voltage_ticks_to_show)
ax0.set_yticks(voltage_ticks_to_show)
ax0.set_xlim(soma_voltage_lims[0],soma_voltage_lims[1])
ax0.set_ylim(soma_voltage_lims[0],soma_voltage_lims[1])
ax0.plot([-90,-50],[-90,-50], ls='-', c='k')
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)

for tick_label in (ax0.get_xticklabels() + ax0.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

ax1.hist(y2_test_for_TCN_hat.ravel() - y2_test_for_TCN.ravel(), bins=300, normed=True)
ax1.set_title('voltage prediction redisduals. RMSE = %.3f [mV]' %(soma_RMSE), fontsize=title_fontsize)
ax1.set_xlabel('$\Delta$V [mV]', fontsize=xylabels_fontsize)
ax1.set_xlim(-20,20)
ax1.set_yticks([])

for tick_label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

if save_figures:
    figure_name = 'figure1C voltage predictions %d' %(np.random.randint(50))

    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
        else:
            subfolder = '%s/' %(file_ending.split('.')[-1])
            fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')


#%% figure 1 replication

xytick_labels_fontsize = 16
title_fontsize = 26
xylabels_fontsize = 17
legend_fontsize = 15

num_spikes_per_simulation = y1_test_for_TCN.sum(axis=1)[:,0]
soma_bias_to_add = y_soma_train.mean()

# get a selected trace
possible_presentable_candidates = np.nonzero(np.logical_and(num_spikes_per_simulation >= 2, num_spikes_per_simulation <= 10))[0]
selected_trace  = np.random.choice(possible_presentable_candidates)
zoomin_fraction = [0.25 + 0.24 * np.random.rand(), 0.51 + 0.24 * np.random.rand()]

selected_trace  = 937
zoomin_fraction = [0.295,0.53]

print('selected_trace = %s' %(selected_trace))
print('zoomin_fraction = %s' %(zoomin_fraction))

# collect everything need for presentation
spike_trace_GT   = y1_test_for_TCN[selected_trace,:,0]
spike_trace_pred = y1_test_for_TCN_hat[selected_trace,:,0] > desired_threshold

output_spike_times_in_ms_GT   = np.nonzero(spike_trace_GT)[0]
output_spike_times_in_ms_pred = np.nonzero(spike_trace_pred)[0]

soma_voltage_trace_GT   = y2_test_for_TCN[selected_trace,:,0] + soma_bias_to_add
soma_voltage_trace_pred = y2_test_for_TCN_hat[selected_trace,:,0] + soma_bias_to_add

soma_voltage_trace_GT[output_spike_times_in_ms_GT] = 40
soma_voltage_trace_pred[output_spike_times_in_ms_pred] = 40

sim_duration_ms = spike_trace_GT.shape[0]
time_in_sec = np.arange(sim_duration_ms) / 1000.0

# for raster plot (scatter)
all_presynaptic_spikes_bin = X_test_for_TCN[selected_trace,:,:]

syn_activation_time, syn_activation_index = np.nonzero(all_presynaptic_spikes_bin)
ex_synapses_inds = syn_activation_index < num_ex_synapses

ex_syn_activation_time   = syn_activation_time[ex_synapses_inds] / 1000.0
ex_syn_activation_index  = num_synapses - syn_activation_index[ex_synapses_inds]
inh_syn_activation_time  = syn_activation_time[~ex_synapses_inds] / 1000.0
inh_syn_activation_index = num_synapses - syn_activation_index[~ex_synapses_inds]


# set up the grid specs
plt.close('all')
fig = plt.figure(figsize=(24,18.5))

gs1 = gridspec.GridSpec(3,1)
gs1.update(left=0.05, right=0.65, bottom=0.05, top=0.45, wspace=0.01, hspace=0.01)

gs2 = gridspec.GridSpec(12,2)
gs2.update(left=0.73, right=0.97, bottom=0.07, top=0.97, wspace=0.58, hspace=1.05)

ax10 = plt.subplot(gs1[0,0])
ax11 = plt.subplot(gs1[1,0])
ax12 = plt.subplot(gs1[2,0])

ax10.axis('off')
ax11.axis('off')
ax12.axis('off')

ax31 = plt.subplot(gs2[:5,:])
ax32 = plt.subplot(gs2[5:7,:])

a33_left  = plt.subplot(gs2[7:9,0])
a33_right = plt.subplot(gs2[7:9,1])

ax34 = plt.subplot(gs2[9:,:])


### left column of the figure

## raster of input exitation and inhibition of the selected trace
ax10.scatter(ex_syn_activation_time, ex_syn_activation_index, s=2, c='r')
ax10.scatter(inh_syn_activation_time, inh_syn_activation_index, s=2, c='b')
ax10.set_xlim(0, sim_duration_sec - 0.01)
ax10.set_ylabel('syn index', fontsize=xylabels_fontsize)
ax10.grid('off')
ax10.set_yticks([])
ax10.set_xticks([])

## ground truth and prediction trace
ax11.plot(time_in_sec,soma_voltage_trace_GT,c='c')
ax11.plot(time_in_sec,soma_voltage_trace_pred,c='m',linestyle=':')
ax11.set_xlim(0,sim_duration_sec)
ax11.set_ylabel('$V_m$ (mV)', fontsize=xylabels_fontsize)
for tick_label in (ax11.get_xticklabels() + ax11.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

# add dashed rectangle
zoomin_xlims = [zoomin_fraction[0] * sim_duration_sec, zoomin_fraction[1] * sim_duration_sec]
zoomin_dur_sec = zoomin_xlims[1] - zoomin_xlims[0]
zoomin_time_in_sec = np.logical_and(time_in_sec >= zoomin_xlims[0], time_in_sec <= zoomin_xlims[1])
zoomin_ylims = [soma_voltage_trace_GT[zoomin_time_in_sec].min() -2.5, -49]
zoomin_scalebar_xloc = zoomin_xlims[1] - 0.05 * zoomin_dur_sec

rect_w = zoomin_xlims[1] - zoomin_xlims[0]
rect_h = zoomin_ylims[1] - zoomin_ylims[0]
rect_bl_x = zoomin_xlims[0]
rect_bl_y = zoomin_ylims[0]
dashed_rectangle = mpatches.Rectangle((rect_bl_x,rect_bl_y),rect_w,rect_h,linewidth=2,edgecolor='k',linestyle='--',facecolor='none')

ax11.add_patch(dashed_rectangle)

## zoomin section of ground truth and prediction trace
ax12.plot(time_in_sec,soma_voltage_trace_GT,c='c')
ax12.plot(time_in_sec,soma_voltage_trace_pred,c='m',linestyle=':')
ax12.set_xlim(zoomin_xlims[0],zoomin_xlims[1])
ax12.set_ylim(zoomin_ylims[0],zoomin_ylims[1])
ax12.set_ylabel('$V_m$ (mV)', fontsize=xylabels_fontsize)
ax12.set_xlabel('time (sec)', fontsize=xylabels_fontsize)

for tick_label in (ax12.get_xticklabels() + ax12.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

# add scale bar to top plot
zoomout_scalebar_xloc = 0.95 * sim_duration_sec

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

# add scalebar to bottom plot
scalebar_loc = np.array([zoomin_scalebar_xloc,-58])
scalebar_size_x = 0.1
scalebar_str_x = '100 ms'
scalebar_size_y = 8
scalebar_str_y = '%d mV' %(scalebar_size_y)

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

### right column of the figure

## ROC curve
a33_left.plot(fpr, tpr, c='k')
a33_left.set_xlabel('False alarm rate', fontsize=xylabels_fontsize)
a33_left.set_ylabel('Hit rate', fontsize=xylabels_fontsize)
a33_left.set_ylim(0,1.05)
a33_left.set_xlim(-0.03,1)

a33_left.spines['top'].set_visible(False)
a33_left.spines['right'].set_visible(False)

for tick_label in (a33_left.get_xticklabels() + a33_left.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

left, bottom, width, height = [0.768, 0.327, 0.047, 0.067]
a33_left_inset = fig.add_axes([left, bottom, width, height])
a33_left_inset.plot(fpr, tpr, c='k')
a33_left_inset.set_ylim(0,1.05)
a33_left_inset.set_xlim(-0.001,0.012)
a33_left_inset.spines['top'].set_visible(False)
a33_left_inset.spines['right'].set_visible(False)
a33_left_inset.scatter(actual_false_positive_rate, tpr[desired_fp_ind + 1], c='r', s=100)

print('at %.4f FP rate, TP = %.4f' %(actual_false_positive_rate, tpr[desired_fp_ind]))

## cross correlation between the spike trains
a33_right.plot(time_axis_cc, 1000 * recall_curve, c='k')
a33_right.set_ylim(0, 1.05 * 1000 * recall_curve.max())
a33_right.set_xlabel('$\Delta t$ (ms)', fontsize=xylabels_fontsize)
a33_right.set_ylabel('spike rate (Hz)', fontsize=xylabels_fontsize)
a33_right.spines['top'].set_visible(False)
a33_right.spines['right'].set_visible(False)
a33_right.spines['left'].set_visible(False)
a33_right.spines['bottom'].set_visible(False)
for tick_label in (a33_right.get_xticklabels() + a33_right.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

## weights heatmap
vmin_max_range = [1.1 * inh_min_avg_w_value, -1.1 * inh_min_avg_w_value]
vmin_max_range_to_plot = (8 * np.array(vmin_max_range)).astype(int) / 10

weights_images = ax31.imshow(first_layer_weights, cmap='jet', aspect='auto', vmin=vmin_max_range[0], vmax=vmin_max_range[1])
ax31.set_xticks([])
ax31.set_ylabel('Synaptic index', fontsize=xylabels_fontsize)
for ytick_label in ax31.get_yticklabels():
    ytick_label.set_fontsize(xytick_labels_fontsize)


ax_colorbar = inset_axes(ax31, width="50%", height="6%", loc=2)
cbar = plt.colorbar(weights_images, cax=ax_colorbar, orientation="horizontal", ticks=[vmin_max_range_to_plot[0], 0, vmin_max_range_to_plot[1]])
ax_colorbar.xaxis.set_ticks_position("bottom")
cbar.ax.tick_params(labelsize=15)

ax31.text(12, 12, 'weight (A.U)', color='k', fontsize=15, ha='left', va='top', rotation='horizontal')

## temporal cross sections of weights
ax32.plot(time_axis_weights, np.flipud(first_layer_weights[:num_ex_synapses,:].T),c='r')
ax32.plot(time_axis_weights, np.flipud(first_layer_weights[num_ex_synapses:,:].T),c='b')
ax32.set_xlim(time_axis_weights.min(),time_axis_weights.max())
ax32.set_xlabel('Time before $t_0$ (ms)', fontsize=xylabels_fontsize)
ax32.set_ylabel('Weight (A.U)', fontsize=xylabels_fontsize)
ax32.set_ylim(vmin_max_range[0], vmin_max_range[1])
ax32.set_yticks([vmin_max_range_to_plot[0],0,vmin_max_range_to_plot[1]])

for ytick_label in ax32.get_yticklabels():
    ytick_label.set_fontsize(xytick_labels_fontsize)
for xtick_label in ax32.get_xticklabels():
    xtick_label.set_fontsize(xytick_labels_fontsize)

# place a text box in upper left in axes coords
ax32.text(-25, 0.5 * vmin_max_range_to_plot[1] - 0.1, 'Exc', color='r', fontsize=20, verticalalignment='bottom')
ax32.text(-25, 0.5 * vmin_max_range_to_plot[0], 'Inh', color='b', fontsize=20, verticalalignment='top')

## voltage predction scatter plot
ax34.scatter(selected_GT,selected_pred, s=1.0, alpha=0.8)
soma_voltage_lims = np.round([np.percentile(selected_pred,0.2),np.percentile(selected_pred,99.8)]).astype(int)
voltage_granularity = 5
voltage_setpoint = -56
voltage_axis = np.arange(soma_voltage_lims[0],soma_voltage_lims[1])
voltage_ticks_to_show = np.unique(((voltage_axis - voltage_setpoint) / voltage_granularity).astype(int) * voltage_granularity + voltage_setpoint)
voltage_ticks_to_show = voltage_ticks_to_show[np.logical_and(voltage_ticks_to_show >= soma_voltage_lims[0],
                                                             voltage_ticks_to_show <= soma_voltage_lims[1])]
ax34.set_xticks(voltage_ticks_to_show)
ax34.set_yticks(voltage_ticks_to_show)
ax34.set_xlim(soma_voltage_lims[0],soma_voltage_lims[1])
ax34.set_ylim(soma_voltage_lims[0],soma_voltage_lims[1])
ax34.plot([-90,-50],[-90,-50], ls='-', c='k')
ax34.set_xlabel('I&F (mV)', fontsize=xylabels_fontsize)
ax34.set_ylabel('ANN (mV)', fontsize=xylabels_fontsize)
ax34.spines['top'].set_visible(False)
ax34.spines['right'].set_visible(False)

for tick_label in (ax34.get_xticklabels() + ax34.get_yticklabels()):
    tick_label.set_fontsize(xytick_labels_fontsize)

# save figure
if save_figures:
    figure_name = 'figure1_full_figure_v3_%d' %(np.random.randint(100))

    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
        else:
            subfolder = '%s/' %(file_ending.split('.')[-1])
            fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

