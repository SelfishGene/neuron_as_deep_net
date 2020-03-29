import numpy as np
import pandas as pd
import glob
import time
import cPickle as pickle
import keras
from keras.models import Model, load_model
from keras.optimizers import SGD, Nadam
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten, Input, TimeDistributed, Reshape, Permute
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv1D, Cropping1D, UpSampling1D, MaxPooling1D, AveragePooling1D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.regularizers import l1,l2,l1_l2
from keras import initializers
from sklearn import decomposition

# NOTE: during this project I've changed my coding style
# and was too lazy to edit the old code to match the new style
# so please ignore any style related wierdness
# thanks for not being petty about unimportant shit

# ALSO NOTE: prints are for logging purposes

print('-----------------------------------')

# ------------------------------------------------------------------
# fit generator params
# ------------------------------------------------------------------
use_multiprocessing = False
num_workers = 1

print('------------------------------------------------------------------')
print('use_multiprocessing = %s, num_workers = %d' %(str(use_multiprocessing), num_workers))
print('------------------------------------------------------------------')
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# basic configurations and directories
# ------------------------------------------------------------------
#synapse_type = 'NMDA'
#synapse_type = 'AMPA'
synapse_type = 'AMPA_SK'

if synapse_type == 'NMDA':
    num_DVT_components = 20

    train_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_NMDA_train/'
    valid_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_NMDA_valid/'
    test_data_dir  = '/Reseach/Single_Neuron_InOut/data/L5PC_NMDA_test/'
    models_dir     = '/Reseach/Single_Neuron_InOut/models/NMDA/'
    
elif synapse_type == 'AMPA':
    num_DVT_components = 30

    train_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_train/'
    valid_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_valid/'
    test_data_dir  = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_test/'
    models_dir     = '/Reseach/Single_Neuron_InOut/models/AMPA/'
    
elif synapse_type == 'AMPA_SK':
    num_DVT_components = 30

    train_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_SK_train/'
    valid_data_dir = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_SK_valid/'
    test_data_dir  = '/Reseach/Single_Neuron_InOut/data/L5PC_AMPA_SK_test/'
    models_dir     = '/Reseach/Single_Neuron_InOut/models/AMPA_SK/'
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# learning schedule params
# ------------------------------------------------------------------

validation_fraction = 0.5
train_file_load = 0.2
valid_file_load = 0.2
num_steps_multiplier = 10

train_files_per_epoch = 6
valid_files_per_epoch = max(1, int(validation_fraction * train_files_per_epoch))

num_epochs = 250

DVT_loss_mult_factor = 0.1

batch_size_per_epoch        = [8] * num_epochs
learning_rate_per_epoch     = [0.0001] * len(batch_size_per_epoch)
loss_weights_per_epoch      = [[1.0, 0.0200, DVT_loss_mult_factor * 0.00005]] * len(batch_size_per_epoch)
num_train_steps_per_epoch   = [100] * len(batch_size_per_epoch)

for i in range(40,num_epochs):
    batch_size_per_epoch[i]    = 8
    learning_rate_per_epoch[i] = 0.00003
    loss_weights_per_epoch[i]  = [2.0, 0.0100, DVT_loss_mult_factor * 0.00003]
    
for i in range(80,num_epochs):
    batch_size_per_epoch[i]    = 8
    learning_rate_per_epoch[i] = 0.00001
    loss_weights_per_epoch[i]  = [4.0, 0.0100, DVT_loss_mult_factor * 0.00001]

for i in range(120,num_epochs):
    batch_size_per_epoch[i]    = 8
    learning_rate_per_epoch[i] = 0.000003
    loss_weights_per_epoch[i]  = [8.0, 0.0100, DVT_loss_mult_factor * 0.0000001]

for i in range(160,num_epochs):
    batch_size_per_epoch[i]    = 8
    learning_rate_per_epoch[i] = 0.000001
    loss_weights_per_epoch[i]  = [9.0, 0.0030, DVT_loss_mult_factor * 0.00000001]

learning_schedule_dict = {}
learning_schedule_dict['train_file_load']           = train_file_load
learning_schedule_dict['valid_file_load']           = valid_file_load
learning_schedule_dict['validation_fraction']       = validation_fraction
learning_schedule_dict['num_epochs']                = num_epochs
learning_schedule_dict['num_steps_multiplier']      = num_steps_multiplier
learning_schedule_dict['batch_size_per_epoch']      = batch_size_per_epoch
learning_schedule_dict['loss_weights_per_epoch']    = loss_weights_per_epoch
learning_schedule_dict['learning_rate_per_epoch']   = learning_rate_per_epoch
learning_schedule_dict['num_train_steps_per_epoch'] = num_train_steps_per_epoch

# ------------------------------------------------------------------



# ------------------------------------------------------------------
# define network architecture params
# ------------------------------------------------------------------

input_window_size = 400
num_segments  = 2*639
num_syn_types = 1

# FCN network architectures
#network_depth = 1
#filter_sizes_per_layer        = [110]
#num_filters_per_layer         = [128]*network_depth

#network_depth = 2
#filter_sizes_per_layer        = [45,1]
#num_filters_per_layer         = [128]*network_depth

#network_depth = 3
#filter_sizes_per_layer        = [25,1,1]
#num_filters_per_layer         = [256]*network_depth


# TCN network architectures
#network_depth = 2
#filter_sizes_per_layer        = [40]*network_depth
#num_filters_per_layer         = [8]*network_depth

#network_depth = 3
#filter_sizes_per_layer        = [20]*network_depth
#num_filters_per_layer         = [8]*network_depth

#network_depth = 4
#filter_sizes_per_layer        = [16]*network_depth
#num_filters_per_layer         = [128]*network_depth

#network_depth = 5
#filter_sizes_per_layer        = [10]*network_depth
#num_filters_per_layer         = [256]*network_depth

#network_depth = 6
#filter_sizes_per_layer        = [13]*network_depth
#num_filters_per_layer         = [128]*network_depth

#network_depth = 7
#filter_sizes_per_layer        = [35]*network_depth
#num_filters_per_layer         = [128]*network_depth

#network_depth = 8
#filter_sizes_per_layer        = [33]*network_depth
#num_filters_per_layer         = [256]*network_depth

# TCN with large first layer filter (for presentation)
#network_depth = 7
#filter_sizes_per_layer        = [54,12,12,12,12,12,12]
#num_filters_per_layer         = [256]*network_depth

network_depth = 3
filter_sizes_per_layer        = [54,24,24]
num_filters_per_layer         = [64] * network_depth

initializer_per_layer         = [0.002] * network_depth
activation_function_per_layer = ['relu'] * network_depth
l2_regularization_per_layer   = [1e-8] * network_depth
strides_per_layer             = [1] * network_depth
dilation_rates_per_layer      = [1] * network_depth


architecture_dict = {}
architecture_dict['network_depth']                 = network_depth
architecture_dict['input_window_size']             = input_window_size
architecture_dict['num_filters_per_layer']         = num_filters_per_layer
architecture_dict['initializer_per_layer']         = initializer_per_layer
architecture_dict['filter_sizes_per_layer']        = filter_sizes_per_layer
architecture_dict['l2_regularization_per_layer']   = l2_regularization_per_layer
architecture_dict['activation_function_per_layer'] = activation_function_per_layer
architecture_dict['strides_per_layer']             = strides_per_layer
architecture_dict['dilation_rates_per_layer']      = dilation_rates_per_layer

print('L2 regularization = %.9f' %(l2_regularization_per_layer[0]))
print('activation function = "%s"' %(activation_function_per_layer[0]))

# ------------------------------------------------------------------


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


def parse_sim_experiment_file_with_DVT(sim_experiment_file, DVT_PCA_model=None, print_logs=False):
    
    if print_logs:
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
    
    # if we recive PCA model of DVTs, then output the projection on that model, else return the full DVTs
    if DVT_PCA_model is not None:
        num_components = DVT_PCA_model.n_components
        y_DVTs  = np.zeros((num_components,sim_duration_ms,num_simulations), dtype=np.float32)
    else:
        y_DVTs  = np.zeros((num_segments,sim_duration_ms,num_simulations), dtype=np.float16)
    
    # go over all simulations in the experiment and collect their results
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex  = dict2bin(sim_dict['exInputSpikeTimes'] , num_segments, sim_duration_ms)
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments, sim_duration_ms)
        X[:,:,k] = np.vstack((X_ex,X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        y_spike[spike_times,k] = 1.0
        y_soma[:,k] = sim_dict['somaVoltageLowRes']
        
        # if we recive PCA model of DVTs, then output the projection on that model, else return the full DVTs
        curr_DVTs = sim_dict['dendriticVoltagesLowRes']
        # clip the DVTs (to mainly reflect synaptic input and NMDA spikes (battery ~0mV) and diminish importance of bAP and calcium spikes)
        curr_DVTs[curr_DVTs > 2.0] = 2.0
        if DVT_PCA_model is not None:
            y_DVTs[:,:,k] = DVT_PCA_model.transform(curr_DVTs.T).T
        else:
            y_DVTs[:,:,k] = curr_DVTs
        
    if print_logs:
        loading_duration_sec = time.time() - loading_start_time
        print('loading took %.3f seconds' %(loading_duration_sec))
        print('-----------------------------------------------------------------')

    return X, y_spike, y_soma, y_DVTs


def parse_multiple_sim_experiment_files_with_DVT(sim_experiment_files, DVT_PCA_model=None):
    
    for k, sim_experiment_file in enumerate(sim_experiment_files):
        X_curr, y_spike_curr, y_soma_curr, y_DVT_curr = parse_sim_experiment_file_with_DVT(sim_experiment_file, DVT_PCA_model=DVT_PCA_model)
        
        if k == 0:
            X       = X_curr
            y_spike = y_spike_curr
            y_soma  = y_soma_curr
            y_DVT   = y_DVT_curr
        else:
            X       = np.dstack((X,X_curr))
            y_spike = np.hstack((y_spike,y_spike_curr))
            y_soma  = np.hstack((y_soma,y_soma_curr))
            y_DVT   = np.dstack((y_DVT,y_DVT_curr))

    return X, y_spike, y_soma, y_DVT


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


def create_temporaly_convolutional_model(max_input_window_size, num_segments, num_syn_types, num_DVT_outputs,
                                                                                             filter_sizes_per_layer,
                                                                                             num_filters_per_layer,
                                                                                             activation_function_per_layer,
                                                                                             l2_regularization_per_layer,
                                                                                             strides_per_layer,
                                                                                             dilation_rates_per_layer,
                                                                                             initializer_per_layer):
    
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
        
        if activation == 'lrelu':
            leaky_relu_slope = 0.25
            activation = lambda x: LeakyReLU(alpha=leaky_relu_slope)(x)
            print('leaky relu slope = %.4f' %(leaky_relu_slope))
            
        if not isinstance(initializer, basestring):
            initializer = initializers.TruncatedNormal(stddev=initializer)
        
        if k == 0:
            x = Conv1D(num_filters, filter_size, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(l2_reg),
                       strides=stride, dilation_rate=dilation_rate, padding='causal', name='layer_%d' %(k + 1))(binary_input_mat)
        else:
            x = Conv1D(num_filters, filter_size, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(l2_reg),
                       strides=stride, dilation_rate=dilation_rate, padding='causal', name='layer_%d' %(k + 1))(x)
        x = BatchNormalization(name='layer_%d_BN' %(k + 1))(x)
        
    output_spike_init_weights = initializers.TruncatedNormal(stddev=0.001)
    output_spike_init_bias    = initializers.Constant(value=-2.0)
    output_soma_init  = initializers.TruncatedNormal(stddev=0.03)
    output_dend_init  = initializers.TruncatedNormal(stddev=0.05)

    output_spike_predictions = Conv1D(1, 1, activation='sigmoid', kernel_initializer=output_spike_init_weights, bias_initializer=output_spike_init_bias,
                                                                  kernel_regularizer=l2(1e-8), padding='causal', name='spikes')(x)
    output_soma_voltage_pred = Conv1D(1, 1, activation='linear' , kernel_initializer=output_soma_init, kernel_regularizer=l2(1e-8), padding='causal', name='somatic')(x)
    output_dend_voltage_pred = Conv1D(num_DVT_outputs, 1, activation='linear' , kernel_initializer=output_dend_init, kernel_regularizer=l2(1e-8), padding='causal', name='dendritic')(x)

    temporaly_convolutional_network_model = Model(inputs=binary_input_mat, outputs=
                                                  [output_spike_predictions, output_soma_voltage_pred, output_dend_voltage_pred])

    optimizer_to_use = Nadam(lr=0.0001)
    temporaly_convolutional_network_model.compile(optimizer=optimizer_to_use,
                                                  loss=['binary_crossentropy','mse','mse'],
                                                  loss_weights=[1.0, 0.006, 0.002])
    temporaly_convolutional_network_model.summary()
    
    return temporaly_convolutional_network_model


# helper function to select random {X,y} window pairs from dataset
def sample_windows_from_sims(sim_experiment_files, batch_size=16, window_size_ms=400, ignore_time_from_start=500, file_load=0.5, 
                             DVT_PCA_model=None, y_train_soma_bias=-67.7, y_soma_threshold=-55.0, y_DTV_threshold=3.0):
    
    while True:
        # randomly sample simulation file
        sim_experiment_file = np.random.choice(sim_experiment_files,size=1)[0]
        print('from %d files loading "%s"' %(len(sim_experiment_files),sim_experiment_file))
        X, y_spike, y_soma, y_DVT = parse_sim_experiment_file_with_DVT(sim_experiment_file, DVT_PCA_model=DVT_PCA_model)
        
        # reshape to what is needed
        X  = np.transpose(X,axes=[2,1,0])
        y_spike = y_spike.T[:,:,np.newaxis]
        y_soma  = y_soma.T[:,:,np.newaxis]
        y_DVT   = np.transpose(y_DVT,axes=[2,1,0])

        # threshold the signals
        y_soma[y_soma > y_soma_threshold] = y_soma_threshold
        y_DVT[y_DVT > y_DTV_threshold] = y_DTV_threshold
        y_DVT[y_DVT < -y_DTV_threshold] = -y_DTV_threshold

        y_soma = y_soma - y_train_soma_bias
        
        # gather information regarding the loaded file
        num_simulations, sim_duration_ms, num_segments = X.shape
        num_output_channels_y1 = y_spike.shape[2]
        num_output_channels_y2 = y_soma.shape[2]
        num_output_channels_y3 = y_DVT.shape[2]
        
        # determine how many batches in total can enter in the file
        max_batches_per_file = (num_simulations * sim_duration_ms) / (batch_size * window_size_ms)
        batches_per_file     = int(file_load * max_batches_per_file)
        
        print('file load = %.4f, max batches per file = %d' %(file_load, max_batches_per_file))
        print('num batches per file = %d. coming from (%dx%d),(%dx%d)' %(batches_per_file, num_simulations, sim_duration_ms,
                                                                         batch_size, window_size_ms))
        
        for batch_ind in range(batches_per_file):
            # randomly sample simulations for current batch
            selected_sim_inds = np.random.choice(range(num_simulations),size=batch_size,replace=True)
            
            # randomly sample timepoints for current batch
            sampling_start_time = max(ignore_time_from_start, window_size_ms)
            selected_time_inds = np.random.choice(range(sampling_start_time,sim_duration_ms),size=batch_size,replace=False)
            
            # gather batch and yield it
            X_batch       = np.zeros((batch_size, window_size_ms, num_segments))
            y_spike_batch = np.zeros((batch_size, window_size_ms, num_output_channels_y1))
            y_soma_batch  = np.zeros((batch_size, window_size_ms, num_output_channels_y2))
            y_DVT_batch   = np.zeros((batch_size, window_size_ms, num_output_channels_y3))
            for k, (sim_ind, win_time) in enumerate(zip(selected_sim_inds, selected_time_inds)):
                X_batch[k,:,:]       = X[sim_ind,win_time - window_size_ms:win_time,:]
                y_spike_batch[k,:,:] = y_spike[sim_ind,win_time - window_size_ms:win_time,:]
                y_soma_batch[k,:,:]  = y_soma[sim_ind,win_time - window_size_ms:win_time,:]
                y_DVT_batch[k,:,:]   = y_DVT[sim_ind,win_time - window_size_ms:win_time,:]
            
            yield (X_batch, [y_spike_batch, y_soma_batch, y_DVT_batch])


class SimulationDataGenerator(keras.utils.Sequence):
    'thread-safe data genertor for network training'

    def __init__(self, sim_experiment_files, num_files_per_epoch=10,
                 batch_size=8, window_size_ms=300, file_load=0.3, DVT_PCA_model=None,
                 ignore_time_from_start=500, y_train_soma_bias=-67.7, y_soma_threshold=-55.0, y_DTV_threshold=3.0):
        'data generator initialization'
        
        self.sim_experiment_files = sim_experiment_files
        self.num_files_per_epoch = num_files_per_epoch
        self.batch_size = batch_size
        self.window_size_ms = window_size_ms
        self.ignore_time_from_start = ignore_time_from_start
        self.file_load = file_load
        self.DVT_PCA_model = DVT_PCA_model
        self.y_train_soma_bias = y_train_soma_bias
        self.y_soma_threshold = y_soma_threshold
        self.y_DTV_threshold = y_DTV_threshold
        
        self.curr_epoch_files_to_use = None
        self.on_epoch_end()
        self.curr_file_index = -1
        self.load_new_file()
        self.batches_per_file_dict = {}
        
        # gather information regarding the loaded file
        self.num_simulations_per_file, self.sim_duration_ms, self.num_segments = self.X.shape
        self.num_output_channels_y1 = self.y_spike.shape[2]
        self.num_output_channels_y2 = self.y_soma.shape[2]
        self.num_output_channels_y3 = self.y_DVT.shape[2]
        
        # determine how many batches in total can enter in the file
        self.max_batches_per_file = (self.num_simulations_per_file * self.sim_duration_ms) / (self.batch_size * self.window_size_ms)
        self.batches_per_file     = int(self.file_load * self.max_batches_per_file)
        self.batches_per_epoch = self.batches_per_file * self.num_files_per_epoch

        print('-------------------------------------------------------------------------')

        print('file load = %.4f, max batches per file = %d, batches per epoch = %d' %(self.file_load,
                                                                                      self.max_batches_per_file,
                                                                                      self.batches_per_epoch))
        print('num batches per file = %d. coming from (%dx%d),(%dx%d)' %(self.batches_per_file, self.num_simulations_per_file,
                                                                         self.sim_duration_ms, self.batch_size, self.window_size_ms))

        print('-------------------------------------------------------------------------')
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch


    def __getitem__(self, batch_ind_within_epoch):
        'Generate one batch of data'
        
        if ((batch_ind_within_epoch + 1) % self.batches_per_file) == 0:
            self.load_new_file()
            
        # randomly sample simulations for current batch
        selected_sim_inds = np.random.choice(range(self.num_simulations_per_file), size=self.batch_size, replace=True)
        
        # randomly sample timepoints for current batch
        sampling_start_time = max(self.ignore_time_from_start, self.window_size_ms)
        selected_time_inds = np.random.choice(range(sampling_start_time, self.sim_duration_ms), size=self.batch_size, replace=False)
        
        # gather batch and yield it
        X_batch       = np.zeros((self.batch_size, self.window_size_ms, self.num_segments))
        y_spike_batch = np.zeros((self.batch_size, self.window_size_ms, self.num_output_channels_y1))
        y_soma_batch  = np.zeros((self.batch_size, self.window_size_ms, self.num_output_channels_y2))
        y_DVT_batch   = np.zeros((self.batch_size, self.window_size_ms, self.num_output_channels_y3))
        for k, (sim_ind, win_time) in enumerate(zip(selected_sim_inds, selected_time_inds)):
            X_batch[k,:,:]       = self.X[sim_ind,win_time - self.window_size_ms:win_time,:]
            y_spike_batch[k,:,:] = self.y_spike[sim_ind,win_time - self.window_size_ms:win_time,:]
            y_soma_batch[k,:,:]  = self.y_soma[sim_ind ,win_time - self.window_size_ms:win_time,:]
            y_DVT_batch[k,:,:]   = self.y_DVT[sim_ind  ,win_time - self.window_size_ms:win_time,:]
        
        # increment the number of batches collected from each file
        try:
            self.batches_per_file_dict[self.curr_file_in_use] = self.batches_per_file_dict[self.curr_file_in_use] + 1
        except:
            self.batches_per_file_dict[self.curr_file_in_use] = 1
        
        # return the actual batch
        return (X_batch, [y_spike_batch, y_soma_batch, y_DVT_batch])


    def on_epoch_end(self):
        'selects new subset of files to draw samples from'

        self.curr_epoch_files_to_use = np.random.choice(self.sim_experiment_files, size=self.num_files_per_epoch, replace=False)

    def load_new_file(self):
        'load new file to draw batches from'

        self.curr_file_index = (self.curr_file_index + 1) % self.num_files_per_epoch
        # update the current file in use
        self.curr_file_in_use = self.curr_epoch_files_to_use[self.curr_file_index]

        # load the file
        X, y_spike, y_soma, y_DVT = parse_sim_experiment_file_with_DVT(self.curr_file_in_use, DVT_PCA_model=self.DVT_PCA_model)

        # reshape to what is needed
        X  = np.transpose(X,axes=[2,1,0])
        y_spike = y_spike.T[:,:,np.newaxis]
        y_soma  = y_soma.T[:,:,np.newaxis]
        y_DVT   = np.transpose(y_DVT,axes=[2,1,0])

        # threshold the signals
        y_soma[y_soma >  self.y_soma_threshold] =  self.y_soma_threshold
        y_DVT[y_DVT   >  self.y_DTV_threshold]  =  self.y_DTV_threshold
        y_DVT[y_DVT   < -self.y_DTV_threshold]  = -self.y_DTV_threshold

        y_soma = y_soma - self.y_train_soma_bias
        
        self.X, self.y_spike, self.y_soma, self.y_DVT = X, y_spike, y_soma, y_DVT


#%% collect a small dataset of {input,output} recordings for constructing DVT PCA model

print('--------------------------------------------------------------------')
print('started calculating PCA for DVT model')

dataset_generation_start_time = time.time()

data_dir = train_data_dir

train_files = glob.glob(data_dir + '*_6_secDuration_*')[:1]

v_threshold = -55
DVT_threshold = 3

# train PCA model
_, _, _, y_DVTs = parse_sim_experiment_file_with_DVT(train_files[0])
X_pca_DVT = np.reshape(y_DVTs, [y_DVTs.shape[0], -1]).T

DVT_PCA_model = decomposition.PCA(n_components=num_DVT_components, whiten=True)
DVT_PCA_model.fit(X_pca_DVT)

total_explained_variance = 100 * DVT_PCA_model.explained_variance_ratio_.sum()
print('finished training DVT PCA model. total_explained variance = %.1f%s' %(total_explained_variance, '%'))
print('--------------------------------------------------------------------')

X_train, y_spike_train, y_soma_train, y_DVT_train = parse_multiple_sim_experiment_files_with_DVT(train_files, DVT_PCA_model=DVT_PCA_model)
# apply symmetric DVT threshold (the threshold is in units of standard deviations)
y_DVT_train[y_DVT_train >  DVT_threshold] =  DVT_threshold
y_DVT_train[y_DVT_train < -DVT_threshold] = -DVT_threshold

y_soma_train[y_soma_train > v_threshold] = v_threshold

sim_duration_ms = y_soma_train.shape[0]
sim_duration_sec = float(sim_duration_ms) / 1000

num_simulations_train = X_train.shape[-1]

#%% train model (in data streaming way)

print('-----------------------------------------------')
print('finding data')
print('-----------------------------------------------')

train_files = glob.glob(train_data_dir + '*_128_simulationRuns*_6_secDuration_*')
valid_files = glob.glob(valid_data_dir + '*_128_simulationRuns*_6_secDuration_*')
test_files  = glob.glob(test_data_dir  + '*_128_simulationRuns*_6_secDuration_*')

data_dict = {}
data_dict['train_files'] = train_files
data_dict['valid_files'] = valid_files
data_dict['test_files']  = test_files

print('number of training files is %d' %(len(train_files)))
print('number of validation files is %d' %(len(valid_files)))
print('number of test files is %d' %(len(test_files)))
print('-----------------------------------------------')

# define model
assert(input_window_size > sum(filter_sizes_per_layer))
temporal_conv_net = create_temporaly_convolutional_model(input_window_size, num_segments, num_syn_types, num_DVT_components,
                                                         filter_sizes_per_layer, num_filters_per_layer,
                                                         activation_function_per_layer, l2_regularization_per_layer,
                                                         strides_per_layer, dilation_rates_per_layer, initializer_per_layer)

is_fully_connected = (network_depth == 1) or sum(filter_sizes_per_layer[1:]) == (network_depth -1)
if is_fully_connected:
    model_prefix = '%s_FCN' %(synapse_type)
else:
    model_prefix = '%s_TCN' %(synapse_type)
network_average_width = int(np.array(num_filters_per_layer).mean())
time_window_T = (np.array(filter_sizes_per_layer) - 1).sum() + 1
architecture_overview = 'DxWxT_%dx%dx%d' %(network_depth,network_average_width,time_window_T)
start_learning_schedule = 0
num_training_samples = 0


print('-----------------------------------------------')
print('about to start training...')
print('-----------------------------------------------')
print(model_prefix)
print(architecture_overview)
print('-----------------------------------------------')

#%% train

num_learning_schedules = len(batch_size_per_epoch)

training_history_dict = {}
for learning_schedule in range(start_learning_schedule, num_learning_schedules):
    epoch_start_time = time.time()
        
    batch_size    = batch_size_per_epoch[learning_schedule]
    learning_rate = learning_rate_per_epoch[learning_schedule]
    loss_weights  = loss_weights_per_epoch[learning_schedule]
    
    # prepare data generators
    if learning_schedule == 0 or (learning_schedule >= 1 and batch_size != batch_size_per_epoch[learning_schedule -1]):
        print('initializing generators')
        train_data_generator = SimulationDataGenerator(train_files, num_files_per_epoch=train_files_per_epoch, batch_size=batch_size,
                                                       window_size_ms=input_window_size, file_load=train_file_load, DVT_PCA_model=DVT_PCA_model)
        valid_data_generator = SimulationDataGenerator(valid_files, num_files_per_epoch=valid_files_per_epoch, batch_size=batch_size,
                                                       window_size_ms=input_window_size, file_load=valid_file_load, DVT_PCA_model=DVT_PCA_model)
    
    train_steps_per_epoch = len(train_data_generator)
    
    optimizer_to_use = Nadam(lr=learning_rate)
    temporal_conv_net.compile(optimizer=optimizer_to_use, loss=['binary_crossentropy','mse','mse'], loss_weights=loss_weights)
    
    print('-----------------------------------------------')
    print('starting epoch %d:' %(learning_schedule))
    print('-----------------------------------------------')
    print('loss weights = %s' %(str(loss_weights)))
    print('learning_rate = %.7f' %(learning_rate))
    print('batch_size = %d' %(batch_size))
    print('-----------------------------------------------')
    
    history = temporal_conv_net.fit_generator(generator=train_data_generator,
                                              epochs=num_steps_multiplier,
                                              validation_data=valid_data_generator,
                                              use_multiprocessing=use_multiprocessing, workers=num_workers)
    
    # store the loss values in training histogry dictionary and add some additional fields about the training schedule
    try:
        for key in history.history.keys():
            training_history_dict[key] += history.history[key]
        training_history_dict['learning_schedule'] += [learning_schedule] * num_steps_multiplier
        training_history_dict['batch_size']        += [batch_size] * num_steps_multiplier
        training_history_dict['learning_rate']     += [learning_rate] * num_steps_multiplier
        training_history_dict['loss_weights']      += [loss_weights] * num_steps_multiplier
        training_history_dict['num_train_samples'] += [batch_size * train_steps_per_epoch] * num_steps_multiplier
        training_history_dict['num_train_steps']   += [train_steps_per_epoch] * num_steps_multiplier
        training_history_dict['train_files_histogram'] += [train_data_generator.batches_per_file_dict]
        training_history_dict['valid_files_histogram'] += [valid_data_generator.batches_per_file_dict]
    except:
        for key in history.history.keys():
            training_history_dict[key] = history.history[key]
        training_history_dict['learning_schedule'] = [learning_schedule] * num_steps_multiplier
        training_history_dict['batch_size']        = [batch_size] * num_steps_multiplier
        training_history_dict['learning_rate']     = [learning_rate] * num_steps_multiplier
        training_history_dict['loss_weights']      = [loss_weights] * num_steps_multiplier
        training_history_dict['num_train_samples'] = [batch_size * train_steps_per_epoch] * num_steps_multiplier
        training_history_dict['num_train_steps']   = [train_steps_per_epoch] * num_steps_multiplier
        training_history_dict['train_files_histogram'] = [train_data_generator.batches_per_file_dict]
        training_history_dict['valid_files_histogram'] = [valid_data_generator.batches_per_file_dict]

    num_training_samples = num_training_samples + num_steps_multiplier * train_steps_per_epoch * batch_size
    
    print('-----------------------------------------------------------------------------------------')
    epoch_duration_sec = time.time() - epoch_start_time
    print('total time it took to calculate epoch was %.3f seconds (%.3f batches/second)' %(epoch_duration_sec, float(train_steps_per_epoch * num_steps_multiplier) / epoch_duration_sec))
    print('-----------------------------------------------------------------------------------------')
    
    # save model every once and a while
    if np.array(training_history_dict['val_spikes_loss'][-3:]).mean() < 0.03:
        model_ID = np.random.randint(100000)
        modelID_str = 'ID_%d' %(model_ID)
        train_string = 'samples_%d' %(num_training_samples)
        if len(training_history_dict['val_spikes_loss']) >= 10:
            train_MSE = 10000 * np.array(training_history_dict['spikes_loss'][-7:]).mean()
            valid_MSE = 10000 * np.array(training_history_dict['val_spikes_loss'][-7:]).mean()
        else:
            train_MSE = 10000 * np.array(training_history_dict['spikes_loss']).mean()
            valid_MSE = 10000 * np.array(training_history_dict['val_spikes_loss']).mean()
            
        results_overview = 'LogLoss_train_%d_valid_%d' %(train_MSE,valid_MSE)
        current_datetime = str(pd.datetime.now())[:-10].replace(':','_').replace(' ','__')
        model_filename    = models_dir + '%s__%s__%s__%s__%s__%s.h5' %(model_prefix,architecture_overview,current_datetime,train_string,results_overview,modelID_str)
        auxilary_filename = models_dir + '%s__%s__%s__%s__%s__%s.pickle' %(model_prefix,architecture_overview,current_datetime,train_string,results_overview,modelID_str)

        print('-----------------------------------------------------------------------------------------')
        print('finished epoch %d/%d. saving...\n     "%s"\n     "%s"' %(learning_schedule +1, num_epochs, model_filename.split('/')[-1], auxilary_filename.split('/')[-1]))
        print('-----------------------------------------------------------------------------------------')

        temporal_conv_net.save(model_filename)
        
        # save all relevent training params (in raw and unprocessed way)
        model_hyperparams_and_training_dict = {}
        model_hyperparams_and_training_dict['data_dict']              = data_dict
        model_hyperparams_and_training_dict['architecture_dict']      = architecture_dict
        model_hyperparams_and_training_dict['learning_schedule_dict'] = learning_schedule_dict
        model_hyperparams_and_training_dict['training_history_dict']  = training_history_dict
        
        pickle.dump(model_hyperparams_and_training_dict, open(auxilary_filename, "wb"), protocol=2)

#%% show learning curves

# gather losses
train_spikes_loss_list    = training_history_dict['spikes_loss']
valid_spikes_loss_list    = training_history_dict['val_spikes_loss']
train_somatic_loss_list   = training_history_dict['somatic_loss']
valid_somatic_loss_list   = training_history_dict['val_somatic_loss']
train_dendritic_loss_list = training_history_dict['dendritic_loss']
valid_dendritic_loss_list = training_history_dict['val_dendritic_loss']
train_total_loss_list     = training_history_dict['loss']
valid_total_loss_list     = training_history_dict['val_loss']

learning_epoch_list        = training_history_dict['learning_schedule']
batch_size_list            = training_history_dict['batch_size']
learning_rate              = training_history_dict['learning_rate']
loss_spikes_weight_list    = [x[0] for x in training_history_dict['loss_weights']]
loss_soma_weight_list      = [x[1] for x in training_history_dict['loss_weights']]
loss_dendrites_weight_list = [x[2] for x in training_history_dict['loss_weights']]

num_iterations = list(range(len(train_spikes_loss_list)))

