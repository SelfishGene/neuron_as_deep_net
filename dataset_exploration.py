import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import time
import pickle
from scipy import signal

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

save_figures = True
all_file_endings_to_use = ['.png', '.pdf', '.svg']

# NOTE: during this project I've changed my coding style
# and was too lazy to edit the old code to match the new style
# so please ignore any style related wierdness
# thanks for not being petty about unimportant shit


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


#%% scrip params

model_string = 'NMDA'
# model_string = 'AMPA'
# model_string = 'AMPA_SK'

morphology_folder = '/Reseach/Single_Neuron_InOut/morphology/'
models_dir        = '/Reseach/Single_Neuron_InOut/best_models/'
data_dir          = '/Reseach/Single_Neuron_InOut/data/'

if model_string == 'NMDA':
    test_data_dir      = data_dir + 'L5PC_NMDA_test/'
    output_figures_dir = '/media/davidb/ExtraDrive1/PhD_research/Figures/NMDA/'
elif model_string == 'AMPA':
    test_data_dir      = data_dir + 'L5PC_AMPA_test/'
    output_figures_dir = '/media/davidb/ExtraDrive1/PhD_research/Figures/AMPA/'
elif model_string == 'AMPA_SK':
    test_data_dir      = data_dir + 'L5PC_AMPA_SK_test/'
    output_figures_dir = '/media/davidb/ExtraDrive1/PhD_research/Figures/AMPA_SK/'

print('-----------------------------------------------')
print('finding data and morphology')
print('-----------------------------------------------')

test_files = glob.glob(test_data_dir  + '*_128_simulationRuns*_6_secDuration_*')
morphology_filename = morphology_folder + 'morphology_dict.pickle'

print('morphology_filename  : "%s"' %(morphology_filename.split('/')[-1]))
print('number of test files is %d' %(len(test_files)))
print('-----------------------------------------------')

#%% load valid and test datasets

print('----------------------------------------------------------------------------------------')
print('loading testing files...')
test_file_loading_start_time = time.time()

# load test data
X_test , y_spike_test , y_soma_test = parse_multiple_sim_experiment_files(test_files)

test_file_loading_duration_min = (time.time() - test_file_loading_start_time) / 60
print('time took to load data is %.3f minutes' %(test_file_loading_duration_min))
print('----------------------------------------------------------------------------------------')

#%% load morphology and display DVTs

## load morphology

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

seg_ind_to_xyz_coords_map = {}
seg_ind_to_sec_ind_map = {}
for k in range(len(allSegmentsType)):
    curr_segment_ind = allSegments_seg_ind_within_sec_ind[k]
    if allSegmentsType[k] == 'basal':
        curr_section_ind = allSegments_SectionInd[k]
        seg_ind_to_xyz_coords_map[k] = all_basal_segment_coords[(curr_section_ind,curr_segment_ind)]
        seg_ind_to_sec_ind_map[k] = ('basal', curr_section_ind)
    elif allSegmentsType[k] == 'apical':
        curr_section_ind = allSegments_SectionInd[k] - len(all_basal_section_coords)
        seg_ind_to_xyz_coords_map[k] = all_apical_segment_coords[(curr_section_ind,curr_segment_ind)]
        seg_ind_to_sec_ind_map[k] = ('apical', curr_section_ind)
    else:
        print('error!')

## load dendritic voltage traces of single simulation file

sim_experiment_file = test_files[0]
experiment_dict = pickle.load(open(sim_experiment_file, "rb" ), encoding='latin1')

X_spikes, _, _ = parse_sim_experiment_file(sim_experiment_file)

# gather params
num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])
num_segments    = len(experiment_dict['Params']['allSegmentsType'])
sim_duration_ms = experiment_dict['Params']['totalSimDurationInSec'] * 1000

# collect X, y_spike, y_soma
sim_dict = experiment_dict['Results']['listOfSingleSimulationDicts'][0]

t_LR = sim_dict['recordingTimeLowRes']
t_HR = sim_dict['recordingTimeHighRes']
y_soma_LR  = np.zeros((sim_duration_ms,num_simulations))
y_nexus_LR = np.zeros((sim_duration_ms,num_simulations))
y_soma_HR  = np.zeros((sim_dict['somaVoltageHighRes'].shape[0],num_simulations))
y_nexus_HR = np.zeros((sim_dict['nexusVoltageHighRes'].shape[0],num_simulations))

y_DVTs  = np.zeros((num_segments,sim_duration_ms,num_simulations), dtype=np.float16)

# go over all simulations in the experiment and collect their results
for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
    y_nexus_LR[:,k] = sim_dict['nexusVoltageLowRes']
    y_soma_LR[:,k] = sim_dict['somaVoltageLowRes']
    y_nexus_HR[:,k] = sim_dict['nexusVoltageHighRes']
    y_soma_HR[:,k] = sim_dict['somaVoltageHighRes']
    y_DVTs[:,:,k] = sim_dict['dendriticVoltagesLowRes']

    output_spike_times = np.int32(sim_dict['outputSpikeTimes'])
    # put "voltage spikes" in low res
    y_soma_LR[output_spike_times,k] = 30

#%% show some colored DVTs with morphology colored with same segment color and soma and nexus voltage at the bottom

time_point_ranges_options = [[2900,3100],[2800,3200],[2700,3300],[2600,3400],[2400,3600],[500,5500]]

for k in range(1):

    # select random simulation to display
    num_spikes_per_sim = (y_soma_LR > 20).sum(axis=0)
    possible_simulations = np.nonzero(np.logical_and(num_spikes_per_sim >= 4, num_spikes_per_sim <= 30))[0]
    random_simulation = np.random.choice(possible_simulations, size=1)[0]
    
    for time_points_ranges in time_point_ranges_options:
        # choose time points to display
        # time_points_ranges = [500,5500]
        # time_points_ranges = [2400,3600]
        # time_points_ranges = [2600,3400]
        # time_points_ranges = [2700,3300]
        # time_points_ranges = [2800,3200]
        # time_points_ranges = [2900,3100]
        width_mult_factor = 1.2
        
        duration_ms = time_points_ranges[1] - time_points_ranges[0]
        
        section_index      = np.array(experiment_dict['Params']['allSegments_SectionInd'])
        distance_from_soma = np.array(experiment_dict['Params']['allSegments_SectionDistFromSoma'])
        is_basal           = np.array([x == 'basal' for x in experiment_dict['Params']['allSegmentsType']])
        
        dend_colors = section_index * 20 + distance_from_soma
        dend_colors = dend_colors / dend_colors.max()
        
        all_seg_inds = seg_ind_to_xyz_coords_map.keys()
        colors = plt.cm.jet(dend_colors)
        
        # assemble the colors for each dendritic segment
        colors_per_segment = {}
        for seg_ind in all_seg_inds:
            colors_per_segment[seg_ind] = colors[seg_ind]
        
        fig = plt.figure(figsize=(32,15))
        fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01,hspace=0.01, wspace=0.2)
        
        ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=1)
        ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)
        
        # plot the cell morphology
        ax1.set_axis_off()
        for key in all_seg_inds:
            seg_color = colors_per_segment[key]
            seg_line_width = width_mult_factor * np.array(seg_ind_to_xyz_coords_map[key]['d']).mean()
            seg_x_coords = seg_ind_to_xyz_coords_map[key]['x']
            seg_y_coords = seg_ind_to_xyz_coords_map[key]['y']
            
            ax1.plot(seg_x_coords,seg_y_coords,lw=seg_line_width,color=seg_color)
        
        # add black soma
        ax1.scatter(x=45.5,y=19.8,s=120,c='k')
        ax1.set_xlim(-180,235)
        ax1.set_ylim(-210,1200)
        
        # plot the DVTs
        dend_colors = section_index * 20 + distance_from_soma
        dend_colors = dend_colors / dend_colors.max()
        sorted_according_to_colors = np.argsort(dend_colors)
        delta_voltage = 700.0 / sorted_according_to_colors.shape[0]
        for k in sorted_according_to_colors:
            ax2.plot(t_LR, 150 + k * delta_voltage + y_DVTs[k,:,random_simulation].T ,c=colors[k], alpha=0.55)
        
        # plot the soma and nexus traces
        ax2.plot(t_HR[:], 2.3 * y_soma_HR[:,random_simulation].T, c='black', lw=2.4)
        ax2.plot(t_HR[:], 2.3 * y_nexus_HR[:,random_simulation].T, c='red', lw=2.4)
        ax2.set_xlim(time_points_ranges[0],time_points_ranges[1])
        ax2.set_axis_off()
        
        if save_figures:
            figure_name = '%s__DVT_%d_ms_%d' %(model_string, duration_ms, np.random.randint(20))
            for file_ending in all_file_endings_to_use:
                if file_ending == '.png':
                    fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
                else:
                    subfolder = '%s/' %(file_ending.split('.')[-1])
                    fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

#%% show input spikes (X) and output somatic voltages (maybe use valens graylevel DVTs and green/red for ex,inh??)

xytick_labels_fontsize = 25
title_fontsize = 30
xylabels_fontsize = 27
legend_fontsize = 28

plt.close('all')

for k in range(1):

    num_spikes_per_sim = y_spike_test.sum(axis=0)
    possible_simulations = np.nonzero(np.logical_and(num_spikes_per_sim >= 3, num_spikes_per_sim <= 30))[0]
    random_simulation = np.random.choice(possible_simulations, size=1)[0]
    
    num_synapses = X_test.shape[0]
    num_ex_synapses  = int(num_synapses / 2)
    num_inh_synapses = int(num_synapses / 2)
    
    # covert binary matrix to dict representation
    all_presynaptic_spikes_bin = X_test[:,:,random_simulation]
    
    exc_presynaptic_spikes = X_test[:num_ex_synapses,:,random_simulation].T
    inh_presynaptic_spikes = X_test[num_ex_synapses:,:,random_simulation].T
    
    #ex_presynaptic_spikes = X_test[random_simulation,:num_ex_synapses,:]
    soma_voltage_trace = y_soma_test[:,random_simulation]
    soma_voltage_trace[soma_voltage_trace > -30] = 30  # fix spikes height (that occurs due to low resolution temporal sampling)
    
    # for raster plot (scatter)
    exc_syn_activation_time, exc_syn_activation_index = np.nonzero(exc_presynaptic_spikes)
    exc_syn_activation_time = exc_syn_activation_time / 1000.0
    
    inh_syn_activation_time, inh_syn_activation_index = np.nonzero(inh_presynaptic_spikes)
    inh_syn_activation_index = inh_syn_activation_index + num_ex_synapses
    inh_syn_activation_time = inh_syn_activation_time / 1000.0
    
    
    exc_instantaneous_firing_rate = exc_presynaptic_spikes.sum(axis=1)
    inh_instantaneous_firing_rate = inh_presynaptic_spikes.sum(axis=1)
    
    tau = 18
    exc_instantaneous_firing_rate = signal.convolve(exc_instantaneous_firing_rate , (1.0 / tau) * np.ones((tau,)), mode='same')
    inh_instantaneous_firing_rate = signal.convolve(inh_instantaneous_firing_rate , (1.0 / tau) * np.ones((tau,)), mode='same')
    
    sim_duration_sec = sim_duration_ms / 1000.0
    time_in_sec = np.arange(sim_duration_ms) / 1000.0
    
    
    fig = plt.figure(figsize=(25,18))
    gs = gridspec.GridSpec(4,1)
    gs.update(left=0.08, right=0.97, bottom=0.05, top=0.97, hspace=0.12)
    ax0 = plt.subplot(gs[0:2,0])
    ax1 = plt.subplot(gs[2,0])
    ax2 = plt.subplot(gs[3,0])
    
    ax0.scatter(exc_syn_activation_time, exc_syn_activation_index, s=2, c='r')
    ax0.scatter(inh_syn_activation_time, inh_syn_activation_index, s=2, c='b')
    ax0.set_xlim(0, sim_duration_sec -0.01)
    ax0.set_ylabel('synapse index \n', fontsize=xylabels_fontsize)
    ax0.grid('off')
    ax0.set_yticks([])
    ax0.set_xticks([])
    
    ax1.plot(time_in_sec, exc_instantaneous_firing_rate, c='r')
    ax1.plot(time_in_sec, inh_instantaneous_firing_rate, c='b')
    ax1.set_xlim(0,sim_duration_sec)
    ax1.set_ylabel('# input spikes\n per milisecond', fontsize=xylabels_fontsize)
    ax1.legend(['excitation', 'inhibition'], fontsize=legend_fontsize, loc='upper left', ncol=2)
    for tick_label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        tick_label.set_fontsize(xytick_labels_fontsize)
    ax1.set_xticks([])

    ax2.plot(time_in_sec, soma_voltage_trace, c='k')
    ax2.set_xlim(0,sim_duration_sec)
    ax2.set_ylabel('Voltage [mV]', fontsize=xylabels_fontsize)
    ax2.set_xlabel('Time [sec]', fontsize=xylabels_fontsize)
    for tick_label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        tick_label.set_fontsize(xytick_labels_fontsize)

    if save_figures:
        figure_name = '%s__input_output_%d' %(model_string, np.random.randint(50))
        for file_ending in all_file_endings_to_use:
            if file_ending == '.png':
                fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
            else:
                subfolder = '%s/' %(file_ending.split('.')[-1])
                fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

#%% create plot of "input ex/ing, bas/apic inst rate" and "output soma voltage" for several simulations

xytick_labels_fontsize = 25
title_fontsize = 30
xylabels_fontsize = 30
legend_fontsize = 30

plt.close('all')

for k in range(1):
    num_plots = 5
    selected_inds = np.random.choice(y_soma_test.shape[1],size=num_plots)
    
    # soma traces
    fig = plt.figure(figsize=(32,15))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.15)
    for k in range(num_plots):
        soma_voltage_trace = y_soma_test[:,selected_inds[k]]
        soma_voltage_trace[soma_voltage_trace > -30] = 30  # fix spikes height (that occurs due to low resolution temporal sampling)
        
        plt.subplot(num_plots, 1, k + 1)
        plt.plot(time_in_sec, soma_voltage_trace)
        plt.xlim(0,sim_duration_sec)
        if k == int(num_plots / 2):
            plt.ylabel('Soma Voltage [mV]', fontsize=xylabels_fontsize)

        if k == num_plots:
            plt.xticks(fontsize=xytick_labels_fontsize)
            plt.yticks(fontsize=xytick_labels_fontsize)
        else:
            plt.xticks([])
            plt.yticks([])

    plt.xlabel('Time [sec]', fontsize=xylabels_fontsize)

    if save_figures:
        figure_name = '%s__soma_traces_%d' %(model_string, np.random.randint(20))
        for file_ending in all_file_endings_to_use:
            if file_ending == '.png':
                fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
            else:
                subfolder = '%s/' %(file_ending.split('.')[-1])
                fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')
    
    # input traces
    num_segments = 639
    basal_cutoff = 262
    tuft_cutoff  = [366,559]
    
    ex_basal_syn_inds    = np.arange(basal_cutoff)
    ex_oblique_syn_inds  = np.hstack((np.arange(basal_cutoff,tuft_cutoff[0]),np.arange(tuft_cutoff[1],num_segments)))
    ex_tuft_syn_inds     = np.arange(tuft_cutoff[0],tuft_cutoff[1])
    inh_basal_syn_inds   = num_segments + ex_basal_syn_inds
    inh_oblique_syn_inds = num_segments + ex_oblique_syn_inds
    inh_tuft_syn_inds    = num_segments + ex_tuft_syn_inds
    
    tau = 18
    
    fig = plt.figure(figsize=(32,15))
    fig.subplots_adjust(left=0.07, bottom=0.05, right=0.95, top=0.9, wspace=0.05, hspace=0.1)
    for k in range(num_plots):
        ex_basal    = X_test[ex_basal_syn_inds,:,selected_inds[k]].sum(axis=0)
        ex_oblique  = X_test[ex_oblique_syn_inds,:,selected_inds[k]].sum(axis=0)
        ex_apical   = X_test[ex_tuft_syn_inds,:,selected_inds[k]].sum(axis=0)
        
        inh_basal   = X_test[inh_basal_syn_inds,:,selected_inds[k]].sum(axis=0)
        inh_oblique = X_test[inh_oblique_syn_inds,:,selected_inds[k]].sum(axis=0)
        inh_apical  = X_test[inh_tuft_syn_inds,:,selected_inds[k]].sum(axis=0)
    
        plt.subplot(num_plots,1,k + 1)
        plt.plot(time_in_sec, signal.convolve(ex_basal   , (1.0 / tau) * np.ones((tau,)), mode='same'), c='red')
        plt.plot(time_in_sec, signal.convolve(ex_oblique , (1.0 / tau) * np.ones((tau,)), mode='same'), c='orange')
        plt.plot(time_in_sec, signal.convolve(ex_apical  , (1.0 / tau) * np.ones((tau,)), mode='same'), c='yellow')
        plt.plot(time_in_sec, signal.convolve(inh_basal  , (1.0 / tau) * np.ones((tau,)), mode='same'), c='darkblue')
        plt.plot(time_in_sec, signal.convolve(inh_oblique, (1.0 / tau) * np.ones((tau,)), mode='same'), c='blue')
        plt.plot(time_in_sec, signal.convolve(inh_apical , (1.0 / tau) * np.ones((tau,)), mode='same'), c='skyblue')
        plt.xlim(0,sim_duration_sec)

        if k == num_plots:
            plt.xticks(fontsize=xytick_labels_fontsize)
            plt.yticks(fontsize=xytick_labels_fontsize)
        else:
            plt.xticks([])
            plt.yticks([])

        if k == 0:
            plt.legend(['exc basal','exc oblique','exc tuft','inh basal','inh oblique','inh tuft'],
                       bbox_to_anchor=(0.05, 1.27, 0.8, 0.25),
                       ncol=6, fontsize=legend_fontsize, loc='upper left')

        if k == int(num_plots / 2):
            plt.ylabel('Instantanous input rate \n[spikes per second per subtree]', fontsize=xylabels_fontsize)

    for k in range(num_plots):
        selected_ind = np.random.choice(y_soma_test.shape[1],size=1)[0]
        
        soma_voltage_trace = y_soma_test[:,selected_ind]
        soma_voltage_trace[soma_voltage_trace > -30] = 30  # fix spikes height (that occurs due to low resolution temporal sampling)
        
        plt.figure(figsize=(32,15))
        plt.subplots_adjust(left=0.07, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.1)
    
        ex_basal    = X_test[ex_basal_syn_inds,:,selected_ind].sum(axis=0)
        ex_oblique  = X_test[ex_oblique_syn_inds,:,selected_ind].sum(axis=0)
        ex_apical   = X_test[ex_tuft_syn_inds,:,selected_ind].sum(axis=0)
        
        inh_basal   = X_test[inh_basal_syn_inds,:,selected_ind].sum(axis=0)
        inh_oblique = X_test[inh_oblique_syn_inds,:,selected_ind].sum(axis=0)
        inh_apical  = X_test[inh_tuft_syn_inds,:,selected_ind].sum(axis=0)
        
        plt.subplot(2,1,1)
        plt.plot(time_in_sec, signal.convolve(ex_basal   , 1000 * (1.0 / tau) * np.ones((tau,)), mode='same'), c='red')[tau:]
        plt.plot(time_in_sec, signal.convolve(ex_oblique , 1000 * (1.0 / tau) * np.ones((tau,)), mode='same'), c='orange')[tau:]
        plt.plot(time_in_sec, signal.convolve(ex_apical  , 1000 * (1.0 / tau) * np.ones((tau,)), mode='same'), c='yellow')[tau:]
        plt.plot(time_in_sec, signal.convolve(inh_basal  , 1000 * (1.0 / tau) * np.ones((tau,)), mode='same'), c='darkblue')[tau:]
        plt.plot(time_in_sec, signal.convolve(inh_oblique, 1000 * (1.0 / tau) * np.ones((tau,)), mode='same'), c='blue')[tau:]
        plt.plot(time_in_sec, signal.convolve(inh_apical , 1000 * (1.0 / tau) * np.ones((tau,)), mode='same'), c='skyblue')[tau:]
        plt.ylabel('Instantanous input rate \n[spikes per second per subtree]', fontsize=xylabels_fontsize)
        plt.legend(['exc basal','exc oblique','exc tuft','inh basal','inh oblique','inh tuft'], ncol=2, fontsize=legend_fontsize, loc='upper left')
        plt.xlim(0,sim_duration_sec)
        plt.subplot(2,1,2)
        plt.plot(time_in_sec, soma_voltage_trace)
        plt.ylabel('Soma Voltage [mV]', fontsize=xylabels_fontsize)
        plt.xlim(0,sim_duration_sec)
        plt.xticks(fontsize=xytick_labels_fontsize)
        plt.yticks(fontsize=xytick_labels_fontsize)

    plt.xlabel('Time [sec]', fontsize=xylabels_fontsize)
    
    if save_figures:
        figure_name = '%s__inst_rates_%d' %(model_string, np.random.randint(50))
        for file_ending in all_file_endings_to_use:
            if file_ending == '.png':
                fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
            else:
                subfolder = '%s/' %(file_ending.split('.')[-1])
                fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')

#%% show histograms and scatter plots of input and output firing rates (x=ex, y=inh, c=output) for different time windows

plt.close('all')

for time_window_size_ms in [100,200,1000]:

    xytick_labels_fontsize = 34
    title_fontsize = 37
    xylabels_fontsize = 34
    legend_fontsize = 32
    
    half_window_size = int(time_window_size_ms / 2)
    
    selected_simulations = np.random.choice(X_test.shape[2], size=1000, replace=False)
    
    X_exc = X_test[:num_ex_synapses,:,selected_simulations].sum(axis=0).astype(np.float32)
    X_inh = X_test[num_ex_synapses:,:,selected_simulations].sum(axis=0).astype(np.float32)
    y_out = y_spike_test[:,selected_simulations]
    
    tau = time_window_size_ms
    
    filter_to_use = np.ones((tau,1))
    
    X_exc_aggregated = signal.convolve(X_exc, filter_to_use, mode='same')[half_window_size::half_window_size,:].ravel()
    X_inh_aggregated = signal.convolve(X_inh, filter_to_use, mode='same')[half_window_size::half_window_size,:].ravel()
    y_out_aggregated = signal.convolve(y_out, filter_to_use, mode='same')[half_window_size::half_window_size,:].ravel()
    
    # histograms
    fig = plt.figure(figsize=(32,15))
    fig.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, wspace=0.2, hspace=0.1)
    plt.suptitle('input and output histograms (window = %d ms)' %(time_window_size_ms), fontsize=title_fontsize)
    plt.subplot(1,3,1); plt.ylabel('Counts', fontsize=xylabels_fontsize)
    plt.hist(X_exc_aggregated, bins=100); plt.xlabel('num excitatory spikes per window', fontsize=xylabels_fontsize)
    xticks = plt.xticks()[0]
    yticks = plt.yticks()[0]
    plt.xticks(np.linspace(0,xticks[-1],5).astype(int), fontsize=xytick_labels_fontsize)
    plt.yticks(np.linspace(0,yticks[-1],5).astype(int), fontsize=xytick_labels_fontsize)

    plt.subplot(1,3,2)
    plt.hist(X_inh_aggregated, bins=100); plt.xlabel('num inhibitory spikes per window', fontsize=xylabels_fontsize)
    xticks = plt.xticks()[0]
    yticks = plt.yticks()[0]
    plt.xticks(np.linspace(0,xticks[-1],5).astype(int), fontsize=xytick_labels_fontsize)
    plt.yticks(np.linspace(0,yticks[-1],5).astype(int), fontsize=xytick_labels_fontsize)

    plt.subplot(1,3,3)
    if time_window_size_ms < 300:
        plt.hist(y_out_aggregated, bins=np.arange(1, 7) - 0.5); plt.xlabel('num output spikes per window', fontsize=xylabels_fontsize)
    else:
        plt.hist(y_out_aggregated, bins=np.arange(1,14) - 0.5); plt.xlabel('num output spikes per window', fontsize=xylabels_fontsize)
    plt.xticks(fontsize=xytick_labels_fontsize)
    plt.yticks(fontsize=xytick_labels_fontsize)

    if save_figures:
        figure_name = '%s__inout_rates_histograms_%d_ms' %(model_string, time_window_size_ms)
        for file_ending in all_file_endings_to_use:
            if file_ending == '.png':
                fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
            else:
                subfolder = '%s/' %(file_ending.split('.')[-1])
                fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')
    
    # scatter
    fig = plt.figure(figsize=(18,18))
    fig.subplots_adjust(left=0.15, bottom=0.10, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
    plt.scatter(x=X_inh_aggregated,y=X_exc_aggregated,c=y_out_aggregated, vmin=0, vmax=3, s=100, alpha=0.55)
    plt.ylabel('num excitatory spikes per window', fontsize=xylabels_fontsize)
    plt.xlabel('num inhibitory spikes per window', fontsize=xylabels_fontsize)
    plt.colorbar()
    plt.title('scatter plot (x=inh, y=exc, color=output)', fontsize=title_fontsize)
    xticks = plt.xticks()[0]
    yticks = plt.yticks()[0]
    plt.xticks(np.linspace(0,xticks[-1],5).astype(int), fontsize=xytick_labels_fontsize)
    plt.yticks(np.linspace(0,yticks[-1],5).astype(int), fontsize=xytick_labels_fontsize)

    if save_figures:
        figure_name = '%s__inout_rates_scatter_%d_ms' %(model_string, time_window_size_ms)
        for file_ending in all_file_endings_to_use:
            if file_ending == '.png':
                fig.savefig(output_figures_dir + figure_name + file_ending, bbox_inches='tight')
            else:
                subfolder = '%s/' %(file_ending.split('.')[-1])
                fig.savefig(output_figures_dir + subfolder + figure_name + file_ending, bbox_inches='tight')


#%%