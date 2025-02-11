# this is just going to be a file to house the useful functions for analysis
import matplotlib.pyplot as plt
import numpy as np
import pickle, re

def plot_stacked_hist(evap, title=None, plot_one=False):
    """ 
    Function to plot histogram of cpd hits, organized by number of bounces

    Args:
        evap (evaporation object): The return object of a detector event. 
    returns: 
        Plot
    """
    cpd_1_ints = np.unique(evap.bounce_flag[0] - 1)

    cpd_2_ints = np.unique(evap.bounce_flag[1] - 1)
    masks_cpd_1 = np.empty((len(cpd_1_ints), len(evap.bounce_flag[0])), dtype = int)
    masks_cpd_2 = np.empty((len(cpd_2_ints), len(evap.bounce_flag[1])), dtype = int)
    cpd1_times = evap.arrivalTimes_us[0]
    cpd2_times = evap.arrivalTimes_us[1]
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1,2,figsize =(16, 4))
    for ii, bounce_num in enumerate(cpd_1_ints):
        mask = evap.bounce_flag[0] -1 == bounce_num 
        ax1.hist(cpd1_times[mask],bins = 200, range = [0,3000], alpha= 0.9, label = 'cpd1, bounce = ' + str(bounce_num - 1))
    ax1.set_title('Arrival times, with bounce')
    if title is not None:
        ax1.set_title(title + ' cpd1')
    ax1.set_xlabel('Time [us]')
    ax1.legend()
    if plot_one:
        return 1
    for ii, bounce_num in enumerate(cpd_2_ints):
        mask = evap.bounce_flag[1] -1 == bounce_num
        ax2.hist(cpd2_times[mask], bins = 200, range = [0.0, 3000.0],alpha = 0.9, label= 'cpd2, bounce = ' + str(bounce_num - 1) )
    ax2.set_title('Arrival Times, with bounce')
    ax2.set_xlabel('Time [us]')
    if title is not None:
        ax2.set_title(title + 'cpd2')
    ax2.legend()



def plot_hist_flavors(evap, title=None, plot_one=False):
    """ Makes two subplots that use the number of bounces as labels

    Args:
        evap (evaporation class): The return object of a detector event. 
    """
    fs = evap.flavor
    
    fig, axs = plt.subplots(1,2, figsize =(16, 4))
    for i in range(2):
        if plot_one and i>0:
            break
        for value in np.unique(fs[i]):
            mask = (fs[i] == value)
            axs[i].hist(evap.arrivalTimes_us[i][mask], bins = 200, range = [0,3000], alpha= 0.7,stacked=True, label = value)
            axs[i].set_title(f'CPD {i+1}')
            if title is not None:
                axs[i].set_title(f'{title} CPD {i+1}' )
            axs[i].legend()
    
            
def generate_waveform(evap):
    """Generates the Waveform from incident events via a convolution of the template and histogram

    Args:
        evap (class): evaporation event class, which contains the arrival times of the many particles

    Returns:
        signal_1: CPD1 detector response
        signal_1: CPD2 detector response
        time: The time of the template
    """
    cpd1_arrival = evap.arrivalTimes_us[0]
    cpd2_arrival = evap.arrivalTimes_us[1]
    template = np.load('/home/cveihmeyer_umass_edu/HeST/data/dispersion_curves/shortened_template.npy')
    template_1 = template[0]
    template_2 = template[1]
    cpd1_hits = np.histogram(cpd1_arrival, bins=12500, range=(0,5000))[0] * 10e-3
    cpd2_hits = np.histogram(cpd2_arrival, bins=12500, range=(0,5000))[0] * 10e-3
    signal_1 = np.convolve(cpd1_hits, template_1)
    signal_2 = np.convolve(cpd2_hits, template_2)
    time = np.arange(0, (len(template_1) + len(cpd1_hits) -1) * 0.4e-6, 0.4e-6)
    return signal_1, signal_2, time

def plot_waveform(evap, title=''):
    signal_1, signal_2, time = generate_waveform(evap)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,4))
    
    ax1.plot(time, signal_1) 
    ax2.plot(time, signal_2) 

    ax1.set_title('CPD1'+title)
    ax2.set_title('CPD2'+title)





def extract_pulse(file_list, plot_one=False):
    heights = []
    for ii, file in enumerate(file_list):
        with open(file, 'rb') as f:
            height = pickle.load(f)
        
        plot_stacked_hist(height, title=file_list[ii][-14:-4], plot_one=plot_one)
        plot_hist_flavors(height,title=file_list[ii][-14:-4] , plot_one=plot_one)
        cpd1 = height.arrivalTimes_us[0]
        cpd2 = height.arrivalTimes_us[1]
        heights.append(cpd1)
        heights.append(cpd2)
    return heights



def extract_number(filename, prefix, delimiter):
    match = re.search(rf"{prefix}_(\d+)\.{delimiter}", filename)
    return int(match.group(1)) if match else None



def get_file_map(config_list, trial_list):
    """ maps all numbered trials to all number configs

    Args:
        config_list (_type_): _description_
        trial_list (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Create a dictionary mapping numbers to file paths
    config_map = {extract_number(f, "config", 'csv'): f for f in config_list}
    print(config_map)
    trial_map = {extract_number(f, "trial", 'pkl'): f for f in trial_list}
    print(trial_map)


    # Match based on extracted numbers
    matched_pairs = {num: (config_map.get(num), trial_map.get(num)) for num in set(config_map) & set(trial_map)}
    return matched_pairs