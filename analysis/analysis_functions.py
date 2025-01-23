# this is just going to be a file to house the useful functions for analysis
import matplotlib.pyplot as plt
import numpy as np

def plot_stacked_hist(evap):
    """ 
    Function to plot histogram of cpd hits, organized by number of bounces

    Args:
        evap (evaporation object): The return object of a detector event. 
    returns: 
        Plot
    """
    cpd_1_ints = np.unique(evap.bounce_flag[0])

    cpd_2_ints = np.unique(evap.bounce_flag[1])
    masks_cpd_1 = np.empty((len(cpd_1_ints), len(evap.bounce_flag[0])), dtype = int)
    masks_cpd_2 = np.empty((len(cpd_2_ints), len(evap.bounce_flag[1])), dtype = int)
    cpd1_times = evap.arrivalTimes_us[0]
    cpd2_times = evap.arrivalTimes_us[1]
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1,2,figsize =(16, 4))
    for ii, bounce_num in enumerate(cpd_1_ints):
        mask = evap.bounce_flag[0] == bounce_num 
        ax1.hist(cpd1_times[mask],bins = 200, range = [0,3000], alpha= 0.9, label = 'cpd1, bounce = ' + str(bounce_num - 1))
    ax1.set_title('Arrival times, with bounce')
    ax1.set_xlabel('Time [us]')
    ax1.legend()
    for ii, bounce_num in enumerate(cpd_2_ints):
        mask = evap.bounce_flag[1] == bounce_num
        ax2.hist(cpd2_times[mask], bins = 200, range = [0.0, 3000.0],alpha = 0.9, label= 'cpd2, bounce = ' + str(bounce_num - 1) )
    ax2.set_title('Arrival Times, with bounce')
    ax2.set_xlabel('Time [us]')
    ax2.legend()



def plot_hist_flavors(evap):
    """ Makes two subplots that use the number of bounces as labels

    Args:
        evap (evaporation class): The return object of a detector event. 
    """
    fs = evap.flavor
    fs = evap.flavor
    fig, axs = plt.subplots(1,2, figsize =(16, 4))
    for i in range(2):
        for value in np.unique(fs[i]):
            mask = (fs[i] == value)
            print(mask)
            axs[i].hist(evap.arrivalTimes_us[i][mask], bins = 200, range = [0,3000], alpha= 0.7,stacked=True, label = value)
            axs[i].set_title(f'CPD {i+1}')
            axs[i].legend()
    
def plot_hist_flavors(evap):
    """ Makes two subplots that use the flavor as the labels

    Args:
        evap (evaporation class): The return object of a detector event.  
    """
    fs = evap.flavor
    fig, axs = plt.subplots(1,2, figsize =(16, 4))
    for i in range(2):
        for value in np.unique(fs[i]):
            mask = (fs[i] == value)
            print(mask)
            axs[i].hist(evap.arrivalTimes_us[i][mask], bins = 200, range = [0,3000], alpha= 0.7,stacked=True, label = value)
            axs[i].set_title(f'CPD {i}')
            axs[i].legend(loc = 'upper right')