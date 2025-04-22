#! /usr/bin/env python
import os
import sys
import HeST as hest
import HeST.Amherst_split_cpd_with_numba as examp
import numpy as np
import matplotlib.pyplot as plt
import HeST.Detection as detection
from numba import jit
# from tqdm import tqdm
from analysis.analysis_functions import *
import astropy.stats as astat
from scipy.interpolate import interp1d
import pickle
import scipy
import glob
import pandas




trial_list = glob.glob('/work/pi_shertel_umass_edu/quasiparticle_simulation/waveform_comparison/further_zoom/trial_*.pkl')
config_list = glob.glob('/work/pi_shertel_umass_edu/quasiparticle_simulation/waveform_comparison/further_zoom/config_*.npy')
matched_pairs = get_file_map(config_list=config_list, trial_list=trial_list)
template_1 = np.load('/home/cveihmeyer_umass_edu/HeST/data/dispersion_curves/extreme_deposition.npy')
chi_squared = np.ones((149, 2), dtype=float)
for num, (config, trial) in matched_pairs.items():
    print(num)
    with open(trial, 'rb') as file:
        evap = pickle.load(file)
    s1, s2 , time =generate_waveform(evap)
    peak = np.max(s1)
    peak_index_signal = np.argmax(s1)
    peak_index_template = np.argmax(template_1)
    diff = peak_index_signal - peak_index_template
    before_pad = int(diff)
    after_pad = len(s1) - len(template_1) -before_pad
    try:

        new_template = np.pad(template_1, (before_pad, after_pad), mode='constant', constant_values=0)  * peak
        chi_squared[num, 0] = np.sum((s1[:1600] - (new_template[:1600]))**2)
        chi_squared[num, 1] = np.sum((s2[:1600] - (new_template[:1600]))**2)
    except:
        print(f"{num} failed while evaluating the new template")
    np.save('/home/cveihmeyer_umass_edu/HeST/data/further_zoom.npy', chi_squared)
