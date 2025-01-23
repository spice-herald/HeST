#! /usr/bin/env python
import os
import HeST as hest
import HeST.Amherst_split_cpd_with_numba as examp
# import HeST.Amherst_split_cpd as examp_test
import numpy as np
import matplotlib.pyplot as plt
import HeST.Detection as detection
from numba import jit
from scipy.interpolate import interp1d
import argparse
import pickle


# first we need to create the detector object. 

if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parser = argparse.ArgumentParser(
        description = "allows configuration of script before-hand"
    )
    parser.add_argument('--diff_prob', type=float, help = 'Diffuse Probability')
    parser.add_argument('--refl_prob', type=float, help = 'Reflection Probability')
    parser.add_argument('--evap_eff', type=float, help = 'Evaporation Probability')
    parser.add_argument('--num_qps', type=int, help = 'Number of Quasiparticles')
    parser.add_argument('--file_path', required=True, type=str, help = 'File path of saved data')
    args = parser.parse_args()


    diff_prob = 0.8
    refl_prob = 0.3
    evap_eff = 0.6
    num_qps = 10000   
    detector = examp.Amherst_split_cpd
    if args.diff_prob:
        diff_prob = args.diff_prob
    if args.refl_prob:
        refl_prob = args.refl_prob
    if args.evap_eff:
        evap_eff = args.evap_eff
    if args.num_qps:
        num_qps = args.num_qps
       
    file_path = args.file_path
    detector.set_QP_reflection_prob(refl_prob)
    detector.set_diffuse_prob(diff_prob)
    detector.set_evaporation_eff(evap_eff)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    num_points=1
    rng = np.random.default_rng()

    r = rng.power(2, size=num_points) * 2.9
    theta = np.random.uniform(0, 2 * np.pi, size=num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(0, 1.48, size=num_points)
    pos = (0,0,.5)


    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    evap = hest.GetEvaporationSignal(detector, num_qps, *pos, useMap=False, verbose=False, flavor_switching=True)
    # then we want to be able save this class
    # will use pickle to do this. Then we can write a post-processing script to move this into a separate file area. 
    with open(os.path.join('/home/cveihmeyer_umass_edu/HeST/data', file_path), 'wb+') as f:
        pickle.dump(evap, f)

    if not os.path.exists(os.path.join('/home/cveihmeyer_umass_edu/HeST/data', file_path)):
        config = {}
        config['diff_prob'] = 0.8
        config['refl_prob'] = 0.3
        config['evap_eff'] = 0.6
        config['num_qps'] = 10000
        with open("config.txt", "w+") as file:
            for key, value in config.items():
                file.write(f"{key}={value}\n")