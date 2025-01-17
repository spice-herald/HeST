#! /cveihmeyer_umass_edu/bin/HeST python
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


# first we need to create the detector object. 

if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parser = argparse.ArgumentParser(
        description = "allows configuration of script before-hand"
    )
    parser.add_argument('--diff_prob', type=float, help = 'Diffuse Probability')
    parser.add_argument('--refl_prob', type=float, help = 'Reflection Probability')
    parser.add_argument('--evap_eff', type=float, help = 'Evaporation Probability')
    parser.add_argument('--num_qps', type=float, help = 'Number of Quasiparticles')
    parser.add_argument('--file_path', type=str, help = 'File path of saved data')
    args = parser.parse_args()
    
    detector = examp.Amherst_split_cpd
    input_params = [args.diff_prob, args.refl_prob, args.evap_eff, args.num_qps]
    default_params = [diff_prob, refl_prob, evap_eff, num_qps]

    for  in params:
        
    diff_prob = args.diff_prob
    refl_prob = args.refl_prob
    evap_eff = args.evap_eff
    num_qps = args.num_qps
    file_path = args.file_path



    print(diff_prob)
    print(refl_prob)
    print(evap_eff)
    print(num_qps)
    print(file_path)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    QP_conditions = detector.get_surface_conditions()
    nCPDs = detector.get_nCPDs()
    for i in range(nCPDs):
        QP_conditions.append( (detector.get_CPD(i)).get_surface_condition())
    QP_conditions.append( detector.liquid_surface )


    # now we define the parameters of interest


