#! /usr/bin/env python
import os, pickle, argparse, pandas

import HeST as hest
import HeST.optimize_6_10 as examp
# import HeST.Amherst_split_cpd as examp_test
import numpy as np
import matplotlib.pyplot as plt
import HeST.Detection as detection

from numba import jit
from scipy.interpolate import interp1d



if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parser = argparse.ArgumentParser(
        description = "allows configuration of script before-hand"
    )
    parser.add_argument('--diff_prob', type=float, help = 'Diffuse Probability')
    parser.add_argument('--refl_prob', type=float, help = 'Reflection Probability')
    parser.add_argument('--evap_eff', type=str, help = 'Evaporation Probability')
    parser.add_argument('--num_qps', type=int, help = 'Number of Quasiparticles')
    parser.add_argument('--file_path', required=True, type=str, help = 'File path of saved data')
    parser.add_argument('--pos', type=str, help='Starting position of quasiparticles')
    args = parser.parse_args()


    diff_prob = 0.8
    refl_prob = 0.2
    evap_eff = [0.6, 0.3, 0.6]
    num_qps = 10000   
    detector = examp.Amherst_split_cpd
    if args.diff_prob:
        diff_prob = args.diff_prob
    if args.refl_prob:
        refl_prob = args.refl_prob
    if args.evap_eff:
        evap_eff = np.fromstring(args.evap_eff[:-1], sep=',', dtype=float)
    if args.num_qps:
        num_qps = args.num_qps
    file_path = args.file_path
    detector.set_QP_reflection_prob(refl_prob)
    detector.set_diffuse_prob(diff_prob)
    detector.set_evaporation_eff(evap_eff)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    num_points=1
    print(args.pos)
    pos = np.fromstring(args.pos[:-1], sep=',', dtype=float)
    print(pos)
    print(tuple(pos))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    evap = hest.GetEvaporationSignal(detector, num_qps, *pos, useMap=False, verbose=False, flavor_switching=True, save_hits_and_paths=False)
    # then we want to be able save this class
    # will use pickle to do this. Then we can write a post-processing script to move this into a separate file area. 
    with open(file_path, 'wb+') as f:
        pickle.dump(evap, f)

    # need to think harder about how to manage this. split on the file name, and go to the last bit
    listed_path = file_path.split('/')[:-1]
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID')) 
    listed_path.append(f'config_{task_id -1}.npy')
    p = os.path.join(*listed_path)
    p = '/' + p
    if not os.path.exists(p):
        config = {}
        config['diff_prob'] = diff_prob
        config['refl_prob'] = refl_prob
        config['evap_eff'] = evap_eff
        config['num_qps'] = num_qps
        config['pos'] = pos
        np.save(p, config)
