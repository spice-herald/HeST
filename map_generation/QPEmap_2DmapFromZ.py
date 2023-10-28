import HeST as hest
import numpy as np
import multiprocessing as mp
import sys

'''
This script takes an example detector, and generates an LCE map using multiprocessing
You can either import an example from HeST, or you can define one, as commented out below
'''

detector = hest.LBNL_Example_Detector.DetectorExample_LBNL

'''
def sensor_conditions(x, y, z):
    boundary_type = "CPD0"
    radius = 3.8
    height = 3.3
    return (x*x + y*y < radius*radius) & (z < height) | (x*x + y*y >= radius*radius) , boundary_type

baseline_noise = [0., 0.]
phonon_conversion = 0.25
cpd = hest.VCPD(sensor_conditions, baseline_noise, phonon_conversion)


def wall_conditions(x, y, z):
    boundary_type = "XY"
    radius = 3. #cm
    height = 2.75 #cm
    return ((x*x + y*y < radius*radius) & (z < height) ) | (z > height), boundary_type

def bottom_conditions(x, y, z):
    boundary_type = "Z"
    bottom = 0. #cm
    return (z > bottom), boundary_type

def liquid_surface(x, y, z):
    boundary_type = "Liquid"
    height = 2.75 #cm
    return (z < height), boundary_type

def liquid_conditions(x, y, z):
    height = 2.75 #cm
    radius = 3. #cm
    bottom = 0. #cm
    return ((x*x + y*y < radius*radius) & (z < height) & (z > bottom))

#create the detector using the conditions above
detector = hest.VDetector( [wall_conditions, bottom_conditions], liquid_surface=liquid_surface, liquid_conditions=liquid_conditions,
                                adsorption_gain=6.0e-3, evaporation_eff=0.60, CPDs=[cpd],
                                photon_reflection_prob=0., QP_reflection_prob=0.)

'''

try:
    z_slice = float(sys.argv[1])
except:
    print("Error! This script requires a z position to be input at run time!")
    exit()

#Define the map's attributes....
#Define the map's attributes....
filestring = "QPE_map_lbnlExample_41x41_z"+str(z_slice)+"_noReflections"
nBins = [51, 51]
radius = 2.4
bottomPos = -8.407 #cm
topPos = -2.791 #cm
det_Zoffset = 3.0 #cm
Z_correction = 0 #cm
x = np.linspace(-radius, radius, nBins[0])
y = np.linspace(-radius, radius, nBins[1])



print("making map at z=%.5f" % z_slice)

nQPs = 200000
reflection_prob = detector.get_QP_reflection_prob()

def fill_XY_array(z_pos):
    nCPDs = detector.get_nCPDs()
    result = np.zeros((len(x), len(y), nCPDs), dtype=float)
    conditions = detector.get_surface_conditions()
    conditions.append( detector.liquid_surface )
    for i in range(nCPDs):
        conditions.append( (detector.get_CPD(i)).get_surface_condition() )
    for xx in range(len(x)):
        print("%.5f: %i / %i" % (z_pos, xx, len(x)))
        for yy in range(len(y)):

           hitProbs = [0.]*nCPDs
           pos = [x[xx], y[yy], z_pos]
           #check if the position is within the liquid volume
           if detector.get_liquid_conditions()(*pos) == False:
               continue

           for n in range(nQPs):
               hit, arrival_time, n, xs, ys, zs, p, surf, cpd_id = hest.QP_propagation(pos, conditions, detector.get_QP_reflection_prob(), evap_eff=detector.get_evaporation_eff())
               if hit > 0.5:
                   hitProbs[cpd_id] += 1.

           hitProbs = np.array(hitProbs)/nQPs

           result[xx][yy] = hitProbs
    return result

m = fill_XY_array(z_slice)

np.save(filestring+".npy", m)
print("Saved map to %s.npy" % filestring)
