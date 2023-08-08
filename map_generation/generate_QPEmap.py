import HeST as hest
import numpy as np
import multiprocessing as mp

'''
This script takes an example detector, and generates an LCE map using multiprocessing
'''

def sensor_conditions(x, y, z):
    boundary_type = "CPD"
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
                                adsorption_gain=6.0, evaporation_eff=0.60, CPDs=[cpd],
                                photon_reflection_prob=0., QP_reflection_prob=0.)

#Define the map's attributes.... 
filestring = "QPE_map_amherstExample_41x41x30_noReflections"
nBins = [41, 41, 30]
radius = 3.
level = 2.75
x = np.linspace(-radius, radius, nBins[0])
y = np.linspace(-radius, radius, nBins[1])
z = np.linspace(0, level, nBins[2])
m = np.zeros((len(x), len(y), len(z)), dtype=float)

nQPs = 20000
reflection_prob = detector.get_QP_reflection_prob()

def fill_XY_array(zz):

    result = np.zeros((len(x), len(y)), dtype=float)
    conditions = detector.get_surface_conditions()
    conditions.append( detector.get_liquid_surface() )
    for i in range(detector.get_nCPDs()):
        conditions.append( (detector.get_CPD(i)).get_surface_condition() )
    for xx in range(len(x)):
        print("%i: %i / %i" % (zz, xx, len(x)))
        for yy in range(len(y)):

           hitProbs = 0
           pos = [x[xx], y[yy], z[zz]]
           #check if the position is within the liquid volume
           if detector.get_liquid_conditions()(*pos) == False:
               continue

           for n in range(nQPs):
               hit, total_time, n, xs, ys, zs, p, surf = hest.QP_propagation(pos, conditions, reflection_prob)
               hitProbs += hit

           hitProbs = hitProbs/nPhotons

           result[xx][yy] = hitProbs
    return result

with mp.Pool() as pool:
    # Map the fill_2D_matrix function to each z index in parallel
    results = pool.map(fill_XY_array, range(len(z)))

for z_index, result in enumerate(results):
    m[:, :, z_index] = result

np.save(filestring+".npy", m)
print("Saved map to %s.npy" % filestring)

