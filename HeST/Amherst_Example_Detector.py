import HeST.Detection as detection
import os
import numpy as np

#The detector geometry is defined from the point of view of particle paths.
# We essentially want to define various "surface conditions" where the particle paths are obstructed
# These functions also carry a "boundary_type", so that we can keep track if the particle is obstructed by
# a CPD, or a wall, and how it may reflect off of a given wall.

def sensor_conditions(x, y, z):
    boundary_type = "CPD0"
    radius = 3.8
    height = 3.3
    return (x*x + y*y < radius*radius) & (z < height) | (x*x + y*y >= radius*radius) , boundary_type

baseline_noise = [0., 0.]
phonon_conversion = 0.25
cpd = detection.VCPD(sensor_conditions, baseline_noise, phonon_conversion)


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
    
DetectorExample_Amherst = detection.VDetector( [wall_conditions, bottom_conditions], liquid_surface=liquid_surface, liquid_conditions=liquid_conditions, adsorption_gain=6.0e-3, evaporation_eff=0.60, CPDs=[cpd] )
DetectorExample_Amherst.load_LCEmap(os.path.dirname(__file__)+'/LCE_map_amherstExample_41x41x30_noReflections.npy')
DetectorExample_Amherst.set_LCEmap_positions([np.linspace(-3., 3., 41), np.linspace(-3., 3., 41), np.linspace(0., 2.75, 30)])
DetectorExample_Amherst.load_QPEmap(os.path.dirname(__file__)+'/QP_map_amherstExample_41x41x30_noReflections.npy')
DetectorExample_Amherst.set_QPEmap_positions([np.linspace(-3., 3., 41), np.linspace(-3., 3., 41), np.linspace(0., 2.75, 30)])



