import HeST.Detection as detection
import os
import numpy as np
from numba import jit
#The detector geometry is defined from the point of view of particle paths.
# We essentially want to define various "surface conditions" where the particle paths are obstructed
# These functions also carry a "boundary_type", so that we can keep track if the particle is obstructed by
# a CPD, or a wall, and how it may reflect off of a given wall.
@jit
def sensor1_conditions(x, y, z):
    boundary_type = "CPD0"
    radius = 3.8
    height = 3.3
    return (x*x + y*y < radius*radius) & (z < height)| (x>=0) & (x*x + y*y < radius*radius) & (z>height) | (x*x + y*y >= radius*radius) , boundary_type
@jit
def sensor2_conditions(x, y, z):
    boundary_type = "CPD1"
    radius = 3.8
    height = 3.3
    return (x*x + y*y < radius*radius) &(z < height)| (x<0) & (x*x + y*y < radius*radius) & (z>height) | (x*x + y*y >= radius*radius) , boundary_type





baseline_noise = [0., 0.]
phonon_conversion = 0.25
cpd1 = detection.VCPD(sensor1_conditions, baseline_noise, phonon_conversion)
cpd2 = detection.VCPD(sensor2_conditions, baseline_noise, phonon_conversion)





@jit
def wall_conditions(x, y, z):
    boundary_type = "XY"
    radius = 3. #cm
    height = 1.0 #cm
    return ((x*x + y*y < radius*radius) & (z < height) ) | (z > height), boundary_type

@jit
def bottom_conditions(x, y, z):
    boundary_type = "Z"
    bottom = 0. #cm
    return (z > bottom), boundary_type

@jit
def liquid_surface(x, y, z):
    boundary_type = "Liquid"
    height = 1.0 #cm
    return (z < height), boundary_type

@jit
def liquid_conditions(x, y, z):
    height =  1.0 #cm
    radius = 3. #cm
    bottom = 0. #cm
    return ((x*x + y*y < radius*radius) & (z < height) & (z > bottom))
   

Amherst_split_cpd = detection.VDetector(wall_conditions=wall_conditions, bottom_conditions=bottom_conditions, liquid_surface=liquid_surface, liquid_conditions=liquid_conditions, CPDs=[cpd1, cpd2], adsorption_gain=6.0e-3, evaporation_eff=0.60)
 