import HeST.Detection as detection
import os
import numpy as np

#The detector geometry is defined from the point of view of particle paths.
# We essentially want to define various "surface conditions" where the particle paths are obstructed
# These functions also carry a "boundary_type", so that we can keep track if the particle is obstructed by
# a CPD, or a wall, and how it may reflect off of a given wall.
def sensor1_conditions(x, y, z):
    boundary_type = "CPD0"
    CPDthickness = .1 #cm
    CPD_length = 1 # cm
    det_Zoffset = 3.0 #cm
    cal_Zoffset = -3.8743 #cm
    CPD_XYpitch = 1.373 #cm
    CPD_Zpitch = 3.878 #cm
    CPD1 = ( (x < CPD_length/2.0 + CPD_XYpitch/2.0) & (x > -CPD_length/2.0 + CPD_XYpitch/2.0) & (y < CPD_length/2.0 + CPD_XYpitch/2.0) & (y > -CPD_length/2.0 + CPD_XYpitch/2.0) & (z > det_Zoffset + cal_Zoffset - CPDthickness/2.0) & (z < det_Zoffset + cal_Zoffset + CPDthickness/2.0) )
    return ~CPD1, boundary_type
    
def sensor2_conditions(x, y, z):
    boundary_type = "CPD1"
    CPDthickness = .1 #cm
    CPD_length = 1 # cm
    det_Zoffset = 3.0 #cm
    cal_Zoffset = -3.8743 #cm
    CPD_XYpitch = 1.373 #cm
    CPD_Zpitch = 3.878 #cm
    CPD2 = ( (x < CPD_length/2.0 - CPD_XYpitch/2.0) & (x > -CPD_length/2.0 - CPD_XYpitch/2.0) & (y < CPD_length/2.0 + CPD_XYpitch/2.0) & (y > -CPD_length/2.0 + CPD_XYpitch/2.0) & (z > det_Zoffset + cal_Zoffset - CPDthickness/2.0) & (z < det_Zoffset + cal_Zoffset + CPDthickness/2.0) )
    return ~CPD2, boundary_type
    
def sensor3_conditions(x, y, z):
    boundary_type = "CPD2"
    CPDthickness = .1 #cm
    CPD_length = 1 # cm
    det_Zoffset = 3.0 #cm
    cal_Zoffset = -3.8743 #cm
    CPD_XYpitch = 1.373 #cm
    CPD_Zpitch = 3.878 #cm
    CPD3 = ( (x < CPD_length/2.0 + CPD_XYpitch/2.0) & (x > -CPD_length/2.0 + CPD_XYpitch/2.0) & (y < CPD_length/2.0 - CPD_XYpitch/2.0) & (y > -CPD_length/2.0 - CPD_XYpitch/2.0) & (z > det_Zoffset + cal_Zoffset - CPDthickness/2.0) & (z < det_Zoffset + cal_Zoffset + CPDthickness/2.0) )
    return ~CPD3, boundary_type
    
def sensor4_conditions(x, y, z):
    boundary_type = "CPD3"
    CPDthickness = .1 #cm
    CPD_length = 1 # cm
    det_Zoffset = 3.0 #cm
    cal_Zoffset = -3.8743 #cm
    CPD_XYpitch = 1.373 #cm
    CPD_Zpitch = 3.878 #cm
    CPD4 = ( (x < CPD_length/2.0 - CPD_XYpitch/2.0) & (x > -CPD_length/2.0 - CPD_XYpitch/2.0) & (y < CPD_length/2.0 - CPD_XYpitch/2.0) & (y > -CPD_length/2.0 - CPD_XYpitch/2.0) & (z > det_Zoffset + cal_Zoffset - CPDthickness/2.0) & (z < det_Zoffset + cal_Zoffset + CPDthickness/2.0) )
    return ~CPD4, boundary_type
    
def sensor5_conditions(x, y, z):
    boundary_type = "CPD4"
    CPDthickness = .1 #cm
    CPD_length = 1 # cm
    det_Zoffset = 3.0 #cm
    cal_Zoffset = -3.8743 #cm
    CPD_XYpitch = 1.373 #cm
    CPD_Zpitch = 3.878 #cm
    CPD5 = ( (x < CPD_length/2.0 + CPD_XYpitch/2.0) & (x > -CPD_length/2.0 + CPD_XYpitch/2.0) & (y < CPD_length/2.0 + CPD_XYpitch/2.0) & (y > -CPD_length/2.0 + CPD_XYpitch/2.0) & (z > det_Zoffset + cal_Zoffset - CPD_Zpitch - CPDthickness/2.0) & (z < det_Zoffset + cal_Zoffset - CPD_Zpitch + CPDthickness/2.0) )
    return ~CPD5, boundary_type
    
def sensor6_conditions(x, y, z):
    boundary_type = "CPD5"
    CPDthickness = .1 #cm
    CPD_length = 1 # cm
    det_Zoffset = 3.0 #cm
    cal_Zoffset = -3.8743 #cm
    CPD_XYpitch = 1.373 #cm
    CPD_Zpitch = 3.878 #cm
    CPD6 = ( (x < CPD_length/2.0 - CPD_XYpitch/2.0) & (x > -CPD_length/2.0 - CPD_XYpitch/2.0) & (y < CPD_length/2.0 + CPD_XYpitch/2.0) & (y > -CPD_length/2.0 + CPD_XYpitch/2.0) & (z > det_Zoffset + cal_Zoffset - CPD_Zpitch - CPDthickness/2.0) & (z < det_Zoffset + cal_Zoffset - CPD_Zpitch + CPDthickness/2.0) )
    return ~CPD6, boundary_type
    
def sensor7_conditions(x, y, z):
    boundary_type = "CPD6"
    CPDthickness = .1 #cm
    CPD_length = 1 # cm
    det_Zoffset = 3.0 #cm
    cal_Zoffset = -3.8743 #cm
    CPD_XYpitch = 1.373 #cm
    CPD_Zpitch = 3.878 #cm
    CPD7 = ( (x < CPD_length/2.0 + CPD_XYpitch/2.0) & (x > -CPD_length/2.0 + CPD_XYpitch/2.0) & (y < CPD_length/2.0 - CPD_XYpitch/2.0) & (y > -CPD_length/2.0 - CPD_XYpitch/2.0) & (z > det_Zoffset + cal_Zoffset - CPD_Zpitch - CPDthickness/2.0) & (z < det_Zoffset + cal_Zoffset - CPD_Zpitch + CPDthickness/2.0) )
    return ~CPD7, boundary_type
    
def sensor8_conditions(x, y, z):
    boundary_type = "CPD7"
    CPDthickness = .1 #cm
    CPD_length = 1 # cm
    det_Zoffset = 3.0 #cm
    cal_Zoffset = -3.8743 #cm
    CPD_XYpitch = 1.373 #cm
    CPD_Zpitch = 3.878 #cm
    CPD8 = ( (x < CPD_length/2.0 - CPD_XYpitch/2.0) & (x > -CPD_length/2.0 - CPD_XYpitch/2.0) & (y < CPD_length/2.0 - CPD_XYpitch/2.0) & (y > -CPD_length/2.0 - CPD_XYpitch/2.0) & (z > det_Zoffset + cal_Zoffset - CPD_Zpitch - CPDthickness/2.0) & (z < det_Zoffset + cal_Zoffset - CPD_Zpitch + CPDthickness/2.0) )
    return ~CPD8, boundary_type
    
    

baseline_noise = [0., 0.]
phonon_conversion = 0.25
cpd1 = detection.VCPD(sensor1_conditions, baseline_noise, phonon_conversion)
cpd2 = detection.VCPD(sensor2_conditions, baseline_noise, phonon_conversion)
cpd3 = detection.VCPD(sensor3_conditions, baseline_noise, phonon_conversion)
cpd4 = detection.VCPD(sensor4_conditions, baseline_noise, phonon_conversion)
cpd5 = detection.VCPD(sensor5_conditions, baseline_noise, phonon_conversion)
cpd6 = detection.VCPD(sensor6_conditions, baseline_noise, phonon_conversion)
cpd7 = detection.VCPD(sensor7_conditions, baseline_noise, phonon_conversion)
cpd8 = detection.VCPD(sensor8_conditions, baseline_noise, phonon_conversion)

def wall_conditions(x, y, z):
    boundary_type = "XY"
    radius = 2.381 #cm
    bottomPos = -8.407 #cm
    topPos = -2.791 #cm
    det_Zoffset = 3.0 #cm
    Z_correction = 0 #cm
    return (x*x + y*y < radius*radius) , boundary_type

def top_conditions(x,y,z):
    boundary_type = "Z"
    radius = 2.381 #cm
    bottomPos = -8.407 #cm
    topPos = -2.791 #cm
    det_Zoffset = 3.0 #cm
    Z_correction = 0 #cm
    return (z < topPos + det_Zoffset + Z_correction), boundary_type

def bottom_conditions(x, y, z):
    boundary_type = "Z"
    radius = 2.381 #cm
    bottomPos = -8.407 #cm
    topPos = -2.791 #cm
    det_Zoffset = 3.0 #cm
    Z_correction = 0 #cm
    return (z > bottomPos + det_Zoffset + Z_correction), boundary_type


def liquid_surface(x, y, z):
    boundary_type = "Liquid"
    height = 2.75 #cm
    bottomPos = -8.407 #cm
    topPos = -2.791 #cm
    det_Zoffset = 3.0 #cm
    Z_correction = 0 #cm
    return (z < bottomPos + det_Zoffset + Z_correction + height), boundary_type

def liquid_conditions(x, y, z):
    height = 2.75 #cm
    radius = 2.381 #cm
    bottomPos = -8.407 #cm
    topPos = -2.791 #cm
    det_Zoffset = 3.0 #cm
    Z_correction = 0 #cm
    return ((x*x + y*y < radius*radius) & (z < bottomPos + det_Zoffset + Z_correction + height) & (z > bottomPos + det_Zoffset + Z_correction))

#create the detector using the conditions above
DetectorExample_LBNL = detection.VDetector( [wall_conditions, top_conditions, bottom_conditions], liquid_surface=liquid_surface, liquid_conditions=liquid_conditions,
                                adsorption_gain=6.0, evaporation_eff=0.60, CPDs=[cpd1, cpd2, cpd3, cpd4, cpd5, cpd6, cpd7, cpd8],
                                photon_reflection_prob=0., QP_reflection_prob=0.)

radius = 2.381
bottomPos = -8.407 #cm
topPos = -2.791 #cm
det_Zoffset = 3.0 #cm
Z_correction = 0 #cm
height = 2.75

nBins = 30
z_slices = np.linspace(det_Zoffset+Z_correction+bottomPos, det_Zoffset+Z_correction+bottomPos+height, nBins)

DetectorExample_LBNL.load_LCEmap(os.path.dirname(__file__)+'/LCE_map_lbnlExample_51x51x30_noReflections.npy')
DetectorExample_LBNL.set_LCEmap_positions([np.linspace(-radius, radius, 51), np.linspace(-radius, radius, 51), z_slices ])
DetectorExample_LBNL.load_QPEmap(os.path.dirname(__file__)+'/QP_map_lbnlExample_51x51x30_noReflections.npy')
DetectorExample_LBNL.set_QPEmap_positions([np.linspace(-radius, radius, 51), np.linspace(-radius, radius, 51), z_slices ])
