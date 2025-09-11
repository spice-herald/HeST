from scipy.interpolate import interpn, interp1d
import numpy as np
import re
import matplotlib.pyplot as plt
from .HeST_Core import HestSignal, Random_QPmomentum, QP_dispersion, QP_velocity, get_phonon_mom_energy, get_rminus_mom_energy, get_rplus_mom_energy, phonon_momentum, rminus_momentum, rplus_momentum
from .HeST_Core import Singlet_PhotonEnergy
from numba import jit
import os
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class VSensor:
    """
    This class is a "virtual sensor". It contains information regarding a single detector's geometry and functionality.

    Attributes
    ----------
    surface_condition : function 
        a function that takes in an (x,y,z) position in cm and returns 
        true outside the photodetector volume and false inside it. 
        (A quasiparticle is said to be detected once it travels from a
        True region in to a False one).
    phononCollectionEfficiency : float
        The phonon collection efficiency of the sensor. (Fraction of 
        energy deposited into the Si that makes it into the TES).

    """
    def __init__(self, surface_condition, phononCollectionEfficiency=0.25, adsorptionGain = 6.0e-3):

        self.surface_condition = surface_condition
        self.phononCollectionEfficiency  = phononCollectionEfficiency
        self.adsoprtionGain = adsorptionGain


    def set_surface_condition(self, f1):
        self.surface_condition = f1
    def set_phononCollectionEfficiency(self, p1):
        self.phononCollectionEfficiency = p1
    def set_adsorptionGain(self, p1):
        self.adsoprtionGain = p1

    def get_surface_condition(self):
        return self.surface_condition
    def get_phononCollectionEfficiency(self):
        return self.phononCollectionEfficiency
    def get_adsorptionGain(self):
        return self.adsoprtionGain
    def check_surface(self, x, y, z):
        return self.surface_condition(x, y, z)



class VDetector:
    """
    This class is a "virtual Detector". It contains all of the geometric 
    and (custom) physics that will be going on in our detector. (Here "detector"
    refers to HeRALD as a whole, potentially comprising many individual sensor)

    Attributes
    ----------
    top_conditions : function
        a function that takes in an (x,y,z) position in cm and returns 
        true before a QP has struck the top surface and False when
        the the QP has pased through the top surface 
    bottom_conditions : function 
        a function that takes in an (x,y,z) position in cm and returns 
        true before a QP has struck the bottom surface and False when
        the the QP has pased through the bottom surface 
    wall_conditions : function 
        a function that takes in an (x,y,z) position in cm and returns 
        true before a QP has struck the lateral face and False when
        the the QP has passed through the lateral face 
    liquid_surface : function
        a function that takes in an (x,y,z) position in cm and returns 
        true when a QP is below the lHe level and false when 
        the QP has crossed into the vacuum
    liquid_conditions : function
        a function that takes in an (x,y,z) position in cm and returns 
        true when a QP is inside the lHe and false when 
        the QP has left the lHe volume 
    evaporation_eff : array
        length-7 array defining the probability of each QP population evaporating atoms
        (FIXME: After measurement, we'll want this defined in a LUT and not by 
        the user)
    sensors : list
        List of the VSensor objects you wish to add to your detector.
    LCEmap : array
        3-dimensional array with float-like elements. Each element describes 
        the probability of a photon at a given position being detected.
    LCEmap_positions : 3-tuple
        3-tuple of 1-D arrays (X Y Z) that correspond to the grid of points (in cm)
        that LCEmap measures light collection at
    QPEmap : array
        3-dimensional array with float-like elements. Each element describes 
        the probability of a phonon at a given position evaporating an atom that is 
        detected. 
    QPEmap_positions : 3-tuple
        3-tuple of 1-D arrays (X Y Z) that correspond to the grid of points (in cm)
        that QPEmap measures phonon collection at   
    UV/IR/QP_sensor_reflection_prob : float
        probability of a UV photon/IR photon/quasiparticle reflecting off of a sensor
        (In the latter case, this is for submerged sensors; evaporated atoms don't reflect)
    UV/IR/QP_sensor_diffuse_prob : float
        The probability of diffuse lambertian reflection off the sensor surface 
        given that reflection will occur
    UV/IR/QP_wall_reflection_prob : float
        probability of a UV photon/IR photon/quasiparticle reflecting off of a wall
        (In the latter case, this is for submerged sensors; evaporated atoms don't reflect)
    UV/IR/QP_wall_diffuse_prob : float
        The probability of diffuse lambertian reflection off a wall surface 
        given that reflection will occur

    """

    def __init__(self, top_conditions, bottom_conditions, wall_conditions, liquid_surface, liquid_conditions,
                 evaporation_eff = np.array([1,1,1,1,1,1,1]), sensors=[], LCEmap=0, LCEmap_positions=0., QPEmap=0, QPEmap_positions=0.,
                 UV_sensor_reflection_prob=0., UV_sensor_diffuse_prob=0., UV_wall_reflection_prob=0., UV_wall_diffuse_prob=0.,
                 IR_sensor_reflection_prob=0., IR_sensor_diffuse_prob=0., IR_wall_reflection_prob=0., IR_wall_diffuse_prob=0.,
                 QP_sensor_reflection_prob=0., QP_sensor_diffuse_prob=0., QP_sensor_Andreev_prob=0.,
                 QP_wall_reflection_prob=0., QP_wall_diffuse_prob=0., QP_wall_Andreev_prob=0.):
        

        
        self.top_condition     = top_conditions
        self.bottom_condition   = bottom_conditions
        self.wall_conditions    = wall_conditions
        self.liquid_surface     = liquid_surface
        self.liquid_conditions  = liquid_conditions
        self.sensors               = sensors
        self.evaporation_eff    = evaporation_eff
        self.LCEmap             = LCEmap
        self.LCEmap_positions   = LCEmap_positions
        self.QPEmap             = QPEmap
        self.QPEmap_positions   = QPEmap_positions
        
        self.UV_sensor_reflection_prob = UV_sensor_reflection_prob
        self.UV_sensor_diffuse_prob = UV_sensor_diffuse_prob
        self.UV_wall_reflection_prob = UV_wall_reflection_prob
        self.UV_wall_diffuse_prob = UV_wall_diffuse_prob

        self.IR_sensor_reflection_prob = IR_sensor_reflection_prob
        self.IR_sensor_diffuse_prob = IR_sensor_diffuse_prob
        self.IR_wall_reflection_prob = IR_wall_reflection_prob
        self.IR_wall_diffuse_prob = IR_wall_diffuse_prob

        self.QP_sensor_reflection_prob = QP_sensor_reflection_prob
        self.QP_sensor_diffuse_prob = QP_sensor_diffuse_prob
        self.QP_sensor_Andreev_prob = QP_sensor_Andreev_prob
        self.QP_wall_reflection_prob = QP_wall_reflection_prob
        self.QP_wall_diffuse_prob = QP_wall_diffuse_prob
        self.QP_wall_Andreev_prob = QP_wall_Andreev_prob

        
        
    """
    Setters
    """   

    def set_surface_conditions(self, f1):
        self.surface_conditions = list(f1)
    def add_surface_condition(self, f1):
        self.surface_conditions.append( f1 ) 
    def set_sensors(self, p1):
        self.sensors = list(p1)
    def add_sensors(self, p1):
        self.sensors.append( p1 ) 
    def set_liquid_surface(self, f1):
        self.liquid_surface = f1
    def set_liquid_conditions(self, f1):
        self.liquid_conditions = f1
    def set_evaporation_eff( self, p1 ):
        self.evaporation_eff = p1
    def set_LCEmap_positions(self, p1):
        if len(p1) == 3:
            self.LCEmap_positions = p1
        else: 
            print("This function requires a list of 3 individual 1-D arrays to associate the LCE map with X,Y,Z coordinates")
    def set_QPEmap_positions(self, p1):
        if len(p1) == 3:
            self.QPEmap_positions = p1
        else: 
            print("This function requires a list of 3 individual 1-D arrays to associate the QP evaporation map with X,Y,Z coordinates")

    def set_UV_sensor_reflection_prob(self, p1):
        self.UV_sensor_reflection_prob = p1
    def set_UV_sensor_diffuse_prob(self, p1):
        self.UV_sensor_diffuse_prob = p1
    def set_UV_wall_reflection_prob(self, p1):
        self.UV_wall_reflection_prob = p1
    def set_UV_wall_diffuse_prob(self, p1):
        self.UV_wall_diffuse_prob = p1

    def set_IR_sensor_reflection_prob(self, p1):
        self.IR_sensor_reflection_prob = p1
    def set_IR_sensor_diffuse_prob(self, p1):
        self.IR_sensor_diffuse_prob = p1
    def set_IR_wall_reflection_prob(self, p1):
        self.IR_wall_reflection_prob = p1
    def set_IR_wall_diffuse_prob(self, p1):
        self.IR_wall_diffuse_prob = p1

    def set_QP_sensor_reflection_prob(self, p1):
        self.QP_sensor_reflection_prob = p1
    def set_QP_sensor_diffuse_prob(self, p1):
        self.QP_sensor_diffuse_prob = p1
    def set_QP_sensor_Andreev_prob(self, p1):
        self.QP_sensor_Andreev_prob = p1
    def set_QP_wall_reflection_prob(self, p1):
        self.QP_wall_reflection_prob = p1
    def set_QP_wall_diffuse_prob(self, p1):
        self.QP_wall_diffuse_prob = p1
    def set_QP_wall_Andreev_prob(self, p1):
        self.QP_wall_Andreev_prob = p1


    """
    Getters
    """

    def get_surface_conditions(self):
        return list(self.surface_conditions)
    def get_liquid_surface(self):
        return self.liquid_surface
    def get_liquid_conditions(self):
        return self.liquid_conditions
    def get_up_conditions(self):
        return [self.liquid_surface, self.wall_conditions, self.top_condition]
    def get_down_conditions(self):
        return [self.wall_conditions, self.bottom_condition]
    def get_LCEmap(self):
        return self.LCEmap
    def get_LCEmap_positions(self):
        return self.LCEmap_positions
    def get_QPEmap(self):
        return self.LCEmap_positions
    def get_QPEmap_positions(self):
        return self.QPEmap_positions  
    def get_nsensors(self):
        return len(self.sensors)
    def get_sensor(self, index):
        return self.sensors[index]
    def get_liquid_surface( self ):
        return self.liquid_surface
    def get_evaporation_eff(self):
        return self.evaporation_eff
    
    def get_UV_sensor_reflection_prob(self):
        return self.UV_sensor_reflection_prob
    def get_UV_sensor_diffuse_prob(self):
        return self.UV_sensor_diffuse_prob
    def get_UV_wall_reflection_prob(self):
        return self.UV_wall_reflection_prob
    def get_UV_wall_diffuse_prob(self):
        return self.UV_wall_diffuse_prob

    def get_IR_sensor_reflection_prob(self):
        return self.IR_sensor_reflection_prob
    def get_IR_sensor_diffuse_prob(self):
        return self.IR_sensor_diffuse_prob
    def get_IR_wall_reflection_prob(self):
        return self.IR_wall_reflection_prob
    def get_IR_wall_diffuse_prob(self):
        return self.IR_wall_diffuse_prob

    def get_QP_sensor_reflection_prob(self):
        return self.QP_sensor_reflection_prob
    def get_QP_sensor_diffuse_prob(self):
        return self.QP_sensor_diffuse_prob
    def get_QP_sensor_Andreev_prob(self):
        return self.QP_sensor_Andreev_prob
    def get_QP_wall_reflection_prob(self):
        return self.QP_wall_reflection_prob
    def get_QP_wall_diffuse_prob(self):
        return self.QP_wall_diffuse_prob
    def get_QP_wall_Andreev_prob(self):
        return self.QP_wall_Andreev_prob

    


    def load_LCEmap(self, filename):
        self.LCEmap = np.load(filename)

    def load_QPEmap(self, filename):
        self.QPEmap = np.load(filename)

    def create_LCEmap(self, x_array, y_array, z_array, max_dist, step_size, nPhotons=10000, filestring="detector_LCEmap"):
        """
        Takes in an x/y/z grid and simulates a large number of
        photons at each one to simulate to get some detection probabilty

        Parameters
        ----------
        x-array : array
            Array of x positions (in cm) at which to simulate quasiparticles 
        y-array : array
            Array of y positions (in cm) at which to simulate quasiparticles 
        z-array : array
            Array of z positions (in cm) at which to simulate quasiparticles 
        nPhotons : int
            Number of quasiparticles to simulate at each point
        filestring : string
            Name for the collection efficiency map.

        """
        print("Creating LCE map for this detector geometry...")
        x, y, z = np.array(x_array), np.array(y_array), np.array(z_array)
        self.set_LCEmap_positions( [x, y, z] )
        nsensors = self.get_nsensors()
        m = np.zeros((len(x), len(y), len(z), nsensors), dtype=float)
        conditions = self.get_surface_conditions()
        for i in range(nsensors):
            conditions.append( (self.get_sensor(i)).get_surface_condition() )
        
        for xx in range(len(x)):
            print("%i / %i" % (xx, len(x)))
            for yy in range(len(y)):
                for zz in range(len(z)):
                    
                    pos = np.array([x[xx], y[yy], z[zz]])
                    if (self.get_liquid_conditions())(*pos) == False:
                        continue
                    hitProbs = [0.]*nsensors

                        
                    energyAtDeath, sensorIdsAll, \
                    total_time, orig_momenta, _ = photon_propagation(nPhotons, pos, self.get_up_conditions(), self.get_down_conditions(),
                                                                     wall_reflection_prob = self.get_UV_wall_reflection_prob(),
                                                                     wall_diffuse_prob = self.get_UV_wall_diffuse_prob(),
                                                                     sensor_reflection_prob= self.get_UV_sensor_reflection_prob(),
                                                                     sensor_diffuse_prob = self.get_UV_sensor_diffuse_prob(),
                                                                     max_dist = max_dist, step_size = step_size, debug = True)
                        
                    for i in range(nsensors):
                         mask = (sensorIdsAll == i)

                        
                                
                    hitProbs = np.array(hitProbs)/nPhotons
                    
                    m[xx][yy][zz] = hitProbs
                    #print(hitProbs)
                    
        self.LCEmap = m
        np.save(filestring+".npy", m)
        print("Saved map to %s.npy" % filestring)

    def get_photon_hits(self, X, Y, Z):
        """
        Return the probability of a photon generated at a particular point,
        interpolated from the pre-made LCE map

        Parameters
        ----------
        X,Y,Z : floats
            Position in cm

        Returns
        -------
        probability : float
            Interpolated probabilty of detection

        """
        if type(self.LCEmap) == int:
            print("No LCE Map Loaded!")
            return -999
        
        positions = self.get_LCEmap_positions()
        x = positions[0]
        y = positions[1]
        z = positions[2]
        return interpn((x, y, z), self.LCEmap, [X, Y, Z])[0]
    
    def create_QPEmap(self, x_array, y_array, z_array, nQPs=10000, filestring="detector_QPEmap", T=2.):
        """
        Takes in an x/y/z grid and simulates a large number of
        quasiparticles at each one to simulate to get some detection probabilty

        Parameters
        ----------
        x-array : array
            Array of x positions (in cm) at which to simulate quasiparticles 
        y-array : array
            Array of y positions (in cm) at which to simulate quasiparticles 
        z-array : array
            Array of z positions (in cm) at which to simulate quasiparticles 
        nQPs : int
            Number of quasiparticles to simulate at each point
        filestring : string
            Name for the collection efficiency map.
        T : float
            Bose-Einstein distribution from which we'll sample quasiparticles

        """
        print("Creating QPE map for this detector geometry...")
        x, y, z = np.array(x_array), np.array(y_array), np.array(z_array)
        self.set_QPEmap_positions( [x, y, z] )
        nsensors = self.get_nsensors()
        m = np.zeros((len(x), len(y), len(z), nsensors), dtype=float)
        conditions = self.get_surface_conditions()
        conditions.append( self.liquid_surface )
        for i in range(nsensors):
            conditions.append( (self.get_sensor(i)).get_surface_condition() )
        
        for xx in range(len(x)):
            print("%i / %i" % (xx, len(x)))
            for yy in range(len(y)):
                for zz in range(len(z)):
                    
                    pos = np.array([x[xx], y[yy], z[zz]])
                    if (self.get_liquid_conditions())(*pos) == False:
                        continue
                    hitProbs = [0.0 for _ in range(nsensors)]

                    energyAtDeath, sensorIdsAll, evaporated, total_time, \
                    step_count, initial_momentum, paths = QP_propagation(nQPs, [x[xx], y[yy], z[zz]], up_conditions=self.get_up_conditions(), 
                                                                         down_conditions=self.get_down_conditions(),
                                                                         evap_eff=self.get_evaporation_eff(), T=T,
                                                                         wall_reflection_prob = self.get_QP_wall_reflection_prob(), 
                                                                         wall_diffuse_prob = self.get_QP_wall_diffuse_prob(),
                                                                         sensor_reflection_prob = self.get_QP_sensor_reflection_prob(), 
                                                                         sensor_diffuse_prob = self.get_QP_sensor_diffuse_prob())
                        
                    for i in range(nsensors):
                        hitProbs[i] = np.sum(np.where(sensorIdsAll == i)[0])
                    hitProbs = np.array(hitProbs)/nQPs
                    
                    m[xx][yy][zz] = hitProbs
                    #print(hitProbs)
                    
        self.QPEmap = m
        np.save(filestring+".npy", m)
        print("Saved map to %s.npy" % filestring)
    
    def get_QP_hits(self, X, Y, Z):
        """
        Return the probability of a QP generated at a particular point,
        interpolated from the pre-made LCE map (FIXME: eventually
        this will need to be momentum-dependent)

        Parameters
        ----------
        X,Y,Z : floats
            Position in cm

        Returns
        -------
        probability : float
            Interpolated probabilty of detection

        """

        if type(self.QPEmap) == int:
            print("No QPE Map Loaded!")
            return -999
        
        positions = self.get_QPEmap_positions()
        x = positions[0]
        y = positions[1]
        z = positions[2]
        return interpn((x, y, z), self.QPEmap, [X, Y, Z])[0]

    def Plot_detector(self, xgrid, ygrid, zgrid, sensor_color = 'darkgreen', Cu_color = 'darkorange', He_color = 'aqua', sensor_alpha = 1, Cu_alpha = .1, He_alpha = .3):
        """
        Plot a 3d image of the detector including photodetectors, boundaries, and helium. 
        Uses the marching cubes algorithm to make a mesh for each volume.
        This can take a bit to run; around a minute for 300^3 points and HeRALD v1

        Parameters
        ----------
        xgrid, ygrid, zgrid : array
            Grid of x, y, z points from which to derive our mesh.
            Step size should be smaller than smallest scale of volume
            (usually thickness of sensors)
        sensor_color, Cu_color, He_color : color
            Colors with which to draw the sensor, copper walls, and helium

        sensor_alpha, Cu_alpha, He_alpha : float
            Opacity with which to draw the sensor, copper walls, and helium

        Returns
        -------
        fig : figure
        ax : axes
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-6,6)
        ax.set_ylim(-6,6)
        ax.set_zlim(-6,6)

        X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')

        if True:
            mask = self.bottom_condition(X, Y, Z)[0].astype(float) * \
            self.top_condition(X, Y, Z)[0].astype(float) * \
            self.wall_conditions(X, Y, Z)[0].astype(float)

            verts, faces, normals, values = measure.marching_cubes(
                mask, level=0.5, spacing=(xgrid[1]-xgrid[0], ygrid[1]-ygrid[0], zgrid[1]-zgrid[0])
            )
            verts[:, 0] += xgrid.min()
            verts[:, 1] += ygrid.min()
            verts[:, 2] += zgrid.min()

            mesh = Poly3DCollection(verts[faces], alpha=Cu_alpha)
            mesh.set_facecolor(Cu_color)
            ax.add_collection3d(mesh)
        
        if True:
            mask = self.liquid_conditions(X, Y, Z).astype(float)

            verts, faces, normals, values = measure.marching_cubes(
                mask, level=0.5, spacing=(xgrid[1]-xgrid[0], ygrid[1]-ygrid[0], zgrid[1]-zgrid[0])
            )
            verts[:, 0] += xgrid.min()
            verts[:, 1] += ygrid.min()
            verts[:, 2] += zgrid.min()

            mesh = Poly3DCollection(verts[faces], alpha=He_alpha)
            mesh.set_facecolor(He_color)
            ax.add_collection3d(mesh)
            
        for sensor in self.sensors:
            mask = (sensor.get_surface_condition()(X, Y, Z))[0].astype(float)

            verts, faces, normals, values = measure.marching_cubes(
                mask, level=0.5, spacing=(xgrid[1]-xgrid[0], ygrid[1]-ygrid[0], zgrid[1]-zgrid[0])
            )
            verts[:, 0] += xgrid.min()
            verts[:, 1] += ygrid.min()
            verts[:, 2] += zgrid.min()

            mesh = Poly3DCollection(verts[faces], alpha=sensor_alpha)
            mesh.set_facecolor(sensor_color)
            ax.add_collection3d(mesh)
        
        ax.view_init(elev=10, azim=0)

        return fig, ax



''' #############################################################################

    Define various functions to calculate particle propagation through the 
    Detector geometry

    ############################################################################# '''


def intersection(start, direction, conditions, max_dist, step_size):
    """
    Finds the surface and point of boundary intersection by  assuming rectilinear motion and interpolating.
    Takes vector-valued positions and directions to track > 1 particle at a time. 
    
    Parameters
    ----------
    start : array
        3xN Array with each column denoting a particle's (x,y,z) initial position
    direction : array
        3xN Array with each column denoting a particle's (x,y,z) initial velocity direction 
    conditions : list
        list of boolean-valued functions of 3-d space.

    Returns
    -------
    X1, Y1, Z1 : array of floats
        length N arrays denoting the X/Y/Z coordinates of interesection with a boundary  
    surface_type : array of strings
        "XY" or "Z"; defining the orientation of the boundary crossed

    """
    if np.isscalar( start[0] ):
        start = np.array([np.array([p]) for p in start])
    if np.isscalar( direction[0] ):
        direction = np.array([np.array([p]) for p in direction])
    t = np.arange(0, max_dist, step_size)
    x_line = start[0][:, None] + direction[0][:, None] * t
    y_line = start[1][:, None] + direction[1][:, None] * t
    z_line = start[2][:, None] + direction[2][:, None] * t

    dist = np.full(start.shape[1], np.inf) # Placeholders
    coords = [np.full(len(start[0]), None), np.full(len(start[0]), None), np.full(len(start[0]), None)]
    surface_type = np.full(len(start[0]), None)
    for cond in conditions:
        cut, surface = cond(x_line, y_line, z_line)
        #if an array's max value is repeated, argmax returns the index of the first 
        first_ints = np.argmax(~cut, axis=1)
        valid = (first_ints > 0)
        d = np.where(valid, t[first_ints], np.inf)
        update = d < dist
        dist[update] = d[update]
        idx = first_ints[update] - 1
        coords[0][update] = x_line[update, idx]
        coords[1][update] = y_line[update, idx]
        coords[2][update] = z_line[update, idx]
        surface_type[update] = surface
    return np.array(coords[0], dtype = float), np.array(coords[1], dtype = float), np.array(coords[2], dtype= float), surface_type

def find_surface_intersection(start, direction, up_conditions, down_conditions, alive, max_dist, step_size):
    """
    Wrapper for intersection that partitions particles as going upward and downward and only checks for the conditions
    relevant to each. 

    Parameters
    ----------
        start : array
            3xN Array with each column denoting a particle's (x,y,z) initial position
        direction : array
            3xN Array with each column denoting a particle's (x,y,z) initial velocity direction 
        up_conditions : list of functions
            List of functions that denote relevant boundaries for upward going particles
        down_conditions : list of functions
            List of functions that denote relevant boundaries for upward going particles

    Returns
    -------
    X1, Y1, Z1 : array of floats
        length N arrays denoting the X/Y/Z coordinates of interesection with a boundary  
    surface_type : array of strings
        "XY" or "Z"; defining the orientation of the boundary crossed

    """
    
    # we need to apply a mask to these things, and then everything should work WONDERFULLY
    up = (direction[2] >0 )
    down = ~up 
    up = (up & alive)
    down = (down & alive)
    surface_type = np.full(len(start[0]), None)
    X1 = np.full(len(start[0]), None, dtype=float)
    Y1 = np.full(len(start[0]), None, dtype=float)
    Z1 = np.full(len(start[0]), None, dtype=float)
    if up.any():
        X1[up], Y1[up], Z1[up], surface_type[up] = intersection(start[:, up], direction[:, up], up_conditions, max_dist, step_size)
    if down.any():
        X1[down], Y1[down], Z1[down], surface_type[down] = intersection(start[:, down], direction[:, down], down_conditions, max_dist, step_size)
    return X1, Y1, Z1, surface_type


def reflection(surface, momenta, energies, velocities, pos, direction, diffuse_prob, Andreev_prob = 0.0):
    """
    Takes in initial conditions of a particle before striking a surface 
    and returns direction directly after reflection. Assumes some probabilty 
    of of diffuse reflection at the wall/floor surfaces

    Parameters
    ----------
        surface : string
            String denoting the surface type; 'Liquid', 'XY', or 'Z'
        momenta : array of floats
            Array of momenta for the quasiparticles (in keV/c)
        energies : array of floats
            Array of energies for the quasiparticles (in eV)
        pos : 3-tuple of arrays
            3-tuple containing arrays the x/y/z position of the reflection
        direction : 3-tuple
            3-tuple containing arrays for the velocity vector's direction
        diffuse_prob : float
            Probability of a QP reflecting diffusely on the wall/floor
        Andreev_prob : float
            Probability of Andreev reflection (only relevant for QPs)

    Returns
    -------
        dx,dy, dz : 3-tuple
            3-tuple correspodning to the velocity vector's direction 
            post-reflection

    """
    x = pos[0]
    y = pos[1]
    z = pos[2]
    dx = direction[0]  
    dy = direction[1]
    dz = direction[2]
    if surface == 'Liquid':
        #We will only do specular off surface.
        dz = -dz
        return dx, dy, dz, momenta, energies, velocities
    elif surface == 'XY' or surface == 'Z':
        #We will do both specular and diffuse reflection
        rs = np.random.random(size = (len(dx),))
        specular_cut = rs > diffuse_prob + Andreev_prob
        diffuse_cut = ( rs < diffuse_prob + Andreev_prob ) & ( rs > Andreev_prob )
        Andreev_cut = (rs < Andreev_prob)
        if specular_cut.any():
            dx[specular_cut], dy[specular_cut], dz[specular_cut] = specular_reflection(surface, pos = (x[specular_cut], y[specular_cut],z[specular_cut]),
                                                                                dir = (dx[specular_cut], dy[specular_cut], dz[specular_cut])) 
        if diffuse_cut.any():
            dx[diffuse_cut], dy[diffuse_cut], dz[diffuse_cut] = diffuse_reflection(surface, pos = (x[diffuse_cut], y[diffuse_cut], z[diffuse_cut]),
                                                                            dir = (dx[diffuse_cut], dy[diffuse_cut], dz[diffuse_cut])) 
        if Andreev_cut.any():
            dx[Andreev_cut], dy[Andreev_cut], dz[Andreev_cut], \
            momenta[Andreev_cut], energies[Andreev_cut], velocities[Andreev_cut]= Andreev_reflection(surface, energies[Andreev_cut], momenta[Andreev_cut],
                                                                                                     pos = (x[Andreev_cut], y[Andreev_cut], z[Andreev_cut]),
                                                                                                     dir = (dx[Andreev_cut], dy[Andreev_cut], dz[Andreev_cut]))        

        return dx, dy, dz, momenta, energies, velocities
    else:
        return ValueError('Reflection off of undefined surface')


def specular_reflection(surface, pos = (0.0,0.0,0.0), dir = (0.0,0.0,-1.0)):
    x = pos[0]
    y = pos[1]
    #Handling reflection off the wall; wall normal unit 
    # vector assumed to be r-hat
    if surface == 'XY':
        #Unit projection of position vector into xy plane
        # i.e. surface unit normal (outward going)
        nx, ny = x/np.sqrt(x**2+y**2), y/np.sqrt(x**2+y**2)
        dx = dir[0]
        dy = dir[1]
        dz = dir[2]

        #Unit projection of direction vector into the xy plane
        rx, ry = dx/np.sqrt(dx**2+dy**2), dy/np.sqrt(dx**2+dy**2)
        r_dot_n = (nx*rx + ny*ry)

        #Subtracting off twice the component
        dx = rx - 2 * r_dot_n * nx
        dy = ry - 2 * r_dot_n * ny

        dx, dy, dz = dx/np.sqrt(dx**2+dy**2+dz**2), dy/np.sqrt(dx**2+dy**2+dz**2), dz/np.sqrt(dx**2+dy**2+dz**2)
        return dx, dy, dz
    #Handling reflection off the floor/ceiling
    if surface == 'Z':
        dx = dir[0]
        dy = dir[1]
        dz = dir[2]
        return dx, dy, -dz

def diffuse_reflection(surface, pos = (0.0,0.0,0.0), dir = (0.0,0.0,-1.0)):
    """
    Generates a reflection direction given a surface orientation assuming
    Lambertian (diffuse) reflection.
    Meant to take in a surface and generate a random direction relative to that. 
    """
    x = pos[0]
    y = pos[1]
    z = pos[2]
    
    if surface == 'XY':

        phi_1, sintheta_1 = 2*np.pi*np.random.random(size = len(x)), np.random.random(size = len(x))
        costheta_1 =np.sqrt(1 - sintheta_1**2)

        phi_pos_1 = np.arctan2(x, y)

        dx = - (np.sin(phi_pos_1) * np.sin(phi_1) * sintheta_1) - (np.cos(phi_pos_1) * costheta_1)
        dy = (np.cos(phi_pos_1) * np.sin(phi_1) * sintheta_1) - (np.sin(phi_pos_1) * costheta_1)
        dz = np.cos(phi_1) * sintheta_1
        return dx, dy, dz

    if surface == 'Z': 
        phi_2, sintheta_2 = 2*np.pi*np.random.random(size = len(x)), np.random.random(size = len(x))
        zdir = dir[2]
        theta_2 = np.arcsin(sintheta_2)
        dx = np.cos( phi_2 ) * np.sin( theta_2 )
        dy  = np.sin( phi_2 ) * np.sin( theta_2 )
        dz  = np.cos(theta_2) * -1 * np.sign(zdir) # determine whether we're hitting the top or bottom
        return dx, dy, dz
    
def Andreev_reflection(surface, momenta, energies,  pos = (0.0,0.0,0.0), dir = (0.0,0.0,-1.0)):
    """
    Generates a reflection direction given a surface orientation assuming
    Lambertian (diffuse) reflection.
    Meant to take in a surface and generate a random direction relative to that. 
    """
    x = pos[0]
    y = pos[1]
    z = pos[2]

    # calculate the momenta each particle would have if it were converted to each flavor
    phonon_momenta = phonon_momentum(energies)
    rminus_momenta = rminus_momentum(energies)
    rplus_momenta = rplus_momentum(energies)

    final_momenta_options = np.stack([phonon_momenta, rminus_momenta, rplus_momenta], axis = 1)

    #Instantiate boolean vector 
    can_convert_to_phonon = np.full_like(momenta, True)
    can_convert_to_rplus = np.full_like(momenta, True)
    can_convert_to_rminus = np.full_like(momenta, True)

    #Particle should not convert to the same flavor if possible; specular reflection handles this
    can_convert_to_phonon = can_convert_to_phonon & (np.abs(momenta) >  2.22538686)
    can_convert_to_rplus = can_convert_to_rplus & (np.abs(momenta) < 3.84035444)
    can_convert_to_rminus = can_convert_to_rminus & ( (np.abs(momenta) < 2.22538686) | (np.abs(momenta) > 3.84035444) )

    #calculating surface normal vectors (outward-going)
    if surface == 'XY':
        nx, ny = x/np.sqrt(x*x + y*y), y/np.sqrt(x*x + y*y)
        nz = x * 0

    elif surface == 'Z':
        nx, ny = x*0, x*0
        nz = np.sign(dir[2])

    cos = nx * dir[0] + ny * dir[1] * nz * dir[2]
    transverse_momenta = np.sqrt(1 - cos**2) * momenta

    #Account for transformations forbidden by conservation laws
    can_convert_to_phonon = can_convert_to_phonon & (transverse_momenta**2 < momenta**2)
    can_convert_to_rplus = can_convert_to_rplus & (transverse_momenta**2 < momenta**2)        
    can_convert_to_rminus = can_convert_to_rminus & (transverse_momenta**2 < momenta**2)

    mask = np.stack([can_convert_to_phonon, can_convert_to_rminus, can_convert_to_rplus], axis = 1)
    rs = np.random.random(final_momenta_options.shape)
    rs[~mask] = -1
    #Store any particles that are have possible way to Andreev reflect
    forbidden = ~mask.any(axis=1)

    #if Andreev reflection is possible, return a randomly selected final states. 
    # Otherwise, default to specular reflection with no flavor switch.
    final_momenta  = np.where(forbidden, momenta, final_momenta_options[:, np.argmax(rs, axis = 1)])

    tx, ty, tz = dir[0] - cos * nx, dir[1] - cos * ny, dir[2] - cos * nz
    tx, ty, tz = tx/np.sqrt(tx*tx + ty*ty + tz*tz), ty/np.sqrt(tx*tx + ty*ty + tz*tz), tz/np.sqrt(tx*tx + ty*ty + tz*tz)
    
    # Calculate component of momenta normal to the surface
    normal_momenta = np.sqrt(final_momenta*final_momenta - transverse_momenta**2)

    sign = np.sign(QP_velocity(momenta)/QP_velocity(final_momenta))

    dx, dy, dz = -nx * normal_momenta + tx * sign * transverse_momenta, -ny * normal_momenta + ty * sign * transverse_momenta, -nz * normal_momenta + tz * sign * transverse_momenta
    dx, dy, dz = dx/np.sqrt(dx*dx + dy*dy + dz*dz), dy/np.sqrt(dx*dx + dy*dy + dz*dz), dz/np.sqrt(dx*dx + dy*dy + dz*dz)
    return dx, dy, dz, momenta, QP_dispersion(momenta), QP_velocity(momenta)


def generate_random_direction(nQPs, bottom_phi = 0, top_phi = 2*np.pi, bottom_theta = -1, top_theta = 1):
    """
    Generates a random direction for a large number of quasiparticles, where that random direction is within ranges.

    Paremeters
    ----------
        nQPs : int
            The number of quasiparticles being generated 
        bottom_phi : float
            Lower bound of the phi angle, which is the azimuthal angle. 
        top_phi : float 
            Upper bound of the phi angle, which is the azimuthal angle
        bottom_theta : float
            Lower bound of the inverse of theta, must be between (-1, 1)
        top_theta : float
            Upper bound of the inverse of theta, must be between (-1, 1)

    Returns
    -------
        tuple: the unit vector in cartesian coordinates, broken up into arrays of X, Y, Z, each of length nQPs. 
    """
    phi, arctheta = np.random.uniform(bottom_phi, top_phi, size=nQPs), np.random.uniform(bottom_theta, top_theta, size=nQPs)
    theta = np.arccos(arctheta)
    #prepare the vecctors of direction
    dx = np.cos( phi ) * np.sin( theta )
    dy = np.sin( phi ) * np.sin( theta )
    dz = np.cos(theta)
    return dx, dy, dz


def evaporation(momentum, energy, direction):
    """
    Gives the momentum/direction of an evaporated He4 atom given the kinematics of the 
    QP that evaporats it
    Parameters
    ----------
        momentum : float/array of floats
            QP momentum in keV/c
        energy : float/array of floats
            QP energy in eV
        velocity : float/array of floats
            QP velocity in 100 m/s
        direction : 3-tuple
            unit vector denoting the QP's direction
    Returns
    -------
        dx,dy,dz : arrays of floats
            unit vectors defining direction of ejected He4 atoms
        momentum : array of floats
            magnitude of momentum of ejected He4 atoms
    """
    dx, dy, dz = direction[0], direction[1], direction[2]
    if np.isscalar(momentum):
        momentum = np.array([momentum])
    if np.isscalar(energy):
        energy = np.array([energy])

    m =  3.725472e9 #He mass in eV/c^2
    c = 2.99792458e8

    # https://journals.aps.org/pr/abstract/10.1103/PhysRev.69.242
    # Interpolation of data to 8 mK. This table is consistent with 
    # other values used in classically modeling quantum evaporation
    Vb = 0.000620 # eV 

    pxi, pyi, pzi = momentum * dx, momentum * dy, momentum * dz
    pxf, pyf = pxi, pyi
    pzf = np.sqrt(2 * m/1000 * (energy/1000 - Vb/1000) - pxi**2 - pyi**2)
    momentum_final = np.sqrt(pxf**2 + pyf**2 + pzf**2)
    dx, dy, dz = pxf/momentum_final, pyf/momentum_final, pzf/momentum_final
    energy_final = energy - Vb #eV
    velocity_final = momentum*1000/m * c
    return dx, dy, dz, momentum_final, energy_final, velocity_final

def evap_prob_of_p_theta(p, theta, evap_eff):
    """"
    I think this (currently barren) function that aims to simulate the kinematic/quantum mechanical 
    probabilities of quantum evaporation

    Parameters
    ----------
        p : array of floats
            Array of QP momenta in keV
        theta : array of floats
            Angle of incidence with the helium surface

        evap_eff : array of floats
            7-term array that indexes the evaporation probability for each of the QP momentum bins
    Returns
    -------
        no_evap_bools : array of bool
            True if evaporation is simulated to occur 
        
    """
    no_evap_bools = np.full_like(p,fill_value= False, dtype=bool)
    the_nums = np.random.random(size = len(p))
    bins = [0.0, 1.1, 1.7, 2.2, 3.834, 4.5, 5.2] #Bins of momentum in keV for different regions of the He dispersion curve
    bin_indices = np.digitize(p, bins) - 1 #Finding which bin each of the quasiparticles falls into

    for ii in np.unique(bin_indices):
        no_evap_bools[ii==bin_indices] = the_nums[ii==bin_indices] < evap_eff[ii] 
    return no_evap_bools
    

def critical_angle(energy, momentum):
    """"
    Calculate the critical angle for quantum evaporation under strictly classical assumptions:
        - Conversation of energy
        - Converse of momentum parallel to surface

    We're also going to use this to impose the energy threshold for evaporation. Any particle
    without sufficient energy to liberate a atom will be given a critical angle of zero

    This currently throws obnoxious erri

    Parameters
    ----------
        energy : float/array
            Incoming quasiparticle's energy in eV
        momentum : float/array
            Incoming quaisparticle's momentum in keV/c
    Returns
    -------
        angle : float/array
            Critical angle of incidence for incoming quasiparticle to produce evaporation
    """

    # https://journals.aps.org/pr/abstract/10.1103/PhysRev.69.242
    # Interpolation of data to 8 mK. This table is consistent with 
    # other values used in classically modeling quantum evaporation
    Vb = .62e-3 # atom/surface binding energy for superfluid helium in eV
    m =  3.725472e9 #He mass in eV/c^2

    sin = np.where( energy >= Vb, np.sqrt(2 * m *(energy - Vb))/np.abs(momentum *1000), 0)
    angle = np.where(np.abs(sin) < 1, np.arcsin(sin), np.arcsin(1))
    return angle

# Define a function to extract the sensor number using regex
#@np.vectorize#(otypes=[str])
def extract_number(s):
    if s is None:
        return -1
    match = re.search(r'\d+', s)  # Find the first occurrence of one or more digits
    if match is not None:
        return int(match.group())  # Convert the matched digits to an integer
    return -1  # Return -1 if no digits are found



"""
##############################
Quasiparticle Propagation
##############################
"""
def QP_propagation(nQPs, start, up_conditions, down_conditions, evap_eff=np.array([1,1,1,1,1,1,1]), 
                   wall_reflection_prob = 0.0, wall_diffuse_prob = 0.0, wall_Andreev_prob = 0.0,
                   sensor_reflection_prob = 0.0, sensor_diffuse_prob = 0.0, sensor_Andreev_prob = 0.0,
                   T=2., max_dist = 10, step_size = .05, plot_3d=False, fixed_dir = None, 
                   fixed_momentum = None, verbose = False):
    """
    Tracking of Quasiparticles through medium. 
   
    Parameters
    ----------

        nQPs : int 
            The number of quasiparticles to be simulated.  
        start : array 
            the initial location of the quasiparticles.
            All are assumed to originate from the same location
        up_conditions : list of functions
            List of functions that denote relevant boundaries for upward going particles
        down_conditions : list of functions
            List of functions that denote relevant boundaries for upward going particles
        reflection_prob : float
            Probability of reflection off a copper surface
        evap_eff : array of floats
            Length 7 array defining evaporation probablities in for each of the QP momentum bins
        diffuse_prob : float
            Probability of QP reflecting diffusely off of a solid surface
        T : float
            effective temperature of momentum distribution
        plot_3d : boolean
            Plot the trajectories of the tracked quasiparticles
        fixed_dir : 3-tuple of floats
            Allow the user to manually set the QP's initial direction's unit vector. Default is isotropy.
        fixed_momentum : float
            Allow the user to manually set a momentum in keV. Default is random sampling



    Returns
    -------
        energyAtDeath : array
            Array of energies (in eV) of the quasiparticles just before death/detection
        sensorIdsAll : array of ints
            Array of integers that define which sensor detected the QP
            -1 if not detected; otherwise the the corresponding integer
        evaporated : array of floats
            Array of floats that defines whether or not each QP is ever
            evaporated
        total_time : array of floats
            Array of times (in us) at which the QPs die or are detected
        step_count : array of ints
            Array of the number of steps before the particle died/collected
        initial_momentum : array of floats
            Array of the inital momenta of all QPs in keV/c
        paths : 3-tuple
            each object is a nQPs x 100 array; each row are the coordinates 
            of each QPs at vertices

    """
    # initializing Variables
    # ~~~~~~~~~~~~~~~~~~~~~~
    if verbose: print(f'Starting at {start}')
    if np.isscalar(nQPs):
        X = np.full(nQPs, start[0])
        Y = np.full(nQPs, start[1])
        Z = np.full(nQPs, start[2])
        start = np.array([X, Y, Z])
    
    # Variables for tracking/storing particle paths
    paths = (0,0,0)
    particles_x = np.zeros(shape=(nQPs, 100))
    particles_y = np.zeros(shape=(nQPs, 100))
    particles_z = np.zeros(shape=(nQPs, 100))

    #Variables for current positions/momenta
    X, Y, Z = start[0], start[1], start[2]
    dx, dy, dz = generate_random_direction(nQPs)
    if fixed_dir is not None: 
        dx = np.full(nQPs, fixed_dir[0]) 
        dy = np.full(nQPs, fixed_dir[1]) 
        dz = np.full(nQPs, fixed_dir[2]) 


    total_time = np.zeros(nQPs, dtype=float)
    n=0

    #Sampling momenta/flavors
    momentum = Random_QPmomentum(nQPs, T=T) #keV/c
    initial_momentum=np.copy(momentum)
    if fixed_momentum is not None:
        momentum = np.full(nQPs, fixed_momentum)

    #prepare the alive tracker, and immeditately kill particles that are 
    #unstable to p -> p + p + ... decays
    alive = np.ones(nQPs, dtype=int)
    cond = (momentum <  1.10) | (momentum  > 4.54) #Lifetime cut
    alive = np.where( cond, 0, alive)

    #velocity and energy, both arrays of length nQP
    velocity = QP_velocity(momentum) #m/s
    energy = QP_dispersion(momentum) #eV
    evaporated = np.zeros( nQPs, dtype=bool)

    # instead of inverting the velocity, we invert the momentum and later, when we use velocity, we take the magnitude of it anyways
    # v_mask = velocity < 0
    # momentum[v_mask] = -momentum[v_mask]
    
    #Arrays for storing detection information
    energyAtDeath = np.zeros(nQPs, dtype=float)
    step_count = np.zeros_like(alive)
    sensorIdsAll = -1*np.ones_like(alive, dtype=int)

    living = alive > 0.5

    #Storing the first vertex
    particles_x[:, 0][living] = X[living]
    particles_y[:, 0][living] = Y[living]
    particles_z[:, 0][living] = Z[living]

    while np.sum(alive) > 0: # loop until all particles are dead
        
        n+=1

        living = ( alive > 0.5 )

        #calculate when each particle will hit its next surface
        X1, Y1, Z1, surface_type = find_surface_intersection(np.array([X, Y, Z]), np.array([dx, dy, dz]), up_conditions, down_conditions, living, max_dist, step_size)

        # Kill particles that don't hit a surface; this shouldn't normally happen
        hit_surface_check= (surface_type != None)
        dx[~hit_surface_check], dy[~hit_surface_check], dz[~hit_surface_check] = np.zeros_like(dx[~hit_surface_check]),np.zeros_like(dx[~hit_surface_check]),np.zeros_like(dx[~hit_surface_check])   
        alive[living] = np.where( hit_surface_check[living], alive[living], 0)
        living = ( alive > 0.5 )

        #Update the step count; total lifetime of each particle 
        step_count[living]  = np.full_like(alive[living], fill_value=n)
        dist_sq = (pow(X1[living]-X[living],2.)+pow(Y1[living]-Y[living], 2.)+pow(Z1[living]-Z[living],2.)).astype(float)      
        total_time[living] = total_time[living] + np.sqrt(dist_sq)/np.abs(velocity[living]*1.0e-4)  #us
        
        
        # Handling quasiparticles that have struck the upper helium surface
        alive_He_surface_check = living & (surface_type == "Liquid")
        if alive_He_surface_check.any():
            if verbose:
                print("Hit Liquid/Vacuum interface")
            critical_angles = critical_angle(energy, momentum) 
            incident_angles = np.arccos(dz)
            evap_bools = evap_prob_of_p_theta(np.abs(momentum), incident_angles, evap_eff) # True if we're simulating to evaporate

            angle_check = (incident_angles < critical_angles) #True if evaporation is kinematically permissible
            no_evap = (~angle_check) | (~evap_bools) #Doesn't evaporate for kinematic or probabilistic reasons
            a_L_noevap = alive_He_surface_check & no_evap # Mask for particles that are alive at the liquid interface but don't evaporate
            a_L_evap = alive_He_surface_check & ~no_evap 
            #Specular reflections for the QPs that don't evaporate atoms
            if a_L_noevap.any():
                dx[a_L_noevap], dy[a_L_noevap], dz[a_L_noevap], \
                momentum[a_L_noevap], energy[a_L_noevap], velocity[a_L_noevap] = reflection('Liquid', momentum[a_L_noevap], energy[a_L_noevap], velocity[a_L_noevap],
                                                                                            (X1[a_L_noevap],Y1[a_L_noevap],Z1[a_L_noevap]), 
                                                                                            (dx[a_L_noevap], dy[a_L_noevap], dz[a_L_noevap]),
                                                                                            diffuse_prob=0)
            #evaporations for the QPs that do evaporate atoms
            if a_L_evap.any():
                dx[a_L_evap], dy[a_L_evap], dz[a_L_evap], \
                momentum[a_L_evap], energy[a_L_evap], velocity[a_L_evap] = evaporation(momentum[a_L_evap],energy[a_L_evap],
                                                                                    direction=(dx[a_L_evap], dy[a_L_evap], dz[a_L_evap]))
                evaporated[a_L_evap] = True

            #Update new positions
            X[alive_He_surface_check] = X1[alive_He_surface_check]
            Y[alive_He_surface_check] = Y1[alive_He_surface_check]
            # If the QP doesn't evaporate, ensure it's below the surface
            Z[a_L_noevap] = Z1[a_L_noevap]
            # If the QP does evaporate, ensure it's above the surface
            Z[a_L_evap] = Z1[a_L_evap] + step_size
               

        # Handling the case in which an evaporated atom hits a dry sensor 
        # (no chance for reflection; these are definitely detected/killed)
        alive_at_dry_sensor_check = living & np.array(["sensor" in str(s) for s in surface_type]) & evaporated
        if alive_at_dry_sensor_check.any():
            if verbose:
                print("Hit a Dry Sensor")
            #Kill off particles; record relevant sensor IDs
            sensorIdsAll[alive_at_dry_sensor_check] = np.vectorize(extract_number)(surface_type[alive_at_dry_sensor_check])
            energyAtDeath[alive_at_dry_sensor_check] = energy[alive_at_dry_sensor_check] 
            alive[alive_at_dry_sensor_check] = 0 

            #Update Positions
            X[alive_at_dry_sensor_check] = X1[alive_at_dry_sensor_check]
            Y[alive_at_dry_sensor_check] = Y1[alive_at_dry_sensor_check]
            Z[alive_at_dry_sensor_check] = Z1[alive_at_dry_sensor_check]


        # Handling the case in which the an un-evaporated quasiparticle hits a wet sensor
        # (these can either reflect or be detected/killed)
        alive_at_wet_sensor_check = living & np.array(["sensor" in str(s) for s in surface_type]) & ~evaporated
        if alive_at_wet_sensor_check.any():
            if verbose:
                print("Hit a Wet Sensor")
            #simulating probability of reflection
            r = np.random.random(len(surface_type[alive_at_wet_sensor_check]))
            absorption_cond = (r > sensor_reflection_prob)


            #Compute reflection kinematics for all particles
            # (moot point for the absorbed ones)
            dx[alive_at_wet_sensor_check], dy[alive_at_wet_sensor_check], dz[alive_at_wet_sensor_check],\
            momentum[alive_at_wet_sensor_check], energy[alive_at_wet_sensor_check],\
            velocity[alive_at_wet_sensor_check] = reflection('Z', momentum[alive_at_wet_sensor_check], energy[alive_at_wet_sensor_check], velocity[alive_at_wet_sensor_check],
                                                             (X1[alive_at_wet_sensor_check],Y1[alive_at_wet_sensor_check],Z1[alive_at_wet_sensor_check]), 
                                                             (dx[alive_at_wet_sensor_check], dy[alive_at_wet_sensor_check], dz[alive_at_wet_sensor_check]),
                                                             diffuse_prob = sensor_diffuse_prob, Andreev_prob = sensor_Andreev_prob)
            
            #Update positions
            X[alive_at_wet_sensor_check] = X1[alive_at_wet_sensor_check]
            Y[alive_at_wet_sensor_check] = Y1[alive_at_wet_sensor_check]
            Z[alive_at_wet_sensor_check] = Z1[alive_at_wet_sensor_check]

            #Update alive/energy at death/sensor ID if absorbed at the detector surface 
            alive[alive_at_wet_sensor_check] = np.where(absorption_cond, 0, alive[alive_at_wet_sensor_check])
            energyAtDeath[alive_at_wet_sensor_check] = np.where(absorption_cond, energy[alive_at_wet_sensor_check], energyAtDeath[alive_at_wet_sensor_check])
            sensorIdsAll[alive_at_wet_sensor_check] = np.vectorize(extract_number)(surface_type[alive_at_wet_sensor_check])
          

        #Handle evaporated He atoms that hit a wall/ceiling 
        #(no probability of reflection)
        living_evaporated_wall = living & ((surface_type == 'XY') | (surface_type == 'Z')) & evaporated
        if living_evaporated_wall.any():
            if verbose:
                print('Hit a wall post-evaporation')
            #Update positions
            X[living_evaporated_wall] = X1[living_evaporated_wall]
            Y[living_evaporated_wall] = Y1[living_evaporated_wall]
            Z[living_evaporated_wall] = Z1[living_evaporated_wall]

            #Update alive/energy at death based on absorption at the surface 
            energyAtDeath[living_evaporated_wall] = energy[living_evaporated_wall]
            alive[living_evaporated_wall] = 0

        #Handle QPs that hit a wall before evaporation
        living_not_evaporated_wall = living & ((surface_type == 'XY') | (surface_type == 'Z')) & ~evaporated
        if living_not_evaporated_wall.any():
            if verbose:
                print('Hit a wall pre-evaporation')            
            # sub-masks for hitting the wall and the floor
            hit_sidewall_check = living_not_evaporated_wall & (surface_type == 'XY')
            hit_floor_check = living_not_evaporated_wall & (surface_type == 'Z')

            #Handle reflections off the sidewalls 
            if hit_sidewall_check.any():
                dx[hit_sidewall_check], dy[hit_sidewall_check], dz[hit_sidewall_check], \
                momentum[hit_sidewall_check], energy[hit_sidewall_check], \
                velocity[hit_sidewall_check]= reflection('XY', momentum[hit_sidewall_check], energy[hit_sidewall_check], velocity[hit_sidewall_check],
                                                         (X1[hit_sidewall_check],Y1[hit_sidewall_check],Z1[hit_sidewall_check]), 
                                                         (dx[hit_sidewall_check], dy[hit_sidewall_check], dz[hit_sidewall_check]), 
                                                         diffuse_prob=wall_diffuse_prob, Andreev_prob = wall_Andreev_prob)
            #Handle reflections off the floor
            if hit_floor_check.any():
                dx[hit_floor_check], dy[hit_floor_check], dz[hit_floor_check], \
                momentum[hit_floor_check], energy[hit_floor_check], \
                velocity[hit_floor_check] = reflection('Z', momentum[hit_floor_check], energy[hit_floor_check], velocity[hit_floor_check],
                                                       (X1[hit_floor_check],Y1[hit_floor_check],Z1[hit_floor_check]),
                                                       (dx[hit_floor_check], dy[hit_floor_check], dz[hit_floor_check]),
                                                       diffuse_prob=wall_diffuse_prob, Andreev_prob = wall_Andreev_prob )
            #Simulate probability of absorption
            r = np.random.random(len(surface_type[living_not_evaporated_wall]))
            cond = (r > wall_reflection_prob)

            #Kill particles that are absorbed
            alive[living_not_evaporated_wall] = np.where(cond, 0, alive[living_not_evaporated_wall])
            energyAtDeath[living_not_evaporated_wall] = np.where(cond, energy[living_not_evaporated_wall],  energyAtDeath[living_not_evaporated_wall])

            X[living_not_evaporated_wall] = X1[living_not_evaporated_wall]
            Y[living_not_evaporated_wall] = Y1[living_not_evaporated_wall]
            Z[living_not_evaporated_wall] = Z1[living_not_evaporated_wall]

        try: 
            #Adding the new vertices to track particle paths
            particles_x[:, n][living] = X1[living]
            particles_y[:, n][living] = Y1[living]
            particles_z[:, n][living] = Z1[living]
        except IndexError:
            print('one reflection has gone on more than 100 times')

    if plot_3d:
        ax = plt.figure().add_subplot(projection ='3d')
        for i in range(nQPs ):
            mask = (particles_x[i,:] == 0) &(particles_y[i,:] == 0) & (particles_z[i,:] == 0) 
            ax.plot(particles_x[i,:][~mask], particles_y[i,:][~mask], particles_z[i,:][~mask], marker = '.', lw = 1)

        ax.view_init(elev = 0,azim = 0)

    
    paths = (particles_x, particles_y, particles_z)

    return energyAtDeath, sensorIdsAll, evaporated, total_time, step_count, initial_momentum, paths
        
def photon_propagation(nPhotons, start, up_conditions, down_conditions, wall_reflection_prob = 0.0, 
                       wall_diffuse_prob = 0.0, sensor_reflection_prob = 0.0,  sensor_diffuse_prob = 0.0,
                       max_dist = 10, step_size = .05, plot_3d=False, fixed_dir = None):
    """
    Tracking of photons through medium. 
   
    Parameters
    ----------
        nPhotons : int 
            The number of quasiparticles to be simulated.  
        start : array 
            the initial location of the quasiparticles.
            All are assumed to originate from the same location
        up_conditions : list of functions
            List of functions that denote relevant boundaries for upward going particles
        down_conditions : list of functions
            List of functions that denote relevant boundaries for upward going particles
        wall_reflection_prob : float
            Probability of reflection off a copper surface
        plot_3d : boolean
            Plot the trajectories of the tracked quasiparticles
        fixed_dir : 3-tuple of floats
            Allow the user to manually set the QP's initial direction's unit vector. Default is isotropy.


    Returns
    -------
        energyAtDeath : array
            Array of energies (in eV) of the quasiparticles just before death/detection
        sensorIdsAll : array of ints
            Array of integers that define which sensor detected the QP
            -1 if not detected; otherwise the the corresponding integer
        total_time : array of floats
            Array of times (in us) at which the QPs die or are detected
        step_count : array of ints
            Array of the number of steps before the particle died/collected
        paths : 3-tuple
            each object is a nQPs x 100 array; each row are the coordinates 
            of each QPs at vertices

    """
    if np.isscalar(nPhotons):
        X = np.full(nPhotons, start[0], dtype = np.float64)
        Y = np.full(nPhotons, start[1], dtype = np.float64)
        Z = np.full(nPhotons, start[2], dtype = np.float64)
        start = np.array([X, Y, Z])

    #initialize arrays for storing particle paths
    paths = (0,0,0)
    particles_x = np.zeros(shape=(nPhotons, 100))
    particles_y = np.zeros(shape=(nPhotons, 100))
    particles_z = np.zeros(shape=(nPhotons, 100))

    #Iinitiallizing current particle positions/momenta
    X, Y, Z = start[0], start[1], start[2]
    dx, dy, dz = generate_random_direction(nPhotons)
    if fixed_dir is not None: 
        dx = np.full(nPhotons, fixed_dir[0]) 
        dy = np.full(nPhotons, fixed_dir[1]) 
        dz = np.full(nPhotons, fixed_dir[2]) 

    total_time = np.zeros(nPhotons, dtype=float)
    n=0
    
    alive = np.ones(nPhotons, dtype=int)
    #FIXME: citation for this index of refraction
    velocity = 29979.2/1.03 #speed of light in He4 cm/us    
    # cond = (velocity > 0.)
    # alive = np.where( cond, alive, 0.)
    energy = np.ones(nPhotons) * 16.0
    energyAtDeath = np.zeros(nPhotons, dtype=float)
    step_count = np.zeros_like(alive)
    #-1 is default, indicating not hitting  a sensor
    sensorIdsAll = -1*np.ones_like(alive, dtype=int)

    living = alive > 0.5
    particles_x[:, 0][living] = X[living]
    particles_y[:, 0][living] = Y[living]
    particles_z[:, 0][living] = Z[living]

    while np.sum(alive) > 0:


        n+=1
        living = ( alive > 0.5 )
   
        X1, Y1, Z1, surface_type = find_surface_intersection(np.array([X, Y, Z]), np.array([dx, dy, dz]), up_conditions, down_conditions, living, max_dist, step_size)
  
        hit_surface_check = (surface_type != None)
        dx[~hit_surface_check], dy[~hit_surface_check], dz[~hit_surface_check] = np.zeros_like(dx[~hit_surface_check]),np.zeros_like(dx[~hit_surface_check]),np.zeros_like(dx[~hit_surface_check])   

        alive[living] = np.where( hit_surface_check[living], alive[living], 0)
        living = ( alive > 0.5 )

        step_count[living] = np.full_like(alive[living], fill_value=n)

        dist_sq = (pow(X1[living]-X[living],2.)+pow(Y1[living]-Y[living], 2.)+pow(Z1[living]-Z[living],2.)).astype(float)      
        total_time[living] = total_time[living] + np.sqrt(dist_sq)/velocity  #us


        #Managing photons crossing the Helium surface 
        alive_He_surface_check = living & (surface_type == "Liquid")
        if alive_He_surface_check.any():
        # FIXME: for now, this doesn't account for refraction
            X[alive_He_surface_check] = X1[alive_He_surface_check]  
            Y[alive_He_surface_check] = Y1[alive_He_surface_check] 
            Z[alive_He_surface_check] = Z1[alive_He_surface_check] + step_size

        #Managing photons that have reached a sensor
        alive_at_sensor_check = living & np.array(["sensor" in str(s) for s in surface_type])
        if alive_at_sensor_check.any():
            #Simulate Reflection probability
            r = np.random.random(len(surface_type[alive_at_sensor_check]))
            absorption_cond = (r > sensor_reflection_prob)

            #Compute reflection kinematics for the surviving partices
            dx[alive_at_sensor_check], dy[alive_at_sensor_check],\
            dz[alive_at_sensor_check],_,_,_ = reflection('Z', energy[alive_at_sensor_check], energy[alive_at_sensor_check], energy[alive_at_sensor_check],
                                                         (X1[alive_at_sensor_check],Y1[alive_at_sensor_check],Z1[alive_at_sensor_check]),
                                                         (dx[alive_at_sensor_check], dy[alive_at_sensor_check], dz[alive_at_sensor_check]),
                                                         diffuse_prob = sensor_diffuse_prob)
            #Update position
            X[alive_at_sensor_check] = X1[alive_at_sensor_check]
            Y[alive_at_sensor_check] = Y1[alive_at_sensor_check]
            Z[alive_at_sensor_check] = Z1[alive_at_sensor_check]

            #Update alive/energyAtDeath/sensor IDs based on absorption at the surface 
            alive[alive_at_sensor_check] = np.where(absorption_cond, 0, alive[alive_at_sensor_check])
            sensorIdsAll[alive_at_sensor_check] = np.vectorize(extract_number)(surface_type[alive_at_sensor_check])
            energyAtDeath[alive_at_sensor_check] = np.where(absorption_cond, energy[alive_at_sensor_check], energyAtDeath[alive_at_sensor_check])

        #Managing photons that have reached a wall/floor/ceiling boundary
        alive_at_cell_check = living & ((surface_type == 'XY') | (surface_type == 'Z'))
        if alive_at_cell_check.any():
            #Simulate reflection probability
            r = np.random.random(len(surface_type[alive_at_cell_check])) 
            absorption_cond = (r > wall_reflection_prob)

            #Submasks for hitting a floor/cieling/sidewall
            hit_sidewall_check = alive_at_cell_check & (surface_type == 'XY')
            hit_floor_or_ceiling_check = alive_at_cell_check & (surface_type == 'Z')
            
            #Compute reflection kinematics for sidewall hits 
            if hit_sidewall_check.any():
                dx[hit_sidewall_check], dy[hit_sidewall_check], \
                dz[hit_sidewall_check],_,_,_ = reflection('XY', energy[hit_sidewall_check], energy[hit_sidewall_check], energy[hit_sidewall_check],
                                                          pos = (X1[hit_sidewall_check],Y1[hit_sidewall_check],Z1[hit_sidewall_check]),
                                                          direction = (dx[hit_sidewall_check], dy[hit_sidewall_check], dz[hit_sidewall_check]),
                                                          diffuse_prob = wall_diffuse_prob)
            #Compute reflection kinematics for floor/ceiling hits 
            if hit_floor_or_ceiling_check.any():
                dx[hit_floor_or_ceiling_check], dy[hit_floor_or_ceiling_check], \
                dz[hit_floor_or_ceiling_check],_,_,_ = reflection('Z', energy[hit_floor_or_ceiling_check], energy[hit_floor_or_ceiling_check], energy[hit_floor_or_ceiling_check],
                                                                  pos = (X1[hit_floor_or_ceiling_check],Y1[hit_floor_or_ceiling_check],Z1[hit_floor_or_ceiling_check]),
                                                                  direction = (dx[hit_floor_or_ceiling_check], dy[hit_floor_or_ceiling_check], dz[hit_floor_or_ceiling_check]),
                                                                  diffuse_prob = wall_diffuse_prob)


            #Update position
            X[alive_at_cell_check] = X1[alive_at_cell_check]
            Y[alive_at_cell_check] = Y1[alive_at_cell_check]
            Z[alive_at_cell_check] = Z1[alive_at_cell_check]
            #Update alive/energyAtDeath based on absorption at the surface 
            alive[alive_at_cell_check] = np.where(absorption_cond, 0, alive[alive_at_cell_check])
            energyAtDeath[alive_at_cell_check] = np.where(absorption_cond, energy[alive_at_cell_check], energyAtDeath[alive_at_cell_check])
       
        try: 
            #Add new poisition to path tracking arrays
            particles_x[:, n][living] = X1[living]
            particles_y[:, n][living] = Y1[living]
            particles_z[:, n][living] = Z1[living]
        except IndexError:
            print('one reflection has gone on more than 100 times')
            break

    if plot_3d:
        ax = plt.figure().add_subplot(projection ='3d')
        for i in range(nPhotons ):
            mask = (particles_x[i,:] == 0) &(particles_y[i,:] == 0) & (particles_z[i,:] == 0) 
            ax.plot(particles_x[i,:][~mask], particles_y[i,:][~mask], particles_z[i,:][~mask], marker = '.', lw = 1)

        ax.view_init(elev = 0,azim = 0)

    
    paths = (particles_x, particles_y, particles_z)
        
    return energyAtDeath, sensorIdsAll, total_time, step_count, paths


''' #########################################################################

    Define functions for getting the energy deposited in the sensors
    
    ######################################################################### '''


def GetEvaporationSignal(detector, QPs, X, Y, Z, useMap=True, T=2.0, max_dist = 10, step_size =.05, 
                         plot_3d = False, fixed_dir = None, fixed_momentum = None, verbose = False,
                         debug = False):
    '''
    Performs a simulation of quasiparticle propagation inside your detector. Depending on whether the useMap is True,
    this will either use pre-generated detection probabilities, or use pregenerated collection efficiency
    FIXME: This will not use a light collection map
    Parameters
    ----------
        detector : vDetector
            virtual detector object inside of which you wish to simulate QP propagation
        QPs : int
            number of quasiparticles to simulate
        X, Y, Z : floats
            coordinates of the QP's origin in cm
        useMap : boolean
            Whether to fully simulate QP motion from first principles, or use-precalculated collectoin efficiency maps
        T : float
            effective temperature from which to sample the quasiparticle momenta 
        plot_3d : boolean
            Generate a 3d plot of QP paths
        fixed_dir : 3-tuple of floats
            Allow the user to manually set the QP's initial direction's unit vector. Default is isotropy.
        fixed_momentum : float
            Allow the user to manually set a momentum in keV. Default is random sampling
        verbose : bool
        debug : bool
            default format the signal as HestSignal() object. If True, print out a bunch of tracking 
            information that our detectors would normally not be privy to

    Returns (default)
    -----------------
        Signal : HestSignal
            Signal object containing hits times and energy deposits

    Returns (if debug == True)
    --------------------------
        energyAtDeath : array
            Array of energies (in eV) of the quasiparticles just before death/detection
        sensorIdsAll : array of ints
            Array of integers that define which sensor detected the QP
            -1 if not detected; otherwise the the corresponding integer
        evaporated : array of floats
            Array of floats that defines whether or not each QP is ever
            evaporated
        total_time : array of floats
            Array of times (in us) at which the QPs die or are detected
        step_count : array of ints
            Array of the number of steps before the particle died/collected
        initial_momentum : array of floats
            Array of the inital momenta of all QPs in keV/c
        paths : 3-tuple
            each object is a nQPs x 100 array; each row are the coordinates 
            of each QPs at vertices

            

        
    '''
    #check to see if there's an LCEmap loaded/generated
    if type(detector.get_QPEmap()) == int:
        useMap = False

    nsensors = detector.get_nsensors()
    arrivalTimes = [[] for x in range(nsensors)]
    energies = [[] for x in range(nsensors)]

    up_conditions = detector.get_up_conditions()

    for i in range(nsensors):
        up_conditions.append( (detector.get_sensor(i)).get_surface_condition() )
    down_conditions = detector.get_down_conditions()

    energyAtDeath, sensorIdsAll, evaporated, \
    total_time, step_count, initial_momentum, paths = QP_propagation(QPs, [X, Y, Z], up_conditions=up_conditions, down_conditions=down_conditions,
                                                                     evap_eff=detector.get_evaporation_eff(), T=T, plot_3d = plot_3d, 
                                                                     fixed_dir = fixed_dir, fixed_momentum = fixed_momentum, verbose=verbose, 
                                                                     wall_reflection_prob = detector.get_QP_wall_reflection_prob(), 
                                                                     wall_diffuse_prob = detector.get_QP_wall_diffuse_prob(),
                                                                     wall_Andreev_prob = detector.get_QP_wall_Andreev_prob(),
                                                                     sensor_reflection_prob = detector.get_QP_sensor_reflection_prob(), 
                                                                     sensor_diffuse_prob = detector.get_QP_sensor_diffuse_prob(),
                                                                     sensor_Andreev_prob = detector.get_QP_sensor_Andreev_prob(),
                                                                     max_dist=max_dist, step_size=step_size)

    for i in range(nsensors):
        hit_sensor_i_and_evaporated = (sensorIdsAll == i) & evaporated
        hit_sensor_i_and_not_evaporated = (sensorIdsAll == i) & ~evaporated

        energies[i] = (energyAtDeath[hit_sensor_i_and_evaporated] + detector.get_sensor(i).get_adsorptionGain()) #* detector.get_sensor(i).get_phononCollectionEfficiency()
        energies[i] = np.append(energies[i], energyAtDeath[hit_sensor_i_and_not_evaporated])# detector.get_sensor(i).get_phononCollectionEfficiency())

        arrivalTimes[i] = total_time[hit_sensor_i_and_evaporated]
        arrivalTimes[i] = np.append(arrivalTimes[i], total_time[hit_sensor_i_and_not_evaporated])


    # return Signal(sum(chAreas), chAreas, coincidence, arrivalTimes, bounced_flag = bounce_flag_with_sensor, paths = paths, flavor=flavor_with_sensor, momentums = momentum_hit, arrivals_unsorted = arrival_times)
    if not debug:
        return HestSignal(energies, arrivalTimes)
    else:
        return energyAtDeath, sensorIdsAll, evaporated, total_time, step_count, initial_momentum, paths

def GetSingletSignal(detector, nPhotons, X, Y, Z, max_dist = 10, step_size = .05, useMap = True, 
                     plot_3d = False, fixed_dir = None, verbose = False, debug = False):
    """
    Parameters
    ----------
        detector : vDetector
            virtual detector object inside of which you wish to simulate QP propagation
        nPhotons : int
            number of quasiparticles to simulate
        X, Y, Z : floats
            coordinates of the QP's origin in cm
        wall_reflection_prob : float
            Probability of a singlet photon reflecting off a wall/floor/ceiling
        useMap : boolean
            Whether to fully simulate QP motion from first principles, or use-precalculated collectoin efficiency maps
        plot_3d : boolean
            Generate a 3d plot of QP paths
        fixed_dir : 3-tuple of floats
            Allow the user to manually set the QP's initial direction's unit vector. Default is isotropy.
        verbose : bool
        debug : bool
            default format the signal as HestSignal() object. If True, print out a bunch of tracking 
            information that our detectors would normally not be privy to

    Returns (default)
    -----------------
        Signal : HestSignal
            Signal object containing hits times and energy deposits

    Returns (if debug == True)
    --------------------------
        energyAtDeath : array
            Array of energies (in eV) of the quasiparticles just before death/detection
        sensorIdsAll : array of ints
            Array of integers that define which sensor detected the QP
            -1 if not detected; otherwise the the corresponding integer
        evaporated : array of floats
            Array of floats that defines whether or not each QP is ever
            evaporated
        total_time : array of floats
            Array of times (in us) at which the QPs die or are detected
        step_count : array of ints
            Array of the number of steps before the particle died/collected
        initial_momentum : array of floats
            Array of the inital momenta of all QPs in keV/c
        paths : 3-tuple
            each object is a nQPs x 100 array; each row are the coordinates 
            of each QPs at vertices    
    """
    #check to see if there's an LCEmap loaded/generated
    if type(detector.get_QPEmap()) == int:
        useMap = False

    nsensors = detector.get_nsensors()
    arrivalTimes = [[] for x in range(nsensors)]
    energies = [[] for x in range(nsensors)]

    up_conditions = detector.get_up_conditions()


    for i in range(nsensors):
        up_conditions.append( (detector.get_sensor(i)).get_surface_condition() )
    down_conditions = detector.get_down_conditions()
    energyAtDeath, sensorIdsAll, total_time, step_count, paths = photon_propagation(nPhotons, [X,Y,Z], up_conditions, down_conditions, 
                                                                                    wall_reflection_prob = detector.get_UV_wall_reflection_prob(),
                                                                                    wall_diffuse_prob = detector.get_UV_wall_diffuse_prob(),
                                                                                    sensor_reflection_prob= detector.get_UV_sensor_reflection_prob(),
                                                                                    sensor_diffuse_prob = detector.get_UV_sensor_diffuse_prob(),
                                                                                    max_dist = max_dist, step_size = step_size,
                                                                                    plot_3d = plot_3d, fixed_dir = fixed_dir )
    
    for i in range(nsensors):
        hit_sensor_i = (sensorIdsAll == i)

        energies[i] = (energyAtDeath[hit_sensor_i]) * detector.get_sensor(i).get_phononCollectionEfficiency()

        arrivalTimes[i] = total_time[hit_sensor_i]
    
    if not debug:
        return HestSignal(energies, arrivalTimes)
    else:
        return energyAtDeath, sensorIdsAll, total_time, step_count, paths
    


def GetIRSignal(detector, nPhotons, X, Y, Z, max_dist = 10, step_size = .05, useMap = True, 
                plot_3d = False, fixed_dir = None, verbose = False, debug = False):
    """
    Parameters
    ----------
        detector : vDetector
            virtual detector object inside of which you wish to simulate QP propagation
        nPhotons : int
            number of quasiparticles to simulate
        X, Y, Z : floats
            coordinates of the QP's origin in cm
        wall_reflection_prob : float
            Probability of a singlet photon reflecting off a wall/floor/ceiling
        useMap : boolean
            Whether to fully simulate QP motion from first principles, or use-precalculated collectoin efficiency maps
        plot_3d : boolean
            Generate a 3d plot of QP paths
        fixed_dir : 3-tuple of floats
            Allow the user to manually set the QP's initial direction's unit vector. Default is isotropy.
        verbose : bool
        debug : bool
            default format the signal as HestSignal() object. If True, print out a bunch of tracking 
            information that our detectors would normally not be privy to

    Returns (default)
    -----------------
        Signal : HestSignal
            Signal object containing hits times and energy deposits

    Returns (if debug == True)
    --------------------------
        energyAtDeath : array
            Array of energies (in eV) of the quasiparticles just before death/detection
        sensorIdsAll : array of ints
            Array of integers that define which sensor detected the QP
            -1 if not detected; otherwise the the corresponding integer
        evaporated : array of floats
            Array of floats that defines whether or not each QP is ever
            evaporated
        total_time : array of floats
            Array of times (in us) at which the QPs die or are detected
        step_count : array of ints
            Array of the number of steps before the particle died/collected
        initial_momentum : array of floats
            Array of the inital momenta of all QPs in keV/c
        paths : 3-tuple
            each object is a nQPs x 100 array; each row are the coordinates 
            of each QPs at vertices    
    """
    #check to see if there's an LCEmap loaded/generated
    if type(detector.get_QPEmap()) == int:
        useMap = False

    nsensors = detector.get_nsensors()
    arrivalTimes = [[] for x in range(nsensors)]
    energies = [[] for x in range(nsensors)]

    up_conditions = detector.get_up_conditions()


    for i in range(nsensors):
        up_conditions.append( (detector.get_sensor(i)).get_surface_condition() )
    down_conditions = detector.get_down_conditions()
    energyAtDeath, sensorIdsAll, total_time, step_count, paths = photon_propagation(nPhotons, [X,Y,Z], up_conditions, down_conditions, 
                                                                                    wall_reflection_prob = detector.get_IR_wall_reflection_prob(),
                                                                                    wall_diffuse_prob = detector.get_IR_wall_diffuse_prob(),
                                                                                    sensor_reflection_prob= detector.get_IR_sensor_reflection_prob(),
                                                                                    sensor_diffuse_prob = detector.get_IR_sensor_diffuse_prob(),
                                                                                    max_dist = max_dist, step_size = step_size,
                                                                                    plot_3d = plot_3d, fixed_dir = fixed_dir )
    
    for i in range(nsensors):
        hit_sensor_i = (sensorIdsAll == i)

        energies[i] = (energyAtDeath[hit_sensor_i]) * detector.get_sensor(i).get_phononCollectionEfficiency()

        arrivalTimes[i] = total_time[hit_sensor_i]
    
    if not debug:
        return HestSignal(energies, arrivalTimes)
    else:
        return energyAtDeath, sensorIdsAll, total_time, step_count, paths

