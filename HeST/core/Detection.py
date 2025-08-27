from scipy.interpolate import interpn, interp1d
import numpy as np
import re
import matplotlib.pyplot as plt
from .HeST_Core import HestSignal, Random_QPmomentum, QP_dispersion, QP_velocity, get_phonon_mom_energy, get_rminus_mom_energy, get_rplus_mom_energy
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
    def __init__(self, surface_condition, baselineNoise, phononCollectionEfficiency=0.25, adsorptionGain = 6.0e-3):

        self.surface_condition = surface_condition
        self.baselineNoise     = np.array( baselineNoise )
        self.phononCollectionEfficiency  = phononCollectionEfficiency
        self.adsoprtionGain = adsorptionGain


    def set_surface_condition(self, f1):
        self.surface_condition = f1
    def set_baselineNoise(self, p1):
        self.baselineNoise = np.array( p1 )
    def set_phononCollectionEfficiency(self, p1):
        self.phononCollectionEfficiency = p1
    def set_adsorptionGain(self, p1):
        self.adsoprtionGain = p1

    def get_surface_condition(self):
        return self.surface_condition
    def get_baselineNoise(self):
        return self.baselineNoise
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
        (FIXME: doesn't this need to be a 4d array if we have > 1 sensor?)
        (FIXME: is the probability of detection or of striking a sensor?)
    LCEmap_positions : 3-tuple
        3-tuple of 1-D arrays (X Y Z) that correspond to the grid of points (in cm)
        that LCEmap measures light collection at
    QPEmap : array
        3-dimensional array with float-like elements. Each element describes 
        the probability of a phonon at a given position evaporating an atom that is 
        detected. 
        (FIXME: doesn't this need to be a 4d array if we have > 1 sensor?)
        (FIXME: is the probability of detection or of striking a sensor?)
    QPEmap_positions : 3-tuple
        3-tuple of 1-D arrays (X Y Z) that correspond to the grid of points (in cm)
        that QPEmap measures phonon collection at   
    photon_reflection_prob : float
        probability of a sensor reflecting a photon
        (FIXME: do we need separate reflection probs for IR and UV?)
    QP_reflection_prob : float
        probability of the top/bottom/wall reflecting a quasiparticle
    QP_diffuse_prob : float
        probability of quasiparticles reflecting diffusley at a wall


    """

    def __init__(self, top_conditions, bottom_conditions, wall_conditions, liquid_surface, liquid_conditions,
                 evaporation_eff = np.array([1,1,1,1,1,1,1]), sensors=[], LCEmap=0, LCEmap_positions=0., QPEmap=0, QPEmap_positions=0.,
                 photon_reflection_prob=0., QP_reflection_prob=0.,QP_diffuse_prob=0.):
        

        
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
        self.photon_reflection_prob = photon_reflection_prob
        self.QP_reflection_prob = QP_reflection_prob
        self.diffuse_prob = QP_diffuse_prob
        
        
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
    def set_photon_reflection_prob(self, p1):
        self.photon_reflection_prob = p1
    def set_QP_reflection_prob(self, p1):
        self.QP_reflection_prob = p1
    def set_diffuse_prob(self, p1):
        self.diffuse_prob = p1


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
    def get_photon_reflection_prob(self):
        return self.photon_reflection_prob
    def get_QP_reflection_prob(self):
        return self.QP_reflection_prob
    def get_diffuse_prob(self):
        return self.diffuse_prob
    


    def load_LCEmap(self, filename):
        self.LCEmap = np.load(filename)

    def load_QPEmap(self, filename):
        self.QPEmap = np.load(filename)

    def create_LCEmap(self, x_array, y_array, z_array, nPhotons=10000, filestring="detector_LCEmap"):
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
        
        refl_prob = self.get_photon_reflection_prob()
        for xx in range(len(x)):
            print("%i / %i" % (xx, len(x)))
            for yy in range(len(y)):
                for zz in range(len(z)):
                    
                    pos = np.array([x[xx], y[yy], z[zz]])
                    if (self.get_liquid_conditions())(*pos) == False:
                        continue
                    hitProbs = [0.]*nsensors
                    for n in range(nPhotons):
                    
                        hit, arrival_time, n, xs, ys, zs, sensor_id = photon_propagation(pos, conditions, refl_prob)
                        if hit > 0:
                            hitProbs[sensor_id] += 1.
                        
                        
                                
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
        
        refl_prob = self.get_QP_reflection_prob()
        for xx in range(len(x)):
            print("%i / %i" % (xx, len(x)))
            for yy in range(len(y)):
                for zz in range(len(z)):
                    
                    pos = np.array([x[xx], y[yy], z[zz]])
                    if (self.get_liquid_conditions())(*pos) == False:
                        continue
                    hitProbs = [0.]*nsensors
                    for n in range(nQPs):
                        hits, times, sensor_ids = QP_propagation(nQPs, pos, conditions, refl_prob, evap_eff=self.evaporation_eff, T=T, debug=True)
                        if len(hits) > 0:
                            hitProbs[sensor_ids] += 1.
                        
                                
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


def intersection(start, direction, conditions):
    """
    Finds the surface and point of boundary intersection by  assuming rectilinear motion and interpolating.
    Takes vector-valued positions and directions to track > 1 particle at a time. 
    
    Parameters
    ----------
    start : array
        3xN Array with each column denoting a particle's (x,y,z) initial position
    direction : array
        3xN Array with each column denoting a particle's (x,y,z) initial velocity direction (FIXME: velocity or momentum?)
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
    t = np.array([np.linspace(0, 10, 500) for i in range(len(start[0]))])  # Parameter range for the line
    # Calculate the line coordinates
    x_line = start[0][:, np.newaxis] + t * direction[0][:, np.newaxis] # Project trajectory forward rectilinearly
    y_line = start[1][:, np.newaxis] + t * direction[1][:, np.newaxis]
    z_line = start[2][:, np.newaxis] + t * direction[2][:, np.newaxis]

    
    dist = np.ones(len(start[0]))*9999. # Placeholders
    coords = [np.full(len(start[0]), None), np.full(len(start[0]), None), np.full(len(start[0]), None)]
    surface_type = np.full(len(start[0]), None)

    for cond in conditions:
        cut, surface = cond(x_line, y_line, z_line)
        #if an array's max value is repeated, argmax returns the index of the first 
        first_ints = np.array([np.argmax(~cut[i]) for i in range(len(cut))]) #indices of first time the condition fails (goes from True to False)
        d = t[np.arange(t.shape[0]), first_ints] # t-value at which the boundary is first crossed
        cond = ( d < dist ) & (first_ints > 0) # boolean of for which tracked particles this is first cut passed
        dist = np.where(cond, d, dist)    
        coords[0] = np.where(cond, x_line[np.arange(x_line.shape[0]), first_ints-1], coords[0])
        coords[1] = np.where(cond, y_line[np.arange(y_line.shape[0]), first_ints-1], coords[1])
        coords[2] = np.where(cond, z_line[np.arange(z_line.shape[0]), first_ints-1], coords[2])
        surface_type = np.where( cond, surface, surface_type )
    return np.array(coords[0], dtype = float), np.array(coords[1], dtype = float), np.array(coords[2], dtype= float), surface_type

def find_surface_intersection(start, direction, up_conditions, down_conditions, alive):
    """
    Wrapper for intersection that partitions particles as going upward and downward and only checks for the conditions
    relevant to each. (FIXME: why the hell would we do this? Does checking too many conditions slow things down?)

    Parameters
    ----------
        start : array
            3xN Array with each column denoting a particle's (x,y,z) initial position
        direction : array
            3xN Array with each column denoting a particle's (x,y,z) initial velocity direction (FIXME: velocity or momentum?)
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
    if any(up):
        X1[up], Y1[up], Z1[up], surface_type[up] = intersection(start[:, up], direction[:, up], up_conditions)
    if any(down):
        X1[down], Y1[down], Z1[down], surface_type[down] = intersection(start[:, down], direction[:, down], down_conditions)
    return X1, Y1, Z1, surface_type




def diffuse_and_specular(surface, pos, direction, diffuse_prob):
    """
    Takes in initial conditions of a particle before striking a surface 
    and returns direction directly after reflection. Assumes some probabilty 
    of of diffuse reflection at the wall/floor surfaces
    (FIXME: Is this for QP or photon or both?)

    Parameters
    ----------
        surface : string
            String denoting the surface type; 'Liquid', 'XY', or 'Z'
        pos : 3-tuple
            3-tuple containing the x/y/z position of the reflection
        direction : 3-tuple
            3-tuple correspodning to the inital velocity vector's direction
        diffuse_prob : float
            Probability of a QP reflecting diffusely on the wall/floor
            (FIXME: check that this isn't passed to photons too? There's 
            no reason to think diffuse probality would be the same )

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
        return dx, dy, dz

    if surface == 'XY':
        #we will do both specular and diffuse here. 
        specular_cut = np.random.uniform(size = (len(dx),)) > diffuse_prob

        dx[specular_cut], dy[specular_cut], dz[specular_cut] = specular_reflection(surface = 'XY',
                                                                                   pos = (x[specular_cut], y[specular_cut],z[specular_cut]),
                                                                                   dir = (dx[specular_cut], dy[specular_cut], dz[specular_cut])) 

        dx[~specular_cut], dy[~specular_cut], dz[~specular_cut] = diffuse_reflection(surface= 'XY',
                                                                                     pos = (x[~specular_cut], y[~specular_cut], z[~specular_cut]),
                                                                                     dir = (dx[~specular_cut], dy[~specular_cut], dz[~specular_cut])) 
        return dx, dy, dz
    if surface == 'Z':
        # we will do both specular and diffuse
        specular_cut = np.random.uniform(size = (len(dx),)) > diffuse_prob

        dx[specular_cut], dy[specular_cut], dz[specular_cut] = specular_reflection(surface = 'XY',
                                                                                   pos = (x[specular_cut], y[specular_cut], z[specular_cut]),
                                                                                   dir = (dx[specular_cut], dy[specular_cut], dz[specular_cut]))
        
        dx[~specular_cut], dy[~specular_cut], dz[~specular_cut] = diffuse_reflection(surface = 'Z',
                                                                                     pos = (x[~specular_cut], y[~specular_cut], z[~specular_cut]),
                                                                                     dir = (dx[~specular_cut], dy[~specular_cut], dz[~specular_cut]))

        return dx, dy, dz

def specular_reflection(surface, pos = (0.0,0.0,0.0), dir = (0.0,0.0,-1.0)):
    x = pos[0]
    y = pos[1]
    z = pos[2]

    if surface == 'XY':
        #Unit projection of position vector into xy plane
        nx, ny = x/np.sqrt(x**2,y**2), y/np.sqrt(x**2+y**2)
        dx = dir[0]
        dy = dir[1]
        dz = dir[2]
        #Unit projection of direction vector into the xy plane
        rx, ry = dx/np.sqrt(dx**2,dy**2), dy/np.sqrt(dx**2+dy**2)
        r_dot_n = (nx*rx + ny*ry)
        #Subtracting off twice the compontentcomponent

        dx = rx - 2 * r_dot_n * nx
        dy = ry - 2 * r_dot_n * ny

        dx, dy, dz = dx/np.sqrt(dx**2+dy**2+dz**2), dy/np.sqrt(dx**2+dy**2+dz**2), dz/np.sqrt(dx**2+dy**2+dz**2)
        return dx, dy, dz
    
    if surface == 'Z':
        dx = dir[0]
        dy = dir[1]
        dz = dir[2]
        dz = -dz
        return dx, dy, dz

def diffuse_reflection(surface, pos = (0.0,0.0,0.0), dir = (0.0,0.0,-1.0)):

    """
    Generates a reflection direction given a surface orientation assuming
    Lambertian (diffuse) reflection.
    Meant to take in a surface and generate a random direction relative to that. 
    Args:
        surface_type (_type_): _description_
    """
    #first we do the case where the surface is XY
    x = pos[0]
    y = pos[1]
    z = pos[2]
    
    if surface == 'XY':
        # ~~~~~~~~~~~ Doing this case by case, dx, dy, dz should already be cut to be alive and diffuse, and at an 'XY' surface
        # polar coordinates in the frame of reference with zhat inward normal from the surface 
        phi_1, sintheta_1 = 2*np.pi*np.random.uniform(0,1, size = len(x)), np.random.uniform(0,1, size = len(x))
        costheta_1 =np.sqrt(1 - sintheta_1**2)

        phi_pos_1 = np.arctan2(x, y)

        dx = - (np.sin(phi_pos_1) * np.sin(phi_1) * sintheta_1) - (np.cos(phi_pos_1) * costheta_1)
        dy = (np.cos(phi_pos_1) * np.sin(phi_1) * sintheta_1) - (np.sin(phi_pos_1) * costheta_1)
        dz = np.cos(phi_1) * sintheta_1
        return dx, dy, dz

    if surface == 'Z': #FIXME: this doesn't account for reflections off the ceiling
        # ~~~~~~~~~ This should where dx, dy, dz and X, Y, Z have already been cut to be ALIVE, DIFFUSE, and at a 'Z' surface
        phi_2, sintheta_2 = 2*np.pi*np.random.uniform(0,1, size = len(x)), np.random.uniform(0,1, size = len(x))
        zdir = dir[2]
        theta_2 = np.arcsin(sintheta_2)
        dx = np.cos( phi_2 ) * np.sin( theta_2 )
        dy  = np.sin( phi_2 ) * np.sin( theta_2 )
        dz  = np.cos(theta_2) * -1 * np.sign(zdir) # determine whether we're hitting the top or bottom
        return dx, dy, dz



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

    pxi, pyi = momentum * dx, momentum * dy
    pxf, pyf = pxi, pyi
    pzf = np.sqrt(2 * m/1000 * (energy/1000 - Vb/1000) - pxi**2 - pyi**2)

    momentum_final = np.sqrt(pxf**2 + pyf**2 + pzf**2)
    dx, dy, dz = pxf/momentum, pyf/momentum, pzf/momentum

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
            FIXME: this currently does nothing, but we probably need a bunch of data to peg this effect
            down. For now, we're doing the right thing
        evap_eff : array of floats
            7-term array that indexes the evaporation probability for each of the QP momentum bins
    Returns
    -------
        no_evap_bools : array of bool
            True if evaporation is simulated to occur 
        
    """
    no_evap_bools = np.full_like(p,fill_value= False, dtype=bool)
    the_nums = np.random.uniform(low = 0.0, high = 1.0, size = len(p))
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
# @np.vectorize
# def extract_number(s):
#     if s == None:
#         return False
#     s = list(s)
#     last_element = s[-1]
#     if last_element.isdigit():
#         return int(last_element)
#     return False


# Define a function to handle reflections off of walls
def wall_reflect(X, Y, dx, dy, dz, diffuse = False):
    #now we need to define a new function which can do specular and diffuse
    #ok so you take the normal vector and normalize it (this means you need some method for coming up with the normal vector. Actually this is easy, because it is a circle)
    normal_vector = np.array([X, Y, 0.])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    incident_vector = np.array([dx, dy, dz])
    reflected_vector = incident_vector - 2. * np.dot(incident_vector, normal_vector) * normal_vector
    dx, dy, dz = reflected_vector / np.linalg.norm(reflected_vector)
    return dx, dy, dz



@np.vectorize
def assign_flavors(p):
    if p < 0.955:
        flavor = 'low energy phonon'
    elif .949 < p <   2.174972:
        flavor = 'phonon'
    elif 2.22538 < p < 3.7899:
        flavor = 'R-'
    elif 3.840<p < 4.555:
        flavor = 'R+'
    elif 4.555 < p:
        flavor = 'high energy Roton'
    else:
        flavor = 'slow moving'
    return flavor

# ~~~~~~~~~~~~ Flavor Switching Functions ~~~~~~~~~~~~~~~
@np.vectorize
def compute_conserved_mom(X, Y, dx, dy, dz, momentum):
    direction = np.array([dx, dy, dz])
    xy_vec = -1 * np.array([X, Y, 0], dtype = float) # compute the normal vector of the wall
    xy_vec = xy_vec/(np.sqrt(np.sum(xy_vec**2))) # normalize this vector
    dir_transverse = direction- np.dot(direction, xy_vec) * xy_vec
    conserved_momentum= momentum * dir_transverse
    return conserved_momentum[0], conserved_momentum[1], conserved_momentum[2]


dispersion_data_path = os.path.dirname(os.path.abspath(__file__))+ '/../dispersion_curves/dispersion_data.csv'

phonon_interp = get_phonon_mom_energy(dispersion_data_path)
rminus_interp = get_rminus_mom_energy(dispersion_data_path)
rplus_interp = get_rplus_mom_energy(dispersion_data_path)
def random_conversion(energy, momentum, old_flavor, X, Y, Z, dx, dy, dz):
    """ This function handles the conversion process, which is done by randomely choosing new momentums, that match the original. 

    Args:
        energy (_type_): The energies of the array of quasiparticles. This should be in meV 
        momentum (_type_): _description_
        flavor (_type_): _description_
        pos (_type_): _description_
        direction (_type_): _description_

    Returns:
        _type_: _description_
    """
    # generate a set of random numbers 
    r_nums = np.random.uniform(low = 0, high= 1, size = (len(energy), 3))
    old_momentum = momentum
    # decide on who goes where
    phonon_mom = phonon_interp(energy)
    rminus_mom = rminus_interp(energy)
    rplus_mom = rplus_interp(energy)
    flavors = np.column_stack((np.full(np.shape(phonon_mom), r'phonon'), np.full(np.shape(phonon_mom), r'R-') , np.full(np.shape(phonon_mom), r'R+')))
    
    cons_x, cons_y, cons_z= compute_conserved_mom(X, Y, dx, dy, dz, old_momentum)
    # compute the conserved momentum, which is the momentum in this direction
    conserved_mom_sq = (cons_x**2 + cons_y**2 + cons_z**2)
    phonon_mask = phonon_mom**2 < conserved_mom_sq 
    rminus_mask = rminus_mom**2 < conserved_mom_sq
    rplus_mask = rplus_mom**2 < conserved_mom_sq
    # generate arrays of flavors
    r_nums[:,0][phonon_mask] = 0
    r_nums[:,1][rminus_mask] = 0
    r_nums[:,2][rplus_mask] = 0
    momentums = np.column_stack((phonon_mom, rminus_mom, rplus_mom))
    max_indices = np.argmax(r_nums, axis=1)  # Correct indexing along rows
    momentum = momentums[np.arange(len(energy)), max_indices]  # Select momenta for each energy
    flavor = flavors[np.arange(len(energy)), max_indices]
    no_change_mask = (flavor == old_flavor)
    if any(dx[~no_change_mask]): 
        dx[~no_change_mask], dy[~no_change_mask], dz[~no_change_mask]= convert_off_XY(cons_x[~no_change_mask],
                                                                    cons_y[~no_change_mask], cons_z[~no_change_mask], momentum[~no_change_mask],
                                                                    X[~no_change_mask], Y[~no_change_mask], Z[~no_change_mask], dx[~no_change_mask],
                                                                    dy[~no_change_mask], dz[~no_change_mask])
    return momentum, flavor, dx, dy, dz


@np.vectorize
def convert_off_XY(conserved_x, conserved_y, conserved_z, new_momentum, X, Y, Z, dx, dy, dz):
    """Handles the converting and reflection off of XY surface (the cylindrical area) based upon QP kinematics. 
    Conserves the translational momentum, but does the longitudinal momentum.

    Args:
        old_momentum (float): This is the old momentum, aka the input momentum to the system. This will be negative if the input QP was previously a R- roton. Should be in units of KeV/c 
        new_momentum (_type_): This is the new momentum, after conversion. This should be negative if the desired QP output is a R- roton.  Should be in units of KeV/c
        pos (tuple or ndarray): This is the position of intersection with the wall
        direction (tuple or ndarray): This is the incident direction 
    """
    direction = np.array([dx, dy, dz])
    xy_vec = -1 * np.array([X, Y, 0], dtype = float) # compute the normal vector of the wall
    xy_vec = xy_vec/(np.sqrt(np.sum(xy_vec**2))) # normalize this vector
    # compute the conserved momentum, which is the momentum in this direction
    new_momentum_parallel= np.sqrt(new_momentum**2 - (conserved_x**2 + conserved_y**2 + conserved_z**2))
    if new_momentum < 0:
        new_momentum_parallel = new_momentum_parallel *-1
    new_total_mom_vec = new_momentum_parallel * xy_vec + np.array([conserved_x, conserved_y, conserved_z])
    new_total_mom_vec = new_total_mom_vec/new_momentum
    direction = new_total_mom_vec/np.linalg.norm(new_total_mom_vec, ord = 2)
    return direction[0], direction[1], direction[2]

def random_conversion_off_z(energy, momentum, old_flavor, dx, dy, dz, verbose = False, debug = False):
    """Handles the random flavor switching off a surface of surface type 'Z'

    Args:
        energy (array): This should be in units of KeV 
        momentum (array): This should be in units of KeV/c, and it should have the same dimension as energy
        old_flavor (_type_): Describes the flavor of each QP, and it has the same dimension as the energy array. 
        dx (array): direction in x 
        dy (array): direction in y
        dz (array): Direction in Z

    Returns:
        _type_: _description_
    """
    r_nums = np.random.uniform(low = 0, high= 1, size = (len(energy), 3))
    old_momentum = momentum
    # decide on who goes where
    phonon_mom = phonon_interp(energy * 1e3)
    rminus_mom = rminus_interp(energy * 1e3)
    rplus_mom = rplus_interp(energy * 1e3)
    flavors = np.column_stack((np.full(np.shape(phonon_mom), r'phonon'), np.full(np.shape(phonon_mom), r'R-') , np.full(np.shape(phonon_mom), r'R+')))
    
    cons_x = momentum * dx
    cons_y = momentum * dy
    # compute the conserved momentum, which is the x,y momentum in this direction
    conserved_mom_sq = (cons_x**2 + cons_y**2) 
    phonon_mask = phonon_mom**2 < conserved_mom_sq 
    rminus_mask = rminus_mom**2 < conserved_mom_sq
    rplus_mask = rplus_mom**2 < conserved_mom_sq
    # generate arrays of flavors
    r_nums[:,0][phonon_mask] = 0
    r_nums[:,1][rminus_mask] = 0
    r_nums[:,2][rplus_mask] = 0 
    momentums = np.column_stack((phonon_mom, -1 * rminus_mom, rplus_mom))
    max_indices = np.argmax(r_nums, axis=1)  # Correct indexing along rows
    momentum = momentums[np.arange(len(energy)), max_indices]  # Select momenta for each energy
    flavor = flavors[np.arange(len(energy)), max_indices]
    no_change_mask = (flavor == old_flavor)
    if any(~no_change_mask): 
        dx[~no_change_mask], dy[~no_change_mask], dz[~no_change_mask] = conserve_z(momentum[~no_change_mask], conserved_mom_sq[~no_change_mask], 
                                                                                   dx[~no_change_mask], dy[~no_change_mask], dx[~no_change_mask])

    if verbose and debug:
        print('conserved momentum:')
        print(conserved_mom_sq)
        print('masks')
        print(phonon_mask, rminus_mask, rplus_mask)    
        print('momentums')
        print(momentums)
        print('random numbers')
        print(r_nums)
        print('final flavor and masks')
        print(flavor, no_change_mask)
        print('final direction vector')
        print(dx, dy, dz)
    return momentum, flavor, dx, dy, dz

@np.vectorize
def conserve_z(momentum, conserved_mom, cons_x, cons_y, dz):
    dz_prime = np.sqrt(momentum**2 - conserved_mom)
    # pack them together

    new_direction_vec = np.array([cons_x, cons_y, dz_prime])
    new_direction_vec = new_direction_vec/momentum
    # normalize
    new_direction_vec = new_direction_vec/np.linalg.norm(new_direction_vec)
    if new_direction_vec[2] < 0:
        new_direction_vec[2] = -1 * new_direction_vec[2]

    return new_direction_vec[0], new_direction_vec[1], new_direction_vec[2] 




"""
##############################
Quasiparticle Propagation
##############################
"""
def QP_propagation(nQPs, start, up_conditions, down_conditions, reflection_prob, evap_eff=np.array([1,1,1,1,1,1,1]), diffuse_prob = 0.0, T=2., plot_3d=False, fixed_dir = None, fixed_momentum = None, verbose = False, flavor_switching = False):
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
        flavor_switching : boolean
            Simulate energy-conserving flavor switching of quasiparticles at reflections 


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
    paths = (0,0,0)

    particles_x = np.zeros(shape=(nQPs, 100))
    particles_y = np.zeros(shape=(nQPs, 100))
    particles_z = np.zeros(shape=(nQPs, 100))
    X, Y, Z = start[0], start[1], start[2]
    dx, dy, dz = generate_random_direction(nQPs)

    if fixed_dir is not None: 
        dx = np.full(nQPs, fixed_dir[0]) 
        dy = np.full(nQPs, fixed_dir[1]) 
        dz = np.full(nQPs, fixed_dir[2]) 

    total_time = np.zeros(nQPs, dtype=float)
    n=0
    #This draws from a k**2 distribution, essentially trying to follow the density of states
    momentum = Random_QPmomentum(nQPs, T=T) #keV/c
    initial_momentum=np.copy(momentum)
    if fixed_momentum is not None:
        momentum = np.full(nQPs, fixed_momentum)
    flavor = assign_flavors(np.abs(momentum))
    #prepare the alive tracker, and then cut out those which we can't sense
    alive = np.ones(nQPs, dtype=int)
    cond = momentum <  1.1 #Lifetime cut
    alive = np.where( cond, 0, alive)
    #velocity and energy, both arrays of length nQP
    velocity = QP_velocity(momentum) #m/s
    energy = QP_dispersion(momentum) #eV
    evaporated = np.zeros( nQPs, dtype=bool)
    # instead of inverting the velocity, we invert the momentum and later, when we use velocity, we take the magnitude of it anyways
    v_mask = velocity < 0
    momentum[v_mask] = -momentum[v_mask]
     
    energyAtDeath = np.zeros(nQPs, dtype=float)
    ids = np.full(nQPs, None)
    step_count = np.zeros_like(alive)
    sensorIdsAll = -1*np.ones_like(alive, dtype=int)

    living = alive > 0.5
    particles_x[:, 0][living] = X[living]
    particles_y[:, 0][living] = Y[living]
    particles_z[:, 0][living] = Z[living]

    while sum(alive) > 0: # loop until all particles are dead
        
        n+=1
        living = ( alive > 0.5 )

        if verbose: 
            print(f'starting point: {np.array([X, Y, Z])}, direction of travel: {np.array([dx, dy, dz])}')
        X1, Y1, Z1, surface_type = find_surface_intersection(np.array([X, Y, Z]), np.array([dx, dy, dz]), up_conditions, down_conditions, living)
        if verbose: print(f'This is the surface type {surface_type}')

        hit_surface_check= (surface_type != None)
        dx[~hit_surface_check], dy[~hit_surface_check], dz[~hit_surface_check] = np.zeros_like(dx[~hit_surface_check]),np.zeros_like(dx[~hit_surface_check]),np.zeros_like(dx[~hit_surface_check])   
        
        alive[living] = np.where( hit_surface_check[living], alive[living], 0)
        living = ( alive > 0.5 )

        step_count[living]  = np.full_like(alive[living], fill_value=n)

        dist_sq = (pow(X1[living]-X[living],2.)+pow(Y1[living]-Y[living], 2.)+pow(Z1[living]-Z[living],2.)).astype(float)      
        total_time[living] = total_time[living] + np.sqrt(dist_sq)/np.abs(velocity[living]*1.0e-4)  #us
        
        check1 = (surface_type == "Liquid")

        alive_He_surface_check = living & check1
        
        # Handling evaporations/reflections at the helium surface
        if any(alive_He_surface_check):
            if verbose: print('~~~~~~~~~~~~~~~~~~~~~~~ Computing Evaporations ~~~~~~~~~~~~~~~~~~~~')

            critical_angles = critical_angle(energy, momentum) 
            incident_angles = np.arccos(dz) #(FIXME: we need to add a bunch of checks to ensure that dx,dy, dz form a unit vector, even when user-defined)
            evap_bools = evap_prob_of_p_theta(np.abs(momentum), incident_angles, evap_eff) # True if we're simulating to evaporate
            if verbose: 
                print('This is evap bool')
                print('\n')
                print(evap_bools)
                print('\n')
                print('This is flavors')
                print(flavor)
                print(flavor_switching) 
            angle_check = (incident_angles < critical_angles) #True if evaporation is kinematically permissible
            no_evap = (~angle_check) | (~evap_bools) #Doesn't evaporate for kinematic or probabilistic reasons
            a_L_noevap = alive_He_surface_check & no_evap # Mask for particles that are alive at the liquid interface but don't evaporate
            if verbose:
                print(f'this is critical angles {critical_angles} and this is incident {incident_angles}')
                print(f'This is no_evap {a_L_noevap}')

            dx[a_L_noevap], dy[a_L_noevap], dz[a_L_noevap] = diffuse_and_specular('Liquid', (X[a_L_noevap], Y[a_L_noevap], 
                                                                                Z[a_L_noevap]), (dx[a_L_noevap], dy[a_L_noevap], dz[a_L_noevap]),
                                                                                 diffuse_prob=diffuse_prob)

            
            a_L_evap = alive_He_surface_check & ~no_evap 

            dx[a_L_evap], dy[a_L_evap], dz[a_L_evap], momentum[a_L_evap], energy[a_L_evap], velocity[a_L_evap] = evaporation(momentum[a_L_evap],energy[a_L_evap],direction=(dx[a_L_evap], dy[a_L_evap], dz[a_L_evap]))

            evaporated[a_L_evap] = True

            # ~~~~~~~~~~ If the QP doesn't evaporate, leave it where it is
            X[a_L_noevap] = X1[a_L_noevap]
            Y[a_L_noevap] = Y1[a_L_noevap]
            Z[a_L_noevap] = Z1[a_L_noevap] - 0.05 

            # ~~~~~~~~~~ If the QP evaporates, move it up above the liquid surface 
            X[a_L_evap] = X1[a_L_evap]
            Y[a_L_evap] = Y1[a_L_evap]
            Z[a_L_evap] = Z1[a_L_evap] + 0.05

                                   
        check2 = np.array(["sensor" in str(s) for s in surface_type])

        # Handling the case in which the QP hits a sensor
        # FIXME: this currently only accomodates sensors above the helium
        alive_at_sensor_check = living & check2
        if any(alive_at_sensor_check):
            if verbose: print('~~~~~~~~~~~~~~~~~~~~~~~ Computing Deposits ~~~~~~~~~~~~~~~~~~~~')

            sensorIdsAll[alive_at_sensor_check] = np.vectorize(extract_number)(surface_type[alive_at_sensor_check])
            energyAtDeath[alive_at_sensor_check] = energy[alive_at_sensor_check] 

            alive[alive_at_sensor_check] = 0 #kill off QPs, they've hit a sensor

            X[alive_at_sensor_check] = X1[alive_at_sensor_check]
            Y[alive_at_sensor_check] = Y1[alive_at_sensor_check]
            Z[alive_at_sensor_check] = Z1[alive_at_sensor_check]

        
        check3 = (surface_type == 'XY') | (surface_type == 'Z') 
        living_not_evaporated_wall = living & check3 & ~evaporated
        living_evaporated_wall = living & check3 & evaporated

        #Handle evaporated He atoms that hit a wall/ceiling
        if any(living_evaporated_wall):
            energyAtDeath[living_evaporated_wall] = energy[living_evaporated_wall]
            alive[living_evaporated_wall] = 0
      
            X[living_not_evaporated_wall] = X1[living_not_evaporated_wall]
            Y[living_not_evaporated_wall] = Y1[living_not_evaporated_wall]
            Z[living_not_evaporated_wall] = Z1[living_not_evaporated_wall]

        #Handle QPs that hit a wall before evaporation
        if any(living_not_evaporated_wall):
            if verbose: print('~~~~~~~~~~~~~~~~~~~~~~~ Computing Wall Reflections ~~~~~~~~~~~~~~~~~~~~')
            # Let's simplify this
            XY_check = (surface_type == 'XY')
            Z_check = (surface_type == 'Z')
            a_xy_check = living_not_evaporated_wall & XY_check
            a_z_check = living_not_evaporated_wall & Z_check
            r = np.random.random(len(surface_type[living_not_evaporated_wall]))
            cond = (r > reflection_prob)
            energyAtDeath[living_not_evaporated_wall] = np.where(cond, energy[living_not_evaporated_wall],  energyAtDeath[living_not_evaporated_wall])
            alive[living_not_evaporated_wall] = np.where(cond, 0, alive[living_not_evaporated_wall]) #kill of those that don't reflect
            converting_range = (flavor == 'phonon') | (flavor == 'R-') | (flavor == 'R+')
            # ~~~~~~~~~~~ First do case of reflection off XY.
            if any(a_xy_check):
                dx[a_xy_check], dy[a_xy_check], dz[a_xy_check] = diffuse_and_specular('XY', pos = (X1[a_xy_check],Y1[a_xy_check],Z1[a_xy_check]), 
                                                direction = (dx[a_xy_check], dy[a_xy_check], dz[a_xy_check]), 
                                                diffuse_prob=diffuse_prob)
                if flavor_switching:
                    a_xy_switch_check = a_xy_check & converting_range
                    if any(a_xy_switch_check):
                        if verbose and debug: hold_flavor = flavor[a_xy_switch_check]
                        momentum[a_xy_switch_check], flavor[a_xy_switch_check], dx[a_xy_switch_check], dy[a_xy_switch_check], dz[a_xy_switch_check] = random_conversion(old_flavor = flavor[a_xy_switch_check], energy=energy[a_xy_switch_check] * 1e6,
                                                                                                                                 momentum=momentum[a_xy_switch_check], 
                                                                                                                                 X = X1[a_xy_switch_check],Y = Y1[a_xy_switch_check],Z  = Z1[a_xy_switch_check], 
                                                                                                                                dx = dx[a_xy_switch_check], dy = dy[a_xy_switch_check], dz = dz[a_xy_switch_check])
                        velocity[a_xy_switch_check] = QP_velocity(np.abs(momentum[a_xy_switch_check]))
                        if verbose and debug: 
                            print(f' old flavors: {hold_flavor} and new flavors  {flavor[a_xy_switch_check]}')
                            print(f' travel direction after flavor switching {np.array([dx, dy, dz])}')

            # ~~~~~~~~~~~ Reflection off Z
            if any(a_z_check):
                dx[a_z_check], dy[a_z_check], dz[a_z_check] = diffuse_and_specular('Z', pos = (X1[a_z_check],Y1[a_z_check],Z1[a_z_check]), 
                                                direction = (dx[a_z_check], dy[a_z_check], dz[a_z_check]), 
                                                diffuse_prob=diffuse_prob)           
                a_z_switch_check = a_z_check & converting_range
                if flavor_switching:
                    if any(a_z_switch_check):
                        if verbose and debug: hold_flavor = flavor[a_z_switch_check]
                        momentum[a_z_switch_check], flavor[a_z_switch_check], dx[a_z_switch_check], dy[a_z_switch_check], dz[a_z_switch_check] = random_conversion_off_z(old_flavor = flavor[a_z_switch_check], energy=energy[a_z_switch_check] * 1e3, 
                                                                                                                                    momentum=momentum[a_z_switch_check], 
                                                                                                                                    dx = dx[a_z_switch_check], dy = dy[a_z_switch_check], dz = dz[a_z_switch_check])                    
                        velocity[a_z_switch_check] = QP_velocity(np.abs(momentum[a_z_switch_check]))     
                        if verbose and debug: 
                            print(f' old flavors: {hold_flavor} and new flavors  {flavor[a_z_switch_check]}')
                            print(f' travel direction after flavor switching {np.array([dx, dy, dz])}')

            X[living_not_evaporated_wall] = X1[living_not_evaporated_wall]
            Y[living_not_evaporated_wall] = Y1[living_not_evaporated_wall]
            Z[living_not_evaporated_wall] = Z1[living_not_evaporated_wall]

        #at end of loop, record new positions
        try: 
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
        
def photon_propagation(nPhotons, start, up_conditions, down_conditions, wall_reflection_prob = 0.0, plot_3d=False, fixed_dir = None):
    """
    Tracking of Quasiparticles through medium. 
   
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
    #make sure the we have an array of starting positions for each photon
    if np.isscalar(nPhotons):
        X = np.full(nPhotons, start[0])
        Y = np.full(nPhotons, start[1])
        Z = np.full(nPhotons, start[2])
        start = np.array([X, Y, Z])
    paths = (0,0,0)

    particles_x = np.zeros(shape=(nPhotons, 100))
    particles_y = np.zeros(shape=(nPhotons, 100))
    particles_z = np.zeros(shape=(nPhotons, 100))
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
    sensorIdsAll = -1*np.ones_like(alive, dtype=int)

    living = alive > 0.5
    particles_x[:, 0][living] = X[living]
    particles_y[:, 0][living] = Y[living]
    particles_z[:, 0][living] = Z[living]

    while sum(alive) > 0:
        n+=1
        living = ( alive > 0.5 )
        
        X1, Y1, Z1, surface_type = find_surface_intersection(np.array([X, Y, Z]), np.array([dx, dy, dz]), up_conditions, down_conditions, living)

        hit_surface_check = (surface_type != None)
        dx[~hit_surface_check], dy[~hit_surface_check], dz[~hit_surface_check] = np.zeros_like(dx[~hit_surface_check]),np.zeros_like(dx[~hit_surface_check]),np.zeros_like(dx[~hit_surface_check])   

        alive[living] = np.where( hit_surface_check[living], alive[living], 0)
        living = ( alive > 0.5 )

        step_count[living] = np.full_like(alive[living], fill_value=n)

        dist_sq = (pow(X1[living]-X[living],2.)+pow(Y1[living]-Y[living], 2.)+pow(Z1[living]-Z[living],2.)).astype(float)      
        total_time[living] = total_time[living] + np.sqrt(dist_sq)/velocity  #us
        
        check1 = (surface_type == "Liquid")

        alive_He_surface_check = living & check1
        # Handling transmission through the Helium surface
        # FIXME: for now, this doesn't account for refraction

        if any(alive_He_surface_check):

            X[alive_He_surface_check] = X1[alive_He_surface_check] 
            Y[alive_He_surface_check] = Y1[alive_He_surface_check] 
            Z[alive_He_surface_check] = Z1[alive_He_surface_check] + .005

        check2 = np.array(["sensor" in str(s) for s in surface_type])

        alive_at_sensor_check = living & check2
        if any(alive_at_sensor_check):

            sensorIdsAll[alive_at_sensor_check] = np.vectorize(extract_number)(surface_type[alive_at_sensor_check])
            energyAtDeath[alive_at_sensor_check] = energy[alive_at_sensor_check] 

            alive[alive_at_sensor_check] = 0 #kill off QPs, they've hit a sensor

            X[alive_at_sensor_check] = X1[alive_at_sensor_check]
            Y[alive_at_sensor_check] = Y1[alive_at_sensor_check]
            Z[alive_at_sensor_check] = Z1[alive_at_sensor_check]
        
        check3 = (surface_type == 'XY') | (surface_type == 'Z') 
        living_wall = living & check3

        if any(living_wall):

            energyAtDeath[living_wall] = energy[living_wall]
            alive[living_wall] = 0

            X[living_wall] = X1[living_wall]
            Y[living_wall] = Y1[living_wall]
            Z[living_wall] = Z1[living_wall]

        try: 
            particles_x[:, n][living] = X1[living]
            particles_y[:, n][living] = Y1[living]
            particles_z[:, n][living] = Z1[living]
        except IndexError:
            print('one reflection has gone on more than 100 times')

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


def GetEvaporationSignal(detector, QPs, X, Y, Z, useMap=True, T=2.0, plot_3d = False, fixed_dir = None, fixed_momentum = None, verbose = False, flavor_switching = False, debug = False):
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
            effective temperature from which to sample the quasiparticle momenta (FIXME: not curretly used)
        plot_3d : boolean
            Generate a 3d plot of QP paths
        fixed_dir : 3-tuple of floats
            Allow the user to manually set the QP's initial direction's unit vector. Default is isotropy.
        fixed_momentum : float
            Allow the user to manually set a momentum in keV. Default is random sampling
        verbose : bool
        flavor_switching : bool
            Flag whether to simulate flavor switching upon reflection 
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

    energyAtDeath, sensorIdsAll, evaporated, total_time, step_count, initial_momentum, paths = QP_propagation(QPs, [X, Y, Z], up_conditions=up_conditions, down_conditions=down_conditions, reflection_prob=detector.get_QP_reflection_prob(), 
                                                                evap_eff=detector.get_evaporation_eff(), T=T, diffuse_prob=detector.get_diffuse_prob(), 
                                                                plot_3d = plot_3d, fixed_dir = fixed_dir, fixed_momentum = fixed_momentum, verbose=verbose, 
                                                                flavor_switching=flavor_switching)

    for i in range(nsensors):
        hit_sensor_i_and_evaporated = (sensorIdsAll == i) & evaporated
        hit_sensor_i_and_not_evaporated = (sensorIdsAll == i) & ~evaporated

        energies[i] = (energyAtDeath[hit_sensor_i_and_evaporated] + detector.get_sensor(i).get_adsorptionGain()) * detector.get_sensor(i).get_phononCollectionEfficiency()
        energies[i] = np.append(energies[i], energyAtDeath[hit_sensor_i_and_not_evaporated] * detector.get_sensor(i).get_phononCollectionEfficiency())

        arrivalTimes[i] = total_time[hit_sensor_i_and_evaporated]
        arrivalTimes[i] = np.append(arrivalTimes[i], total_time[hit_sensor_i_and_evaporated])


    # return Signal(sum(chAreas), chAreas, coincidence, arrivalTimes, bounced_flag = bounce_flag_with_sensor, paths = paths, flavor=flavor_with_sensor, momentums = momentum_hit, arrivals_unsorted = arrival_times)
    if not debug:
        return HestSignal(energies, arrivalTimes)
    else:
        return energyAtDeath, sensorIdsAll, evaporated, total_time, step_count, initial_momentum, paths

def GetSingletSignal(detector, nPhotons, X, Y, Z, wall_reflection_prob = 0.0, useMap = True, plot_3d = False, fixed_dir = None, verbose = False, debug = False):
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
        flavor_switching : bool
            Flag whether to simulate flavor switching upon reflection 
            (FIXME: I'm guessing this doesn't when using the map)
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
                                                                 wall_reflection_prob, plot_3d = plot_3d, fixed_dir = fixed_dir )
    
    for i in range(nsensors):
        hit_sensor_i = (sensorIdsAll == i)

        energies[i] = (energyAtDeath[hit_sensor_i]) * detector.get_sensor(i).get_phononCollectionEfficiency()

        arrivalTimes[i] = total_time[hit_sensor_i]
        arrivalTimes[i] = np.append(arrivalTimes[i], total_time[hit_sensor_i])
    
    if not debug:
        return HestSignal(energies, arrivalTimes)
    else:
        return energyAtDeath, sensorIdsAll, total_time, step_count, paths

''' #########################################################################
    Define functions for turning lists of arrival times into pulse shapes (and generating pulse shapes for LEE events)
    ######################################################################### '''
    
def GetLEEPulse(area):
    template_gen = Template(verbose=True)
    template_gen.create_template('sensor_1',A=1,B=1,
                                 trace_length_msec=5, #msec
                                 pretrigger_length_msec=2, #msec
                                 sample_rate=1.25e6, #Hz
                                 tau_r=65.00e-6, #sec
                                 tau_f1=190.0e-6, #sec
                                 tau_f2=596.0e-6) #sec
    template_array1, time_array1 = template_gen.get_template('sensor_1')
    normalization = np.sum(template_array1)*(time_array1[1] - time_array1[0])
    
    return template_array1*area/normalization, time_array1

def GetDetResponse(arrivalTimes_us):
    '''
    First draft of a function that takes in arrival times in us and spits out a TES waveform in us
    '''
    minTime = min(arrivalTimes_us)
    maxTime = max(arrivalTimes_us)
    print(minTime)
    print(maxTime)
    
    PulseTime_us = 5*(maxTime) #take to be ~5x range of arrival times; this will be tuned later
    print("pulseTime_us: "+str(PulseTime_us))
    sample_count = 10000
    data_freq = sample_count/PulseTime_us*1000000 #Hz
    print("Frequency: "+str(data_freq))
    # sample_count = int(PulseTime_us*1e-6*data_freq)  #Get the number of time samples in the array
    print("Sample Count: "+str(sample_count))
    template_array_total = np.zeros(sample_count)
    time_array = np.linspace(0, PulseTime_us, sample_count)

    template_gen = Template()
    for time in arrivalTimes_us:
            template_gen.create_template('sensorx',A=1,B=1, #All of these parameters will need to be tuned; currently copied from the LEE template
                                 trace_length_msec=.001*PulseTime_us, #msec;
                                 pretrigger_length_msec=time*.001, #msec
                                 sample_rate= data_freq, #Hz
                                 tau_r=65.00e-6, #sec
                                 tau_f1=190.0e-6, #sec
                                 tau_f2=596.0e-6) #sec
            template_array1, time_array1 = template_gen.get_template('sensorx')
            normalization = 1 #Needs tuning
            template_array_total += template_array1/normalization
    # print(template_array_total)
    return time_array, template_array_total
