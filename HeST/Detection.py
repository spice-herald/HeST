from scipy.interpolate import interpn, interp1d
from detprocess import Template
import numpy as np
import re
import matplotlib.pyplot as plt
from .HeST_Core import CPD_Signal, Random_QPmomentum, QP_dispersion, QP_velocity 
from .HeST_Core import Singlet_PhotonEnergy

class VCPD:
    def __init__(self, surface_condition, baselineNoise, phononConversion=0.25, bounced_flag = 0):
        '''
        Create a Virtual CPD class to keep track of individual CPDs
        and add them to the VDetector object.

        Args:
        surface_condition: a function that tracks where photon/QP paths are obstructed by
                          the CPD surface. Returns (True, "CPD") if the photon is unobstructed;
                          Returns (False, "CPD") if the photon path hits the CPD surface.
        baselineNoise: [mean, width] of the CPDs baseline noise
        phononConversion=0.25: fraction of photon energy that creates a phonon signal in the CPD

        '''

        self.surface_condition = surface_condition
        self.baselineNoise     = np.array( baselineNoise )
        self.phononConversion  = phononConversion
        self.bounce_flag       = bounced_flag


    def set_surface_condition(self, f1):
        self.surface_condition = f1
    def set_baselineNoise(self, p1):
        self.baselineNoise = np.array( p1 )
    def set_phononConversion(self, p1):
        self.phononConversion = p1

    def get_surface_condition(self):
        return self.surface_condition
    def get_baselineNoise(self):
        return self.baselineNoise
    def get_phononConversion(self):
        return self.phononConversion
    def get_bounces(self):
        return self.bounce_flag

    def check_surface(self, x, y, z):
        return self.surface_condition(x, y, z)



class VDetector:
    def __init__(self, surface_conditions, liquid_surface, liquid_conditions,
                 adsorption_gain, evaporation_eff, CPDs=[], LCEmap=0, LCEmap_positions=0., QPEmap=0, QPEmap_positions=0.,
                 photon_reflection_prob=0., QP_reflection_prob=0.):
        
        '''
        Create a Virtual Detector class.
        
        Args:
        surface_conditions: a list of functions that tracks where photon/QP paths are obstructed by 
                            the Detector surfaces. Each function returns (True, *SurfaceType) if the photon is unobstructed; 
                            Returns (False, *surfaceType) if the photon path hits the detector surface.
                            *SurfaceType is a way of tracking what type of boundary this is: e.g. "X", "Y", "Z", or "XY", making it 
                            easy to track how the particle may reflect off of that boundary
                            
        liquid_surface:     A function that tracks where the liquid surface is, similar to the surface conditions above. This is used in QP propagation/evaporation calculations.
                            The surfaceType here must be "Liquid"
        liquid_conditions:  A function returning true if the X,Y,Z position is inside the LHe volume, and False if outside the volume

        CPDs:               A list of VCPD objects to keep track of
        
        adsorption_gain:    Added energy from adsorption of evaporated QPs, per QP, in eV
        
        self.evaporation_eff:  Flat efficiency factor on QP evaporation, between 0 and 1
        
        LCEmap:             A 3D array tracking positions with mean photon collection probability, dimensionality (M x N x L)
        
        LCEmap_positions:   A set of three 1-D arrays (x, y, z) to associate the LCE map entries in (M x N x L) discrete bins to individual x,y,z coordinates.
                            If loading a premade map, these must match what was used for that map's generator. 
        
        QPEmap:             A 3D array tracking position with mean QP evaporation probability. Different from LCEmap due to liquid surface physics
        
        QPEmap_positions:   A set of three 1-D arrays (x, y, z) to associate the QPE map entries in (M x N x L) discrete bins to individual x,y,z coordinates. 
                            If loading a premade map, these must match what was used for that map's generator. The dimensionality of the QP map doesn't need to
                            match the dimensionality of the LCE map.
                            
        photon_reflection_prob: probability that a photon reflects off of detector surfaces, between 0 and 1
        QP_reflection_prob: probability that a QP reflects off of detector surfaces, between 0 and 1
        '''
        
        self.surface_conditions = surface_conditions
        self.liquid_surface     = liquid_surface
        self.liquid_conditions  = liquid_conditions
        self.CPDs               = CPDs
        self.adsorption_gain    = adsorption_gain #eV
        self.evaporation_eff    = evaporation_eff
        self.LCEmap             = LCEmap
        self.LCEmap_positions   = LCEmap_positions
        self.QPEmap             = QPEmap
        self.QPEmap_positions   = QPEmap_positions
        self.photon_reflection_prob = photon_reflection_prob
        self.QP_reflection_prob = QP_reflection_prob
        
        
    
        
    def set_surface_conditions(self, f1):
        self.surface_conditions = list(f1)
    def add_surface_condition(self, f1):
        self.surface_conditions.append( f1 )
        
    def set_CPDs(self, p1):
        self.CPDs = list(p1)
    def add_CPD(self, p1):
        self.CPDs.append( p1 )
        
    def set_liquid_surface(self, f1):
        self.liquid_surface = f1
    def set_liquid_conditions(self, f1):
        self.liquid_conditions = f1
    def set_adsorption_gain( self, p1 ):
        self.adsorption_gain = p1
    def set_evaporation_eff( self, p1 ):
        self.evaporation_eff = p1
        
        
    def load_LCEmap(self, filename):
        self.LCEmap = np.load(filename)
    def set_LCEmap_positions(self, p1):
        if len(p1) == 3:
            self.LCEmap_positions = p1
        else: 
            print("This function requires a list of 3 individual 1-D arrays to associate the LCE map with X,Y,Z coordinates")
    def load_QPEmap(self, filename):
        self.QPEmap = np.load(filename)
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
        
    def get_surface_conditions(self):
        return list(self.surface_conditions)
    def get_liquid_surface(self):
        return self.liquid_surface
    def get_liquid_conditions(self):
        return self.liquid_conditions
    
    def get_LCEmap(self):
        return self.LCEmap
    
    def get_LCEmap_positions(self):
        return self.LCEmap_positions
    
    def get_QPEmap(self):
        return self.LCEmap_positions
    
    def get_QPEmap_positions(self):
        return self.QPEmap_positions
        
        
        
    def get_nCPDs(self):
        return len(self.CPDs)
    
    def get_CPD(self, index):
        return self.CPDs[index]
    
    def get_liquid_surface( self ):
        return self.liquid_surface
    def get_adsorption_gain(self):
        return self.adsorption_gain
    def get_evaporation_eff(self):
        return self.evaporation_eff
    
    def get_photon_reflection_prob(self):
        return self.photon_reflection_prob
    def get_QP_reflection_prob(self):
        return self.QP_reflection_prob
    def get_diffuse_prob(self):
        return self.diffuse_prob
    
    
    def create_LCEmap(self, x_array, y_array, z_array, nPhotons=10000, filestring="detector_LCEmap"):
        print("Creating LCE map for this detector geometry...")
        x, y, z = np.array(x_array), np.array(y_array), np.array(z_array)
        self.set_LCEmap_positions( [x, y, z] )
        nCPDs = self.get_nCPDs()
        m = np.zeros((len(x), len(y), len(z), nCPDs), dtype=float)
        conditions = self.get_surface_conditions()
        for i in range(nCPDs):
            conditions.append( (self.get_CPD(i)).get_surface_condition() )
        
        refl_prob = self.get_photon_reflection_prob()
        for xx in range(len(x)):
            print("%i / %i" % (xx, len(x)))
            for yy in range(len(y)):
                for zz in range(len(z)):
                    
                    pos = np.array([x[xx], y[yy], z[zz]])
                    if (self.get_liquid_conditions())(*pos) == False:
                        continue
                    hitProbs = [0.]*nCPDs
                    for n in range(nPhotons):
                    
                        hit, arrival_time, n, xs, ys, zs, cpd_id = photon_propagation(pos, conditions, refl_prob)
                        if hit > 0:
                            hitProbs[cpd_id] += 1.
                        
                        
                                
                    hitProbs = np.array(hitProbs)/nPhotons
                    
                    m[xx][yy][zz] = hitProbs
                    #print(hitProbs)
                    
        self.LCEmap = m
        np.save(filestring+".npy", m)
        print("Saved map to %s.npy" % filestring)
        
    
        
        
    def get_photon_hits(self, X, Y, Z):
        if type(self.LCEmap) == int:
            print("No LCE Map Loaded!")
            return -999
        
        positions = self.get_LCEmap_positions()
        x = positions[0]
        y = positions[1]
        z = positions[2]
        return interpn((x, y, z), self.LCEmap, [X, Y, Z])[0]
    
    def create_QPEmap(self, x_array, y_array, z_array, nQPs=10000, filestring="detector_QPEmap", T=2.):
        print("Creating QPE map for this detector geometry...")
        x, y, z = np.array(x_array), np.array(y_array), np.array(z_array)
        self.set_QPEmap_positions( [x, y, z] )
        nCPDs = self.get_nCPDs()
        m = np.zeros((len(x), len(y), len(z), nCPDs), dtype=float)
        conditions = self.get_surface_conditions()
        conditions.append( self.liquid_surface )
        for i in range(nCPDs):
            conditions.append( (self.get_CPD(i)).get_surface_condition() )
        
        refl_prob = self.get_QP_reflection_prob()
        for xx in range(len(x)):
            print("%i / %i" % (xx, len(x)))
            for yy in range(len(y)):
                for zz in range(len(z)):
                    
                    pos = np.array([x[xx], y[yy], z[zz]])
                    if (self.get_liquid_conditions())(*pos) == False:
                        continue
                    hitProbs = [0.]*nCPDs
                    for n in range(nQPs):

                        hits, times, cpd_ids = QP_propagation(nQPs, pos, conditions, refl_prob, evap_eff=self.evaporation_eff, T=T, debug=True)
                        print(cpd_ids)
                        if len(hits) > 0:
                            hitProbs[cpd_ids] += 1.
                        
                                
                    hitProbs = np.array(hitProbs)/nQPs
                    
                    m[xx][yy][zz] = hitProbs
                    #print(hitProbs)
                    
        self.QPEmap = m
        np.save(filestring+".npy", m)
        print("Saved map to %s.npy" % filestring)
    
    def get_QP_hits(self, X, Y, Z):
        if type(self.QPEmap) == int:
            print("No QPE Map Loaded!")
            return -999
        
        positions = self.get_QPEmap_positions()
        x = positions[0]
        y = positions[1]
        z = positions[2]
        return interpn((x, y, z), self.QPEmap, [X, Y, Z])[0]




''' #############################################################################

    Define various functions to calculate particle propagation through the 
    Detector geometry

    ############################################################################# '''

def find_surface_intersection(start, direction, conditions):
    """Finds the surface intersection by calculating the path, and finding the point right before the first intersection point.

    Args:
        start (_type_): _description_
        direction (_type_): _description_
        conditions (_type_): _description_

    Returns:
        _type_: _description_
    """
    print('~~~~~~~~~~~~~~~~ Finding Surface Intersection ~~~~~~~~~~~~~~~~~')
    print('\n')
    print(start)
    print(direction)
    if np.isscalar( start[0] ):
        start = np.array([np.array([p]) for p in start])
    if np.isscalar( direction[0] ):
        direction = np.array([np.array([p]) for p in direction])
    t = np.array([np.linspace(0, 12, 300) for i in range(len(start[0]))])  # Parameter range for the line
    # Calculate the line coordinates
    x_line = start[0][:, np.newaxis] + t * direction[0][:, np.newaxis]
    y_line = start[1][:, np.newaxis] + t * direction[1][:, np.newaxis]
    z_line = start[2][:, np.newaxis] + t * direction[2][:, np.newaxis]

    
    dist = np.ones(len(start[0]))*9999.
    coords = [np.full(len(start[0]), None), np.full(len(start[0]), None), np.full(len(start[0]), None)]
    surface_type = np.full(len(start[0]), None)
    for cond in conditions:
        cut, surface = cond(x_line, y_line, z_line)
        print(surface, cut)
        # the below line is meant to calculate the indices of the past point
        first_ints = np.array([np.argmax(~cut[i]) for i in range(len(cut))])
        #if first_int == 0:
        # continue
        # this determines how far the particle is ??
        d = t[np.arange(t.shape[0]), first_ints]
        cond = ( d < dist ) & (first_ints > 0)
        dist = np.where(cond, d, dist)        
       
        #get the coords of the first point *before* the interaction
        coords[0] = np.where(cond, x_line[np.arange(x_line.shape[0]), first_ints-1], coords[0])
        coords[1] = np.where(cond, y_line[np.arange(y_line.shape[0]), first_ints-1], coords[1])
        coords[2] = np.where(cond, z_line[np.arange(z_line.shape[0]), first_ints-1], coords[2])
        surface_type = np.where( cond, surface, surface_type )
        
    return np.array(coords[0], dtype = float), np.array(coords[1], dtype = float), np.array(coords[2], dtype= float), surface_type


def diffuse_and_specular(surface, pos, direction, diffuse_prob):
    """To calculate the diffuse and specular reflections. You specify a surface, and it works directly off the cases presented there. 


    Args:
        surface (_type_): _description_
        pos (_type_): _description_
        direction (_type_): _description_
        diffuse_prob (_type_): _description_

    Returns:
        _type_: _description_
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
        if list(specular_cut).count(True)> 0:
            dx[specular_cut], dy[specular_cut], dz[specular_cut] = np.vectorize(wall_reflect)(x[specular_cut], y[specular_cut], dx[specular_cut], dy[specular_cut], dz[specular_cut]) 
        dx[~specular_cut], dy[~specular_cut], dz[~specular_cut] = generate_random_direction_off_surface(surface='XY',pos = (x[~specular_cut], y[~specular_cut], z[~specular_cut])) 
        return dx, dy, dz
    if surface == 'Z':
        # we will do both specular and diffuse
        specular_cut = np.random.uniform(size = (len(direction[0]),)) > diffuse_prob
        dz[specular_cut] = - dz[specular_cut]

        dx[~specular_cut], dy[~specular_cut], dz[~specular_cut] = generate_random_direction_off_surface('Z', pos = (dx[~specular_cut], dy[~specular_cut], dx[~specular_cut]))

        return dx, dy, dz

        


def generate_random_direction_off_surface(surface, pos = (0.0,0.0,0.0)):
    """Meant to take in a surface and generate a random direction relative to that. 
    I want to make this function more case by case too. `
    Args:
        surface_type (_type_): _description_
    """
    #first we do the case where the surface is XY
    x = pos[0]
    y = pos[1]
    z = pos[2]
    if surface == 'XY':
        # ~~~~~~~~~~~ Doing this case by case, dx, dy, dz should already be cut to be alive and diffuse, and at an 'XY' surface
        phi_1, arctheta_1 = np.random.uniform(np.pi/10, 9 * np.pi/10, size=len(x)), np.random.uniform(-1., 1, size=len(x))
        offset_angles_1 = np.arctan2(x, y)
        phi_1 += offset_angles_1
        theta_1 = np.arccos(arctheta_1)
        dx = np.cos( phi_1 ) * np.sin( theta_1 )
        dy  = np.sin( phi_1 ) * np.sin( theta_1 )
        dz  = np.cos(theta_1)
        return dx, dy, dz


    if surface == 'Z':
        # ~~~~~~~~~ This should where dx, dy, dz and X, Y, Z have already been cut to be ALIVE, DIFFUSE, and at a 'Z' surface
        phi_2, arctheta_2 = np.random.uniform(0, 2 * np.pi, size=len(x)), np.random.uniform(0, 1.0, size=len(x))
        theta_2 = np.arccos(arctheta_2)
        dx = np.cos( phi_2 ) * np.sin( theta_2 )
        dy  = np.sin( phi_2 ) * np.sin( theta_2 )
        dz  = np.cos(theta_2)
        return dx, dy, dz



def generate_random_direction(nQPs, bottom_phi, top_phi, bottom_theta, top_theta):
    """
    This function generates a random direction for a large number of quasiparticles, where that random direction is within ranges. See arguments for details.
    Phi is normally distributed from the two bounds, and theta is normally distributed across the inverse. 

    Args:
        nQPs (int): The number of quasiparticles being generated 
        bottom_phi (Float): the bottom bound of the phi angle, which is the azimuthal angle. 
        top_phi (float): the top bound of the phi angle, which is the azimuthal angle
        bottom_theta (float): bottom bound of the inverse of theta, must be between (-1, 1)
        top_theta (_type_): upper bound of the inverse of theta, must be between (-1, 1)

    Returns:
        tuple: the unit vector in cartesian coordinateds, broken up into arrays of X, Y, Z, each of length nQPs. 
    """
    phi, arctheta = np.random.uniform(bottom_phi, top_phi, size=nQPs), np.random.uniform(bottom_theta, top_theta, size=nQPs)
    theta = np.arccos(arctheta)
    #prepare the vecctors of direction
    dx = np.cos( phi ) * np.sin( theta )
    dy = np.sin( phi ) * np.sin( theta )
    dz = np.cos(theta)
    return dx, dy, dz





def evaporation(momentum, energy, velocity, direction):
    """Calculate the kinematics of a particle, after evaporation. This means that every argument here
    is assumed to be only the ones that have already evaporated. 

    Args:
        momentum (array): 
        energy (array): _description_
        velocity (array): _description_
        direction (array): _description_
    """
    dx, dy, dz = direction[0], direction[1], direction[2]
    if np.isscalar(momentum):
        momentum = np.array([momentum])
    if np.isscalar(energy):
        energy = np.array([energy])
    if np.isscalar(velocity):
        velocity = np.array([velocity])

    m =  3.725472e6 #He mass in keV/c^2
    E_binding = 0.00062
    pxi, pyi, pzi = momentum * dx, momentum * dy, momentum * dz
    pxf, pyf =pxi, pyi
    pzf = np.sqrt(2 * m * energy - pxi**2 - pyi**2)

    momentum = np.sqrt(pxf**2 + pyf**2 + pzf**2)
    dx, dy, dz = pxf/momentum, pyf/momentum, pzf/momentum
    # once again, just a reminder that these should only be for the QP element after evaporation
    return dx, dy, dz, momentum

def evap_prob_of_p_theta(p, theta):
    # For now, we are just going to do a uniform distribution, but this is to build this later
    return np.random.uniform(size = len(theta))

    

def critical_angle(Energy, momentum, binding_energy = 0.00062e-3):
    m =  3.725472e6 #He mass in keV/c^2
    c = 2.998e8

    print(np.sqrt(2 * m * (Energy - binding_energy))/momentum)
    return np.arcsin(np.sqrt(2 * m * (Energy - binding_energy))/momentum)


# Define a function to extract the CPD number using regex
# @np.vectorize(otypes=[str])
# def extract_number(s):
#     match = re.search(r'\d+', s)  # Find the first occurrence of one or more digits
#     if match:
#         return int(match.group())  # Convert the matched digits to an integer
#     return None  # Return None if no digits are found
@np.vectorize
def extract_number(s):
    if s == None:
        return False
    s = list(s)
    last_element = s[-1]
    if last_element.isdigit():
        return int(last_element)
    return False


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


"""
##############################
Quasiparticle Propagation
##############################
"""
def QP_propagation(nQPs, start, conditions, reflection_prob, evap_eff=1.0, diffuse_prob = 0.0, T=2., debug= False, debug_dir = (0,0,1), plot_3d=False, choose_momentum = False, momentum_choice = 0.0):
    """Tracking of Quasiparticles through medium. I'm going to add a debug flag where we can specify the direction that the particles go in, in this case I am first going to make all of them go straight down. 
   
    Args:
        nQPs (int): The number of quasiparticles being tracked at a given time.  
        start (ndarray): the initial location of the quasiparticles.  
        conditions (func): list of function assigning the conditions on the particles
        reflection_prob (float): the probability of reflection on certain surfaces (we don't really know what this is, and I don't ) 
        evap_eff (float, optional): The evaporation efficiency through the surface. Defaults to 0.60.
        T (float, optional): the thermal temperature of the helium I think, probably in mk. Defaults to 2..

    Returns:
        _type_: _description_
    """
    #if there is only one quasiparticle, we still need to prepare an array for it to be tracked
    print(f'Starting at {start}')
    if np.isscalar(nQPs):
        X = np.full(nQPs, start[0])
        Y = np.full(nQPs, start[1])
        Z = np.full(nQPs, start[2])
        start = np.array([X, Y, Z])

    #if the plot3d is true, then we want to record the path traveled. This means, for each particle, recording it's starting and ending point. Sadly, this also means lists. 
    if plot_3d:
        particles_x = np.zeros(shape=(nQPs, 100))
        particles_y = np.zeros(shape=(nQPs, 100))
        particles_z = np.zeros(shape=(nQPs, 100))
 
    #assign the starting values of the array
    X, Y, Z = start[0], start[1], start[2]
    #randomely assign the phi and theta directions, each with an array the size of the number of quasiparticles. 
    # I almost feel like this should be a function, where we just input the range of these things
    dx, dy, dz = generate_random_direction(nQPs, 0.0, 2.0 * np.pi, -1.0, 1.0)

    if debug: 
        dx = np.full(nQPs, debug_dir[0]) 
        dy = np.full(nQPs, debug_dir[1]) 
        dz = np.full(nQPs, debug_dir[2]) 

    total_time = np.zeros(nQPs, dtype=float)
    n=0
    #This draws from a k**2 distribution, essentially trying to follow the density of states
    momentum = Random_QPmomentum(nQPs) #keV/c
    if choose_momentum:
        momentum = np.full(nQPs, momentum_choice)
    #prepare the alive tracker, and then cut out those which we can't sense
    alive = np.ones(nQPs, dtype=int)
    cond = momentum <  1.1
    alive = np.where( cond, 0, alive)

    #velocity and energy, both arrays of length nQP
    velocity = QP_velocity(momentum) #m/s
    energy = QP_dispersion(momentum) *1e-3 #KeV
    evaporated = np.zeros( nQPs, dtype=bool)
    # instead of inverting the velocity, we invert the momentum and later, when we use velocity, we take the magnitude of it anyways
    v_mask = velocity < 0
    momentum[v_mask] = -momentum[v_mask]
    
    deposits = np.zeros(nQPs, dtype=float)
    ids = np.full(nQPs, None)
    bounced_flag = np.zeros_like(alive)
    #we prepare a boolean for evaporated, meaning that evaporated will be a mask that gets added to, then deposits is a float so idk what that is for, ids is for which cpd?
    #here we can also add the particle locations
    if plot_3d:
        living = alive > 0.5
        particles_x[:, 0][living] = X[living]
        particles_y[:, 0][living] = Y[living]
        particles_z[:, 0][living] = Z[living]
    while sum(alive) > 0:
        n+=1
        #prepare an array for the living ones (alive is bool so this is really just not 0)
        living = ( alive > 0.5 )
        #print(Z[living])
        #find a surface intersection
        X1, Y1, Z1, surface_type = find_surface_intersection(np.array([X, Y, Z]), np.array([dx, dy, dz]), conditions)
        print(surface_type)
        #I'm still confused on how this find_surfface_intersection could ever be None. Ok so it's automatically set to none
        hit_surface_check= (surface_type != None)
        # print(list(hit_surface_check).count(False))
        # we are just setting these things to 0. Hopefully this doesn't destroy our efficiency
        X[~hit_surface_check], Y[~hit_surface_check], Z[~hit_surface_check] = np.zeros_like(X[~hit_surface_check]),np.zeros_like(X[~hit_surface_check]),np.zeros_like(X[~hit_surface_check])   
        dx[~hit_surface_check], dy[~hit_surface_check], dz[~hit_surface_check] = np.zeros_like(dx[~hit_surface_check]),np.zeros_like(dx[~hit_surface_check]),np.zeros_like(dx[~hit_surface_check])   
        

        #eliminate (from the living) the values that have escaped without intersecting a surface (which should be none for this geometry)
        alive = np.where(hit_surface_check, alive, 0)
        #reset the living parameter and indices
        living = ( alive > 0.5 )
        # ~~~~~~~~~~~~~ To maintain the surface length is the right size
        # surface_type = surface_type[hit_surface_check]
        # print(len(living_indices) == len(surface_type))
        # print(len(living_indices))
        # print(len(surface_type))
        # print(np.shape(living))
        bounced_flag[living]  = np.full_like(alive[living], fill_value=n)
        #set the new X1, Y1, Z1 space to exclude any particles that did not make it to a wall 
        
        #finding the time it took to go from initial point to intersection point
        dist_sq = (pow(X1[living]-X[living],2.)+pow(Y1[living]-Y[living], 2.)+pow(Z1[living]-Z[living],2.)).astype(float)      
        total_time[living] = total_time[living] + np.sqrt(dist_sq)/np.abs(velocity[living]*1.0e-4)  #us
        
        check1 = (surface_type == "Liquid")
        
        #checking if it hit the surface of the liquid helium
        if len(surface_type[check1]) > 0:
            print('~~~~~~~~~~~~~~~~~~~~~~~ Computing Evaporations ~~~~~~~~~~~~~~~~~~~~')
            # cut to discuss only the living qps that hit the surface of the helium
            alive_surface_check = living & check1
            # print(np.shape(alive_surface_check))
            # ~~~~~~~~~~~~~ this is the point where the particle is ALIVE and HIT THE SURFACE OF HELIUM
            # evap, velocity[alive_surface_check], dx[alive_surface_check], dy[alive_surface_check], dz[alive_surface_check] = evaporation(momentum[alive_surface_check], energy[alive_surface_check], velocity[alive_surface_check], [dx[alive_surface_check], dy[alive_surface_check], dz[alive_surface_check]])
            # calc critical angle
            critical_angles = critical_angle(energy, momentum) 
            print(f' this is critical angles {critical_angles}')
            incident_angles = np.arccos(dz)
            print(incident_angles > critical_angles)
            no_evap = (incident_angles > critical_angles) & (evap_prob_of_p_theta(momentum, incident_angles) > evap_eff)
            # print(len(no_evap) == len(alive_surface_check))
            # print(len(no_evap) == list(alive_surface_check).count(True))
            # Calculate reflections for the ones that don't evaporate. We must define a clear boolean for this 
            a_L_noevap = alive_surface_check & no_evap # This is the mask that selects just the particles that are ALIVE, on the LIQUID SURFACE, and REFLECT
            dx[a_L_noevap], dy[a_L_noevap], dz[a_L_noevap] = diffuse_and_specular('Liquid', (X[a_L_noevap], Y[a_L_noevap], 
                                                                                Z[a_L_noevap]), (dx[a_L_noevap], dy[a_L_noevap], dz[a_L_noevap]),
                                                                                 diffuse_prob=0.0)

            
            a_L_evap = alive_surface_check & ~no_evap # THis is the mask that selects just the particles that are ALIVE, on the LIQUID SURFACE, and EVAPORATE
            dx[a_L_evap], dy[a_L_evap], dz[a_L_evap], momentum[a_L_evap] = evaporation(momentum[a_L_evap], 
                                                                                            energy[a_L_evap],
                                                                                            velocity[a_L_evap], 
                                                                                            direction=(dx[a_L_evap], dy[a_L_evap], dz[a_L_evap]))



            # point of this line is to set this big array equal to True if it evaporated
            evaporated[a_L_evap] = True
            # evaporated[alive_surface_check] = np.where( no_evap, evaporated[alive_surface_check], True )
            # ~~~~~~~~~~ If the QP doesn't evaporate, leave it where it is
            X[a_L_noevap] = X1[a_L_noevap]
            Y[a_L_noevap] = Y1[a_L_noevap]
            Z[a_L_noevap] = Z1[a_L_noevap] - 0.1

            # ~~~~~~~~~~ If the QP evaporates, move it up above the liquid surface 
            X[a_L_evap] = X1[a_L_evap]
            Y[a_L_evap] = Y1[a_L_evap]
            Z[a_L_evap] = Z1[a_L_evap] + 0.1

       
            
                                            
        check2 = np.array(["CPD" in str(s) for s in surface_type])
        #print("Hit %i CPDs" % len(surface_type[check2]))
        if list(living & check2).count(True)> 0:
            print('~~~~~~~~~~~~~~~~~~~~~~~ Computing Deposits ~~~~~~~~~~~~~~~~~~~~')
            cpd_id = np.empty_like(surface_type, dtype=int)
            for ii, surface in enumerate(surface_type):
                cpd_id[ii] = extract_number(surface)
            cond = evaporated & living & check2
            #print("nDeps", len(deposits[living_indices[check2]][cond]))
            # deposits[living & check2] = np.where( cond, energy[living & check2], deposits[living & check2])
            deposits[cond] = energy[cond] 
            
            #going to store the CPD intersection point here for plotting purposes. 

            alive[cond] = np.zeros(len(alive[cond]), dtype=int) #kill off QPs, they've hit a CPD
            ids[cond] = cpd_id[cond] #store the CPD IDs for evaporated QPs that hit a CPD
            # ids[cond] = cpd_id[cond]
        check3 = (surface_type == 'XY') | (surface_type == 'Z') #doesn't reach liquid or CPD
        if len(surface_type[check3]) > 0:
            print('~~~~~~~~~~~~~~~~~~~~~~~ Computing Wall Reflections ~~~~~~~~~~~~~~~~~~~~')
            # Let's simplify this
            XY_check = (surface_type == 'XY')
            Z_check = (surface_type == 'Z')
            a_xy_check = living & check3 & XY_check
            a_z_check = living & check3 & Z_check
            r = np.random.random(len(surface_type[living & check3]))
            cond = (r > reflection_prob)
            alive[living &check3] = np.where(cond, 0, alive[living & check3]) #kill of those that don't reflect
            # ~~~~~~~~~~~ First do case of reflection off XY.
            if list(a_xy_check).count(True) > 0:
                dx[a_xy_check], dy[a_xy_check], dz[a_xy_check] = diffuse_and_specular('XY', pos = (X1[a_xy_check],Y1[a_xy_check],Z1[a_xy_check]), 
                                                direction = (dx[a_xy_check], dy[a_xy_check], dz[a_xy_check]), 
                                                diffuse_prob=diffuse_prob)
            # ~~~~~~~~~~~ Reflection off Z
            if list(a_z_check).count(True) > 0:
                dx[a_z_check], dy[a_z_check], dz[a_z_check] = diffuse_and_specular('Z', pos = (X1[a_z_check],Y1[a_z_check],Z1[a_z_check]), 
                                                direction = (dx[a_z_check], dy[a_z_check], dz[a_z_check]), 
                                                diffuse_prob=diffuse_prob)           
            X[living & check3] = X1[living &check3]
            Y[living & check3 ] = Y1[living &check3]
            Z[living & check3] = Z1[living &check3]
        if plot_3d: 
            try: 
                particles_x[:, n][living] = X1[living]
                particles_y[:, n][living] = Y1[living]
                particles_z[:, n][living] = Z1[living]
            except IndexError:
                    print('one reflection has gone on more than 20 times')
    if plot_3d:
        print(f'this is the bounced flag {bounced_flag}')
        ax = plt.figure().add_subplot(projection ='3d')
        for i in range(nQPs ):
            mask = (particles_x[i,:] == 0) &(particles_y[i,:] == 0) & (particles_z[i,:] == 0) 
            ax.plot(particles_x[i,:][~mask], particles_y[i,:][~mask], particles_z[i,:][~mask], '-o', label = f'bounced {bounced_flag[i]} times')
        ax.set_xlim(-3.8, 3.8)
        ax.set_ylim(-3.8, 3.8)
        def walls(points, radius): 
            theta = np.linspace(0, 2 * np.pi, points)

            x = radius *np.cos(theta)
            y = radius * np.sin(theta)
            return x,y
        x,y= walls(100, 3.)
        ax.plot(x,y)
        xx, yy = np.meshgrid(np.linspace(-3.0, 3.0, 50), np.linspace(-3.0, 3.0, 50))
        z = np.ones_like(xx) 
        ax.plot_surface(xx, yy, z * 2.75, alpha = 0.2,  label = 'Liquid Surface')
        z_cpd = z * 3.3
        ax.plot_surface(xx, yy, z_cpd, alpha = 0.2, label = 'CPD Level' )

        # ax.legend()
        print(f'x: {particles_x}, y: {particles_y}, z: {particles_z}')
    hit = (deposits > 0.)
    print(deposits[hit], total_time[hit])
    return deposits[hit], total_time[hit], ids[hit], bounced_flag, hit
        
def photon_propagation(nPhotons, start, conditions, reflection_prob):
        
    #make sure the we have an array of starting positions for each photon
    if np.isscalar(nPhotons):
        X = np.full(nPhotons, start[0])
        Y = np.full(nPhotons, start[1])
        Z = np.full(nPhotons, start[2])
        start = np.array([X, Y, Z])
    
    X, Y, Z = start[0], start[1], start[2]
    phi, arctheta = np.random.uniform(0., 2.*np.pi, size=nPhotons), np.random.uniform(-1., 1, size=nPhotons)
    theta = np.arccos(arctheta)
    dx = np.cos( phi ) * np.sin( theta )
    dy = np.sin( phi ) * np.sin( theta )
    dz = np.cos(theta)

    #dx, dy, dz = 0., 0., 1
    total_time = np.zeros(nPhotons, dtype=float)
    n=0
    alive = np.ones(nPhotons, dtype=int)

    velocity = 29979.2/1.03 #speed of light in He4 cm/us    
    cond = (velocity > 0.)
    alive = np.where( cond, alive, 0.)
    
    deposits = np.zeros(nPhotons, dtype=float)
    ids = np.full(nPhotons, None)
    while sum(alive) > 0:
        n+=1
        living = ( alive > 0.5 )
        
        #print(Z[living])
        X1, Y1, Z1, surface_type = find_surface_intersection(np.array([X[living], Y[living], Z[living]]), np.array([dx[living], dy[living], dz[living]]), conditions)
        cond = (surface_type != None)
        alive[living] = np.where( cond, alive[living], 0)
        living = ( alive > 0.5 )
        living_indices = np.where(living)[0]

        X1, Y1, Z1 = X1[cond], Y1[cond], Z1[cond]
        surface_type = surface_type[cond]
        
        dist_sq = (pow(X1-X[living],2.)+pow(Y1-Y[living], 2.)+pow(Z1-Z[living],2.)).astype(float)      
        total_time[living] = total_time[living] + np.sqrt(dist_sq)/velocity  #us
        
        check1 = np.array(["CPD" in str(s) for s in surface_type])
        #print("Hit %i CPDs" % len(surface_type[check2]))
        if len(surface_type[check1]) > 0:
            cpd_id = np.vectorize(extract_number)(surface_type[check1])
            deposits[living_indices[check1]] = 1.
            alive[living_indices[check1]] = 0 #kill off photons, they've hit a CPD
            ids[living_indices[check1]] = cpd_id #store the CPD IDs for photons that hit a CPD
            
        check3 = (check1 == False) #doesn't reach CPD
        if len(surface_type[check3]) > 0:
            r = np.random.random(len(surface_type[check3]))
            cond = (r > reflection_prob)
            alive[living_indices[check3]] = np.where(cond, 0, alive[living_indices[check3]]) #kill of those that don't reflect
            
            #handle reflections
            checkX = (surface_type == "X")
            checkY = (surface_type == "Y")
            checkZ = (surface_type == "Z")
            checkXY = (surface_type == "XY")
            
            dx[living_indices[checkX]] = -1.*dx[living][checkX] 
            dy[living_indices[checkY]] = -1.*dx[living][checkY] 
            dz[living_indices[checkZ]] = -1.*dx[living][checkZ] 
            
            if len(dz[living][checkXY]) > 0:
                dx[living_indices[checkXY]], dy[living_indices[checkXY]], dz[living_indices[checkXY]] = np.vectorize(wall_reflect)(X1[checkXY], Y1[checkXY], dx[living][checkXY], dy[living][checkXY], dz[living][checkXY] )

        
            X[living_indices[check3]] = X1[check3]
            Y[living_indices[check3]] = Y1[check3]
            Z[living_indices[check3]] = Z1[check3]
        #print(alive)
    hit = (deposits > 0.)
    return deposits[hit], total_time[hit], ids[hit]


''' #########################################################################

    Define functions for getting the energy deposited in the CPDs
    
    ######################################################################### '''

def GetSingletSignal(detector, photons, X, Y, Z, useMap=True):
    '''
    Attempt to simulate the CPD response for singlet photons.

    If useMap == True, the detector's LCEmap is loaded and the number of photon hits are
    calculated using that light collection efficiency given the interaction location.
    If False, loop through individual photons, calculating the number of hits
    and the arrival times.
    Number of hits are converted to pulse areas using the Singlet Photon Energy
    '''
    #check to see if there's an LCEmap loaded/generated
    if type(detector.get_LCEmap()) == int:
        useMap = False

    nCPDs = detector.get_nCPDs()
    nHitsPerCPD = [0]*nCPDs
    arrivalTimes = [[] for x in range(nCPDs)]
    chAreas = [0.]*nCPDs
    #check to see if a map is loaded
    if useMap:
        hitProbabilities = detector.get_photon_hits( X, Y, Z )
        coincidence = 0
        for i in range(nCPDs):
            nHits = np.random.binomial(photons, hitProbabilities[i])
            coincidence += min([nHits, 1]) # 0 or 1
            chAreas[i] = nHits*Singlet_PhotonEnergy
            thisCPD = detector.get_CPD(i)
            noiseBaseline = thisCPD.get_baselineNoise()
            noise = np.random.normal(noiseBaseline[0], noiseBaseline[1], size=nHits)
            chAreas[i] += np.sum(noise)

        return CPD_Signal(sum(chAreas), chAreas, coincidence, arrivalTimes)

    #if the map hasn't been loaded or we want arrival times...
    conditions = detector.get_surface_conditions()
    for i in range(nCPDs):
        conditions.append( (detector.get_CPD(i)).get_surface_condition() )
    hits, arrival_times, cpd_ids = photon_propagation(photons, [X, Y, Z], conditions, detector.get_photon_reflection_prob())
    
    coincidence = 0
    #add in baseline noise for each detected photon
    for i in range(nCPDs):
        cond = (cpd_ids == i)
        chAreas[i] = sum(hits[cond])*Singlet_PhotonEnergy
        coincidence += min([len(hits[cond]), 1]) # 0 or 1
        thisCPD = detector.get_CPD(i)
        noiseBaseline = thisCPD.get_baselineNoise()
        noise = np.random.normal(noiseBaseline[0], noiseBaseline[1], size=len(hits[cond]))
        chAreas[i] += np.sum(noise)
        arrivalTimes[i] = arrival_times[cond]


    return CPD_Signal(sum(chAreas), chAreas, coincidence, arrivalTimes)


def GetEvaporationSignal(detector, QPs, X, Y, Z, useMap=True, T=2., debug = False, debug_dir = (0,0,1), plot_3d = False, choose_momentum = False, momentum_choice = 0.0):
    '''
    Attempt to simulate the CPD response for quasiparticles.

    If useMap == True, the detector's QPEmap is loaded and the number of QP hits are
    calculated using that evaporation efficiency given the interaction location.
    If False, loop through individual QPs, calculating the number of hits
    and the arrival times.
    Hits are floating point numbers containing the amount of energy deposited on the CPD
    and then increased using the detector's adsorption gain.

    '''
    #check to see if there's an LCEmap loaded/generated
    if type(detector.get_QPEmap()) == int:
        useMap = False

    nCPDs = detector.get_nCPDs()
    nHitsPerCPD = [0]*nCPDs
    arrivalTimes = [[] for x in range(nCPDs)]
    bounce_flag_with_cpd = [[] for x in range(nCPDs)]
    chAreas = [0.]*nCPDs
    #check to see if a map is loaded
    if useMap:
        hitProbabilities = detector.get_QP_hits( X, Y, Z)
        coincidence = 0

        for i in range(nCPDs):
            nHits = np.random.binomial( QPs, hitProbabilities[i] )
            coincidence += min([nHits, 1]) # 0 or 1
            chAreas[i] = nHits * detector.get_adsorption_gain() # eV
            thisCPD = detector.get_CPD(i)
            noiseBaseline = thisCPD.get_baselineNoise()
            noise = np.random.normal(noiseBaseline[0], noiseBaseline[1], size=nHits)
            chAreas[i] += np.sum(noise)

        return CPD_Signal(sum(chAreas), chAreas, coincidence, arrivalTimes)

    #if the map hasn't been loaded or we want arrival times...
    conditions = detector.get_surface_conditions()
    conditions.append( detector.liquid_surface )
    for i in range(nCPDs):
        conditions.append( (detector.get_CPD(i)).get_surface_condition() )
        
    hits, arrival_times, cpd_ids, bounced_flag, hit = QP_propagation(QPs, [X, Y, Z], conditions, detector.get_QP_reflection_prob(), 
                                                                evap_eff=detector.get_evaporation_eff(), T=T, diffuse_prob=detector.get_diffuse_prob(), 
                                                                debug=debug, debug_dir = debug_dir, plot_3d = plot_3d, choose_momentum = choose_momentum, momentum_choice = momentum_choice)
    bounce_nums = bounced_flag
    bounced_flag = bounced_flag[hit]
    coincidence = 0
    for i in range(nCPDs):
        cond = (cpd_ids == i)
        chAreas[i] = sum(hits[cond] + detector.get_adsorption_gain()) # eV
        coincidence += min([len(hits[cond]), 1]) # 0 or 1
        thisCPD = detector.get_CPD(i)
        noiseBaseline = thisCPD.get_baselineNoise()
        noise = np.random.normal(noiseBaseline[0], noiseBaseline[1], size=len(hits[cond]))
        chAreas[i] += np.sum(noise)
        arrivalTimes[i] = arrival_times[cond]
        bounce_flag_with_cpd[i] = bounced_flag[cond]

    return CPD_Signal(sum(chAreas), chAreas, coincidence, arrivalTimes, bounced_flag = bounce_flag_with_cpd, num_bounces = bounce_nums)

''' #########################################################################
    Define functions for turning lists of arrival times into pulse shapes (and generating pulse shapes for LEE events)
    ######################################################################### '''
    
def GetLEEPulse(area):
    template_gen = Template(verbose=True)
    template_gen.create_template('CPD_1',A=1,B=1,
                                 trace_length_msec=5, #msec
                                 pretrigger_length_msec=2, #msec
                                 sample_rate=1.25e6, #Hz
                                 tau_r=65.00e-6, #sec
                                 tau_f1=190.0e-6, #sec
                                 tau_f2=596.0e-6) #sec
    template_array1, time_array1 = template_gen.get_template('CPD_1')
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
            template_gen.create_template('CPDx',A=1,B=1, #All of these parameters will need to be tuned; currently copied from the LEE template
                                 trace_length_msec=.001*PulseTime_us, #msec;
                                 pretrigger_length_msec=time*.001, #msec
                                 sample_rate= data_freq, #Hz
                                 tau_r=65.00e-6, #sec
                                 tau_f1=190.0e-6, #sec
                                 tau_f2=596.0e-6) #sec
            template_array1, time_array1 = template_gen.get_template('CPDx')
            normalization = 1 #Needs tuning
            template_array_total += template_array1/normalization
    # print(template_array_total)
    return time_array, template_array_total
