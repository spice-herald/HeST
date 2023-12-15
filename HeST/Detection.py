from scipy.interpolate import interpn
from detprocess import Template
import numpy as np
import re
from .HeST_Core import CPD_Signal, Random_QPmomentum, QP_dispersion, QP_velocity 
from .HeST_Core import Singlet_PhotonEnergy

class VCPD:
    def __init__(self, surface_condition, baselineNoise, phononConversion=0.25):
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

                        hit, time, n, xs, ys, zs, p, surf, cpd_id = QP_propagation(pos, conditions, refl_prob, evap_eff=self.evaporation_eff, T=T)
                        if hit > 0:
                            hitProbs[cpd_id] += 1.
                        
                                
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
    if np.isscalar( start[0] ):
        start = np.array([np.array([p]) for p in start])
    if np.isscalar( direction[0] ):
        direction = np.array([np.array([p]) for p in direction])
    t = np.array([np.linspace(0, 8, 250) for i in range(len(start[0]))])  # Parameter range for the line
    # Calculate the line coordinates
    x_line = start[0][:, np.newaxis] + t * direction[0][:, np.newaxis]
    y_line = start[1][:, np.newaxis] + t * direction[1][:, np.newaxis]
    z_line = start[2][:, np.newaxis] + t * direction[2][:, np.newaxis]

    dist = np.ones(len(start[0]))*9999.
    coords = [np.full(len(start[0]), None), np.full(len(start[0]), None), np.full(len(start[0]), None)]
    surface_type = np.full(len(start[0]), None)
    for cond in conditions:
        cut, surface = cond(x_line, y_line, z_line)
        first_ints = np.array([np.argmax(~cut[i]) for i in range(len(cut))])
        #if first_int == 0:
        #    continue
        d = t[np.arange(t.shape[0]), first_ints]
        cond = ( d < dist ) & (first_ints > 0)
        dist = np.where(cond, d, dist)        
       
        #get the coords of the first point *before* the interaction
        coords[0] = np.where(cond, x_line[np.arange(x_line.shape[0]), first_ints-1], coords[0])
        coords[1] = np.where(cond, y_line[np.arange(y_line.shape[0]), first_ints-1], coords[1])
        coords[2] = np.where(cond, z_line[np.arange(z_line.shape[0]), first_ints-1], coords[2])
        surface_type = np.where( cond, surface, surface_type )
        
    return coords[0], coords[1], coords[2], surface_type

def evaporation(momentum, energy, velocity, direction):
    if np.isscalar(momentum):
        momentum = np.array([momentum])
    if np.isscalar(energy):
        energy = np.array([energy])
    if np.isscalar(velocity):
        velocity = np.array([velocity])
    
    #adapted from Pratyush's code
    mass = 3.725472e9 #He mass in eV/c^2
    c = 2.998e8 #m/s
    Eb = 0.00062

    
    theta = np.arccos(direction[2]) #radians
    #sin_critical_angle = np.sqrt( 2*(energy-0.00062)/(4.002603254e9) )/(np.abs(velocity)/3e8)
    sin_critical_angle = np.sqrt(2.*mass*(energy - Eb))/momentum/1000. #Eq 2.19 from J. Adams thesis
    
    crit_angle = np.arcsin( sin_critical_angle ) #radians
    
    Velocity_He_atom=np.sqrt( 2.*(energy-Eb)/(mass) )*c

    sin_theta_R = (velocity/c)*np.sin((theta))/np.sqrt( 2*(energy-Eb)/(mass) ) #Eq 2.18 from Adams thesis (q = mv/hbar*c)
    theta_R = np.arcsin(sin_theta_R)
    new_Vz=Velocity_He_atom*np.cos(theta_R)
    cond = (direction[1] == 0.)
    tan = np.where( cond, np.pi/2., np.arctan(np.abs(direction[0]/direction[1]) ))
    new_Vy = np.where( cond, 0., direction[1]/np.abs(direction[1])*Velocity_He_atom*np.sin(theta_R)*np.sin((tan)) )
    
    cond = (direction[0] == 0.)
    new_Vx = np.where( cond, 0., direction[0]/np.abs(direction[0])*Velocity_He_atom*np.sin(theta_R)*np.cos((tan)))

    magnitude=np.sqrt(new_Vx*new_Vx+new_Vy*new_Vy+new_Vz*new_Vz)
    
    cond = (velocity <= 0) | (sin_critical_angle > 1.) | (theta > crit_angle) | ( energy < Eb )
    energy = np.where( cond, 0., energy - Eb)
  
    Velocity_He_atom = np.where( cond, 0., Velocity_He_atom )
    new_Vx = np.where( cond, 0., new_Vx/magnitude )
    new_Vy = np.where( cond, 0., new_Vy/magnitude )
    new_Vz = np.where( cond, 0., new_Vz/magnitude )
    
    return np.where(cond, 0, 1), Velocity_He_atom, new_Vx, new_Vy, new_Vz

# Define a function to extract the CPD number using regex
def extract_number(s):
    match = re.search(r'\d+', s)  # Find the first occurrence of one or more digits
    if match:
        return int(match.group())  # Convert the matched digits to an integer
    return None  # Return None if no digits are found

# Define a function to handle reflections off of walls
def wall_reflect(X, Y, dx, dy, dz):
    normal_vector = np.array([X, Y, 0.])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    incident_vector = np.array([dx, dy, dz])
    reflected_vector = incident_vector - 2. * np.dot(incident_vector, normal_vector) * normal_vector
    dx, dy, dz = reflected_vector / np.linalg.norm(reflected_vector)
    return dx, dy, dz

def QP_propagation(nQPs, start, conditions, reflection_prob, evap_eff=0.60, T=2.):
    
    #make sure the we have an array of starting positions for each QP
    if np.isscalar(nQPs):
        X = np.full(nQPs, start[0])
        Y = np.full(nQPs, start[1])
        Z = np.full(nQPs, start[2])
        start = np.array([X, Y, Z])
    
    X, Y, Z = start[0], start[1], start[2]
    phi, arctheta = np.random.uniform(0., 2.*np.pi, size=nQPs), np.random.uniform(-1., 1, size=nQPs)
    theta = np.arccos(arctheta)
    dx = np.cos( phi ) * np.sin( theta )
    dy = np.sin( phi ) * np.sin( theta )
    dz = np.cos(theta)

    #dx, dy, dz = 0., 0., 1
    total_time = np.zeros(nQPs, dtype=float)
    n=0
    #xs, ys, zs = [start[0]], [start[1]], [start[2]]
    momentum = Random_QPmomentum(T=T, size=nQPs) #keV/c
    Eb = 0.00062
    alive = np.ones(nQPs, dtype=int)
    cond = momentum < 1.1
    alive = np.where( cond, 0, alive)

    velocity = QP_velocity(momentum) #m/s
    energy = QP_dispersion(momentum) #eV
    cond = (velocity > 0.)
    alive = np.where( cond, alive, 0.)
    evaporated = np.zeros( nQPs, dtype=bool)
    
    deposits = np.zeros(nQPs, dtype=float)
    ids = np.full(nQPs, None)
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
        total_time[living] = total_time[living] + np.sqrt(dist_sq)/(velocity[living]*1.0e-4)  #us
        
        check1 = ( surface_type == "Liquid")
        #print(surface_type)
        if len(surface_type[check1]) > 0:
            
            evap, velocity[living_indices[check1]], dx[living_indices[check1]], dy[living_indices[check1]], dz[living_indices[check1]] = evaporation(momentum[living][check1], energy[living][check1], velocity[living][check1], [dx[living][check1], dy[living][check1], dz[living][check1]])
            #print("Evap", sum(evap))
            cond = (evap < 0.5) | (np.random.random(len(evap)) > evap_eff ) #check if evaporation was successful
            alive[living_indices[check1]] = np.where(cond, 0, alive[living_indices[check1]]) #kill off QPs that didn't evaporate
            evaporated[living_indices[check1]] = np.where( cond, evaporated[living_indices[check1]], True )
            X[living_indices[check1]] = X1[check1]
            Y[living_indices[check1]] = Y1[check1]
            Z[living_indices[check1]] = Z1[check1]+0.1
                                            
        #print(surface_type)
        check2 = np.array(["CPD" in str(s) for s in surface_type])
        #print("Hit %i CPDs" % len(surface_type[check2]))
        if len(surface_type[check2]) > 0:
            cpd_id = np.vectorize(extract_number)(surface_type[check2])
            cond = (evaporated[living][check2])
            #print("nDeps", len(deposits[living_indices[check2]][cond]))
            deposits[living_indices[check2]] = np.where( cond, energy[living_indices[check2]], deposits[living_indices[check2]])
            alive[living_indices[check2]] = np.zeros(len(alive[living_indices[check2]]), dtype=int) #kill off QPs, they've hit a CPD
            ids[living_indices[check2]] = np.where( cond, cpd_id, None ) #store the CPD IDs for evaporated QPs that hit a CPD
            
        check3 = (check1 == False) & (check2 == False) #doesn't reach liquid or CPD
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


def GetEvaporationSignal(detector, QPs, X, Y, Z, useMap=True, T=2.):
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
        
    hits, arrival_times, cpd_ids = QP_propagation(QPs, [X, Y, Z], conditions, detector.get_QP_reflection_prob(), evap_eff=detector.get_evaporation_eff(), T=T)
    
    coincidence = 0
    #add in baseline noise for each detected photon
    for i in range(nCPDs):
        cond = (cpd_ids == i)
        chAreas[i] = sum(hits[cond] + detector.get_adsorption_gain()) # eV
        coincidence += min([len(hits[cond]), 1]) # 0 or 1
        thisCPD = detector.get_CPD(i)
        noiseBaseline = thisCPD.get_baselineNoise()
        noise = np.random.normal(noiseBaseline[0], noiseBaseline[1], size=len(hits[cond]))
        chAreas[i] += np.sum(noise)
        arrivalTimes[i] = arrival_times[cond]

    return CPD_Signal(sum(chAreas), chAreas, coincidence, arrivalTimes)

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
