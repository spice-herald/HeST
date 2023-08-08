from scipy.interpolate import interpn
import numpy as np
from .HeST_Core import CPD_Signal 
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
        self.adsorption_gain    = adsorption_gain #meV
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
        
    def set_liquid_surface( f1 ):
        self.liquid_surface = f1
    def set_liquid_conditions( f1 ):
        self.liquid_conditions = f1
    def set_adsorption_gain( p1 ):
        self.adsorption_gain = p1
    def set_evaporation_eff( p1 ):
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
        m = np.zeros((len(x), len(y), len(z)), dtype=float)
        conditions = self.get_surface_conditions()
        for i in range(self.get_nCPDs()):
            conditions.append( (self.get_CPD(i)).get_surface_condition() )
        
        refl_prob = self.get_photon_reflection_prob()
        for xx in range(len(x)):
            print("%i / %i" % (xx, len(x)))
            for yy in range(len(y)):
                for zz in range(len(z)):
                    
                    hitProbs = 0
                    pos = np.array([x[xx], y[yy], z[zz]])
                    if (self.get_liquid_conditions())(*pos) == False:
                        continue
                    for n in range(nPhotons):
                    
                        hit, arrival_time, n, xs, ys, zs = photon_propagation(pos, conditions, refl_prob)
                        hitProbs += hit
                        
                                
                    hitProbs = hitProbs/nPhotons
                    
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
    
    def create_QPEmap(self, x_array, y_array, z_array, nQPs=10000, filestring="detector_QPEmap"):
        print("Creating QPE map for this detector geometry...")
        x, y, z = np.array(x_array), np.array(y_array), np.array(z_array)
        self.set_QPEmap_positions( [x, y, z] )
        m = np.zeros((len(x), len(y), len(z)), dtype=float)
        conditions = self.get_surface_conditions()
        conditions.append( self.liquid_surface )
        for i in range(self.get_nCPDs()):
            conditions.append( (self.get_CPD(i)).get_surface_condition() )
        
        refl_prob = self.get_QP_reflection_prob()
        for xx in range(len(x)):
            print("%i / %i" % (xx, len(x)))
            for yy in range(len(y)):
                for zz in range(len(z)):
                    
                    hitProbs = 0
                    pos = np.array([x[xx], y[yy], z[zz]])
                    if (self.get_liquid_conditions())(*pos) == False:
                        continue
                    for n in range(nQPs):

                        hit, time, n, xs, ys, zs, p, surf = QP_propagation(pos, conditions, refl_prob, evap_eff=self.evaporation_eff)
                        hitProbs += hit
                        
                                
                    hitProbs = hitProbs/nQPs
                    
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
    t = np.linspace(0, 8, 500)  # Parameter range for the line
    # Calculate the line coordinates
    x_line = start[0] + t * direction[0]
    y_line = start[1] + t * direction[1]
    z_line = start[2] + t * direction[2]
    intersection_points = []
    dist = 999.
    coords = [None, None, None]
    surface_type = None
    for cond in conditions:
        cut, surface = cond(x_line, y_line, z_line)
        first_int = np.argmax(~cut)
        if first_int == 0:
            continue
        d = t[first_int]
        if d < dist:
            dist = d
            #get the coords of the first point *before* the interaction
            coords = [x_line[first_int-1], y_line[first_int-1], z_line[first_int-1]]
            surface_type = surface
    return coords[0], coords[1], coords[2], surface_type

def evaporation(momentum, energy, velocity, direction):
    #adapted from Pratyush's code
    mass = 3.725472e9 #He mass in eV/c^2
    c = 2.998e8 #m/s
    Eb = 0.00062
    
    if velocity <= 0.:
        return 0., 0., 0., 0., 0.
    theta = np.arccos(direction[2]) #radians
    #sin_critical_angle = np.sqrt( 2*(energy-0.00062)/(4.002603254e9) )/(np.abs(velocity)/3e8)
    sin_critical_angle = np.sqrt(2.*mass*(energy - Eb))/momentum/1000. #Eq 2.19 from J. Adams thesis
    if sin_critical_angle > 1.:
        return 0., 0., 0., 0., 0.
    crit_angle = np.arcsin( sin_critical_angle ) #radians
    if theta > crit_angle:
        return 0., 0., 0., 0., 0.
    Velocity_He_atom=np.sqrt( 2.*(energy-Eb)/(mass) )*c
    
    #sin_theta_R = (velocity/c)*np.sin((theta))/np.sqrt( 2*(energy-Eb)/(mass) )
    sin_theta_R = (momentum/1000.)*np.sin(theta)/np.sqrt(2.*mass*(energy-Eb)) #Eq 2.18 from Adams thesis
    theta_R = np.arcsin(sin_theta_R)
    new_Vz=Velocity_He_atom*np.cos(theta_R)
    if direction[1] == 0.:
        tan = np.pi/2.
        new_Vy = 0.        
    else:
        tan = np.arctan(np.abs(direction[0]/direction[1]) )
        new_Vy= direction[1]/np.abs(direction[1])*Velocity_He_atom*np.sin(theta_R)*np.sin((tan))
    if direction[0] == 0.:
        new_Vx = 0.
    else:
        new_Vx= direction[0]/np.abs(direction[0])*Velocity_He_atom*np.sin(theta_R)*np.cos((tan))
        
    magnitude=np.sqrt(new_Vx*new_Vx+new_Vy*new_Vy+new_Vz*new_Vz)
    return 1., Velocity_He_atom, new_Vx/magnitude, new_Vy/magnitude, new_Vz/magnitude


def QP_propagation(start, conditions, reflection_prob, evap_eff=0.60):
    phi, arctheta = np.random.uniform(0., 2.*np.pi), np.random.uniform(-1., 1)
    theta = np.arccos(arctheta)
    dx = np.cos( phi ) * np.sin( theta )
    dy = np.sin( phi ) * np.sin( theta )
    dz = np.cos(theta)
    
    #dx, dy, dz = 0., 0., 1
    total_time = 0.
    n=0
    xs, ys, zs = [start[0]], [start[1]], [start[2]]
    momentum = Random_QPmomentum() #keV/c
    if momentum < 1.1:
        return 0, total_time, n, xs, ys, zs, momentum, None
    velocity = QP_velocity(momentum) #m/s
    #print(velocity)
    energy = QP_dispersion(momentum) #eV
    while True:
        n+=1
        X1, Y1, Z1, surface_type = find_surface_intersection(start, [dx, dy, dz], conditions)
        if surface_type == None:
            return 0, total_time, n, xs, ys, zs, momentum, None
        xs.append(X1)
        ys.append(Y1)
        zs.append(Z1)
        total_time += (np.sqrt(pow(X1-start[0],2.)+pow(Y1-start[1], 2.)+pow(Z1-start[2],2.)))/(velocity*1.0e-4) #us
        if surface_type == "Liquid":
            evap, velocity, dx, dy, dz = evaporation(momentum, energy, velocity, [dx, dy, dz])
            #evap = 1.
            #print("Liquid", velocity)
            #if evap < 0.5:
            if  (evap < 0.5) or (np.random.random() > evap_eff ):
                return 0, total_time, n, xs, ys, zs, momentum, surface_type
            else:
                Z1 = 2.77 #just above the liquid surface
                start = [X1, Y1, Z1]
                continue
            
        if "CPD" in surface_type:
            return 1, total_time, n, xs, ys, zs, momentum, surface_type
        r = np.random.random()
        if r > reflection_prob:
            return 0, total_time, n, xs, ys, zs, momentum, surface_type
        if surface_type == "X":
            dx *= -1.
        if surface_type == "Y":
            dy *= -1.
        if surface_type == "Z":
            dz *= -1.
        if surface_type == "XY":
            normal_vector = np.array([X1, Y1, 0.])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            incident_vector = np.array([dx, dy, dz])
            reflected_vector = incident_vector - 2. * np.dot(incident_vector, normal_vector) * normal_vector
            dx, dy, dz = reflected_vector / np.linalg.norm(reflected_vector)
        start = [X1, Y1, Z1]
        
        
def photon_propagation(start, conditions, reflection_prob):
    phi, arctheta = np.random.uniform(0., 2.*np.pi), np.random.uniform(-1., 1)
    theta = np.arccos(arctheta)
    dx = np.cos( phi ) * np.sin( theta )
    dy = np.sin( phi ) * np.sin( theta )
    dz = np.cos(theta)
    #dx, dy, dz = 0., 0., 1.
    total_time = np.random.exponential(0.004)
    n=0
    xs, ys, zs = [start[0]], [start[1]], [start[2]]
    velocity = 29979.2/1.03 #speed of light in He4 cm/us
    while True:
        n+=1
        X1, Y1, Z1, surface_type = find_surface_intersection(start, [dx, dy, dz], conditions)
        if surface_type == None:
            return 0,total_time, n, xs, ys, zs
        xs.append(X1)
        ys.append(Y1)
        zs.append(Z1)
        total_time += np.sqrt(pow(X1-start[0],2.)+pow(Y1-start[1], 2.)+pow(Z1-start[2],2.))/velocity
        if "CPD" in surface_type:
            return 1, total_time, n, xs, ys, zs
        r = np.random.random()
        if r > reflection_prob:
            return 0, total_time, n, xs, ys, zs
        if surface_type == "X":
            dx *= -1.
        if surface_type == "Y":
            dy *= -1.
        if surface_type == "Z":
            dz *= -1.
        if surface_type == "XY":
            normal_vector = np.array([X1, Y1, 0.])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            incident_vector = np.array([dx, dy, dz])
            reflected_vector = incident_vector - 2. * np.dot(incident_vector, normal_vector) * normal_vector
            dx, dy, dz = reflected_vector / np.linalg.norm(reflected_vector)
        start = [X1, Y1, Z1]

''' #########################################################################

    Define functions for getting the energy deposited in the CPDs
    
    ######################################################################### '''

def GetSingletSignal(detector, photons, X, Y, Z, useMap=True):
    '''
    Attempt to simulate the CPD response for singlet photons.
    
    If useMap == True, the detector's LCEmap is loaded and the number of photon hits are 
    calculated using that light collection efficiency given the interaction location.
    If False, loop through individual photons, calculating the number of hits 
    and the arrival times. This takes substantially more time: ~1.3s for 10k photons.
    Number of hits are converted to pulse areas using the Singlet Photon Energy
    '''
    #check to see if there's an LCEmap loaded/generated
    if type(detector.get_LCEmap()) == int:
        useMap = False
        
    nHits = 0
    arrivalTimes = []
    
    nCPDs = detector.get_nCPDs()
        
    #check to see if a map is loaded
    if useMap:
        hitProbability = detector.get_photon_hits( X, Y, Z)
        nHits = np.random.binomial(photons, hitProbability)
        
        area = nHits*Singlet_PhotonEnergy
        for i in range(nCPDs):
            thisCPD = detector.get_CPD(i)
            noiseBaseline = thisCPD.get_baselineNoise()
            noise = np.random.normal(noiseBaseline[0], noiseBaseline[1], size=nHits)
            area += np.sum(noise)
            
        return CPD_Signal(area, arrivalTimes)
    
    #if the map hasn't been loaded or we want arrival times...
    conditions = detector.get_surface_conditions()
    for i in range(nCPDs):
        conditions.append( (detector.get_CPD(i)).get_surface_condition() )
    for n in range(photons):
        hit, arrival_time, n, xs, ys, zs = photon_propagation([X, Y, Z], conditions, detector.get_photon_reflection_prob())
        nHits += hit
        arrivalTimes.append( arrival_time )
    area = nHits*Singlet_PhotonEnergy
    
    #add in baseline noise for each detected photon
    for i in range(nCPDs):
        thisCPD = detector.get_CPD(i)
        noiseBaseline = thisCPD.get_baselineNoise()
        noise = np.random.normal(noiseBaseline[0], noiseBaseline[1], size=nHits)
        area += np.sum(noise)     
        
    return CPD_Signal(area, arrivalTimes)  



def GetEvaporationSignal(detector, QPs, X, Y, Z, useMap=True):
    '''
    Attempt to simulate the CPD response for quasiparticles.
    
    If useMap == True, the detector's QPEmap is loaded and the number of QP hits are 
    calculated using that evaporation efficiency given the interaction location.
    If False, loop through individual QPs, calculating the number of hits 
    and the arrival times. This takes substantially more time: ~1.5s for 10k particles.
    Number of hits are converted to pulse areas using the detector's adsorption gain. 
    
    '''
    #check to see if there's an LCEmap loaded/generated
    if type(detector.get_QPEmap()) == int:
        useMap = False
        
    nHits = 0
    arrivalTimes = []
    
    nCPDs = detector.get_nCPDs()
        
    #check to see if a map is loaded
    if useMap:
        hitProbability = detector.get_QP_hits(X, Y, Z)
        nHits = np.random.binomial(QPs, hitProbability)
        
        area = nHits*detector.get_adsorption_gain()
        for i in range(nCPDs):
            thisCPD = detector.get_CPD(i)
            noiseBaseline = thisCPD.get_baselineNoise()
            noise = np.random.normal(noiseBaseline[0], noiseBaseline[1], size=nHits)
            area += np.sum(noise)
            
        return CPD_Signal(area, arrivalTimes)
    
    #if the map hasn't been loaded or we want arrival times...
    conditions = detector.get_surface_conditions()
    conditions.append( detector.liquid_surface )
    for i in range(nCPDs):
        conditions.append( (detector.get_CPD(i)).get_surface_condition() )
    for n in range(QPs):
        hit, arrival_time, n, xs, ys, zs, p, surf = QP_propagation([X, Y, Z], conditions, detector.get_QP_reflection_prob(), evap_eff=detector.get_evaporation_eff())
        nHits += hit
        arrivalTimes.append( arrival_time )
    area = nHits*detector.get_adsorption_gain()
    
    #add in baseline noise for each detected photon
    for i in range(nCPDs):
        thisCPD = detector.get_CPD(i)
        noiseBaseline = thisCPD.get_baselineNoise()
        noise = np.random.normal(noiseBaseline[0], noiseBaseline[1], size=nHits)
        area += np.sum(noise)     
        
    return CPD_Signal(area, arrivalTimes) 

