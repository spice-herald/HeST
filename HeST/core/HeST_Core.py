import scipy.special as scispec
from scipy.interpolate import CubicSpline, interp1d
import numpy as np
import os
from qetpy.utils import fft, ifft, fftfreq, rfftfreq
from tqdm import tqdm
from numba import njit

#Global constants
# Singlet_PhotonEnergy = 15.5 #eV
# Triplet_PhotonEnergy = 18.0 #eV

# Excitation energies for yields; these numbers are 
# geared toward the "observed" energy in that the singlet 
# one is the photon energy while the triplet one is the 
# total molecular excitation energy
Singlet_ExcitationEnergy = 15.5 # eV; derived from potential curves
Triplet_ExcitationEnergy = 18.0 # eV; derived from potential curves
IR_ExcitationEnergy = 1 # eV


# Singlet_ExcitationEnergy = 18.1 # eV; derived from potential curves
# Triplet_ExcitationEnergy = 17.8 # eV; derived from potential curves


class QuantaResult:
    """
    Simple Class for storing the number of singlet/IR photons, triplet molecules, and IR photons generated

    Attributes
    ----------
        SingletPhotons : float
            # of singlet photons generated in a recoil
        TripletMolecules : float
            # of long-lived triplet photons generated
        IRPhotons : float
            @ of IR photons generated
        Quasiparticles : float
            # of quasiparticles generated in the 
    """
    def __init__(self, SingletPhotons, TripletMolecules, IRPhotons, Quasiparticles):
        self.SingletPhotons = SingletPhotons
        self.TripletMolecules = TripletMolecules
        self.IRPhotons = IRPhotons
        self.Quasiparticles = Quasiparticles

    def get_nSingletPhotons(self):
        return self.SingletPhotons
    
    def get_nIRPhotons(self):
        return self.IRPhotons
    
    def get_nTripletMolecules(self):
        return self.TripletMolecules
    
    def get_nQuasiparticles(self):
        return self.Quasiparticles
    

class HestSignal:
    """
    Signal is an object that 

    Attributes
    ----------
        energies : list of floats
             List of length equal to number of sensors. Each element in a list is a quantum's
            energy observed (after phonon collection efficiency/adsorption gain) at the sensor in µs
        arrivalTimes : List of arrays
            List of length equal to number of sensors. Each element in a list is a quantum's
            arrival time at the sensor in µs
        templates
            
    """
    def __init__(self, energies = [[]], arrivalTimes_us=[[]]):

        self.energies = energies #total pulse area
        self.arrivalTimes = arrivalTimes_us #time at which the particle hits the sensor
    
    def __add__(self, other):
        if isinstance(other, HestSignal):
            if ( len(self.energies) == len(other.energies) ) and ( len(self.arrivalTimes) == len(other.arrivalTimes) ):

                new_energies = []
                for i in range(len(self.energies)):
                    new_energies.append(np.append(self.energies[i], other.energies[i]))

                new_times = []
                for i in range(len(self.arrivalTimes)):
                    new_times.append(np.append(self.arrivalTimes[i], other.arrivalTimes[i]))
                
                return HestSignal(new_energies, new_times)
            
            else:

                raise ValueError("Number of channels must match")
            

# Polynomial functions to get the energy channel partitioning for ERs and NRs; outdated Work by Greg R but we'll leave it for posterity (for now)

# def ER_QP_eFraction(energy):
#     """
#     Returns fraction of a electroic recoil energy partitioning into quasiparticles
#     """
#     a, b, c, d = 100., 2.98839372, 13.89437102, 0.33504361
#     frac =  a/((energy-c)**b) + d
#     condition = (frac > 1.0) | (energy < 19.77)
#     return np.where( condition, 1.0, frac )

# def ER_triplet_eFraction(energy):
#     """
#     Returns fraction of a electronic recoil energy partitioning into triplets
#     """
#     a, b, c, d, e, f, g = 14.69346932, 1.17858385, 17.60481951, 0.95286617, 0.23929604, 13.81618099, 4.92551215
#     frac = f/(energy-a)**b - g/(energy-c)**d + e
#     condition = ( frac < 0. ) | (energy < 19.77 )
#     return np.where( condition, 0., frac )

# def ER_singlet_eFraction(energy):
#     """
#     Returns fraction of a electronic recoil energy partitioning into singlets
#     """
#     a, b, c, d = 2.05492076e+03, 5.93097885, 3.3091563, 0.31773768
#     frac = -a/(energy-b)**c + d
#     condition = (frac < 0.) | (energy < 19.77)
#     return np.where( condition, 0., frac )

# def NR_QP_eFraction(energy):
#     """
#     Returns fraction of a nuclear recoil energy partitioning into quasiparticles
#     """
#     logE = np.log10(energy)
#     a, b, c, d, e, f, g =  1.37928294, 1.62782641, -3.64361207, 2.25802903, 0.07210282, -0.31126613, -0.00495869
#     frac = a/logE/logE + b/logE + c + d*logE + e*logE*logE + f*logE*logE*logE + \
#            e*logE*logE*logE*logE + g*logE*logE*logE*logE*logE
#     condition = (frac > 1.0) | (energy < 19.77)
#     return np.where( condition, 1.0, frac )

# def NR_triplet_eFraction(energy):
#     """
#     Returns fraction of a nuclear recoil energy partitioning into triplets
#     """
#     x = np.log10(energy)
#     a, b, c, d, e, f, g, h =  -7.73003451, 17.1117164, -12.0306409, 4.388470, -2.5361635, 0.661017, -8.41417e-2,  4.2301e-03
#     frac =  a/x + b + c*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x + g*x*x*x*x*x*x + h*x*x*x*x*x*x*x
#     condition = (frac < 0.) | (energy < 19.77)
#     return np.where( condition, 0., frac)

# def NR_singlet_eFraction(energy):
#     """
#     Returns fraction of a nuclear recoil energy partitioning into singlets
#     """

#     x = np.log10(energy)
#     a, b, c, d = -6.74959165e+01,  1.70997665e+02, -1.42571001e+02,  8.45017681e+01
#     e, f, g, h = -6.92781245e+01,  2.87318953e+01, -7.05306162e+00,  1.03483204e+00
#     i, j  = -8.39958299e-02,  2.90462675e-03
#     frac = a/x + b + c*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x + g*x*x*x*x*x*x + h*x*x*x*x*x*x*x + i*x*x*x*x*x*x*x*x + j*x*x*x*x*x*x*x*x*x
#     condition = (frac < 0.) | (energy < 19.77)
#     return np.where( condition, 0., frac)

#Interpolated functions digitized from Hertel et. al

singlet_ER_x = np.array([0.0, 19.77, 21.63, 23.22, 25.15, 
                         30.60, 39.39, 48.88, 72.51, 100.93, 
                         155.05, 242.75, 354.78, 593.74, 1015.77,
                         2351.09, 4698.44, 10186.55, 32983.81, 100000.0])
singlet_ER_y = np.array([0.0, 0.00, 0.086, 0.147, 0.200, 
                         0.253, 0.295, 0.320, 0.325, 0.321, 
                         0.315, 0.313, 0.311, 0.310, 0.311, 
                         0.314, 0.315, 0.317, 0.320, 0.321])

singlet_ER = interp1d(singlet_ER_x, singlet_ER_y)

def ER_singlet_eFraction(energy):
    return np.where(energy > 19.77, singlet_ER(energy), 0)

triplet_ER_x = np.array([0.0, 19.77, 20.47, 20.93, 21.32, 
                         21.89, 23.24, 24.93, 27.24, 29.35, 
                         33.46, 40.26, 51.43, 65.68, 91.09, 
                         133.70, 203.26, 380.10, 691.15, 1333.10, 
                         3014.93, 7330.22, 14626.38, 27595.60, 100000.0])
triplet_ER_y = np.array([0.0, 0.0, 0.1062, 0.194, 0.276, 
                         0.347, 0.381, 0.398, 0.381, 0.357, 
                         0.326, 0.298, 0.265, 0.241, 0.237, 
                         0.239, 0.240, 0.242, 0.241, 0.241, 
                         0.240, 0.239, 0.237, 0.237, 0.235])

triplet_ER = interp1d(triplet_ER_x, triplet_ER_y)

def ER_triplet_eFraction(energy):
    return np.where(energy > 19.77, triplet_ER(energy), 0)

qp_ER_x = np.array([0.0, 19.77, 20.48, 20.77, 21.04, 
                    21.49, 21.90, 23.16, 24.32, 25.42, 
                    30.20, 47.20, 59.84, 91.77, 158.16, 
                    410.34, 891.33, 2009.58, 4556.31, 10420.55, 
                    22034.62, 52973.97, 100000.0])
qp_ER_y = np.array([1.0, 1.0, 0.865, 0.775, 0.702, 
                    0.641, 0.553, 0.467, 0.395, 0.382, 
                    0.373, 0.346, 0.339, 0.333, 0.334, 
                    0.334, 0.334, 0.334, 0.334, 0.334, 
                    0.334, 0.334, 0.334])

qp_ER = interp1d(qp_ER_x, qp_ER_y)

def ER_QP_eFraction(energy):
    return np.where(energy > 19.77, qp_ER(energy), 1)

singlet_NR_x = np.array([0.0, 19.77, 21.21, 24.29, 28.33, 
                         35.70, 54.04, 105.43, 242.56, 415.38, 
                         716.00, 1275.54, 2083.48, 2989.18, 3960.88, 
                         5021.45, 6538.47, 8513.37, 12679.53, 15885.66, 
                         21654.95, 30359.14, 50407.77, 100000.0])
singlet_NR_y = np.array([0.0, 0.0, 0.023, 0.064, 0.088, 
                         0.109, 0.125, 0.142, 0.164, 0.183, 
                         0.208, 0.244, 0.274, 0.293, 0.297, 
                         0.294, 0.282, 0.259, 0.217, 0.192, 
                         0.163, 0.140, 0.120, 0.106])

singlet_NR = interp1d(singlet_NR_x, singlet_NR_y)

def NR_singlet_eFraction(energy):
    return np.where(energy > 19.77, singlet_NR(energy), 0)

triplet_NR_x = np.array([0.0, 19.77, 24.88, 37.04, 62.83, 
                         155.50, 391.65, 653.32, 1028.13, 1627.49, 
                         2760.35, 4480.65, 7169.79, 11793.22, 18683.25, 
                         29959.84, 46394.00, 73580.83, 100000.0])
triplet_NR_y = np.array([0.0, 0.0, 0.0104, 0.0186, 0.0251, 
                         0.0286, 0.0350, 0.0406, 0.0481, 0.0588, 
                         0.0817, 0.115, 0.167, 0.229, 0.279, 
                         0.309, 0.317, 0.317, 0.313])

triplet_NR = interp1d(triplet_NR_x, triplet_NR_y)

def NR_triplet_eFraction(energy):
    return np.where(energy > 19.77, triplet_NR(energy), 0)

qp_NR_x = np.array([0.0, 19.77, 21.53, 23.72, 27.54, 
                    34.60, 48.80, 69.98, 99.78, 158.60, 
                    219.41, 318.34, 507.09, 749.17, 1137.36, 
                    1957.43, 3517.02, 5763.96, 9036.91, 16679.51, 
                    29633.11, 47310.22, 100000.0])
qp_NR_y = np.array([1.0, 1.0, 0.970, 0.933, 0.900, 
                    0.872, 0.852, 0.840, 0.826, 0.813, 
                    0.803, 0.787, 0.764, 0.738, 0.704, 
                    0.650, 0.584, 0.534, 0.497, 0.468, 
                    0.464, 0.469, 0.487])

qp_NR = interp1d(qp_NR_x, qp_NR_y)

def NR_QP_eFraction(energy):
    return np.where(energy > 19.77, qp_NR(energy), 1)

#combine the above functions into a single function
def GetEnergyChannelFractions(energy, interaction):
    """
    Wrapper function to get the fractional yields of singlet, triplet, QP, and IR

    Parameters 
    ----------
    energy : float
        energy in eV
    interaction : string
        "ER" or "NR"

    Returns
    -------
    singlet, triplet, QP, IR : floats
        fractions of energy in each channel; should sum to 1


    """

    maxEnergy = 1.0e5
    condition = (energy > maxEnergy)

    energy = np.where( condition, maxEnergy, energy )
    
    # energy -- recoil energy in eV
    if interaction == "ER":
        singlet = ER_singlet_eFraction(energy)
        triplet = ER_triplet_eFraction(energy)
        QP      = ER_QP_eFraction(energy)
        res     = 1-singlet-triplet-QP
        cond    = (res < 0.)
        IR      = np.where( cond, 0., res )
    else:
        if interaction == "NR":
            singlet = NR_singlet_eFraction(energy)
            triplet = NR_triplet_eFraction(energy)
            QP      = NR_QP_eFraction(energy)
            res     = 1-singlet-triplet-QP
            cond    = (res < 0.)
            IR      = np.where( cond, 0., res )
        else:
            print("Please specify ER or NR for interaction type!")
            return 1

    return singlet, triplet, QP, IR

def Average_QPEnergy(T = 2., upper_bound = 4.54):
    """
    Estimates the number of quasiparticles that will be created given the total Quasiparticle energy.
    Defaults to assuming that the QPs follow a BE distribution 

    Parameters
    ----------
        E_total : float
            Sum of all of the QP's energies in eV
        T : float or None
            Effective temperature of the BE distribution. Should be O(1) K; see arXiv 2208.14474
            If None, defaults to limit of n(p)~p^2

    Returns
    -------
        avg_num : float
            Average number of quasiparticles to expect
    """

    k = 8.617e-5
    p = np.linspace(0.001, upper_bound, 1000) 
    if T is not None:
        probs = p*p/(np.exp(QP_dispersion(p)/(k*T)) - 1)/np.sum(p*p/(np.exp(QP_dispersion(p)/(k*T)) - 1))
    else:
        probs = p*p/np.sum(p*p)

    avg_energy = np.sum( probs * QP_dispersion(p) )
    return avg_energy


#combine the above functions into a single function
def Sim_AbsYields(energy, interaction):
    """
    Wrapper function to get the 
    ----------
    energy : float
        energy in eV
    interaction : string
        "ER" or "NR"

    Returns
    -------
    singlet, triplet, QP, IR : floats
        fractions of energy in each channel; should sum to 1


    """

    maxEnergy = 1.0e5
    condition = (energy > maxEnergy)

    energy = np.where( condition, maxEnergy, energy )
    
    # energy -- recoil energy in eV
    if interaction == "ER":
        singlet = ER_singlet_eFraction(energy)
        triplet = ER_triplet_eFraction(energy)
        QP      = ER_QP_eFraction(energy)
        res     = energy-singlet-triplet-QP
        cond    = (res < 0.)
        IR      = np.where( cond, 0., res )
    else:
        if interaction == "NR":
            singlet = NR_singlet_eFraction(energy)
            triplet = NR_triplet_eFraction(energy)
            QP      = NR_QP_eFraction(energy)
            res     = energy-singlet-triplet-QP
            cond    = (res < 0.)
            IR      = np.where( cond, 0., res )
        else:
            print("Please specify ER or NR for interaction type!")
            return 1

    return singlet, triplet, QP, IR

def GetQuanta(energy, interaction, T=2., atomic_fano = 1.0, asQuantaResult = True, track_unstable_QPs = False):
    if np.isscalar(energy):
        energy = np.array([energy])
    
    singlet_fraction, triplet_fraction, _, IR_fraction = GetEnergyChannelFractions(energy, interaction)

    singlet_energy = singlet_fraction*energy
    triplet_energy = triplet_fraction*energy

    IR_energy = IR_fraction*energy
    nSingletExcitations_mean = singlet_energy / Singlet_ExcitationEnergy
    nTripletExcitations_mean = triplet_energy / Triplet_ExcitationEnergy
    nIRExcitations_mean = IR_energy / IR_ExcitationEnergy

    nSingExcitations_actual = (np.random.normal(nSingletExcitations_mean, np.sqrt(atomic_fano*nSingletExcitations_mean))).astype(int)
    nTripletExcitations_actual = (np.random.normal(nTripletExcitations_mean, np.sqrt(atomic_fano*nTripletExcitations_mean))).astype(int)
    nIRExcitations_actual = (np.random.normal(nIRExcitations_mean, np.sqrt(atomic_fano*nIRExcitations_mean))).astype(int)

    QP_energy = energy - (singlet_energy + triplet_energy + IR_energy)
    
    if track_unstable_QPs:
        QP_avg_energy = Average_QPEnergy(T=T, cutoff = 6.5)
    else:
        QP_avg_energy = Average_QPEnergy(T=T)

    nQPs_actual = (QP_energy/QP_avg_energy).astype(int)
    if asQuantaResult:
        return QuantaResult( nSingExcitations_actual[0], nTripletExcitations_actual[0], nIRExcitations_actual[0], nQPs_actual[0] )
    else:
        return nSingExcitations_actual, nTripletExcitations_actual, nIRExcitations_actual, nQPs_actual

# Quasiparticle functions
def GetInterpFunc(d_path, lower_bound = None, upper_bound = None, reverse_xy = False, flip_order = False):

    data = np.load(d_path)


    if reverse_xy == False:
        X = data[0,lower_bound:upper_bound]
        Y = data[1,upper_bound:upper_bound]
    else:
        X = data[1,lower_bound:upper_bound]
        Y = data[0,lower_bound:upper_bound]
        
    if flip_order:
        X, Y = np.flip(X), np.flip(Y)
    
    return CubicSpline(X,Y, extrapolate = False)
   


def get_phonon_mom_energy(d_path):
    data = np.loadtxt(d_path, delimiter=',')
    X = data[0:62,1]
    Y = data[0:62,0]
    return interp1d(X,Y, kind = 'linear')

def get_rminus_mom_energy(d_path):
    data = np.loadtxt(d_path, delimiter=',')
    X = data[61:102,1]
    Y = data[61:102,0]
    return interp1d(X,Y, kind = 'linear')

def get_rplus_mom_energy(d_path):
    data = np.loadtxt(d_path, delimiter=',')
    X = data[101:,1]
    Y = data[101:,0]
    return interp1d(X,Y, kind = 'linear')
    

dispersion_data_path = os.path.dirname(os.path.abspath(__file__))+ '/../dispersion_curves/QP_dispersion_curve.npy'
velocity_data_path = os.path.dirname(os.path.abspath(__file__))+ '/../dispersion_curves/QP_velocity_curve.npy'

QP_dispersion_base = GetInterpFunc(dispersion_data_path)
QP_velocity_base = GetInterpFunc(velocity_data_path)

def QP_dispersion(p ):
    """
    Takes in quasiparticle momentum in keV/c; spits out quasiparticle energy in eV
    
    Cubic spline of data from https://link.springer.com/article/10.1007/BF00117839
    ("Specific heat and dispersion curve for helium II" by Donnelly et. al)

    Parameters
    ----------
        p : array/float
            QP momentum in keV/c
    Returns
    -------
        energy : array/float
            QP energy in eV

    """

    return QP_dispersion_base(p) 

def QP_velocity(p):
    """
    Takes in quasiparticle momentum in keV/c; spits out quasiparticle velocity in m/s

    Cubic spline of digitized data from https://link.springer.com/article/10.1007/BF00117839
    ("Specific heat and dispersion curve for helium II" by Donnelly et. al)

    Parameters
    ----------
        p : array/float
            QP momentum in keV/c
    Returns
    -------
        velocity : array/float
            QP velocity in m/s
            
    """

    return QP_velocity_base(p) 


phonon_momentum_base = GetInterpFunc(dispersion_data_path, reverse_xy = True, upper_bound = 23)
rot_minus_momentum = GetInterpFunc(dispersion_data_path, reverse_xy = True, lower_bound = 22, upper_bound = 40, flip_order=True)
rot_plus_momentum = GetInterpFunc(dispersion_data_path, reverse_xy = True, lower_bound = 39)

def phonon_momentum(E):
    """
    Takes in quasiparticle energy in eV; spits out quasiparticle 
    momentum under the assumption of it being a phonon in keV/c

    Parameters
    ----------
        p : array/float
            QP energy in eV
    Returns
    -------
        momentum : array/float
            QP momentum in keV/c
            
    """
    return phonon_momentum_base(E) 

def rminus_momentum(E):
    """
    Takes in quasiparticle energy in eV; spits out quasiparticle 
    momentum under the assumption of it being a R- in keV/c

    Parameters
    ----------
        p : array/float
            QP energy in eV
    Returns
    -------
        momentum : array/float
            QP momentum in keV/c
            
    """
    return rot_minus_momentum(E) 

def rplus_momentum(E):
    """
    Takes in quasiparticle energy in eV; spits out quasiparticle 
    momentum under the assumption of it being a R+ in keV/c

    Parameters
    ----------
        p : array/float
            QP energy in eV
    Returns
    -------
        momentum : array/float
            QP momentum in keV/c
            
    """
    return rot_plus_momentum(E) 


def Random_QPmomentum(nQPs, T=2, pmin = .001, pmax = 4.54):
    """"
    Randomly sample Quasiparticles from a Bose-Einstein distribution with some effective temperature

    Parameters
    ----------
        nQPs : int
            Number of qps to sample
        T : float
            Effective temperature of the BE distribution. Should be O(1) K; see arXiv 2208.14474
        pmin : float
            minimum momentum to sample in keV/c. As close to zero as possible
        pmax : float
            maximum momentum to sample in keV/c. Should be the highest momentum while still being stable.
            (It's assumed that higher energy QPs will decay down and thermalize.)

    Return
    ------
        inverse_cdf : array
            array of length nQPs that gives quasiparticle momenta in keV/C

    """

    k = 8.617e-5
    p = np.linspace(pmin, pmax, 1000) 
    probabilities = p*p/(np.exp(QP_dispersion(p)/(k*T)) - 1)
    cumulative_probabilities = np.cumsum(probabilities) / np.sum(probabilities)
    inverse_cdf = np.interp(np.random.rand(nQPs), cumulative_probabilities, p)
    return inverse_cdf

