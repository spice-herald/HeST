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
            


class HestNoiseFactory:
    """
    Noise that handles the noise across our CPDs. Creates sample noise from a single PSD 
    (assuming all channels have identical noise spectra and that noise is uncorrelated 
    between the two

    Parameters
    ----------
        CSD : array
            1-D array (in the case of PSD) or 3-D array (in the case of CSD) that defines the noise
        fs : float
            Sampling frequency with which the CSD is recorded
        nsamples : int
            Number of samples for the desired noise. For now it is mandated that this be an integer 
            multiple of the number of samples over which the user-give CSD is defined  

    """
    def __init__(self, PSD, fs, nsamples):
        if not PSD.ndim == 1:
            raise ValueError("PSD should have 1 dimension ")
        if nsamples % len(PSD) != 0:
            raise ValueError("nsamples should be an integer multiple of the PSD sample count")

        self.nsamples = nsamples
        self.fs = fs
        self.PSD = PSD
  
        self.gen_interpolated_PSD()



    def gen_interpolated_PSD(self):
        """
        Generate PSD that's interpolated at the necessary frequencies to produce noise of 
        the length nsamples. Frequencies that would require extrapolation are manually set
        to zero

        """
        self.PSD_interp = np.ones( self.nsamples )*(0+0j)
        freqs = fftfreq(self.PSD.size, self.fs)
        freqs_interp = fftfreq(self.nsamples, self.fs)
        self.freqs = freqs
        self.freqs_interp = freqs_interp

        self.PSD_interp = np.where( (0 < np.abs(freqs_interp)) & (np.abs(freqs_interp) < freqs[1]), 0+0j, interp1d(freqs, self.PSD, bounds_error=False, fill_value = 0+0j)(freqs_interp))

        norm = len(self.PSD_interp) * self.fs
        self.fourier_stds = np.sqrt(self.PSD_interp * norm)

    def gen_noise(self):
        """
        Generate noise from our interpolated PSD
        """
        n = self.nsamples
        half = (n - 1) // 2
        even = (n % 2 == 0)


        zs = (np.random.normal(size=half) + 1j * np.random.normal(size=half)) / np.sqrt(2)

        if even:
            zs_final = np.empty(n, dtype=np.complex128)
            zs_final[0] = 0.0
            zs_final[1:half+1] = zs
            zs_final[half+1] = np.random.normal()  # Nyquist frequency term
            zs_final[half+2:] = np.conjugate(zs[::-1])
        else:
            zs_final = np.empty(n, dtype=np.complex128)
            zs_final[0] = 0.0
            zs_final[1:half+1] = zs
            zs_final[half+1:] = np.conjugate(zs[::-1])

        zs_final *= self.fourier_stds

        return np.fft.ifft(zs_final).real


# Polynomial functions to get the energy channel partitioning for ERs and NRs

def ER_QP_eFraction(energy):
    """
    Returns fraction of a electroic recoil energy partitioning into quasiparticles
    """
    a, b, c, d = 100., 2.98839372, 13.89437102, 0.33504361
    frac =  a/((energy-c)**b) + d
    condition = (frac > 1.0) | (energy < 19.77)
    return np.where( condition, 1.0, frac )

def ER_triplet_eFraction(energy):
    """
    Returns fraction of a electronic recoil energy partitioning into triplets
    """
    a, b, c, d, e, f, g = 14.69346932, 1.17858385, 17.60481951, 0.95286617, 0.23929604, 13.81618099, 4.92551215
    frac = f/(energy-a)**b - g/(energy-c)**d + e
    condition = ( frac < 0. ) | (energy < 19.77 )
    return np.where( condition, 0., frac )

def ER_singlet_eFraction(energy):
    """
    Returns fraction of a electronic recoil energy partitioning into singlets
    """
    a, b, c, d = 2.05492076e+03, 5.93097885, 3.3091563, 0.31773768
    frac = -a/(energy-b)**c + d
    condition = (frac < 0.) | (energy < 19.77)
    return np.where( condition, 0., frac )

def NR_QP_eFraction(energy):
    """
    Returns fraction of a nuclear recoil energy partitioning into quasiparticles
    """
    logE = np.log10(energy)
    a, b, c, d, e, f, g =  1.37928294, 1.62782641, -3.64361207, 2.25802903, 0.07210282, -0.31126613, -0.00495869
    frac = a/logE/logE + b/logE + c + d*logE + e*logE*logE + f*logE*logE*logE + \
           e*logE*logE*logE*logE + g*logE*logE*logE*logE*logE
    condition = (frac > 1.0) | (energy < 19.77)
    return np.where( condition, 1.0, frac )

def NR_triplet_eFraction(energy):
    """
    Returns fraction of a nuclear recoil energy partitioning into triplets
    """
    x = np.log10(energy)
    a, b, c, d, e, f, g, h =  -7.73003451, 17.1117164, -12.0306409, 4.388470, -2.5361635, 0.661017, -8.41417e-2,  4.2301e-03
    frac =  a/x + b + c*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x + g*x*x*x*x*x*x + h*x*x*x*x*x*x*x
    condition = (frac < 0.) | (energy < 19.77)
    return np.where( condition, 0., frac)

def NR_singlet_eFraction(energy):
    """
    Returns fraction of a nuclear recoil energy partitioning into singlets
    """

    x = np.log10(energy)
    a, b, c, d = -6.74959165e+01,  1.70997665e+02, -1.42571001e+02,  8.45017681e+01
    e, f, g, h = -6.92781245e+01,  2.87318953e+01, -7.05306162e+00,  1.03483204e+00
    i, j  = -8.39958299e-02,  2.90462675e-03
    frac = a/x + b + c*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x + g*x*x*x*x*x*x + h*x*x*x*x*x*x*x + i*x*x*x*x*x*x*x*x + j*x*x*x*x*x*x*x*x*x
    condition = (frac < 0.) | (energy < 19.77)
    return np.where( condition, 0., frac)


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

def Average_QPEnergy(T= 2., upper_bound = 4.54):
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

#HeST main yields function
def GetSingletYields(energy, interaction):


    detection_gain = 0.052 #to get the LBL yields to match with the ~35% energy fraction in prompt yields

    meanERyield = 1.25049e-3# +/- 0.03153 phe/eV
    meanNRyield = 0.477545e-3 # +/- 0.006395 phe/eV

    ERsingletFraction = 0.85824614
    NRsingletFraction = 0.72643866
    cond = (energy < 19.77)
    energy = np.where(cond, 0., energy)
    if interaction == "ER":
        return meanERyield*energy/detection_gain * ERsingletFraction #convert from phe/keV to n_photons
    else:
        if interaction == "NR":
            return meanNRyield*energy/detection_gain *NRsingletFraction #convert from phe/keV to n_photons
        else:
            print("Please specify ER or NR for interaction type!")


def Get_Quasiparticles(qp_energy, T=2.): 
    #returns the mean number of quasiparticles given 
    # the energy in eV in the QP channel
    
    #based on a power law fit after calculating the nQPs
    # using random sampling of the dispersion relation
    # given a Bose-Einstein distribution for nD/dp
    Coeff =  1156.25/pow(T, 3.58) + 1137.3
    Pow   = 1.0 - 8.74e-4*np.exp(-T/0.264) 
 
    #slope_t2, b_t2 = 1244.04521473, 29.10987933 #linear fit params
    #return slope_t2*qp_energy + b_t2
    return  Coeff*pow(qp_energy,Pow) 


def GetTripletEnergy( triplet_fraction, singlet_fraction, singlet_energy ):
    return (triplet_fraction/singlet_fraction) * singlet_energy

def GetQPEnergy( QP_fraction, singlet_fraction, singlet_energy ):
    return (QP_fraction/singlet_fraction) * singlet_energy


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

