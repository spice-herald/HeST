import scipy.special as scispec
from scipy.interpolate import interp1d
import numpy as np
import os
from qetpy.utils import fft, ifft, fftfreq, rfftfreq
from tqdm import tqdm

#Global constants
Singlet_PhotonEnergy = 15.5 #eV
Triplet_PhotonEnergy = 18.0 #eV

#create some other classes to store results
#unlike NEST, we won't have a mean yields object; just the quanta object with fluctuations applied
class QuantaResult:
    def __init__(self, SingletPhotons, TripletPhotons, Quasiparticles):
        self.SingletPhotons = SingletPhotons
        self.TripletPhotons = TripletPhotons
        self.Quasiparticles = Quasiparticles



class HestSignal:
    """
    Signal is an object that 

    attributes
    ----------
        energies : list of floats
             List of length equal to number of sensors. Each element in a list is a quantum's
            energy observed (after phonon collection efficiency/adsorption gain) at the sensor in µs
        arrivalTimes : List of arrays
            List of length equal to number of sensors. Each element in a list is a quantum's
            arrival time at the sensor in µs
        templates
            
    """
    def __init__(self, energies = [[]], arrivalTimes_us=[[]], templates = None, eV_amp_ratios = None):

        self.energies = energies #total pulse area
        self.arrivalTimes = arrivalTimes_us #time at which the particle hits the sensor

class HestNoise:
    """
    Noise that handles the noise across our CPDs. Creates sample noise from PSDs/CSDs

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
    def __init__(self, CSD, fs, nsamples):
        if CSD.ndim == 3:
            self.nchannels = CSD.shape[1]
            if CSD.shape[0] != CSD.shape[1]:
                raise ValueError("Inconsistently defined CSD")
        elif CSD.ndim == 1:
            
            self.nchannels = 1
            CSD = np.array([[CSD]])
        else:
            raise ValueError("CSD should have either 1 dimension (PSD) or 3")
        if nsamples % CSD.shape[2] != 0:
            raise ValueError("nsamples should be an integer multiple of the CSD sample count")

        self.nsamples = nsamples
        self.fs = fs
        self.CSD = CSD
        self.CSD_interp = None
        self.L_matrices = None
        self.L_matrices_interp = None
        self.waveform = None


    def gen_interpolated_CSD(self):
        """
        Generate CSD that's interpolated at the necessary frequencies to produce noise of 
        the length nsamples. Frequencies that would require extrapolation are manually set
        zero

        """
        self.CSD_interp = np.ones((self.nchannels, self.nchannels, self.nsamples))*(0+0j)
        freqs = fftfreq(self.CSD.shape[2], self.fs)
        freqs_interp = fftfreq(self.nsamples, self.fs)

        for i in range(self.nchannels):
            for j in range(self.nchannels):
                if j >= i:
                    self.CSD_interp[i,j] = np.where((0 < np.abs(freqs_interp)) & (np.abs(freqs_interp) < freqs[1]), 0+0j, interp1d(freqs, self.CSD[i,j], bounds_error=False, fill_value = 0+0j )(freqs_interp))
                    #FIXME: the above line doesn't handle the highest frequencies well. If the nyquist frequecny has no multiplicity, then
                else:
                    self.CSD_interp[j,i] = np.conjugate(self.CSD_interp[i,j])

    
    def gen_noise_interpolation(self, calc_Ls = False ):
        """
        Generate noise of a desired length by using an CSD defined over longer time periods.
        """
        if self.CSD_interp is None:
            self.gen_interpolated_CSD()
        norm = self.CSD.shape[2] * self.fs

        #(Re)calculate the L matrices as needed/requested by the user
        print(calc_Ls)
        print(self.L_matrices_interp is None)
        if calc_Ls or self.L_matrices_interp is None:
            n=0
            self.L_matrices_interp = np.ones_like(self.CSD_interp)*(0+0j)
            for f in tqdm(range(self.nsamples)):
                try:
                    L = np.linalg.cholesky(norm*self.CSD_interp[:,:,f])
                except:
                    L = np.ones((self.nchannels,self.nchannels))*(0+0j)
                    n+=1
                self.L_matrices_interp[:,:,f] = L
            print("failed on "+str(n))

        zs = np.random.normal(loc =0, scale= np.sqrt(2), size= (self.nchannels,self.nsamples)) 
        zs = zs * np.exp( 2 * np.pi * 1j * np.random.uniform(0, 1, size= (self.nchannels,self.nsamples)))
        if self.nsamples % 2:
            zs[:,0] /= np.sqrt(2)
            zs[:,int(self.nsamples/2)] /= np.sqrt(2)
        else:
            zs[:0] /= np.sqrt(2)

        fourier_coefficients = np.einsum('ijk,jk->ik',self.L_matrices_interp, zs)
        waveform = np.real(ifft(fourier_coefficients,axis = 1))
        self.waveform = waveform
    
    def gen_noise_stitching(self):
        """
        Generate noise of a desired length by generating a sufficient number of traces of length defined 
        by the user-defined CSD and stitching them together 
        """
        count = self.nsamples // self.CSD.shape[2]
        waveforms = []
        for i in range(count):
            waveforms.append(self.gen_noise())

        self.waveform = np.concatenate(waveforms)

    def gen_noise(self, calc_Ls=False):
        """
        Generate noise of length given by the user-defined CSD.

        Returns
        -------
            noise : array
                Array whose row vectors are sampled noise on each of the channels
        """   
        norm = self.CSD.shape[2] * self.fs

        #(Re)calculate the L matrices as needed/requested by the user
        if calc_Ls or self.L_matrices is None:
            n=0
            self.L_matrices = np.ones_like(self.CSD)*(0+0j)
            for f in tqdm(range(self.CSD.shape[2])):
                try:
                    L = np.linalg.cholesky(norm*self.CSD[:,:,f])
                except:
                    L = np.ones((self.nchannels,self.nchannels))*(0+0j)
                    n+=1
                self.L_matrices[:,:,f] = L
            print('failed on '+str(n))

        zs = np.random.normal(loc =0, scale= np.sqrt(2), size= (self.nchannels,self.CSD.shape[2])) 
        zs = zs * np.exp( 2 * np.pi * 1j * np.random.uniform(0, 1, size= (self.nchannels,self.CSD.shape[2])))
        if self.CSD.shape[2] % 2 == 0:
            zs[:,0] /= np.sqrt(2)
            zs[:,int(self.CSD.shape[2]/2)] /= np.sqrt(2)
        else:
            zs[:,0] /= np.sqrt(2)
        

        fourier_coefficients = np.einsum('ijk,jk->ik',self.L_matrices, zs)
        noise = np.real(ifft(fourier_coefficients,axis = 1))
        return noise
    
    
    

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
    # returns the mean prompt scintillation yield for
    # the prompt + exponential + 1/t components

    # energy -- recoil energy in eV
    # interaction -- "ER" or "NR"

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

def GetQuanta(energy, interaction, T=2.):
    # energy -- recoil energy in eV
    # interaction -- "ER" or "NR"

    if np.isscalar(energy):
        energy = np.array([energy])
        
    singlet_fraction, triplet_fraction, QP_fraction, IR_fraction = GetEnergyChannelFractions(energy, interaction)
    #nSingletPhotons = GetSingletYields(energy*singlet_fraction, interaction)
    #singlet_energy = nSingletPhotons * Singlet_PhotonEnergy
    singlet_energy = singlet_fraction*energy
    nSingletPhotons = singlet_energy / Singlet_PhotonEnergy
    
    Fano = 1.0
    cond = (singlet_fraction > 0.)
    triplet_energy = np.where( cond, GetTripletEnergy( triplet_fraction, singlet_fraction, singlet_energy ), 0. )
    QP_energy = np.where( cond, GetQPEnergy( QP_fraction, singlet_fraction, singlet_energy ), energy )

    nTripletPhotons = triplet_energy / Triplet_PhotonEnergy #(eV / (eV/ph))

    scint_energy = singlet_energy + triplet_energy

    nPhotons = nSingletPhotons + nTripletPhotons

    nSing_actual = (np.random.normal(nSingletPhotons, np.sqrt(Fano*nSingletPhotons))).astype(int)
    cond = ( nSing_actual > 0 )
    nSing_actual = np.where( cond, nSing_actual, 0 ).astype(int)

    nTrip_actual = (np.random.normal(nTripletPhotons, np.sqrt(Fano*nTripletPhotons)) ).astype(int)
    cond = ( nTrip_actual > 0 )
    nTrip_actual = np.where( cond, nTrip_actual, 0 ).astype(int)

    scint_energy_actual = nSing_actual * Singlet_PhotonEnergy + nTrip_actual * Triplet_PhotonEnergy

    #assume the fano fluctuations are anti-correlated with the QP energy
    QP_energy += scint_energy - scint_energy_actual
    cond = (QP_energy > 0)
    QP_energy = np.where( cond, QP_energy, 0. )

    nQP = (Get_Quasiparticles(QP_energy, T=T)).astype(int)
    #if there are no photons, apply fano fluctuations directly to QPs
    cond = ( scint_energy > 0. )
    nQP_actual = np.where( cond, nQP, ( np.random.normal(nQP, np.sqrt(Fano*nQP)) ).astype(int) )

    cond = (nQP_actual > 0)
    nQP_actual = np.where( cond, nQP_actual, 0 ).astype(int)

    return QuantaResult( nSing_actual, nTrip_actual, nQP_actual )

# Quasiparticle functions
def GetInterpFunc(d_path):
    """Creates an linear interpolation function from data found at the file path below,, giving us the ability to convert from resistance to temperature. 
    returns:
        Interpoltion function: If input exceeds range of the data function returns a NaN"""
    data = np.loadtxt(d_path, delimiter=',')
    X = data[:,0]
    Y = data[:,1]
    return interp1d(X,Y, kind = 'linear')

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
    

def QP_dispersion(p ):
    """
    Takes in quasiparticle momentum in keV/c; spits out quasiparticle energy in eV

    Parameters
    ----------
        p : array/float
            QP momentum in keV/c
    Returns
    -------
        energy : array/float
            QP energy in eV

    """
    dispersion_data_path = os.path.dirname(os.path.abspath(__file__))+ '/../dispersion_curves/dispersion_data.csv'
    interp = GetInterpFunc(dispersion_data_path)

    energy = interp(p)
    return energy * 1e-3 

def QP_velocity(p ):
    """
    Takes in quasiparticle momentum in keV/c; spits out quasiparticle velocity in m/s

    Parameters
    ----------
        p : array/float
            QP momentum in keV/c
    Returns
    -------
        velocity : array/float
            QP velocity in m/s
            
    """
    dispersion_data_path = os.path.dirname(os.path.abspath(__file__))+ '/../dispersion_curves/velocity_data.csv'
    interp = GetInterpFunc(dispersion_data_path)
    velocity = interp(p)
    return velocity 


def Random_QPmomentum(nQPs, T=2, pmin = .15, pmax = 4.6):
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
    p = np.linspace(.15, 4.7, 1000) 
    probabilities = p*p/(np.exp(QP_dispersion(p)/(k*T)) - 1)
    cumulative_probabilities = np.cumsum(probabilities) / np.sum(probabilities)
    inverse_cdf = np.interp(np.random.rand(nQPs), cumulative_probabilities, p)
    return inverse_cdf


def est_QPcount(E_total, T= 2.):
    """
    Estimates the number of quasiparticles that will be created given the total Quasiparticle energy.
    Assumes that the QPs follow a BE distribution

    Parameters
    ----------
        E_total : float
            Sum of all of the QP's energies in eV
        T : float
            Effective temperature of the BE distribution. Should be O(1) K; see arXiv 2208.14474
        FIXME : this needs some sort of Fano factor to conserve energy

    Returns
    -------
        avg_num : float
            Average number of quasiparticles to expect
    """

    k = 8.617e-5
    p = np.linspace(.15, 4.7, 1000) 
    avg_energy = np.sum(p*p/(np.exp(QP_dispersion(p)/(k*T)) - 1) * QP_dispersion(p)) / np.sum(p*p/(np.exp(QP_dispersion(p)/(k*T)) - 1))
    avg_num = E_total/avg_energy
    return avg_num
