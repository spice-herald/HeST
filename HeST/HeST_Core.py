import scipy.special as scispec
import numpy as np

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

class CPD_Signal:
    def __init__(self, area_eV, chArea_eV, coincidence, arrivalTimes_us=[[]]):
        if arrivalTimes_us == [[]]:
            arrivalTimes_us = [[]]*len(chArea_eV)
        self.area_eV = area_eV #total pulse area
        self.chArea_eV = chArea_eV # individual CPD pulse areas
        self.coincidence = coincidence #number of CPDs hit
        self.arrivalTimes_us = arrivalTimes_us #time at which the particle hits the CPD

# Polynomial functions to get the energy channel partitioning for ERs and NRs

def ER_QP_eFraction(energy):
    a, b, c, d = 100., 2.98839372, 13.89437102, 0.33504361
    frac =  a/((energy-c)**b) + d
    condition = (frac > 1.0) | (energy < 19.77)
    return np.where( condition, 1.0, frac )

def ER_triplet_eFraction(energy):
    a, b, c, d, e, f, g = 14.69346932, 1.17858385, 17.60481951, 0.95286617, 0.23929604, 13.81618099, 4.92551215
    frac = f/(energy-a)**b - g/(energy-c)**d + e
    condition = ( frac < 0. ) | (energy < 19.77 )
    return np.where( condition, 0., frac )

def ER_singlet_eFraction(energy):
    a, b, c, d = 2.05492076e+03, 5.93097885, 3.3091563, 0.31773768
    frac = -a/(energy-b)**c + d
    condition = (frac < 0.) | (energy < 19.77)
    return np.where( condition, 0., frac )

def NR_QP_eFraction(energy):
    logE = np.log10(energy)
    a, b, c, d, e, f, g =  1.37928294, 1.62782641, -3.64361207, 2.25802903, 0.07210282, -0.31126613, -0.00495869
    frac = a/logE/logE + b/logE + c + d*logE + e*logE*logE + f*logE*logE*logE + \
           e*logE*logE*logE*logE + g*logE*logE*logE*logE*logE
    condition = (frac > 1.0) | (energy < 19.77)
    return np.where( condition, 1.0, frac )

def NR_triplet_eFraction(energy):
    x = np.log10(energy)
    a, b, c, d, e, f, g, h =  -7.73003451, 17.1117164, -12.0306409, 4.388470, -2.5361635, 0.661017, -8.41417e-2,  4.2301e-03
    frac =  a/x + b + c*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x + g*x*x*x*x*x*x + h*x*x*x*x*x*x*x
    condition = (frac < 0.) | (energy < 19.77)
    return np.where( condition, 0., frac)

def NR_singlet_eFraction(energy):
    x = np.log10(energy)
    a, b, c, d = -6.74959165e+01,  1.70997665e+02, -1.42571001e+02,  8.45017681e+01
    e, f, g, h = -6.92781245e+01,  2.87318953e+01, -7.05306162e+00,  1.03483204e+00
    i, j  = -8.39958299e-02,  2.90462675e-03
    frac = a/x + b + c*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x + g*x*x*x*x*x*x + h*x*x*x*x*x*x*x + i*x*x*x*x*x*x*x*x + j*x*x*x*x*x*x*x*x*x
    condition = (frac < 0.) | (energy < 19.77)
    return np.where( condition, 0., frac)


#combine the above functions into a single function
def GetEnergyChannelFractions(energy, interaction):
    #returns the mean fraction of energy in
    # the singlet, triplet, QP, and IR channels
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

def QP_dispersion(p):
    params = [-2.26246976e-03, 9.13297690e-01, -9.43774817e-01, 2.81293178, -3.64233027, \
              2.38490219, -8.76986870e-01, 1.82682488e-01, -2.00519847e-02, 8.99140180e-04]
    e = 0.
    for i in range(len(params)):
        e += params[i]*pow(p, i)
    return e/1000. #returns energy in eV

def QP_velocity(p):
    params = [236.47194598487116, 142.7789374022186, -760.901788379717, 2197.6009596586828, \
              -3370.919165124788, 2734.173890494752, -1262.64114986808, 341.97960568115644,\
              -53.286858540286154, 4.369764313313597, -0.14355574751607653]

    v = 0.
    for i in range(len(params)):
        v += params[i]*pow(p, i)
    return v


def getMaxY_Temp(T):
    p = np.linspace(0, 5, 50)
    f = p*p/(np.exp(QP_dispersion(p)/(T/11606.)) -1.)
    return  max(f)*1.05


def Random_QPmomentum_old(T=2.):
    maxY = getMaxY_Temp(T)
    while True:
        pTry = np.random.uniform(0., 4.7)
        yTry = np.random.uniform(0., maxY)
        if yTry <  pTry*pTry/(np.exp(QP_dispersion(pTry)/(T/11606.)) -1.):
            return pTry

def Random_QPmomentum(T=2., size=1):
    p = np.linspace(0, 4.7, 1000)  # Adjust the range as needed
    probabilities = p*p/(np.exp(QP_dispersion(p)/(T/11606.)) -1.)
    cumulative_probabilities = np.cumsum(probabilities) / np.sum(probabilities)
    inverse_cdf = np.interp(np.random.rand(size), cumulative_probabilities, p)
    return inverse_cdf

def generate_quasiparticles(energy, T=2.):
    #make test draws for momentum and probability
    p = np.linspace(0, 4.7, 5000) #momentum range to consider

    f = p*p/(np.exp(QP_dispersion(p)/(T/11606.)) -1.)

    c = (f > 0.) #make sure there are no zero or negative values
    p, f = p[c], f[c]
    sumf = sum(f)
    f = f/sumf #normalize probabilities to unity

    #get the number of expected draws based on total energy
    expectedDraws = int(energy/0.0005 )
    initDraws = np.random.choice(p, p=f, size=expectedDraws)
    initEnergies = QP_dispersion(initDraws)
    QP_energies = initEnergies
    eSum = sum(initEnergies)
    nQP = expectedDraws

    if eSum > energy:
        cumEnergies = np.cumsum(QP_energies)
        truncate = ( cumEnergies < energy )
        truncEnergies = QP_energies[truncate]
        return len(truncEnergies)#, truncEnergies

    #QP_energies = list(QP_energies)
    #fill in the rest of the energies
    while eSum < energy:
        q = QP_dispersion(np.random.choice(p, p=f, size=1)[0])
        eSum += q
        nQP += 1
        #QP_energies.append( q )
    return nQP#, np.array(QP_energies)



