import scipy.special as scispec
from VDetector import *


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
    def __init__(self, area_eV, chArea_eV, coincidence):
        self.area_eV = area_eV #total pulse area
        self.chArea_eV = chArea_eV #array of pulse areas per channel
        self.coincidence = coincidence #total CPDs triggered



#polynomial fits to get the energy fractions in the Singlet, Triplet, QP, and IR channels

#separate functions for ERs and NRs
def ER_QP_eFraction(energy):
    a, b, c, d = 100., 2.98839372, 13.89437102, 0.33504361
    frac =  a/((energy-c)**b) + d
    if frac > 1.0 or energy < 19.77:
        return 1.0
    return frac

def ER_triplet_eFraction(energy):
    a, b, c, d, e, f, g = 14.69346932, 1.17858385, 17.60481951, 0.95286617, 0.23929604, 13.81618099, 4.92551215
    frac = f/(energy-a)**b - g/(energy-c)**d + e
    if frac < 0. or energy < 19.77:
        return 0.
    return frac

def ER_singlet_eFraction(energy):
    a, b, c, d = 2.05492076e+03, 5.93097885, 3.3091563, 0.31773768
    frac = -a/(energy-b)**c + d
    if frac < 0. or energy < 19.77:
        return 0.
    return frac

def NR_QP_eFraction(energy):
    logE = np.log10(energy)
    a, b, c, d, e, f, g =  1.37928294, 1.62782641, -3.64361207, 2.25802903, 0.07210282, -0.31126613, -0.00495869
    frac = a/logE/logE + b/logE + c + d*logE + e*logE*logE + f*logE*logE*logE + \
           e*logE*logE*logE*logE + g*logE*logE*logE*logE*logE
    if frac > 1.0 or energy < 19.77:
        return 1.0
    return frac

def NR_triplet_eFraction(energy):
    x = np.log10(energy)
    a, b, c, d, e, f, g, h =  -7.73003451, 17.1117164, -12.0306409, 4.388470, -2.5361635, 0.661017, -8.41417e-2,  4.2301e-03
    frac =  a/x + b + c*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x + g*x*x*x*x*x*x + h*x*x*x*x*x*x*x
    if frac < 0 or energy < 19.77:
        return 0.
    return frac

def NR_singlet_eFraction(energy):
    x = np.log10(energy)
    a, b, c, d = -6.74959165e+01,  1.70997665e+02, -1.42571001e+02,  8.45017681e+01
    e, f, g, h = -6.92781245e+01,  2.87318953e+01, -7.05306162e+00,  1.03483204e+00
    i, j  = -8.39958299e-02,  2.90462675e-03
    frac = a/x + b + c*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x + g*x*x*x*x*x*x + h*x*x*x*x*x*x*x + i*x*x*x*x*x*x*x*x + j*x*x*x*x*x*x*x*x*x
    if frac < 0. or energy < 19.77:
        return 0.
    return frac


### Put all of the energy channel functions together:

def GetEnergyChannelFractions(energy, interaction):
    #returns the mean fraction of energy in
    # the singlet, triplet, QP, and IR channels
    energy = min([energy, 1.0e5])
    # energy -- recoil energy in eV
    if interaction == "ER":
        singlet = ER_singlet_eFraction(energy)
        triplet = ER_triplet_eFraction(energy)
        QP      = ER_QP_eFraction(energy)
        IR      = max([0., energy-singlet-triplet-QP])
    else:
        if interaction == "NR":
            singlet = NR_singlet_eFraction(energy)
            triplet = NR_triplet_eFraction(energy)
            QP      = NR_QP_eFraction(energy)
            IR      = max([0., energy-singlet-triplet-QP])
        else:
            print("Please specify ER or NR for interaction type!")
            return 1

    return singlet, triplet, QP, IR

def GetSingletYields(energy, interaction):
    # returns the mean prompt scintillation yield for 
    # the prompt + exponential + 1/t components
    # based on the UCB yields paper arxiv.2108.02176
    
    # energy -- recoil energy in eV
    # interaction -- "ER" or "NR"
    
    detection_gain = 0.052 #to get the LBL yields to match with the ~35% energy fraction in prompt yields    
    
    meanERyield = 1.25049e-3# +/- 0.03153 phe/eV
    meanNRyield = 0.477545e-3 # +/- 0.006395 phe/eV
    
    #assumes the "prompt" light is the 1:1 with the singlet light
    ERsingletFraction = 0.85824614
    NRsingletFraction = 0.72643866
    if energy < 19.77:
        return 0.
    if interaction == "ER":
        return meanERyield*energy/detection_gain * ERsingletFraction #convert from phe/keV to n_photons 
    else:
        if interaction == "NR":
            return meanNRyield*energy/detection_gain *NRsingletFraction #convert from phe/keV to n_photons 
        else:
            print("Please specify ER or NR for interaction type!")

def Get_Quasiparticles(qp_energy):
    #returns the mean number of quasiparticles given
    # the energy in eV in the QP channel

    #based on a linear fit after calculating the nQPs
    # using random sampling of the dispersion relation
    # given a Bose-Einstein distribution for nD/dp

    slope_t2, b_t2 = 1244.04521473, 29.10987933 #linear fit params
    return slope_t2*qp_energy + b_t2



def GetQuanta(energy, interaction, Fano=1.0):
    '''
    # energy -- recoil energy in eV
    # interaction -- "ER" or "NR"
    # Fano -- fano-like factor for statistical fluctuations on singlet and triplet photons

    Returns a quanta object storing the number of singlet and triplet photons and the quasiparticles produced.
    '''

    #get the fractions in each energy channel, and convert to number of singlet photons
    singlet_fraction, triplet_fraction, QP_fraction, IR_fraction = GetEnergyChannelFractions(energy, interaction)
    nSingletPhotons = GetSingletYields(energy, interaction)
    singlet_energy = nSingletPhotons * Singlet_PhotonEnergy

    #get the number of triplets based on the singlet photons
    # based on the polynomial fits, this can breakdown as singlet_energy --> 0
    # this happens around 20 eV

    triplet_energy = (triplet_fraction/singlet_fraction) * singlet_energy
    QP_energy = (QP_fraction/singlet_fraction) * singlet_energy
    nTripletPhotons = triplet_energy / Triplet_PhotonEnergy #(eV / (eV/ph))

    scint_energy = singlet_energy + triplet_energy #total scintillation energy
    nPhotons = nSingletPhotons + nTripletPhotons #total photons

    #apply stat fluctuations
    nSing_actual = max([0, int( np.random.normal(nSingletPhotons, np.sqrt(Fano*nSingletPhotons)) )])
    nTrip_actual = max([0, int( np.random.normal(nTripletPhotons, np.sqrt(Fano*nTripletPhotons)) )])


    scint_energy_actual = nSing_actual * Singlet_PhotonEnergy + nTrip_actual * Triplet_PhotonEnergy

    #assume the fano fluctuations are anti-correlated with the QP energy
    QP_energy += (scint_energy - scint_energy_actual)
    nQP_actual = max([0, int(Get_Quasiparticles(QP_energy))])

    return QuantaResult( nSing_actual, nTrip_actual, nQP_actual )


def GetCPDSignal(detector, photons, X, Y, Z):
    '''
    Attempt to simulate the CPD response;     not pusle shapes, just areas.
    This assumes that any photons incident on the CPDs lead contribute to the
    observed pulse area, and any photons incident on detector walls are absorbed.
    
    '''
    #photons -- number of true prompt photons produced
    #gain  -- (CPD_Area / total_surface_area)
    #X, Y, Z -- interaction coordinates (the origin 0,0,0 is at the detector center)
        
    areaPerCPD = [0.]*detector.get_nCPDs()
    coincidence = [0]*detector.get_nCPDs()
    
    noiseBaseline = detector.get_baselineNoise() 
    
    #call the get_CPD_hits function from VDetector.py
    CPDhits = get_CPD_hits(detector, photons, X, Y, Z)
    totalHits = sum(CPDhits)
    
    
    #get fractional coverage of each CPD based on interaction location:
    totalCoverage = 0.
    coverage_fraction = []
    
    
    for i in range(detector.get_nCPDs() ):
        
        if CPDhits[i] < 1:
            continue
        areaPerCPD[i]  = CPDhits[i] * Singlet_PhotonEnergy*detector.get_phononConversion() 
        
        #add in baseline noise
        noise = np.random.normal(noiseBaseline[0][i], noiseBaseline[1][i], size=CPDhits[i])
        areaPerCPD[i] += np.sum(noise)
        
        #convert back to deposited energy
        areaPerCPD[i] /= detector.get_phononConversion() 
        coincidence[i] += 1        
        
    return CPD_Signal( sum(areaPerCPD), np.array(areaPerCPD), sum(coincidence) )# area in eV, and total CPD coincidence 


##### Create some functions for simulating the WIMP recoil spectrum
# Taken from NEST's TestSpectra.cpp, switched to Helium-4 and made pythonic

#This WIMP_spectrum(...) function requires knowing the min/max energy range to sample over
# so I've made a WIMP_spectrum_prep(...) function that gets this for a given mass 

''' Usage: 
    prep = WIMP_spectrum_prep(mass_MeV, minE = 21.)
    random_energy = WIMP_spectrum(mass_MeV, *prep) #a single random energy 
'''

def WIMP_dRate(recoilEnergy_eV, mass_MeV):
    # We are going to hard code in the astrophysical halo for now.  This may be
    # something that we make an argument later, but this is good enough to start.
    # Some constants:

    ER = recoilEnergy_eV / 1000.
    mWimp = mass_MeV / 1000.
    V_SUN = 250.2     # +/- 1.4: arXiv:2105.00599, page 12 (V_EARTH 29.8km/s)
    V_WIMP = 238.     # +/- 1.5: page 12 and Table 1
    V_ESCAPE = 544.   # M.C. Smith et al. RAVE Survey
    RHO_NAUGHT = 0.3  # Local dm density in GeV/cc
    M_N = 0.9395654                  # Nucleon mass [GeV]
    N_A = 6.0221409e23              # Avogadro's number [atoms/mol]
    c = 2.99792458e10                # Speed of light [cm/s]
    GeVperAMU = 0.9315               # Conversion factor
    SecondsPerDay = 60. * 60. * 24.  # Conversion factor
    KiloGramsPerGram = 0.001         # Conversion factor
    keVperGeV = 1.0e6                 # Conversion factor
    cmPerkm = 1.0e5                   # Conversion factor
    SqrtPi = pow(np.pi, 0.5)
    root2 = np.sqrt(2.);
    #// Convert all velocities from km/s into cm/s
    v_0 = V_WIMP * cmPerkm;      # peak WIMP velocity
    v_esc = V_ESCAPE * cmPerkm;  # escape velocity
    v_e = (V_SUN + (0.49 * 29.8 * np.cos(-0.415 * 2. * np.pi))) * cmPerkm # the Earth's velocity

    #// Define the detector Z and A and the mass of the target nucleus
    Z = 2. #ATOM_NUM;
    A = 4. #Atom mass
    M_T = A * GeVperAMU;

    # Calculate the number of target nuclei per kg
    N_T = N_A / (A * KiloGramsPerGram);

    # Rescale the recoil energy and the inelastic scattering parameter into GeV
    ER /= keVperGeV;
    delta = 0. / keVperGeV #Setting this to a nonzero value will allow
    # for inelastic dark matter...
    # Set up your dummy WIMP model (this is just to make sure that the numbers
    # came out correctly for definite values of these parameters, the overall
    # normalization of this spectrum doesn't matter since we generate a definite
    # number of events from the macro).
    m_d = mWimp       # [GeV]
    sigma_n = 1.0e-36  #[cm^2] 1 pb reference

    #// Calculate the other factors in this expression
    mu_ND = mWimp * M_N / (mWimp + M_N) #WIMP-nucleON reduced mass
    mu_TD = mWimp * M_T / (mWimp + M_T) #WIMP-nucleUS reduced mass
    fp = 1.  #// Neutron and proton coupling constants for WIMP interactions.
    fn = 1.

    # Calculate the minimum velocity required to give a WIMP with energy ER
    v_min = 0.
    if (ER != 0.):
        v_min = c * (((M_T * ER) / mu_TD) + delta) / (root2 * np.sqrt(M_T * ER))

    bet = 1.

    # Start calculating the differential rate for this energy bin, starting
    # with the velocity integral:
    x_min = v_min / v_0 # Use v_0 to rescale the other velocities
    x_e = v_e / v_0
    x_esc = v_esc / v_0

    # Calculate overall normalization to the velocity integral
    N = SqrtPi * SqrtPi * SqrtPi * v_0 * v_0 * v_0
    N *= (scispec.erf(x_esc) - (4. / SqrtPi) * np.exp(-x_esc * x_esc) * (x_esc / 2. + bet * x_esc * x_esc * x_esc / 3.))
    # Calculate the part of the velocity integral that isn't a constant
    zeta = 0.
    thisCase = -1
    if ((x_e + x_min) < x_esc):
        zeta = ((SqrtPi * SqrtPi * SqrtPi * v_0 * v_0) / (2. * N * x_e))
        zeta *= (scispec.erf(x_min + x_e) - scispec.erf(x_min - x_e) - ((4. * x_e) / SqrtPi) * np.exp(-x_esc * x_esc) * (1 + bet * (x_esc * x_esc - x_e * x_e / 3. - x_min * x_min)))

    if ((x_min > abs(x_esc - x_e)) and ((x_e + x_esc) > x_min)):
        zeta = ((SqrtPi * SqrtPi * SqrtPi * v_0 * v_0) / (2. * N * x_e))
        zeta *= (scispec.erf(x_esc) + scispec.erf(x_e - x_min) - (2. / SqrtPi) * np.exp(-x_esc * x_esc) * (x_esc + x_e - x_min - (bet / 3.) * (x_e - 2. * x_esc - x_min) * (x_esc + x_e - x_min) * (x_esc + x_e - x_min)))

    if (x_e > (x_min + x_esc)):
        zeta = 1. / (x_e * v_0)

    if ((x_e + x_esc) < x_min):
        zeta = 0.


    a = 0.52                           # in fm
    C = 1.23 * pow(A, 1. / 3.) - 0.60  # fm
    s = 0.9  # skin depth of nucleus in fm. Originally used by Karen Gibson; XENON100 1fm; 2.30 acc. to Lewin and Smith maybe?
    rn = np.sqrt(C * C + (7. / 3.) * np.pi * np.pi * a * a - 5. * s * s);
    ''' alternatives: 1.14*A^1/3 given in L&S, or
        rv=1.2*A^1/3 then rn =
        sqrt(pow(rv,2.)-5.*pow(s,2.)); used by
        XENON100 (fm)
    '''
    q = 6.92 * np.sqrt(A * ER) # in units of 1 over distance or length
    if (q * rn > 0.):
        FormFactor = 3. * np.exp(-0.5 * q * q * s * s) * (np.sin(q * rn) - q * rn * np.cos(q * rn))
        FormFactor /= (q * rn * q * rn * q * rn) # qr and qs unitless inside Bessel function, which is dimensionless too
    else:
        FormFactor = 1.;

    # Now, the differential spectrum for this bin!
    dSpec = 0.5 * (c * c) * N_T * (RHO_NAUGHT / m_d) * (M_T * sigma_n / (mu_ND * mu_ND));
    # zeta=1.069-1.4198*ER+.81058*pow(ER,2.)-.2521*pow(ER,3.)+.044466*pow(ER,4.)-0.0041148*pow(ER,5.)+0.00013957*pow(ER,6.)+2.103e-6*pow(ER,7.);
    # if ( ER > 4.36 ) squiggle = 0.; //parameterization for 7 GeV WIMP using
    # microMegas
    dSpec *= (((Z * fp) + ((A - Z) * fn)) / fn) * (((Z * fp) + ((A - Z) * fn)) / fn) * zeta * FormFactor * FormFactor * SecondsPerDay / keVperGeV;

    return dSpec;


def WIMP_spectrum_prep(mass_MeV, minE = 21):
    test_e = np.geomspace(minE, 10000, 100)
    f = []
    for i in test_e:
        f.append( WIMP_dRate(i, mass_MeV) )
    f = np.array(f)

    cut_off = (f > f[0]/1e7)
    maxE = test_e[cut_off][-1]
    return test_e[0], maxE, f[0]

def WIMP_spectrum(mass_MeV, minE_eV, maxE_eV, maxY):

    maxE = np.log10(maxE_eV * 1.2)
    minE = np.log10(minE_eV)
    #print( minE, maxE)
    while True:
        yTry = np.random.random()*maxY
        xTry = pow(10., np.random.uniform(minE, maxE) )
        if (yTry < WIMP_dRate(xTry, mass_MeV)):
            return xTry


#function for random generation of heat-based background
def random_heatNoise():
    #E is the energy in eV
    scale = 20. # eV
    offset = 10 # eV
    norm = 3000. # 1/eV/day
    maxY = np.exp( 0.5 )
    while True:
        eTry = np.random.uniform(0., 200.)
        yTry = np.random.uniform(0., maxY)
        if yTry <  np.exp(-(eTry)/scale):
            return eTry


#Additional functions for studying quasiparticle generation -- not needed for yields and signals yet

def QP_dispersion(p):
    params = [-2.26246976e-03, 9.13297690e-01, -9.43774817e-01, 2.81293178, -3.64233027, \
              2.38490219, -8.76986870e-01, 1.82682488e-01, -2.00519847e-02, 8.99140180e-04]
    e = 0.
    for i in range(len(params)):
        e += params[i]*pow(p, i)
    return e/1000. #returns energy in eV

def getMaxY_Temp(T):
    p = np.linspace(0, 5, 50)
    f = p*p/(np.exp(QP_dispersion(p)/(T/11606.)) -1.)
    return  max(f)*1.05


def Random_QPmomentum(T=2., maxY=0.2):
    while True:
        pTry = np.random.uniform(0., 4.7)
        yTry = np.random.uniform(0., maxY)
        if yTry <  pTry*pTry/(np.exp(QP_dispersion(pTry)/(T/11606.)) -1.):
            return pTry

def GetQuasiParticles(energy, T=2.):    
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

