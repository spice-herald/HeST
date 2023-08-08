import scipy.special as scispec
import numpy as np


#WIMP generator function taken from NEST, but pythonified


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
