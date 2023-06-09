import numpy as np
import numericalunits as nu
import matplotlib.pyplot as plt
import Particle_creation
#polynomial fits to get the energy fractions in the Singlet, Triplet, QP, and IR channels

 

#separate functions for ERs and NRs
def ER_QP_eFraction(energy):
    if energy < 19.77: 
        return 1
    a, b, c, d = 100., 2.98839372, 13.89437102, 0.33504361
    frac =  a/((energy-c)**b) + d
    if frac > 1.0 :
        return 1.0
    return frac

def ER_triplet_eFraction(energy):
    if energy < 19.77: 
        return 0
    a, b, c, d, e, f, g = 14.69346932, 1.17858385, 17.60481951, 0.95286617, 0.23929604, 13.81618099, 4.92551215
    frac = f/(energy-a)**b - g/(energy-c)**d + e
    if frac < 0 :
        return 0.
    return frac

def ER_singlet_eFraction(energy):
    if energy < 19.77: 
        return 0
    a, b, c, d = 2.05492076e+03, 5.93097885, 3.3091563, 0.31773768
    frac = -a/(energy-b)**c + d
    if frac < 0. or energy < 19.77:
        return 0.
    return frac

def NR_QP_eFraction(energy):
    if energy < 19.77: 
        return 1.0
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
        IR      = max([0., 1.0-singlet-triplet-QP])
    else:
        if interaction == "NR":
            singlet = NR_singlet_eFraction(energy)
            triplet = NR_triplet_eFraction(energy)
            QP      = NR_QP_eFraction(energy)
            IR      = max([0., 1.0-singlet-triplet-QP])
        else:
            print("Please specify ER or NR for interaction type!")
            return 1

    return singlet, triplet, QP, IR

def GetPossianSmeared_energy_in_each_channel(Energy_deposited, singlet_frac, triplet_frac, IR_frac, QP_frac):
    '''
    should provide the Energy_deposited in eV
    '''
    singlet_photon_mean = Energy_deposited*singlet_frac/Particle_creation.Singlet4He().get_energy()
    singlet =  np.random.poisson(singlet_photon_mean)*Particle_creation.Singlet4He().get_energy()


    IR_photon_mean=Energy_deposited*IR_frac/Particle_creation.IR4He().get_energy()
    IR =  np.random.poisson(IR_photon_mean)*Particle_creation.IR4He().get_energy() 

    triplet_photon_mean=Energy_deposited*triplet_frac/Particle_creation.Triplet4He().get_energy()
    triplet = np.random.poisson(triplet_photon_mean)*Particle_creation.Triplet4He().get_energy()

    QP = Energy_deposited-triplet-singlet-IR


    return singlet, triplet, IR, QP






if __name__ == "__main__":
    
    fig, ax= plt.subplots(figsize=(12,8))
    Energy=np.linspace(0.1,1.0e5,num=1000000) 
    singlet=np.zeros(np.size(Energy))
    triplet=np.zeros(np.size(Energy))
    QP=np.zeros(np.size(Energy))
    IR=np.zeros(np.size(Energy))
    i=0
    for energy in Energy:
        singlet[i],  triplet[i], QP[i], IR[i]=GetEnergyChannelFractions(energy, "ER")
        i=i+1

    plt.plot(Energy, singlet, 'g')
    plt.plot(Energy, triplet,'r')
    plt.plot(Energy, QP,'b')
    plt.plot(Energy, IR,'k')


    #xtick=np.arange(1e1,1e7,1e4)
    #ytick=np.arange(-1,1.1,0.2)
    plt.minorticks_on()
    plt.xlabel(r'E$_{\mathregular{Recoil}}$[eV]',fontsize=30,fontweight='bold',fontname="Courier New")
    plt.ylabel('Partition',fontsize=30,fontweight='bold',fontname="Courier New")
    plt.tick_params(direction='in',which='minor',length=10,bottom=True, top =True, left=True, right=True)
    plt.tick_params(direction='in',which='major',length=20,bottom=True, top =True, left=True, right=True)
    plt.xticks(fontsize=30,fontname="Courier New")
    plt.yticks(fontsize=30,fontname="Courier New")
    plt.xscale("log")
    plt.xlim(0.1,1.0e5)
    plt.ylim(0,1)
    plt.show()
