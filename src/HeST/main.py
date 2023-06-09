import numpy as np
import Geometry_creation
import Particle_creation 
import Energy_partioning 

def singlet_steps(detector,Esinglet,x,y,z):
    singlet=Particle_creation.Singlet4He()
    #while loop for propagation and reflection till reach the He surface
    # refraction  


def triplet_steps(detector,Etriplet,x,y,z):
    triplet=Particle_creation.Triplet4He()
    # involves some quenching
     
     
def IR_steps(detector,EIR,x,y,z):
    IR=Particle_creation.IR4He()
    # for loop for propagation and reflection
    # refraction 

def QP_steps(detector,EQP,x,y,z):
    QP=Particle_creation.QuasiParticle4He()
    # for loop for propagation and reflection till reach the He surface 
    # Refraction 
    
if __name__ == "__main__":
    
    #this all information at the end will be extracted from yaml file in the end
    No_of_Event = 1
    detector = Geometry_creation.UMassDetector(27.5, 30, 24, 1, 1, 38, 6, "4He", "Silicon", "Aluminum" , "Copper")  
    x,y,z = detector.get_random_position_in_Target(No_of_Event)
    
    Energy=1600*np.ones(np.size(x))
    singlet_frac, triplet_frac, IR_frac, QP_frac= Energy_partioning.GetEnergyChannelFractions(Energy, "ER")
    Esinglet, Etriplet, EIR, EQP= Energy_partioning.GetPossianSmeared_energy_in_each_channel(Energy, singlet_frac, triplet_frac, IR_frac, QP_frac)
    
    Singlet_photons= singlet_steps(detector,Esinglet,x,y,z)

    Triplet_photons= triplet_steps(detector,Etriplet,x,y,z)

    IR_photons= IR_steps(detector,EIR,x,y,z)

    No_ofQP= QP_steps(detector,EQP,x,y,z)

    #print(No_ofQP)
    #Energy,Momentum,Velocity= Particle_creation.Get_Energy_Velocity_Momentum_Position_df_from_recoil(EQP)
    #print(Energy,Momentum,Velocity)