import numpy as np
import Geometry_creation
import Particle_creation 
import Energy_partioning 
import Propagation
import Optics
import QPEvaporation
import matplotlib.pyplot as plt

def singlet_steps(detector,Esinglet,x,y,z,N):
    if N is None:
            N= Particle_creation.Singlet4He.Get_NoOfSinglet(Esinglet)
    detected_time=np.zeros(N)
    detected_energy=np.zeros(N) 
    for i in range(N):
        singlet=Particle_creation.Singlet4He(x,y,z)
        u=detector.get_random_direction_in_Target(1)
        singlet.set_Direction(u[0],u[1],u[2])
        #while loop for propagation and reflection till reach the Cu_Vessel_Rim
        while detector.get_which_surface(singlet.get_Position()) != "Cu_Vessel_Rim"  and singlet.get_Alive_Status()== "Alive" : 
            
            singlet= Propagation.GetNextsurface(singlet,detector,detector.get_height())
            where_is_Particle=detector.get_which_surface(singlet.get_Position())
            #print(where_is_Particle)
            if  where_is_Particle != "Cu_Vessel_Rim":
                #print("Undergoing Reflection")
                singlet=Optics.GetReflection(singlet,detector, where_is_Particle)

        if singlet.get_Alive_Status()=="Alive":       
            #refraction some day one might include refraction 
            singlet= Propagation.FromCuVesselRim_To_Sensor(singlet,detector,detector.get_height()+detector.get_distance_between_CPD_and_Target()) 
            #print(singlet.get_Position())
    detected_time[i]=singlet.get_Time()*1e6 #sec to microseconds 
    detected_energy[i]=singlet.get_energy()
    return detected_time, detected_energy


def triplet_steps(detector,Etriplet,x,y,z):
    triplet=Particle_creation.Triplet4He()
    # involves some quenching
     
     
def IR_steps(detector,EIR,x,y,z):
    IR=Particle_creation.IR4He()
    # for loop for propagation and reflection
    # refraction 

def QP_steps(detector,Energy,Momentum,Velocity,x,y,z,N):
    detected_time=np.zeros(N)
    detected_energy=np.zeros(N)
    for i in range(N): 
        QP=Particle_creation.QuasiParticle4He(x,y,z)
        u=detector.get_random_direction_in_Target(1)
        QP.set_energy(Energy[i])
        QP.set_momentum(Momentum[i])
        QP.set_velocity(Velocity[i])
        QP.set_Direction(u[0],u[1],u[2])
        #while loop for propagation and reflection till reach the Cu_Vessel_Rim
        while detector.get_which_surface(QP.get_Position()) != "He_Vaccum_interface"  and QP.get_Alive_Status()== "Alive" :     
   
            QP= Propagation.GetNextsurface(QP,detector,detector.get_fill_height())
            where_is_Particle=detector.get_which_surface(QP.get_Position())
            #print(where_is_Particle)
            if  where_is_Particle != "He_Vaccum_interface":
                #print("Undergoing Reflection")
                QP=Optics.GetReflection(QP,detector, where_is_Particle)
     
      
        if QP.get_Alive_Status()=="Alive":
            QP=QPEvaporation.Evaporation_He_Surface(QP,detector,where_is_Particle)

            if QP.get_Alive_Status()=="Alive":
                # Cu_vessel_Rim to Sensor 
                QP= Propagation.FromCuVesselRim_To_Sensor(QP,detector,detector.get_height()+detector.get_distance_between_CPD_and_Target())
                #if QP.get_Alive_Status()=="Alive":
                #   print(detector.get_which_surface(QP.get_Position()))
        detected_time[i]=QP.get_Time()*1e6 #sec to microseconds 
        detected_energy[i]=QP.get_energy()
        #print(i,QP.get_Alive_Status())
    return detected_time, detected_energy
    
        
        
    
if __name__ == "__main__":
    
    #this all information at the end will be extracted from yaml file in the end
    No_of_Event = 1
    detector = Geometry_creation.UMassDetector(27.5, 30.0, 24.0, 1.0, 1.0, 38.0, 6.0, "Helium", "Silicon", "Aluminum" , "Copper")  
    x,y,z = detector.get_random_position_in_Target(No_of_Event)
    
    Energy=1600*np.ones(np.size(x))
    singlet_frac, triplet_frac, IR_frac, QP_frac= Energy_partioning.GetEnergyChannelFractions(Energy, "ER")
    Esinglet, Etriplet, EIR, EQP= Energy_partioning.GetPossianSmeared_energy_in_each_channel(Energy, singlet_frac, triplet_frac, IR_frac)
    #N=Particle_creation.Singlet4He(x,y,z).Get_NoOfSinglet(Esinglet)
    #for efficiency maps we can do N = 1000 
    N=1000
    t, E =singlet_steps(detector,Esinglet,x,y,z,N)
    print("Efficency of Singlet detection is {0}".format(np.size(np.sum(np.array(E) > 0, axis=0))*100/N))
    #Triplet= triplet_steps(detector,Etriplet,x,y,z)

    #IR= IR_steps(detector,EIR,x,y,z)

    

    N=100000
    Energy,Momentum,Velocity= Particle_creation.QuasiParticle4He(x,y,z).Get_Energy_Velocity_Momentum_from_recoil(EQP,N)
    #Energy,Momentum,Velocity= Particle_creation.QuasiParticle4He(x,y,z).Get_Energy_Velocity_Momentum_from_recoil(EQP)
    #plt.hist(Velocity, bins=np.arange(-300,300,10),histtype='stepfilled', alpha=0.7, color='red')
    #plt.scatter(Momentum, Velocity,color='red', s=0.02)
    #plt.show()
    print(np.size(Energy),np.size(Momentum),np.size(Velocity))
    time,Energy= QP_steps(detector,Energy,Momentum,Velocity,x,y,z,np.size(Energy))
    plt.hist(time, bins=np.arange(0,1000,10),histtype='stepfilled', alpha=0.7, color='red')
    plt.show()