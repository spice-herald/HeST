import numpy as np
import random
import pandas as pd 
import matplotlib.pyplot as plt


def Get_randomDirection():
    u = np.random.uniform(low=0, high=1)
    theta=2*np.pi*u
    phi= np.arccos(1-2*u)

    nx=np.sin(phi)*np.cos(theta)
    ny=np.sin(phi)*np.sin(theta)
    nz=np.cos(phi)
    return nx,ny,nz


class QuasiParticle4He:
    def __init__(self,XPosition, YPosition, ZPosition):
        self.momentum = 0.0
        self.energy   = 0.0
        self.velocity = 0.0
        self.XPosition = XPosition 
        self.YPosition = YPosition 
        self.ZPosition = ZPosition 
        self.XDirection = 0.0 
        self.YDirection = 0.0 
        self.ZDirection = 1.0 
        self.Time = 0.0
        self.IsAlive= "Alive"
        self.type="QP"
        self.momentum_cutoff = 1100
        self.energy_cutoff = 0.00062
        self.momentum_bins = 4700
        self.Temperature_in_K = 2 

    def set_momentum(self, p1):
        self.momentum = p1
    def set_energy(self, p1):
        self.energy = p1
    def set_velocity(self, p1):
        self.velocity = p1
    def set_momentum_cutoff(self, p1):
        self.momentum_cutoff = p1
    def set_energy_cutoff(self, p1):
        self.energy_cutoff = p1
    def set_momentum_bins(self, p1):
        self.momentum_bins = int(p1)
    def set_Position(self, p1, p2, p3):
        self.XPosition = p1
        self.YPosition = p2
        self.ZPosition = p3
    def set_Direction(self, p1,p2,p3):
        self.XDirection = p1
        self.YDirection = p2
        self.ZDirection = p3
    def set_Time(self, p1):
            self.Time = p1
    def set_Alive_Status(self, p1): 
            self.IsAlive = p1
    def set_Temperature_in_K(self, p1): 
            self.Temperature_in_K = p1

    def get_momentum(self):
        return self.momentum
    def get_energy(self):
        return self.energy
    def get_velocity(self):
        return self.velocity
    def get_momentum_cutoff(self):
        return self.momentum_cutoff
    def get_energy_cutoff(self):
        return self.energy_cutoff
    def get_momentum_bins(self):
        return int(self.momentum_bins)
    def get_type(self):
        return self.type
    def get_Temperature_in_K(self):
        return self.Temperature_in_K
    def get_Position(self):
        return self.XPosition,self.YPosition,self.ZPosition
    def get_Direction(self):
        return self.XDirection, self.YDirection, self.ZDirection 
    def get_Time(self):
        return self.Time 
    def get_Alive_Status(self):
        return self.IsAlive
    
    def make_dead(self):
        #print("killing particle")
        self.set_Alive_Status("Dead")
        self.set_energy(-1)
        self.set_Time(-1)
        self.set_Position(-1,-1,-1)
        return self
    
    def Is_Detected(self): 
        self.set_Alive_Status(self, "Detected")
        return self
    
    
    def Get_Velocity_m_per_s_from_Momentum_eV(self,p):
        domega_params = [2.47255840e+02, -6.41503187e-02, 2.95712310e-04,\
                                           -1.61653459e-07, -5.52938644e-10, 7.46760531e-13, -3.95194092e-16,\
                                           1.05157859e-19,-1.39334819e-23, 7.30704618e-28]
        e=np.zeros(np.size( p ))
        for i in range(len(domega_params)):
            e += domega_params[i]*pow( p , i)
        return e
    
    def Get_EnergyQP_eV_from_Momentum_eV(self,p):
        omega_params = [1.70059532e-05, 4.44224343e-07, 1.38203928e-09, -1.84747959e-12,\
                                           1.11630438e-15, -3.64966200e-19, 6.00657923e-23, -3.84221025e-27 ]
        e = np.zeros(np.size(p))
        for i in range(len(omega_params)):
            e += omega_params[i]*pow(p, i)
        return e #returns momentum in eV
    
    def Guess_from_theory(self,p,Temp_QP_in_K):
        '''
        This part is where you guess the theory 
        '''
        Boltzman_factor_without_V= 1.0/(np.exp(self.Get_EnergyQP_eV_from_Momentum_eV(p)/(Temp_QP_in_K*0.025/300))-1)
        functional_form=p**2*Boltzman_factor_without_V
        return functional_form/np.sum(functional_form)
    
    def Get_MomentumDistribution_UsingTemp_RandomSampling(self,N,Temp_QP_in_K):
        '''
         Based on the above guess construct a function that can map out the intial momentum of 
         QPs that are generated due to the recoil. But thsi makes use of random sampling from numpy. 
    
        '''
        p= np.linspace(0, 4700, num=int(self.get_momentum_bins())) 
        #print(self.get_momentum_bins())
        y=self.Guess_from_theory(p,Temp_QP_in_K)

        return np.asarray(random.choices(p, y, k=N))
    
    def Get_MomentumDistribution_UsingTemp(self,N,Temp_QP_in_K):
        '''
         Based on the above guess construct a function that can map out the intial momentum of 
         QPs that are generated due to the recoil 
    
        '''
        p= np.linspace(0, 4700, num=int(self.get_momentum_bins())) 
        #print(self.get_momentum_bins())
        y=self.Guess_from_theory(p,Temp_QP_in_K)

        x=np.int_(y*N)
        x[-1]=x[-1]+ (N-np.sum(x))
        df = pd.DataFrame(({'Number': x[1:] ,\
            'momentum': p[1:]}) )
        df = df.loc[df.index.repeat(df['Number'])]
        return np.asarray(df['momentum'])  
    
    def Get_NoOfQuasiparticles(self, qp_energy):
        '''
        returns the mean number of quasiparticles given
        the energy in eV in the QP channel
        based on a linear fit after calculating the nQPs
        using random sampling of the dispersion relation
        given a Bose-Einstein distribution for nD/dp
        '''
        slope_t2, b_t2 = 1244.04521473, 29.10987933 #linear fit params
        return int(slope_t2*qp_energy + b_t2)
    

    def Get_Energy_Velocity_Momentum_from_recoil(self, Ein_QP_channel, N=None):
        '''
        We generate the intial QP particle velocity, momentum and energy information
        '''
        if N is None:
            N= self.Get_NoOfQuasiparticles(Ein_QP_channel)
        

        #Momentum = self.Get_MomentumDistribution_UsingTemp(int(N),2.0)
        #Momentum=np.random.shuffle(Momentum)
        Momentum =self.Get_MomentumDistribution_UsingTemp_RandomSampling(int(N),self.get_Temperature_in_K())
        Energy = self.Get_EnergyQP_eV_from_Momentum_eV(Momentum)
        Velocity = self.Get_Velocity_m_per_s_from_Momentum_eV(Momentum)



        Energy=Energy[Momentum>self.get_momentum_cutoff()]  # no finite lifetime QP
        Velocity=Velocity[Momentum>self.get_momentum_cutoff()]
        Momentum=Momentum[Momentum>self.get_momentum_cutoff()]

        Momentum=Momentum[Energy>self.get_energy_cutoff()] # QE threshold
        Velocity=Velocity[Energy>self.get_energy_cutoff()]
        Energy=Energy[Energy>self.get_energy_cutoff()]

        Momentum=Momentum[abs(Velocity)>0]
        Energy=Energy[abs(Velocity)>0]
        Velocity=Velocity[abs(Velocity)>0]

        return Energy,Momentum,Velocity

    
    
class Singlet4He:
    def __init__(self,XPosition, YPosition, ZPosition):
        self.energy         = 16
        self.velocity       = 3e8
        self.XPosition = XPosition 
        self.YPosition = YPosition 
        self.ZPosition = ZPosition 
        self.XDirection = 0.0 
        self.YDirection = 0.0 
        self.ZDirection = 1.0 
        self.Time = 0.0
        self.type="Singlet"
        self.IsAlive= "Alive"
        
    def set_energy(self, p1):
        self.energy = p1
    def set_velocity(self, p1):
        self.velocity = p1
    def set_Position(self, p1, p2, p3):
        self.XPosition = p1
        self.YPosition = p2
        self.ZPosition = p3
    def set_Direction(self, p1,p2,p3):
        self.XDirection = p1
        self.YDirection = p2
        self.ZDirection = p3
    def set_Time(self, p1):
            self.Time = p1
    def set_Alive_Status(self, p1):
            self.IsAlive = p1


    def get_energy(self):
        return self.energy
    def get_velocity(self):
        return self.velocity
    def get_Alive_Status(self):
        return self.IsAlive
    def get_type(self):
        return self.type

    def Get_NoOfSinglet(self, E):
        return int(E/self.get_energy())
    def get_Position(self):
        return self.XPosition,self.YPosition,self.ZPosition
    def get_Direction(self):
        return self.XDirection, self.YDirection, self.ZDirection 
    def get_Time(self):
        return self.Time 
    
    def make_dead(self):
        #print("killing particle")
        self.set_Alive_Status("Dead")
        self.set_energy(-1)
        self.set_Time(-1)
        self.set_Position(-1,-1,-1)
        return self
    
    def Is_Detected(self): 
        self.set_Alive_Status(self, "Detected")
        return self
    




class Triplet4He:
    def __init__(self,XPosition, YPosition, ZPosition):
        self.energy         = 16
        self.velocity       = 2  
        self.decaytime      = 13 #seconds
        self.XPosition = XPosition 
        self.YPosition = YPosition 
        self.ZPosition = ZPosition 
        self.XDirection = 0.0 
        self.YDirection = 0.0 
        self.ZDirection = 1.0 
        self.type="Triplet"
        self.Time = 0.0
        self.IsAlive= "Alive"
        
    def set_energy(self, p1):
        self.energy = p1
    def set_velocity(self, p1):
        self.velocity = p1
    def set_decaytime(self, p1):
        self.decaytime = p1
    def set_Position(self, p1, p2, p3):
        self.XPosition = p1
        self.YPosition = p2
        self.ZPosition = p3
    def set_Direction(self, p1,p2,p3):
        self.XDirection = p1
        self.YDirection = p2
        self.ZDirection = p3
    def set_Time(self, p1):
            self.Time = p1
    def set_Alive_Status(self, p1):
            self.IsAlive = p1


    def get_energy(self):
        return self.energy
    def get_velocity(self):
        return self.velocity
    def get_decaytime(self):
        return self.decaytime  
    def get_Position(self):
        return self.XPosition,self.YPosition,self.ZPosition
    def get_Direction(self):
        return self.XDirection, self.YDirection, self.ZDirection 
    def get_Time(self):
        return self.Time
    def get_Alive_Status(self):
        return self.IsAlive  
    def get_type(self):
        return self.type


    def Get_NoOfTriplet(self, E):
        return int(E/self.get_energy())
    def make_dead(self):
        #print("killing particle")
        self.set_Alive_Status("Dead")
        self.set_energy(-1)
        self.set_Time(-1)
        self.set_Position(-1,-1,-1)
        return self
    
    def Is_Detected(self): 
        self.set_Alive_Status(self, "Detected")
        return self

    


class IR4He:
    def __init__(self,XPosition, YPosition, ZPosition):
        self.energy         = 0.01
        self.velocity       = 3e8
        self.XPosition = XPosition 
        self.YPosition = YPosition 
        self.ZPosition = ZPosition  
        self.XDirection = 0.0 
        self.YDirection = 0.0 
        self.ZDirection = 1.0
        self.Time = 0.0 
        self.type="IR"
        self.IsAlive= "IsAlive" 
        
    def set_energy(self, p1):
        self.enenrgy = p1
    def set_velocity(self, p1):
        self.velocity = p1
    def set_Position(self, p1, p2, p3):
        self.XPosition = p1
        self.YPosition = p2
        self.ZPosition = p3
    def set_Direction(self, p1,p2,p3):
        self.XDirection = p1
        self.YDirection = p2
        self.ZDirection = p3
    def set_Time(self, p1):
        self.Time = p1
    def set_Alive_Status(self, p1):
            self.IsAlive = p1
    def get_type(self):
        return self.type



    def get_energy(self):
        return self.energy
    def get_velocity(self):
        return self.velocity
    def get_Position(self):
        return self.XPosition,self.YPosition,self.ZPosition
    def get_Direction(self):
        return self.XDirection, self.YDirection, self.ZDirection 
    def get_Time(self):
        return self.Time 
    def get_Alive_Status(self):
        return self.IsAlive 

    def Get_NoIR(self, E):
        return int(E/self.get_energy())
    def make_dead(self):
        #print("killing particle")
        self.set_Alive_Status("Dead")
        self.set_energy(-1)
        self.set_Time(-1)
        self.set_Position(-1,-1,-1)
        return self
    def Is_Detected(self): 
        self.set_Alive_Status(self, "Detected")
        return self


    
    




if __name__ == "__main__":
    '''
    one can check if these function are doing theirs job correctly or not
    '''
    fig, ax= plt.subplots(figsize=(12,8))
    p= np.linspace(0, 4700, num=QuasiParticle4He(0,0,0).get_momentum_bins())

    Energy,Momentum,Velocity= QuasiParticle4He(0,0,0).Get_Energy_Velocity_Momentum_from_recoil(3000)

    #plt.plot(p, QuasiParticle4He(0,0,0).Get_EnergyQP_eV_from_Momentum_eV(p), 'g')
    plt.plot(p, QuasiParticle4He(0,0,0).Get_Velocity_m_per_s_from_Momentum_eV(p), 'g')
    #plt.hist(Momentum, bins=np.arange(0,4700,1),histtype='stepfilled', alpha=0.7, color='red')
    #plt.plot(p, QP.Get_EnergyQP_eV_from_Momentum_eV(p), 'r')
    plt.show()
    