import numpy as np 
import Geometry_creation
import Particle_creation

def GetNextsurface(particle,detector,Z_position_Surface):
    nx,ny,nz = particle.get_Direction()
    dummy =particle.get_Position()
    X=dummy[0]
    Y=dummy[1]
    Z=dummy[2]
    if  nz>0 : 
        Z1= Z_position_Surface
        X1= X + nx*(Z1-Z)/nz
        Y1= Y + ny*(Z1-Z)/nz
    else:
        Z1= 0
        X1= X + nx*(Z1-Z)/nz
        Y1= Y + ny*(Z1-Z)/nz
        
    if X1**2+Y1**2 < detector.get_radius()**2: 
        #time in seconds, x are in mm and velocity is m/s
        time= particle.get_Time() + np.sqrt( (X1-X)**2\
            + (Y1-Y)**2 + (Z1-Z)**2 )*1e-3/np.abs(particle.get_velocity()) 
        particle.set_Time(time) 
        particle.set_Position(X1,Y1,Z1)

    else:
        a= nx*nx + ny*ny
        b= 2*(nx*X + ny*Y) 
        c= X**2 + Y**2 - detector.get_radius()**2
        sol1 = ( -b + np.sqrt(b**2 -4*a*c) )/ (2*a)
        sol2 = ( -b - np.sqrt(b**2 -4*a*c) )/ (2*a)
        k = max(sol1,sol2); #because x0,y0 <R therefore products of root <0
        X1 = X + k*nx
        Y1 = Y + k*ny
        Z1 = Z + k*nz
        #time in seconds, x are in mm and velocity is m/s
        time= particle.get_Time() + np.sqrt( (X1-X)**2\
            + (Y1-Y)**2 + (Z1-Z)**2 )*1e-3/np.abs(particle.get_velocity()) 
        particle.set_Time(time) 
        particle.set_Position(X1,Y1,Z1)
    return particle   

def FromCuVesselRim_To_Sensor(particle,detector,Z_position_Surface):
    nx,ny,nz = particle.get_Direction()
    dummy =particle.get_Position()
    X=dummy[0]
    Y=dummy[1]
    Z=dummy[2]
    Z1= Z_position_Surface
    X1= X + nx*(Z1-Z)/nz
    Y1= Y + ny*(Z1-Z)/nz
    particle.set_Position(X1,Y1,Z1) 
    if detector.get_which_surface(particle.get_Position()) != "CPD":
        particle=particle.make_dead()
        return particle 
    else:
        time= particle.get_Time() + np.sqrt( (X1-X)**2\
            + (Y1-Y)**2 + (Z1-Z)**2 )*1e-3/np.abs(particle.get_velocity()) 
        particle.set_Time(time)         
        return particle



         

if __name__ == "__main__":
    '''
    one can check if these function are doing theirs job correctly or not
    '''
    singlet=Particle_creation.Singlet4He(0,0,0)
    detector = Geometry_creation.UMassDetector(27.5, 30, 24, 1, 1, 38, 6, "4He", "Silicon", "Aluminum" , "Copper") 

    singlet = FromCuVesselRim_To_Sensor(singlet,detector,10) 
    