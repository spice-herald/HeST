import numpy as np 
import Particle_creation
import Geometry_creation

class Propagator:
    def __init__(self):{}
        
    def GetNextsurface(self,particle,detector):
        nx,ny,nz = particle.get_Direction()
        X,Y,Z =particle.get_Position()
        if  nz>0 : 
            Z1= detector.get_fill_height()
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
            k = np.max(sol1,sol2); #because x0,y0 <R therefore products of root <0
            X1 = X + k*nx
            Y1 = Y + k*ny
            Z1 = Z + k*nz
            #time in seconds, x are in mm and velocity is m/s
            time= particle.get_Time() + np.sqrt( (X1-X)**2\
                + (Y1-Y)**2 + (Z1-Z)**2 )*1e-3/np.abs(particle.get_velocity()) 
            particle.set_Time(time) 
            particle.set_Position(X1,Y1,Z1)
         




if __name__ == "__main__":
    '''
    one can check if these function are doing theirs job correctly or not
    '''
    