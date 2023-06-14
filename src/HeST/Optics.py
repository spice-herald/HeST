import Material
import numpy as np 



def get_Cosinverse_range_0_to_360( X,  Y):
    mean_phi= np.arccos( X/np.sqrt(X*X+Y*Y) )
    if Y>=0: 
        return mean_phi
    else:
        return np.pi+ mean_phi
  
def get_unitvector(x):
    magnitude=np.sqrt(x[0]**2+x[1]**2)
    return x[0]/magnitude, x[1]/magnitude,0

def SpecularReflection_Off_PlanarSurfaceArea(particle,detector):
    X,Y,Z=particle.get_Direction()
    particle.set_Direction(X,Y,-Z)
    return particle
def SpecularReflection_Off_CurveSurfaceArea(particle,detector):
    #https://physics.stackexchange.com/questions/497605/3d-reflection-of-a-ray-on-a-cylindrical-mirror
    n=get_unitvector(particle.get_Position())
    i=particle.get_Direction()
    e0=i[0] -2*( n[0]*i[0]+n[1]*i[1]+n[2]*i[2] )*n[0]
    e1=i[1] -2*( n[0]*i[0]+n[1]*i[1]+n[2]*i[2] )*n[1]
    e2=i[2] -2*( n[0]*i[0]+n[1]*i[1]+n[2]*i[2] )*n[2]
    particle.set_Direction(e0,e1,e2)
    return particle
def DiffusiveReflection_Off_PlanarSurfaceArea(particle,detector):
    u=np.random.uniform(low=0,high=1) 
    phi= 2*np.pi*u #0 to 2*pi
    theta = np.arccos( np.random.uniform(low=0,high=1) ) #0 to pi/2
    X = np.sin(theta)*np.cos(phi)
    Y = np.sin(theta)*np.sin(phi)
    Z = np.cos(theta)
    particle.set_Direction(X,Y,Z)
    return particle
def DiffusiveReflection_Off_CurveSurfaceArea(particle,detector):
    phi= np.pi*np.random.uniform(low=-0.5,high=0.5) #-pi/2 to pi/2
    theta = np.arccos(np.random.uniform(low=-1,high=1) );#0 to pi
    x=particle.get_Position()
    i0= -x[0]/np.sqrt(x[0]**2+x[1]**2)
    i1= -x[1]/np.sqrt(x[0]**2+x[1]**2)
    phi_mean= get_Cosinverse_range_0_to_360(i0,i1)
    phi= phi + phi_mean
    e0 = np.sin(theta)*np.cos(phi)
    e1 = np.sin(theta)*np.sin(phi)
    e2 = np.cos(theta)
    particle.set_Direction(e0,e1,e2)
    return particle

def Refraction_Off_Planar_Surface(particle,detector):
    pass
    
def GetReflection(particle,detector, where_is_Particle):
    if  particle.type == 'Triplet' or particle.type == 'Singlet':
        Material_build = Material.For_Singlet_Triplet[detector.get_Material_of_Surface(where_is_Particle)] 
    elif particle.type == 'IR':  
        Material_build = Material.For_IR[detector.get_Material_of_Surface(where_is_Particle)]  
    else:
        Material_build = Material.For_QP[detector.get_Material_of_Surface(where_is_Particle)]

    reflection_coeff=Material_build.getReflectionFraction()
    specular_coeff=Material_build.getSpecularFraction()

    if np.random.choice([0,1], 1, p=[1-reflection_coeff,reflection_coeff]):
        if np.random.choice([0,1], 1,  p=[1-specular_coeff,specular_coeff]):
            if  where_is_Particle == "He_Al_interace": 
                particle = SpecularReflection_Off_PlanarSurfaceArea(particle,detector)
            else:
                particle = SpecularReflection_Off_CurveSurfaceArea(particle,detector)
        else:       
            if  where_is_Particle == "He_Al_interace": 
                particle = DiffusiveReflection_Off_PlanarSurfaceArea(particle,detector)
            else:
                particle = DiffusiveReflection_Off_CurveSurfaceArea(particle,detector)
    else: 
            particle=particle.make_dead()

    return particle
    



if __name__ == "__main__":
    '''
    one can check if these function are doing theirs job correctly or not
    '''
    Material_build = Material.For_Singlet_Triplet["Aluminum"]
    reflection_coeff=Material_build.getReflectionFraction()
    print(np.random.choice([0,1], 1, p=[1-reflection_coeff,reflection_coeff]))