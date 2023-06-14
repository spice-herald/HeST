import numpy as np 
import Material

'''

code that will contain on how quantum evaporation happens  

'''

#right now the code is pretty simplistic and only includes the 

def Evaporation_He_Surface(particle,detector,where_is_Particle):
    direction=particle.get_Direction()
    velocity=particle.get_velocity()
    energy=particle.get_energy()
    

    theta_I = np.deg2rad(np.arccos(direction[2]))
    sin_critical_angle = np.sqrt( 2*(energy-0.00062)/(4.002603254*1e9) )/(np.abs(velocity)/3e8)
    if sin_critical_angle >1.0:
        particle=particle.make_dead()
        return particle             
    #print(sin_critical_angle)
    critical_angle_deg=  np.rad2deg(np.arcsin(sin_critical_angle))
    if theta_I > critical_angle_deg: 
        particle=particle.make_dead()
        return particle 
    else:
        Velocity_He_atom=np.sqrt( 2*(energy-0.00062)/(4.002603254*1e9) ) *3e8
        sin_theta_R = (velocity/3e8)*np.sin(np.deg2rad(theta_I))/np.sqrt( 2*(energy-0.00062)/(4.002603254*1e9) )

        theta_R = np.arcsin(sin_theta_R)

        new_Vz=Velocity_He_atom*np.cos(theta_R)

        tan_degree=np.deg2rad( np.arctan(np.abs(direction[0]/direction[1])) )


        new_Vx= direction[0]/np.abs(direction[0])*Velocity_He_atom*np.sin(theta_R)*np.cos((tan_degree))

        new_Vy= direction[1]/np.abs(direction[1])*Velocity_He_atom*np.sin(theta_R)*np.sin((tan_degree))


        magnitude=np.sqrt(new_Vx*new_Vx+new_Vy*new_Vy+new_Vz*new_Vz)
        particle.set_Direction(new_Vx/magnitude,new_Vy/magnitude,new_Vz/magnitude)
        particle.set_velocity(Velocity_He_atom)

        Material_build = Material.For_QP[detector.get_Material_of_Surface(where_is_Particle)]
        evaporation_probability=Material_build.getEvaporation_probability()

        if np.random.choice([0,1], 1, p=[1-evaporation_probability,evaporation_probability]):
            return particle
        else:
            particle=particle.make_dead()
            return particle 
