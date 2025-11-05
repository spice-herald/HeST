from .Detection import VSensor, VDetector
import numpy as np
from numba import njit

def HeRALD_v1(fill_height = 4.8):
    sqcm_thickness = .1 #cm
    sqcm_width = 1 #cm
    sqcm_pitch = 1.1 #cm
    sqcm_height = 5.2 #cm; height of bottom of sensors

    cell_rad = 3.5 #cm
    cell_rad_sq = cell_rad*cell_rad


    array_map = np.array([[0,0,1,1,0,0],
                         [0,1,1,1,1,0],
                         [1,1,1,1,1,1],
                         [1,1,1,1,1,1],
                         [0,1,1,1,1,0],
                         [0,0,1,1,0,0]])


    @njit #True outside the detector; false inside
    def prelim(x,y,z,x0,y0,z0):
        return (np.abs(x-x0) > sqcm_width/2) | (np.abs(y-y0) > sqcm_width/2) | (z < z0 - sqcm_thickness/2)

    def sensor_condition(x,y,z,x0,y0,z0, name):
        return prelim(x,y,z,x0,y0,z0), name

    def create_sensors_condition(x0,y0,z0, name):
        return lambda x,y,z: sensor_condition(x,y,z, x0, y0, z0, name)

    sensor_conditions = []
    sensor_locations = {}
    sensor_id = 0
    for i in range(array_map.shape[0]):
        for j in range(array_map.shape[1]):
            if array_map[i,j] > .5:
                x0, y0 = (i - 2.5)*sqcm_pitch, (j - 2.5)*sqcm_pitch, 
                # print(x0,y0)
                sensor_conditions.append(create_sensors_condition(x0,y0, sqcm_height+ .5*sqcm_thickness, 'sensor_'+str(sensor_id)))
                sensor_locations[str(sensor_id)] = (x0,y0)
                sensor_id += 1

                # continue

    sensors = []
    for condition in sensor_conditions:
        sensors.append(VSensor(condition))
        
    #True above the bottom; false below
    def bottom_conditions(x, y, z):
        boundary_type = "Z"
        return (z > 0), boundary_type

    #True below the top; false above
    def top_conditions(x,y,z):
        boundary_type = "Z"
        return (z < sqcm_height + .5*sqcm_thickness), boundary_type

    #True inside the walls; false outside the walls
    @njit
    def wall(x,y,z):
        return (x*x + y*y < cell_rad_sq)
    
    def wall_conditions(x, y, z):
        boundary_type = "XY"
        return wall(x,y,z), boundary_type

    #True below the liquid level; false above
    def liquid_surface(x, y, z):
        boundary_type = "Liquid"
        return (z < fill_height), boundary_type

    #True inside the liquid volume; false outside
    def liquid_conditions(x, y, z):
        return ((x*x + y*y < cell_rad_sq) & (z < fill_height) & (z > 0))

    return VDetector(top_conditions,
                       bottom_conditions,
                       wall_conditions,
                       liquid_surface,
                       liquid_conditions,
                       sensors = sensors,
                       QP_wall_reflection_prob=.3,
                       QP_wall_diffuse_prob=1.0,
                       QP_wall_Andreev_prob=0.0,
                       QP_sensor_reflection_prob = 0,
                       QP_sensor_diffuse_prob = 0,
                       QP_sensor_Andreev_prob=0.0)
                    
def HeRALD_LBNL(fill_height=3):

    cell_rad = 2.381 # cm
    cell_rad_sq = cell_rad*cell_rad
    sensor_Z_pitch = 3.73
    sensor_XY_pitch = 1.14
    sensor_thickness = .1 # cm
    sensor_width = 1
    
    @njit
    def sensor_top(x,y,z,x0,y0,z0):
        return (np.abs(x-x0) > sensor_width/2) | (np.abs(y-y0) > sensor_width/2) | (z < z0 - sensor_thickness/2)
    @njit
    def sensor_bot(x,y,z,x0,y0,z0):
        return (np.abs(x-x0) > sensor_width/2) | (np.abs(y-y0) > sensor_width/2) | (z > z0 + sensor_thickness/2)
    
    def sensor_condition_top(x,y,z,x0,y0,z0, name):
        return sensor_top(x,y,z,x0,y0,z0), name
    
    def sensor_condition_bot(x,y,z,x0,y0,z0, name):
        return sensor_bot(x,y,z,x0,y0,z0), name
    
    def create_sensors_condition_top(x0,y0,z0, name):
        return lambda x,y,z: sensor_condition_top(x,y,z, x0, y0, z0, name)

    def create_sensors_condition_bot(x0,y0,z0, name):
        return lambda x,y,z: sensor_condition_bot(x,y,z, x0, y0, z0, name)

    array_map = np.array([[1,1],
                          [1,1]])
    
    sensor_conditions = []
    sensor_locations = {}
    sensor_id = 0

    for i in range(array_map.shape[0]):
        for j in range(array_map.shape[1]):
            if array_map[i,j] > .5:
                x0, y0 = (i - .5)*sensor_XY_pitch, (j - .5)*sensor_XY_pitch, 
                # print(x0,y0)
                sensor_conditions.append(create_sensors_condition_top(x0,y0, sensor_Z_pitch, 'sensor_'+str(sensor_id)))
                sensor_locations[str(sensor_id)] = (x0,y0)
                sensor_id += 1

    for i in range(array_map.shape[0]):
        for j in range(array_map.shape[1]):
            if array_map[i,j] > .5:
                x0, y0 = (i - .5)*sensor_XY_pitch, (j - .5)*sensor_XY_pitch, 
                # print(x0,y0)
                sensor_conditions.append(create_sensors_condition_bot(x0,y0, 0, 'sensor_'+str(sensor_id)))
                sensor_locations[str(sensor_id)] = (x0,y0)
                sensor_id += 1
    locations = ['top', 'top', 'top', 'top', 'bottom', 'bottom', 'bottom', 'bottom']
    sensors = []
    i=0
    for condition in sensor_conditions:
        sensors.append(VSensor(condition, location = locations[i]))
        i+=1

    @njit
    def bottom(x,y,z):
        return (z > 0)

    def bottom_conditions(x, y, z):
        return bottom(x,y,z), "Z"
    
    @njit    
    def top(x,y,z):
        return (z < sensor_Z_pitch)

    def top_conditions(x,y,z):
        return top(x,y,z), "Z"
    
    @njit  
    def wall(x,y,z):
        return (x*x + y*y < cell_rad_sq)

    def wall_conditions(x, y, z):
        return wall(x,y,z), "XY"
    
    @njit    
    def liquid(x,y,z):
        return (z < fill_height)

    def liquid_surface(x, y, z):
        return liquid(x,y,z), "Liquid"
    
    @njit
    def liquid_conditions(x, y, z):
        return ((x*x + y*y < cell_rad_sq) & (z < fill_height) & (z > 0))
    
    return VDetector(top_conditions,
                     bottom_conditions,
                     wall_conditions,
                     liquid_surface,
                     liquid_conditions,
                     sensors = sensors,
                     QP_wall_reflection_prob=.3,
                     QP_wall_diffuse_prob=1.0,
                     QP_wall_Andreev_prob=0.0,
                     QP_sensor_reflection_prob = 0,
                     QP_sensor_diffuse_prob = 0,
                     QP_sensor_Andreev_prob=0.0,
                     UV_wall_reflection_prob = .3,
                     UV_wall_diffuse_prob = 1.0)


def HeRALD_UMass_splitCPD(fill_height = 3):
    cell_rad = 3.5 #cm
    CPD_height = 3
    CPD_rad = 3.5

    @njit
    def sensor0_conditions(x, y, z):
        boundary_type = "sensor_0"
        # return ((x*x + y*y < CPD_rad*CPD_rad) & (z < CPD_height)) | (x>=0) & (x*x + y*y < CPD_rad*CPD_rad) & (z>CPD_height) | (x*x + y*y >= CPD_rad*CPD_rad) , boundary_type
        return (z < CPD_height) | (x < .05) | (x*x+y*y > CPD_rad*CPD_rad), boundary_type
    @njit
    def sensor1_conditions(x, y, z):
        boundary_type = "sensor_1"
        # return (x*x + y*y < CPD_rad*CPD_rad) &(z < CPD_height)| (x<0) & (x*x + y*y < CPD_rad*CPD_rad) & (z>CPD_height) | (x*x + y*y >= CPD_rad*CPD_rad) , boundary_type
        return (z < CPD_height) | (x > -.05) | (x*x+y*y > CPD_rad*CPD_rad), boundary_type

    def bottom_conditions(x, y, z):
        boundary_type = "Z"
        return (z > 0), boundary_type
    
    def top_conditions(x,y,z):
        boundary_type = "Z"
        return (z < CPD_height + 1), boundary_type

    def wall_conditions(x, y, z):
        boundary_type = "XY"
        return (x*x + y*y < cell_rad*cell_rad), boundary_type
    
    def liquid_surface(x, y, z):
        boundary_type = "Liquid"
        return (z < fill_height), boundary_type

    def liquid_conditions(x, y, z):
        return ((x*x + y*y < cell_rad*cell_rad) & (z < fill_height) & (z > 0))

    sensors = []
    locations = ['top', 'top']
    sensor_conditions = [sensor0_conditions,sensor1_conditions]
    i=0
    for condition in sensor_conditions:
        sensors.append(VSensor(condition, location = locations[i]))
        i+=1


    return VDetector(top_conditions,
                     bottom_conditions,
                     wall_conditions,
                     liquid_surface,
                     liquid_conditions,
                     sensors = sensors,
                     QP_wall_reflection_prob=.3,
                     QP_wall_diffuse_prob=1.0,
                     QP_wall_Andreev_prob=0.0,
                     QP_sensor_reflection_prob = 0,
                     QP_sensor_diffuse_prob = 0,
                     QP_sensor_Andreev_prob=0.0,
                     UV_wall_reflection_prob = .3,
                     UV_wall_diffuse_prob = 1.0)