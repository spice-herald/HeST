import scipy.special as scispec
import numpy as np


class UMassDetector:
    def __init__(self, height, radius, fill_height,
                 nCPDs, heightCPD, radiusCPD,
                 distance_between_CPD_and_Target, targetMaterial, cpdMaterial,BaseMaterial,CurveSurfaceMaterial):
        self.height           = height
        self.radius            = radius
        self.fill_height       = fill_height #this is only important for Helium 
        self.nCPDs            = nCPDs
        self.heightCPD        = heightCPD
        self.radiusCPD         = radiusCPD
        self.distance_between_CPD_and_Target    = distance_between_CPD_and_Target
        self.targetMaterial         = targetMaterial
        self.cpdMaterial         = cpdMaterial
        self.BaseMaterial         = BaseMaterial
        self.CurveSurfaceMaterial = CurveSurfaceMaterial
        print("\n\n\nA Umass version 1 type detector has been constructed\n")
        print("targetMaterial  : {0} modeled as cyclinder with radius {1}mm and height {2}mm".format(targetMaterial,radius,height))
        print("SensorMaterial  : {0} modeled as cylinder with radius {1}mm and height {2}mm at distance of {3}mm from target top".format(cpdMaterial, radiusCPD,heightCPD,distance_between_CPD_and_Target))
        print("Configuration has {0} sensor".format(nCPDs))
        print("Helium is stored in a {0} vessel that has a {1} layer at the bottom\n".format(CurveSurfaceMaterial,BaseMaterial))

    def set_height(self, p1):
        self.height = p1
    def set_fill_height(self, p1):
        self.fill_height = p1
    def set_radius(self, p1):
        self.radius = p1
    def set_nCPDs(self, p1):
        self.nCPDs = int(p1)
    def set_heightCPD(self, p1):
        self.heightCPD = p1
    def set_radiusCPD(self, p1):
        self.radiusCPD = p1
    def set_distance_between_CPD_and_Target(self, p1):
        self.distance_between_CPD_and_Target = p1
    def set_targetMaterial(self, p1):
        self.targetMaterial = p1
    def set_cpdMaterial(self, p1):
        self.cpdMaterial = p1
    def set_BaseMaterial(self, p1):
        self.BaseMaterial = p1
    def set_CurveSurfaceMaterial(self, p1):
        self.CurveSurfaceMaterial = p1


    def get_height(self):
        return self.height
    def get_radius(self):
        return self.radius
    def get_fill_height(self):
        return self.fill_height
    def get_nCPDs(self):
        return self.nCPDs
    def get_heightCPD(self):
        return self.heightCPD
    def get_radiusCPD(self):
        return self.radiusCPD
    def get_distance_between_CPD_and_Target(self):
        return self.distance_between_CPD_and_Target
    def get_targetMaterial(self):
        return self.targetMaterial 
    def get_cpdMaterial(self):
        return self.cpdMaterial
    def get_BaseMaterial(self):
        return self.BaseMaterial
    def get_CurveSurfaceMaterial(self):
        return self.CurveSurfaceMaterial
 

    def get_which_surface(self,position):
        #print(round((position[0]**2+position[1]**2)[0]),int(self.get_radius()*self.get_radius()),(int(self.get_radius()*self.get_radius())==round((position[0]**2+position[1]**2)[0])))
        if position[2]==self.get_fill_height() and round((position[0]**2+position[1]**2)[0]) < self.get_radius() * self.get_radius():
            status="He_Vaccum_interface"
            return status 
        elif position[2]==0 and round((position[0]**2+position[1]**2)[0]) < self.get_radius() * self.get_radius():
            #include here more if statement so as to make the right geometry if the base has CPD 
            status="He_Al_interace"
            return status 
        elif position[2]==self.get_height():
            status="Cu_Vessel_Rim"
            return status 
        elif ((position[2] > 0 and position[2] < self.get_fill_height() ) and round((position[0]**2+position[1]**2)[0]) == int(self.get_radius()*self.get_radius())):
            status="He_Cu_interface"
            return status 
        elif ((position[2] > self.get_fill_height() and position[2] < self.get_height() ) and round((position[0]**2+position[1]**2)[0]) == int(self.get_radius()*self.get_radius())):
            status="Vacumm_Cu_interface"
            return status 
        elif (position[2] > 0 and position[2] < self.get_fill_height() and round((position[0]**2+position[1]**2)[0]) < int(self.get_radius()*self.get_radius())):
            status="Inside Helium"
            return status 
        elif position[2] == self.get_height() + self.get_distance_between_CPD_and_Target() and round((position[0]**2+position[1]**2)[0]) < int(self.get_radiusCPD() * self.get_radiusCPD()):
            status="CPD" 
            return status  
        elif position[2] == self.get_height() + self.get_distance_between_CPD_and_Target() and round((position[0]**2+position[1]**2)[0]) > int(self.get_radiusCPD() * self.get_radiusCPD()):
            status="outside_CPD" 
            return status   
        else:
            print(position[0]**2+position[1]**2,position[2])
            #return "Unknown"
            raise Exception("Unkown interface")
    
    def get_Material_of_Surface(self,status):
        if status=="He_Al_interace":
            return self.get_BaseMaterial()
        if status=="He_Cu_interface" or status=="Vacumm_Cu_interface":
            #include here more if statement so as to make the right geometry 
            return self.get_CurveSurfaceMaterial()
        if status=="Inside Helium" or status=="He_Vaccum_interface":
            return self.get_targetMaterial()
        if status=="CPD":
            return self.get_cpdMaterial()
        return "Unknown" 
    

    
    def get_random_position_in_Target(self, N):
        h, r = self.get_fill_height() , self.get_radius()
        Z = np.random.uniform(low=0,high=h,size=np.size(N))
        #https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
        R = r * np.sqrt(np.random.uniform(low=0,high=1,size=np.size(N)))
        theta = np.random.uniform(low=0, high=1,size=np.size(N)) * 2 * np.pi
        return R*np.cos(theta), R*np.sin(theta), Z
    
    def get_random_direction_in_Target(self,N):
        phi= 2*np.pi*np.random.uniform(low=0, high=1,size=np.size(N)) #0 to 2*pi
        theta = np.arccos(np.random.uniform(low=-1, high=1,size=np.size(N)) ) #0 to pi
        return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)

    
    
if __name__ == "__main__":
    #print("This code defines the detector Class for the experiment, for now this includes only Detector geometry similar to UMass verison 1 detector")
    
    detector = UMassDetector(27.5, 30, 24, 1, 1, 38, 6, "4He", "Silicon", "Aluminum" , "Copper") 
  