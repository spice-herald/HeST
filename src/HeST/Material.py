from scipy.interpolate import interp1d
import pandas as pd
class MaterialProperty:
    def __init__(self,reflection_frac, specular_frac,refractive_index,evaporation_probability):
          self.reflection_frac = reflection_frac
          self.specular_frac = specular_frac
          self.refractive_index = refractive_index
          self.evaporation_probability = evaporation_probability

    def setReflectionFraction(self,p1):
        self.reflection_frac= p1
    def setSpecularFraction(self,p1):
        self.specular_frac= p1
    def setDiffusive_frac(self,p1):
        self.diffusive_frac = p1
    def setRefractive_index(self,p1):
        self.refractive_index = p1
    def setEvaporation_probability(self,p1):
        self.evaporation_probability = p1

    def getReflectionFraction(self):
        return self.reflection_frac  
    def getAbsorptionFraction(self):
        return 1.0 - self.reflection_frac  
    def getSpecularFraction(self):
        return self.specular_frac
    def getDiffusive_frac(self):
        return 1-self.specular_frac 
    def getRefractive_index(self,):
        return self.refractive_index 
    def getEvaporation_probability(self,):
        return self.evaporation_probability 
          
    #we can do fancy stuff here later on maybe...
    #like actually import the varaiation with wavlength 
    #def SetSpectrumVsEnergy(self,file):
    #    data = pd.read_table(file, sep=",", usecols=['Energy', 'Reflection', 'Absorption'])
    #    reflection = interp1d(data['Energy'],data['Reflection'],fill_value=(0, 0), bounds_error=False)
    #    absorption = interp1d(data['Energy'],data['Absorption'],fill_value=(0, 0), bounds_error=False)
    #    return reflection, absorption 


#reflection_frac, specular_frac,refractive_index, evaporation_probability
 
For_Singlet_Triplet= dict([
    ("Aluminum",MaterialProperty(0.9,0.5, 1.37,0)),
    ("Copper",MaterialProperty(0.9,0.5,1.3,0)),
    ("Helium",MaterialProperty(0.9,0.5,1.038,0)),
    ("Silicon",MaterialProperty(0.0,0.0,1.038,0))
])

For_IR= dict([
    ("Aluminum",MaterialProperty(0.90,0.5, 1.37,0)),
    ("Copper",MaterialProperty(0.9,0.5,1.3,0)),
    ("Helium",MaterialProperty(0.9,0.5,1.038,0)),
    ("Silicon",MaterialProperty(0.0,0.0,1.038,0))
])

For_QP= dict([
    ("Aluminum",MaterialProperty(0.30,0.5, 1.37,0)),
    ("Copper",MaterialProperty(0.20,0.5,1.3,0)),
    ("Helium",MaterialProperty(0.9,0.5,1.038,0.6)),
    ("SiliconInHelium",MaterialProperty(0.3,0.5,3.88,0)), #immmersed CPD 
    ("Silicon",MaterialProperty(0,0,3.88,0))
])


if __name__ == "__main__":
    '''
    one can check if these function are doing theirs job correctly or not
    '''
    x= For_Singlet_Triplet["Aluminum"].getReflectionFraction()
    print(x) 