class Singlet4HeReflector:
    def __init__(self):
        {

        }
    def set_energy(self, p1):
        self.energy = p1
    def set_velocity(self, p1):
        self.velocity = p1

    def get_energy(self):
        return self.energy
    def get_velocity(self):
        return self.velocity

    def Get_NoOfSinglet(self, E):
        return int(E/self.get_energy())
    

class Triplet4HeReflector:
    def __init__(self):
        {
            
        }
    def set_energy(self, p1):
        self.energy = p1
    def set_velocity(self, p1):
        self.velocity = p1

    def get_energy(self):
        return self.energy
    def get_velocity(self):
        return self.velocity

    def Get_NoOfSinglet(self, E):
        return int(E/self.get_energy())
    
class QP4HeReflector:
    def __init__(self):
        {
            
        }
    def set_energy(self, p1):
        self.energy = p1
    def set_velocity(self, p1):
        self.velocity = p1

    def get_energy(self):
        return self.energy
    def get_velocity(self):
        return self.velocity

    def Get_NoOfSinglet(self, E):
        return int(E/self.get_energy())
    

if __name__ == "__main__":
    '''
    one can check if these function are doing theirs job correctly or not
    '''
