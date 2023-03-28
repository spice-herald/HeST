import numpy as np

#create a detector class

'''
Start with a very simple detector geometry: Box detector with width*length*height being the X,Y,Z dimensions.
Origin is at the detector center. 
6 CPDs (one per side) with equal dimensions, given by heightCPD and widthCPD. 
heightCPD is the Z-dimension, except for the top/bottom sides, where it is the X-dimension.
Each CPD can have it's own baseline noise mean and width.

Optional argument: phononConversion = 0.25 -- the conversion from incident photon energy to phonon energy that produces the readout signal. 

Example Usage:

    height, width, length = 5., 5., 5.
    heightCPD, widthCPD = 5., 5., 5.
    nCPDs = 6
    baselineMean = [0.]*nCPDs
    baselineWidth = [1.]*nCPDs
    baselineNoise = [baselineMean, baselineWidth]
    phononConversion = 0.25

    detector = VDetector(height, width, length, nCPDs, heightCPD, widthCPD, baselineNoise, phononConversion)

'''

class VDetector:
    def __init__(self, height, width, length, 
                 nCPDs, heightCPD, widthCPD,
                 baselineNoise, phononConversion=0.25):
        self.height           = height
        self.width            = width
        self.length           = length
        self.nCPDs            = nCPDs
        self.heightCPD        = heightCPD
        self.widthCPD         = widthCPD
        self.baselineNoise    = baselineNoise
        self.phononConversion = phononConversion
        
    def set_height(self, p1):
        self.height = p1
    def set_width(self, p1):
        self.width = p1
    def set_length(self, p1):
        self.length = p1
    def set_nCPDs(self, p1):
        self.nCPDs = int(p1)
    def set_heightCPD(self, p1):
        self.heightCPD = p1
    def set_widthCPD(self, p1):
        self.widthCPD = p1
    def set_baselineNoise(self, p1, p2):
        if len(p1) != self.nCPDs or len(p2) != self.nCPDs:
            print("Lists of noise means and widths must equal the number of CPDs!")
        else:
            self.baselineNoise = [p1, p2]
    def set_phononConversion(self, p1):
        self.phononConversion = p1
        
        
    def get_height(self):
        return self.height
    def get_width(self):
        return self.width
    def get_length(self):
        return self.length
    def get_nCPDs(self):
        return self.nCPDs
    def get_heightCPD(self):
        return self.heightCPD
    def get_widthCPD(self):
        return self.widthCPD
    def get_baselineNoise(self):
        return self.baselineNoise
    def get_phononConversion(self):
        return self.phononConversion
    
    def get_volume(self):
        return self.height * self.width * self.length
    def get_surfaceArea(self):
        wall1 = self.width * self.length
        wall2 = self.width * self.height
        wall3 = self.length * self.height
        return 2. * ( wall1 + wall2 + wall3 )
    def get_areaCPD(self):
        return self.heightCPD * self.widthCPD
    def get_CPDcoverage(self):
        return self.nCPDs * self.get_areaCPD() / self.get_surfaceArea()


'''Add some useful functions for doing detector specific things'''


def get_distance(pos1, pos2):
    return np.sqrt( (pos1[0] - pos2[0])**2. + (pos1[1] - pos2[1])**2. + (pos1[2] - pos2[2])**2. )

def get_random_position(detector):
    h, w, l = detector.get_height(), detector.get_width(), detector.get_length()
    X = np.random.uniform( -w/2., w/2. )
    Y = np.random.uniform( -l/2., l/2. )
    Z = np.random.uniform( -h/2., h/2. )
    return X, Y, Z

def get_CPD_hits(detector, photons, X, Y, Z):
    '''
    #This is hard-coded to 6 CPDs (one per side). This needs to be overhalued eventually to allow for 
    a more sophisticated CPD handling.

    Use the detector and CPD geometry to get the light collection based on X,Y,Z position.
    This simply takes the 4pi random scattering angle and checks if the photon hits a CPD.

    We'll start by sending the photon the max distance of the geometry d^2 = h^2 + w^2 + l^2,
    and see if we've crossed any walls (possible to cross the plane of more than one wall.)

    The minimum distance to wall will be the first wall plane that's crossed, and then check
    if it crossed within the CPD geometry.
    '''

    h, w, l = detector.get_height(), detector.get_width(), detector.get_length()
    cpd_h, cpd_w = detector.get_heightCPD(), detector.get_widthCPD()

    hitsPerCPD = [0, 0, 0, 0, 0, 0]
    #X, Y, Z = np.random.uniform(-w/2., w/2.), np.random.uniform(-l/2., l/2.), np.random.uniform(-h/2., h/2)
    for n in range(photons):


        phi, arctheta = np.random.uniform(0., 2.*np.pi), np.random.uniform(-1., 1)
        theta = np.arccos(arctheta)
        distance = np.sqrt( h*h + w*w + l*l )
        X1 = X + distance*np.cos( phi ) * np.sin( theta )
        Y1 = Y + distance*np.sin( phi ) * np.sin( theta )
        Z1 = Z + distance*arctheta

        #temporary arrays to track where the photon is going
        wallDistances = [999, 999, 999, 999, 999, 999]
        temp_hits = [0, 0, 0, 0, 0, 0]
        #pos = [0, 0, 0, 0, 0, 0]

        if Z1 > h/2.: #crossed top wall
            wallDistances[0] = abs( ( h/2. - Z)/np.cos(theta) )
            X2 = X + wallDistances[0]*np.cos( phi ) * np.sin( theta )
            Y2 = Y + wallDistances[0]*np.sin( phi ) * np.sin( theta )
            if (X2 < cpd_h/2.) & (X2 > -cpd_h/2.) & (Y2 < cpd_w/2.) & (Y2 > -cpd_w/2.):
                temp_hits[0] += 1
            #pos[0] = [X2, Y2, h/2.]
        else:
            if Z1 < -h/2.: #crossed bottom wall
                wallDistances[1] = abs( ( -h/2. - Z)/np.cos(theta) )
                X2 = X + wallDistances[1]*np.cos( phi ) * np.sin( theta )
                Y2 = Y + wallDistances[1]*np.sin( phi ) * np.sin( theta )
                if (X2 < cpd_h/2.) & (X2 > -cpd_h/2.) & (Y2 < cpd_w/2.) & (Y2 > -cpd_w/2.):
                    temp_hits[1] += 1
                #pos[1] = [X2, Y2, -h/2.]

        if X1 > w/2.: #crossed top wall
            wallDistances[2] = abs( ( w/2. - X)/np.sin(theta)/np.cos(phi) )
            Y2 = Y + wallDistances[2]*np.sin( phi ) * np.sin( theta )
            Z2 = Z + wallDistances[2]*np.cos(theta)
            if (Z2 < cpd_h/2.) & (Z2 > -cpd_h/2.) & (Y2 < cpd_w/2.) & (Y2 > -cpd_w/2.):
                temp_hits[2] += 1
            #pos[2] = [w/2., Y2, Z2]
        else:
            if X1 < -w/2.: #crossed bottom wall
                wallDistances[3] = abs( ( -w/2. - X)/np.sin(theta)/np.cos(phi) )
                Y2 = Y + wallDistances[3]*np.sin( phi ) * np.sin( theta )
                Z2 = Z + wallDistances[3]*np.cos(theta)
                if (Z2 < cpd_h/2.) & (Z2 > -cpd_h/2.) & (Y2 < cpd_w/2.) & (Y2 > -cpd_w/2.):
                    temp_hits[3] += 1
                #pos[3] = [-w/2., Y2, Z2]
        if Y1 > l/2.: #crossed top wall
            wallDistances[4] = abs( ( l/2. - Y)/np.sin(theta)/np.sin(phi) )
            X2 = X + wallDistances[4]*np.cos( phi ) * np.sin( theta )
            Z2 = Z + wallDistances[4]*np.cos(theta)
            if (Z2 < cpd_h/2.) & (Z2 > -cpd_h/2.) & (X2 < cpd_w/2.) & (X2 > -cpd_w/2.):
                temp_hits[4] += 1
            #pos[4] = [X2, l/2., Z2]
        else:
            if Y1 < -l/2.: #crossed bottom wall
                wallDistances[5] = abs( ( -l/2. - Y)/np.sin(theta)/np.sin(phi) )
                X2 = X + wallDistances[5]*np.cos( phi ) * np.sin( theta )
                Z2 = Z + wallDistances[5]*np.cos(theta)
                if (Z2 < cpd_h/2.) & (Z2 > -cpd_h/2.) & (X2 < cpd_w/2.) & (X2 > -cpd_w/2.):
                    temp_hits[5] += 1
                #pos[5] = [X2, -l/2., Z2]

        index = wallDistances.index(min(wallDistances))
        distance = wallDistances[index]
        #position = pos[index]
        hitsPerCPD[index] += temp_hits[index]

    return np.array(hitsPerCPD)
