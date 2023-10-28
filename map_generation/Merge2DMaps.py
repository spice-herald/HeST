import numpy as np
import os
import sys



# LCE_map_amherstExample_41x41_z2.275862068965517_noReflections.npy

try:
    file_prefix = sys.argv[1]
    xBins = int(sys.argv[2])
    yBins = int(sys.argv[3])
    nCPDs = int(sys.argv[4])
except:
    print("Error: This file takes 4 input arguments -- common map file prefix; number of xbins; number of ybins; number of CPDs")
    exit()


files = np.array([f for f in os.listdir('.') if file_prefix in f])
print("Merging %i 2D maps..." % len(files))
z = []

for f in files:
    string = f.split('z')[1]
    zSlice = string.split('_')[0]
    z.append( float(zSlice) )

z = np.array(z)

sorted_indices = np.argsort(z)

z_sorted = z[sorted_indices]
f_sorted = files[sorted_indices]

m = np.zeros((xBins, yBins, len(z), nCPDs), dtype=float)

for i in range(len(z)):
    print(z_sorted[i])
    matrix = np.load( f_sorted[i] )
    m[:,:,i] = matrix

output_name = file_prefix + "Merged.npy"

np.save( output_name, m )
print("Saved map to %s" % output_name)
