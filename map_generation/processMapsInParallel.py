import sys
import os
import subprocess
import numpy as np

try:
    script = sys.argv[1]
except:
    print("Error! This script requires you to provide the name of the python script you want to submit with parallel jobs!")
    exit()

bottomPos = -8.407 #cm
topPos = -2.791 #cm
det_Zoffset = 3.0 #cm
Z_correction = 0 #cm
height = 2.75

nBins = 51
z_slices = np.linspace(det_Zoffset+Z_correction+bottomPos, det_Zoffset+Z_correction+bottomPos+height, nBins)


print("Submitting %i jobs!" % len(z_slices))
for zz in z_slices:
    exec_string = "sbatch submit_parallelJob.sh "+script+" "+str(zz)
    print("Submitting %s at Z=%s" % (script, str(zz) ) )
    process = subprocess.Popen(exec_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Wait for the command to finish and get the output and errors
stdout, stderr = process.communicate()

# Check the return code to see if the command was successful (0 indicates success)
return_code = process.returncode

# Print the output, errors, and return code
print("Output:")
print(stdout)
print("\nErrors:")
print(stderr)
print("\nReturn Code:", return_code)
