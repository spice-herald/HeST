# HeST
Fast python-based yields simulation technique for producing LHe yields

This is an early developmental version of a yields simulation package. Further development will continue. 

## Install from pip

In the terminal, simply use: 

`pip install HeST==0.0.7`

Then in your python scripts you can use: 

`from HeST import HeST`

which will get you all of the functions in src/HeST/HeST.py

## Simple Example

```
from HeST import HeST

# set detector params
height, length, width = 5., 5., 5
nCPDs = 6
heightCPD, widthCPD = 3., 3.
baselineMean = [0.]*nCPDs
baselineWidth = [0.]*nCPDs
baselineNoise = [baselineMean, baselineWidth]

detector = HeST.VDetector( height, width, length, nCPDs, heightCPD, widthCPD, baselineNoise )

print( "Using a %.0f x %.0f x %.0f size detector" % (detector.get_width(), detector.get_length(), detector.get_height() ) )

#generate quanta

energy = 1000. # eV
type = "ER" # or "NR"
q = HeST.GetQuanta( energy, type )

print( "Yields at %.1f eV -- Singlet: %i  Triplet: %i   QP: %i " % (energy, q.SingletPhotons, q.TripletPhotons, q.Quasiparticles) )

#get the CPD response

X, Y, Z = HeST.get_random_position( detector )
signal = HeST.GetCPDSignal(detector, q.SingletPhotons, X, Y, Z )

print( "CPD pulse area = %.1f eV" % signal.area_eV )
print( "Individual pulse areas: ", signal.chArea_eV )
print( "Coincidence = %i" % signal.coincidence )

```


## Help

If you need help or have suggestions, reach out to me, Greg Rischbieter via slack or at rischbie@umich.edu

