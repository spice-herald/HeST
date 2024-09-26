# HeST
Fast python-based yields simulation technique for producing LHe yields

This is an early developmental version of a yields simulation package. Further development will continue. 

## Install from pip

In the terminal, simply use: 

`pip install HeST==0.2.0`

Then in your python scripts you can use: 

`from HeST import HeST`

which will get you all of the functions in HeST

## Install from source (Recommended)

You'll need to clone the code from git:

`git clone git@github.com:spice-herald/HeST.git`

Then from within the newly cloned "HeST" directory, install with:

`pip install .`

This is recommended simply so that the user has access to the source files, see how things are defined, and make custom changes. 

## Simple Example

A jupyter notebook, "ExampleUsage.ipynb" has been included for an example of how to generate events.

## Map generation

Detector geometries require light collection efficiency (LCE) and QP evaporation (QPE) maps to speed up the yields generation. 
Example detector geometry (Amherst and LBNL designs) have been included. The "map_generation" directory has example scripts
for how these maps were generated. It's recommended to run these in an environment where you can submit jobs with many nodes, such as 
NERSC or Great Lakes. 

To run these scripts, verify that `LCEmap_2DmapFromZ.py` and `QPEmap_2DmapFromZ.py` are using the proper detector geometry, and the X,Y binning is 
set properly. 
Then, make sure that the script `processMapsInParallel.py` uses the right Z binning. Running 
    ``` python processMapsInParallel.py LCEmap_2DmapFromZ```
will submit one 2D map-making job for each z-bin. You can then use the script `Merge2DMaps.py` to combine the parallel outputs into a single numpy array. 

## Help

If you need help or have suggestions, reach out to me, Greg Rischbieter via slack or at rischbie@umich.edu


## Operation

So currently, I'm focusing purely on the quasiparticle modeling, as this is the most important area of development, and the most interesting physics that we can get from this sim. 

The question is how does this whole thing work: Everything is an array. Each particle exists as an index at every array. alive: determines whether or not the particle is alive (1 if it is, 0 if it is dead). 
X, Y, Z: position
X1, Y1, Z1: Position of intersection with wall. You can think of the sim as constantly calculating intersections with walls, so we always need a starting and ending location. 

dx, dy, dz: These are the direction vectors. 

Velocity: velocity, momentum: momentum, energy: energy. 

Everything is one-to-one. 

living_indices: the _indices_ of each living particle. 


### Theory of Operation

Each particle has all of it's attributes in this chain of arrays, all of which are connected via this 1-1 nature. 
We let the sim run until the sum of the alive array is 0, meaning that all quasiparticles are dead. 