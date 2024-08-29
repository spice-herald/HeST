# Documentation
I'm going to write Documentation for the HeST system, because I feel like it really should have some. When I talk to Greg, I'll see if we want to implement it into the functions themselves (we should).


## HeST_Core.py 
CPD_Signal 

This is a class, dedicated towards finding the response of a CPD to an incident signal, be it either evaporated Helium atoms or photons.

Initialization requires 3 parameters, area_eV, chArea_eV, and coincidence. 
There is an additional optional parameter: arrivalTimes_us

area_eV: the area of the total signal response, which should be the total energy absorbed by the CPD.

chArea_eV: Individual CPD pulse areas. 

arrivalTimes_us: The arrival times of the energy deposition (either photon or QP.)


**GetEnergyChannelFractions**:
Returns the mean energy in singlet, triplet, QP and IR channels.

_energy_: the parameter indicating how much energy is included. 

_interaction_: keyword, indicating which channel. Could be "ER", "NR".

returns singlet, triplet, QP, IR. 

**GetSingletYields**
I don't get this one, I'm not even sure what it is trying to do. 

**Get_Quasiparticles**: 
Returns the mean number of quasiparticles given the energy in the QP channel. 

This is based on a power law fit after calculating the number of Quasiparticles. It uses a random sampling of the dispersion relation, given a Bose-Einstein distribution for nD/dp
Once again this just has a lot of math with no justification of where the math is coming from. This is a perfect example of how phycisists write bad code. 

**GetQuanta**: 
Returns the number of events in the three signal channels: singlets, triplets, QP. 

Params: 

_energy_: the recoil energy of the event

_interaction_: the type of recoil, either electron recoil ('ER') or nuclear ('NR')


Calculates singlet energy by taking it's fraction of the total, then calculates number by dividing the amount per singlet photon energy. 

Triplet energy is created similarly, basically saying that we have some full energy, and the proportion coming from triplets is _____. It then normalizes to the singlet fraction, then multiplies by the singlet energy. 

This is also done for the quasiparticle. Why normalize by the singlet fraction ??

**generate_quasiparticles**: 
Returns the number of quasiparticles, and used to return the array of quasiparticle energies
This is done randomely, this is probably where the 'monte carlo' description comes into play.

I have a specific question about line 248 in HeST_Core

params:
_energy_: This is the total energy of the quasiparticles. Found by taking the total energy of the event, minus the amount taken by singlet and triplet. 

It then randomly samples from a distribution what energies to use, until the sum of the energy is met. 


## Detection.py 
This file works a lot different from the other ones I think. It requires some knowledge of Template, from detprocess (which I think is just a detector thing). 

### VCPD:
'virtual CPD', keeps track of individual cpds and adds them to the VDetector object. 
parameters (for init):

this one is actually documented, this is a function that keeps track of where the photon/QP paths are obstructed by the CPD surface. 


### VDetector: 
Creates the Virtual detector. 

This function is pretty well documented, so I'll just copy it in here. 

Create a Virtual Detector class.
        
Args:
surface_conditions: a list of functions that tracks where photon/QP paths are obstructed by 
                    the Detector surfaces. Each function returns (True, *SurfaceType) if the photon is unobstructed; 
                    Returns (False, *surfaceType) if the photon path hits the detector surface.
                    *SurfaceType is a way of tracking what type of boundary this is: e.g. "X", "Y", "Z", or "XY", making it 
                    easy to track how the particle may reflect off of that boundary
                    
liquid_surface:     A function that tracks where the liquid surface is, similar to the surface conditions above. This is used in QP propagation/evaporation calculations.
                    The surfaceType here must be "Liquid"
liquid_conditions:  A function returning true if the X,Y,Z position is inside the LHe volume, and False if outside the volume

CPDs:               A list of VCPD objects to keep track of

adsorption_gain:    Added energy from adsorption of evaporated QPs, per QP, in eV

self.evaporation_eff:  Flat efficiency factor on QP evaporation, between 0 and 1

LCEmap:             A 3D array tracking positions with mean photon collection probability, dimensionality (M x N x L)

LCEmap_positions:   A set of three 1-D arrays (x, y, z) to associate the LCE map entries in (M x N x L) discrete bins to individual x,y,z coordinates.
                    If loading a premade map, these must match what was used for that map's generator. 

QPEmap:             A 3D array tracking position with mean QP evaporation probability. Different from LCEmap due to liquid surface physics

QPEmap_positions:   A set of three 1-D arrays (x, y, z) to associate the QPE map entries in (M x N x L) discrete bins to individual x,y,z coordinates. 
                    If loading a premade map, these must match what was used for that map's generator. The dimensionality of the QP map doesn't need to
                    match the dimensionality of the LCE map.
                    
photon_reflection_prob: probability that a photon reflects off of detector surfaces, between 0 and 1
QP_reflection_prob: probability that a QP reflects off of detector surfaces, between 0 and 1

**create_QPEmap**:
Function dedicated towards creating the quasiparticle map, which is a 3D array with corresponding probabilities of hitProbabilities. These are a measure of any sort of obstruction in the particles path. 

you set the positions of the map by generating your x, y and z array. 
then you set the number of cpds you have. 
you have your list of conditions, which you then generate for each cpd. 
then you get your reflection probability
then loop over the entire space. 
then, for every point in the space, you create a quasiparticle there, and determine the probability that it collides with something? then you save that, and you then don't need to recalculate this ??

I'm not sure what the point of it is, honestly. I haven't found evidence yet that this needs to be used, or will be used at all. I think I will understand that a little later. 


**get_QP_hits**: Finds the number of QP hits against something, presumably. It's not clear to me what this actually does, honeslty.

**find_surface_intersection**:
Function to find the location of the intersections with walls, liquid surface, or CPD. 

params:

_start_: The starting location.

_direction_ the direction that the QP travels in. 

_conditions_: list of the functions determining surface conditions. 

How it works:
It creates a time array, then creates three additional arrays, which is the three dimensional path that the particle follows through space. 
It then individually checks each of the surface conditions. It finds the first value which is within the wall. It then finds the time that this is at. 
Then it records the points just before the interaction, and returns that as X, Y, Z, surface_type (the thing it collided with). 

**evaporation** (I'm trusting that this is mostly written well)


Weirdly, it doesn't seem like the wall reflection works correclty for the QP. I'm checking this now

**QP_propagation**:
Function which handles the propagation of Quasiparticles through the medium. 

Params:
_nQPs_: The number of quasiparticles
_start_: the starting position of the QPs
_conditions_: list of functions which set surface conditions.


Operation:
First we start at the position, then define a random direction (uniformely distributed), then choose quasiparticle momentum from a random distribution. 

Immediately, there is a cut on the type of momentum, so it doesn't go above some specified level. Weirldy enough, I don't thnik that this momentum is good. This only allows for phonon reflection, which is probably why the reflection code is written the way it is. 

additionally, create the energy from the momentum. 
the first condition is that the velocity is greater than 0. Another weird thing, because it's not clear how they're calculated.

So we set a condition where the momentum must be less than 1.1 and the velocity must be greater than 0. I really don't know why that is required. 

then we begin a loop, which lasts as long as there are still alive quasiparticles. It ends when one of them is absorbed or reflected. 

It searches for a surface intersection, then sets a condition that cuts out any particle that didn't hit a surface (which isn't really possible in my opinion). 

It tracks it through the liquid, and finds the surface intersection (I think there should always be one). 
Then it starts checking specific surfaces, meaning the top surface, and the walls. 

