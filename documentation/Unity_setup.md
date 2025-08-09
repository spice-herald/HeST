Hello world

This is going to be a quick explanation of how to set the HeST simulation to run on unity, as this is not the easiest thing. 

## Step One
go to the URL unity.rc.umass.edu, and look at the getting started with unity.
This will require you to request an account which then gets PI approval. 
After this, go through the initial directions for connecting to unity as outlined there. 

These are much better directions than anything I will write, so please follow them carefully. 

## Step Two 
Go to the section on connecting to unity with vscode, and follow through with them. I suggest writing a script that contains the line 
 salloc --gpus=1 --partition-gpu-preempt -q short -t 4:00:00

and then making an executable. 
(To run an executable, you can do: _./<script.name>_)
This will allow you to quickly open an interactive session. 

**IMPORTANT**: if you are going to use vscode, you MUST open an interactive session in order to connect with it. Do not bog down the cluster by running commands directly on the unity connection. 

## Step 3
To then connect to vscode, in the interactive terminal (or powershell) session (the terminal that opens after a salloc command, or the above mentioned script), type _hostname -f_. This is the _host-name_ of the  node you are logging into. 

This will print out the hostname of the node that you will connect to.

Copy this hostname, go to the bottom left corner of your vscode window, and select the little triangle thingy. 
Click on connect to remote window, and then type 

_user_name_@_node-hostname_


## Step 4
Open the folder you care to work on, I suggest always opening up HeST. 



## Step 5
Now we need to make a conda environment. There are better instructions online again, however the important thing to get is jupyter. 
This will allow us to run the analysis notebooks inside the computing cluster, which is a very useful task. 

