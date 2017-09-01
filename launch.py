from __future__ import division
import numpy as np
import os, sys

#-------------------------------------------------------------------------------
# Set 1
#-------------------------------------------------------------------------------
#N = 20
#D = 10
#Vs = [0.1] #5, 1.0, 2.0, 5.0]
#gammal = 0.0
#gammad = 0.0
#seeds = range(100)
#
#for seed in seeds:
#    for V in Vs:
#        command = "bsub -W 168:00 -J Set1 python TEBD_run.py {0} {1} {2} {3} {4} {5}".format(N,D,V,gammad,gammal,seed)
#        os.system(command)
#
##-------------------------------------------------------------------------------
## Set 2
##-------------------------------------------------------------------------------
#N = 20
#D = 10
#Vs = [1, 2, 5, 10, 50, 100]
#gammal = 0.01
#gammad = 0.1
#seeds = range(100)
#
#for seed in seeds:
#    for V in Vs:
#        command = "bsub -W 168:00 -J Set2 python TEBD_run.py {0} {1} {2} {3} {4} {5}".format(N,D,V,gammad,gammal,seed)
#        os.system(command)

##-------------------------------------------------------------------------------
## Set 3
##-------------------------------------------------------------------------------
#N = 20
#D = 10
#V = 100
#gammals = [0.0, 0.01, 0.05]
#gammad = 0.1
#seeds = range(100)
#
#for seed in seeds:
#    for gammal in gammals:
#        command = "bsub -W 168:00 -J Set3 python TEBD_run.py {0} {1} {2} {3} {4} {5}".format(N,D,V,gammad,gammal,seed)
#        os.system(command)

#-------------------------------------------------------------------------------
# Set 4
#-------------------------------------------------------------------------------
#N = 20
#D = 10
#Vs = [100.0] #0.5, 1.0, 2.0, 5.0, 100.0]
#gammal = 0.0
#gammad = 0.0
#seeds = range(100)

#for seed in seeds:
#    for V in Vs:
#        command = "bsub -W 168:00 -J Set1 python TEBD_run_nolindblad.py {0} {1} {2} {3} {4} {5}".format(N,D,V,gammad,gammal,seed)
#        os.system(command)

###-------------------------------------------------------------------------------
### Set 5
###-------------------------------------------------------------------------------
#N = 20
#D = 10
#V = 100
#gammal  = 0.02
#gammads = [0.0, 0.1]
#seeds = range(100)
#
#for seed in seeds:
#    for gammad in gammads:
#        command = "bsub -W 168:00 -J Set5 python TEBD_run.py {0} {1} {2} {3} {4} {5}".format(N,D,V,gammad,gammal,seed)
#        os.system(command)

##-------------------------------------------------------------------------------
## Set 6
##-------------------------------------------------------------------------------
#N = 20
#D = 10
##Vs = [50.0, 200.0]
#Vs = [75.0]
#gammal = 0.01
#gammad = 0.0
#seeds = range(100)
#
#for seed in seeds:
#    for V in Vs:
#        command = "bsub -W 168:00 -J Set6 python TEBD_run.py {0} {1} {2} {3} {4} {5}".format(N,D,V,gammad,gammal,seed)
#        os.system(command)

##-------------------------------------------------------------------------------
## Set 7
##-------------------------------------------------------------------------------
#N = 20
#D = 10
#Vs = [50.0, 75.0, 100.0]
#gammal = 0.01
#gammad = 0.0
#seeds = range(100)
#
#for seed in seeds:
#    for V in Vs:
#        command = "bsub -W 168:00 -J Set6 python TEBD_run.py {0} {1} {2} {3} {4} {5}".format(N,D,V,gammad,gammal,seed)
#        os.system(command)
#

#-------------------------------------------------------------------------------
# Set 8
#-------------------------------------------------------------------------------
N = 20
D = 10
V = 100.0
gammas = [(0.1, 0.0), (0.0, 0.01), (0.0, 0.02), (0.1, 0.01), (0.1, 0.02)]
seeds = range(100)

for seed in seeds:
    for gamma in gammas:
        command = "bsub -W 168:00 -J Set8 python TEBD_run.py {0} {1} {2} {3} {4} {5}".format(N,D,V,gamma[0],gamma[1],seed)
        os.system(command)

