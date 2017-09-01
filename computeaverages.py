from __future__ import division
import numpy as np
import numpy.ma
import os,sys
import glob

timesteps = 20001

savedir = "/cluster/scratch/evertv/conserve"
def loaddata(savedir,N,D,V,gammad,gammal,seed):
    data = np.load("{0}/results-N-{1}-h-{2}-V-{3}-gammas-{4}-{5}-seed-{6}.npy".format(savedir,N,D,V,gammad,gammal,seed))
    times = data[0]; ntot = data[1]; cdw = data[2]; ent = data[3]; err = data[4]; Qnum = data[5]
    return times,ntot,cdw,ent,err,Qnum

def average(savedir, N, D, V, gammad, gammal, seeds):
    num_seeds = len(seeds)

    ntot = np.zeros( (num_seeds, timesteps) )
    cdw  = np.zeros( (num_seeds, timesteps) )
    ent  = np.zeros( (num_seeds, timesteps) )
    err  = np.zeros( (num_seeds, timesteps) )

    t0 = timesteps
    for s,seed in enumerate(seeds):
        try:
            print "Loading seed %d"%seed
            t,n,c,en,er,Qnum = loaddata(savedir,N,D,V,gammad,gammal,seed)
            t0 = len(t)

            ntot[s][:t0] = n
            cdw[s][:t0] = c
            ent[s][:t0] = en
            err[s][:t0] = er
        except Exception as e:
            print "Error in reading seed %d: %s"%(seed,e)
            continue

    ntot = np.ma.masked_equal(ntot, 0) 
    cdw  = np.ma.masked_equal(cdw, 0) 
    ent  = np.ma.masked_equal(ent, 0) 
    err  = np.ma.masked_equal(err, 0) 

    ntot_avg = np.mean(ntot, axis=0)[:t0]
    cdw_avg = np.mean(cdw, axis=0)[:t0]
    ent_avg = np.mean(ent, axis=0)[:t0]
    err_avg = np.mean(err, axis=0)[:t0]

    ntot_std = np.std(ntot, axis=0)[:t0]
    cdw_std = np.std(cdw, axis=0)[:t0]
    ent_std = np.std(ent, axis=0)[:t0]
    err_std = np.std(err, axis=0)[:t0]

    return t, [ntot_avg, ntot_std], [cdw_avg, cdw_std], [ent_avg, ent_std], [err_avg, err_std]

#-------------------------------------------------------------------------------
# Figure 1
#-------------------------------------------------------------------------------
N = 20
D = 10.0
Vs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 50.0]
gammal = 0.01
gammad = 0.0

for V in Vs:
    print "Loading files {0}/results-N-{1}-h-{2}-V-{3}-gammas-{4}-{5}-seed-*.npy".format(savedir,N,D,V,gammad,gammal)
    files = glob.glob("{0}/results-N-{1}-h-{2}-V-{3}-gammas-{4}-{5}-seed-*.npy".format(savedir,N,D,V,gammad,gammal))

    if files == []:
        os.system("scp evert@acheron.dhcp.phys.ethz.ch:~/groupshare/Users/evert/projects/OpenFermions/results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-seed-*.npy {5}".format(N,D,V,gammad,gammal,savedir))
        files = glob.glob("{0}/results-N-{1}-h-{2}-V-{3}-gammas-{4}-{5}-seed-*.npy".format(savedir,N,D,V,gammad,gammal))

    available_seeds = []
    for f in files:
        available_seeds.append( int(f.split("-")[-1].split(".")[0]) )
    available_seeds = np.sort(available_seeds)
    print "Available seeds: ", available_seeds

    times,ntot,cdw,ent,err = average(savedir,N,D,V,gammad,gammal,available_seeds)
    np.save("results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}.npy".format(N,D,V,gammad,gammal), np.array([times,ntot,cdw,ent,err], dtype=object))

#-------------------------------------------------------------------------------
# Figure 2
#-------------------------------------------------------------------------------
N = 20
D = 10.0
V = 2.0
gammas = [(0.0, 0.0), (0.02, 0.0), (0.0, 0.02), (0.02, 0.02)] 

for gamma in gammas:
    gammad, gammal = gamma
    files = glob.glob("{0}/results-N-{1}-h-{2}-V-{3}-gammas-{4}-{5}-seed-*.npy".format(savedir,N,D,V,gammad,gammal))

    if files == []:
        os.system("scp evert@acheron.dhcp.phys.ethz.ch:~/groupshare/Users/evert/projects/OpenFermions/results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-seed-*.npy {5}".format(N,D,V,gammad,gammal,savedir))
        files = glob.glob("{0}/results-N-{1}-h-{2}-V-{3}-gammas-{4}-{5}-seed-*.npy".format(savedir,N,D,V,gammad,gammal))

    available_seeds = []
    for f in files:
        available_seeds.append( int(f.split("-")[-1].split(".")[0]) )
    available_seeds = np.sort(available_seeds)
    print "Available seeds: ", available_seeds

    times,ntot,cdw,ent,err = average(savedir,N,D,V,gammad,gammal,available_seeds)
    np.save("results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}.npy".format(N,D,V,gammad,gammal), np.array([times,ntot,cdw,ent,err], dtype=object))

##-------------------------------------------------------------------------------
## Figure 3
##-------------------------------------------------------------------------------
#N = 20
#Ds = [8.0, 10.0, 15.0, 20.0]
#V = 2.0
#gammas = [(0.02, 0.0), (0.0, 0.02), (0.02, 0.02)] 
#
#for D in Ds:
#    for gamma in gammas:
#        gammad, gammal = gamma
#        files = glob.glob("{0}/results-N-{1}-h-{2}-V-{3}-gammas-{4}-{5}-seed-*.npy".format(savedir,N,D,V,gammad,gammal))
#
#    if files == []:
#        os.system("scp evert@acheron.dhcp.phys.ethz.ch:~/groupshare/Users/evert/projects/OpenFermions/results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-seed-*.npy {5}".format(N,D,V,gammad,gammal,savedir))
#        files = glob.glob("{0}/results-N-{1}-h-{2}-V-{3}-gammas-{4}-{5}-seed-*.npy".format(savedir,N,D,V,gammad,gammal))
#
#        available_seeds = []
#        for f in files:
#            available_seeds.append( int(f.split("-")[-1].split(".")[0]) )
#        available_seeds = np.sort(available_seeds)
#        print "Available seeds: ", available_seeds
#
#        times,ntot,cdw,ent,err = average(savedir,N,D,V,gammad,gammal,available_seeds)
#        np.save("results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}.npy".format(N,D,V,gammad,gammal), np.array([times,ntot,cdw,ent,err], dtype=object))
#
##-------------------------------------------------------------------------------
## Figure 4
##-------------------------------------------------------------------------------
#N = 20
#D = 10.0
#V = 50.0
#densities = [0.3, 0.4, 0.5]
#gammas = [(0.02, 0.0), (0.0, 0.02), (0.02, 0.02)] 
#
#for dens in densities:
#    for gamma in gammas:
#        gammad, gammal = gamma
#        files = glob.glob("{0}/results-N-{1}-h-{2}-V-{3}-gammas-{4}-{5}-den-{6}-seed-*.npy".format(savedir,N,D,V,gammad,gammal,dens))
#        available_seeds = []
#        for f in files:
#            available_seeds.append( int(f.split("-")[-1].split(".")[0]) )
#        available_seeds = np.sort(available_seeds)
#        print "Available seeds: ", available_seeds
#
#        times,ntot,cdw,ent,err = average(savedir,N,D,V,gammad,gammal,available_seeds)
#        np.save("results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-den-{5}.npy".format(N,D,V,gammad,gammal,dens), np.array([times,ntot,cdw,ent,err], dtype=object))
