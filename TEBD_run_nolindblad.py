from __future__ import division
import numpy as np
from scipy import linalg as lg
import math
import random
import TEBD_core as tebdf
from MPS import MPS as MPS
import model
import sys

Pi=math.pi

#Simple namespace to store parameters as param.parameter_name
class SimpleNamespace(object):
    """A simple container for parameters."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def build_evolution_op(p, delta):
    '''
    Function to construct factorized evolution opearator gate
    Returns:
    List of evolution operator for pairs of sites
    '''
    Uth=[]
    for i in range(int(np.shape(p.H)[0]/2)+1):
        Ub = np.reshape(lg.expm2(-1j*delta/2.*p.H[2*i]), (p.d, p.d, p.d, p.d))
        #Ub = np.reshape(lg.expm2(-1j*delta/2.*np.eye(4)), (p.d, p.d, p.d, p.d))
        Uth.append([Ub, 2*i])
        
    # Full timestep on odd
    for i in range(int(np.shape(p.H)[0]/2)):
        Ub = np.reshape(lg.expm2(-1j*delta*p.H[2*i+1]), (p.d, p.d, p.d, p.d))
        #Ub = np.reshape(lg.expm2(-1j*delta*np.eye(4)), (p.d, p.d, p.d, p.d))
        Uth.append([Ub, 2*i+1])
        
    # Another half on even
    for i in range(int(np.shape(p.H)[0]/2)+1):
        Ub = np.reshape(lg.expm2(-1j*delta/2.*p.H[2*i]), (p.d, p.d, p.d, p.d))
        #Ub = np.reshape(lg.expm2(-1j*delta/2.*np.eye(4)), (p.d, p.d, p.d, p.d))
        Uth.append([Ub, 2*i])
    return Uth

#This implementation is only for NO PERIODIC BOUNDARY CONDITIONS!
def build_sys(nsteps=50,chi_max=4,deltalist=[0.01],Nsites=20, D=8, V=0, gammad=0,gammal=0,seed=0):
    '''
    parm - is a container (Namespace) for the system information (parameters, hamiltonian)
    mps - is a namespace for G and L matrices,
    evolution oparators - U[:]
    Returns:
        parm, mps
    '''
    parm=SimpleNamespace()
    #print all kinds of stuff
    parm.verb = False
    parm.max_error = 1e-8
    #local Hilbert space dim
    parm.d=2
    parm.disorder = "Anderson"
    parm.seed = seed #In order to compare better, this allows to seed the random
                  #number generator
    #set hamiltonian parameters
    parm.t = 1.
    parm.V = V
    parm.Delta=D
    #number of sites and bonds for PBC and OBC on a chain
    parm.Nsites=Nsites
    # I add 'dummy' Lambdas at the begining and the end of the chain.
    parm.Nbonds = Nsites+1
    #set two site Hamiltonians for every bond
    #function returns a list of hamiltonians for every bond
    parm.H=model.spinless_hopping_ham(parm)
    #set TEBD parameters
    parm.chi_max=chi_max # maximal bond dimension
    parm.deltalist=deltalist #list of time steps
    parm.nsteps=nsteps #number of time steps
    mps = MPS(parm.Nsites,parm.d,parm.chi_max)
    # initial state CDW
    state = [0,1]*int(parm.Nsites/2)
    mps.set_initial_state( state )
    
    return parm,mps
'''
MAIN functions to do stuff
'''
def num(p, mps):
    return np.sum(np.array([np.real(mps.site_expectation_value(site_nr, model.N)) for site_nr in range(p.Nsites)]))

def CDW(p, mps):
    rho = 0
    ntot = np.sum( num(p, mps) )
    for site_nr in range(0, p.Nsites):
        rho -= (-1)**site_nr * mps.site_expectation_value(site_nr, model.N)
    return rho/ntot

def time_evolution(p,mps):
    #time evolve initial system
    times = []
    ntot = []
    cdw = []
    ent = []
    err = []
    blocksizes = []
    times.append( 0 )
    ntot.append(num(p, mps))
    cdw.append( CDW(p,mps) )
    ent.append(mps.entanglement_entropy(int(p.Nsites/2)))
    err.append(0)
    blocksizes.append( mps.Q[int(p.Nsites/2)] )

    for delta in p.deltalist:
        Ut=build_evolution_op(p,delta)
        for timestep in range(p.nsteps):
            Err = 0
            for U in Ut:
                Err+=tebdf.bond_update_conserve(U[0], U[1]+1, mps, p)
                
            times.append( (timestep+1) * delta )
            ntot.append( num(p, mps) )
            cdw.append( CDW(p, mps) )
            ent.append(mps.entanglement_entropy(int(p.Nsites/2)))
            blocksizes.append( mps.Q[int(p.Nsites/2)] )
            err.append(Err)

            #------------------------------------------------------------------
            # Checkpointing every 100 steps (including the very first)
            #------------------------------------------------------------------
            if timestep % 100 == 0:
                print "Saving timestep %d..."%timestep
                np.save("/cluster/scratch_xp/public/evertv/conserve/results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-seed-{5}.npy".format(N,D,V,gammad,gammal,seed), np.array([np.real(times),np.real(ntot),np.real(cdw),np.real(ent),np.real(err),np.real(blocksizes)], dtype=object))
                mps.saveToFile("/cluster/scratch_xp/public/evertv/conserve/results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-seed-{5}.mps".format(N,D,V,gammad,gammal,seed))

            if p.verb:
                print "Error: ", Err
                print p.chi

    return times, ntot, cdw, ent, err, blocksizes
 
def main(Tsteps=3, chi_max=100, deltalist=[0.1], D=10., V=0.7, N=4, gammad=0.2,
        gammal=0.0, seed=3, run=0):
    p,mps=build_sys(Tsteps, chi_max,deltalist, N, D,V, gammad, gammal,seed)
    t, ntot,cdw,ent,err, blocks = time_evolution(p,mps)
    return np.real(t), np.real(ntot), np.real(cdw), np.real(ent), np.real(err), np.real(blocks), mps

Tsteps      = 1000
chi_max     = 100
deltalist   = [0.1]
N           = int(sys.argv[1])
D           = float(sys.argv[2])
V           = float(sys.argv[3])
gammad      = float(sys.argv[4])
gammal      = float(sys.argv[5])
seed        = int(sys.argv[6])
times, ntot, cdw, ent, err, blocksizes, mps = main(Tsteps=Tsteps, chi_max=chi_max,
        deltalist=deltalist, D=D, V=V, N=N, gammad=gammad, gammal=gammal, seed=seed)
np.save("/cluster/scratch_xp/public/evertv/conserve/results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-seed-{5}.npy".format(N,D,V,gammad,gammal,seed), np.array([np.real(times),np.real(ntot),np.real(cdw),np.real(ent),np.real(err),np.real(blocksizes)], dtype=object))
mps.saveToFile("/cluster/scratch_xp/public/evertv/conserve/results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-seed-{5}.mps".format(N,D,V,gammad,gammal,seed))
#np.savetxt("res.txt",
#        np.column_stack([np.real(times),np.real(ntot),np.real(cdw),np.real(ent),np.real(err)]))
