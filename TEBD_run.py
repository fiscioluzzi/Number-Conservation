from __future__ import division
import numpy as np
from scipy import linalg as lg
import math
import random
import TEBD_core as tebdf
from MPS import spMPS as MPS
import model
import sys

Pi=math.pi
savedir = "/cluster/scratch/evertv/conserve"

#Simple namespace to store parameters as param.parameter_name
class SimpleNamespace(object):
    """A simple container for parameters."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def build_basis(d):
    '''
    Ones the other basis works, this will not be used anymore, I guess...
    
    Build a basis for the matrices for the density matrix
    Physical basis has dimension: d
    '''
    basisdD = []
    basisdD.append(np.eye(d, dtype=np.complex128))
    
    # Compute other diagonal matrices
    for i in range(1, d):
        diag = [1 for n in range(d-i)] + [1-d+(i-1)] + [0 for n in range(i-1)]
        mat = np.zeros( (d,d), dtype = np.complex128 )
        for j in range(d):
            mat[j,j] = diag[j]
        basisdD.append( mat )
    
    # Add the other symmetric & anti-symmetric matrices
    for i in range(1, d): 
        for j in range(0, i):
            mat = np.zeros( (d,d), dtype = np.complex128 )
            mat[i,j] = 1; mat[j,i] = 1
            basisdD.append(mat)
            
            mat = np.zeros( (d,d), dtype = np.complex128 )
            mat[i,j] = 1j; mat[j,i] = -1j
            basisdD.append(mat)
    
    return basisdD

def build_simple_basis(d):
    '''
    in order to track quantum numbers, we introduce the simple basis where one
    entry is 1, all the others 0.
    Note that the diagonal elements will be at n = i*d + i
    '''
    basisdD = []
    for i in range(d):
        for j in range(d):
            mat = np.zeros((d,d), dtype=np.complex128)
            mat[i,j] = np.sqrt(d)
            basisdD.append(mat)
    
    return basisdD

# some simple functions that make life easier 
def inner_product(A,B):
    return np.trace(np.dot(np.transpose(np.conjugate(A)), B))/A.shape[0]

def mult(A,B):
    return 0.5 * (np.dot(A,B) + np.dot(B,A))
    
def commutator(A,B):
    return np.dot(A,B) - np.dot(B,A)

def full_lindblad(A,B):
    '''
    the jump operators act as: A B A^dag - 1/2 {A^dag A, B}
    '''
    return np.dot(A, np.dot(B,np.conjugate(A.T))) - mult(np.dot(np.conjugate(A.T), A), B)

def jump_lindblad(A,B):
    '''
    the jump operators act as: A B A^dag
    '''
    return np.dot(A, np.dot(B,np.conjugate(A.T)))

def build_superop(A, p, func=mult):
    '''
    given an operator A and how it acts on a matrix (func), build the superoperator 
    '''
    basis = p.basis
    d = p.d
    
    opsize = int(np.log(A.shape[0])/np.log(d))
    
    if(opsize == 1):
         return np.array([[inner_product(b, func(A, c)) for b in basis] for c in basis])
 
    if(opsize == 2):
        return np.array([[[[inner_product(np.kron(k,l), func(A, np.kron(m,n))) for n in basis] for m in basis] for l in basis] for k in basis])  
    
    else:
        print 'there seems to be a problem'
        return 0

def build_lindbladian(p):
    '''
    Function to construct Lindbladian jump operators for site and bond
    operators. Note that site updates are only necessary for fermionic jump
    operators that come with a 'string'.
    Return:
    Dictionary of Jump operators for every site and every bond 
    '''
    BondLin={}
    SiteLin={}
    for k in p.Lindblad.keys():
        supLB = np.zeros((p.d**4, p.d**4), np.complex128)
        supLSp = np.zeros((p.d**2, p.d**2), np.complex128)
        supLSm = np.zeros((p.d**2, p.d**2), np.complex128)
        for op in p.Lindblad[k]:
            if np.shape(op)[0] == p.d:
                supLSp += np.reshape(build_superop(op, p, jump_lindblad) -
                build_superop(np.dot(np.conjugate(op.T), op), p, mult), (p.d**2, p.d**2)) 
                supLSm += np.reshape(- build_superop(op, p, jump_lindblad) -
                build_superop(np.dot(np.conjugate(op.T), op), p, mult), (p.d**2, p.d**2))
            if np.shape(op)[0] == p.d**2:
                supLB += np.reshape(build_superop(op, p, full_lindblad),
                    (p.d**4, p.d**4)) 
        BondLin[k] = supLB # the bond lindbladian should always be set.
        if not np.sum(supLSp)==0:
            SiteLin[k] = [supLSp, supLSm]
        
    return SiteLin, BondLin
   
def build_lndbld_evolution_op(p, delta):
    '''
    Function to construct factorized evolution opearator gate
    Returns:
    List of evolution operator for pairs of sites
    '''
    Uth=[]
    for k in p.Lsite.keys():
        singlesiteL = p.Lsite[k]
        Ubp = lg.expm(delta/2.*singlesiteL[0])
        Ubm = lg.expm(delta/2.*singlesiteL[1])
        Uth.append( [1, Ubp, Ubm, k] )

    for i in range(int(np.shape(p.H)[0]/2)+1):
        supH = -1j * np.reshape(build_superop(p.H[2*i], p, commutator), (p.d**4, p.d**4))
        supH += p.Lbond[2*i+1]
        Ub = np.reshape(lg.expm(delta/2.*supH), (p.d**2, p.d**2, p.d**2, p.d**2))
        Uth.append([2, Ub, 2*i])
    for i in range(int(np.shape(p.H)[0]/2)):
        supH = -1j * np.reshape(build_superop(p.H[2*i+1], p, commutator), (p.d**4, p.d**4))
        supH += p.Lbond[2*i+2]
        Ub = np.reshape(lg.expm(delta*supH), (p.d**2, p.d**2, p.d**2, p.d**2))
        Uth.append([2, Ub, 2*i+1])
    for i in range(int(np.shape(p.H)[0]/2)+1):
        supH = -1j * np.reshape(build_superop(p.H[2*i], p, commutator), (p.d**4, p.d**4))
        supH += p.Lbond[2*i+1]
        Ub = np.reshape(lg.expm(delta/2.*supH), (p.d**2, p.d**2, p.d**2, p.d**2))
        Uth.append([2, Ub, 2*i])

    for k in p.Lsite.keys():
        singlesiteL = p.Lsite[k]
        Ubp = lg.expm(delta/2.*singlesiteL[0])
        Ubm = lg.expm(delta/2.*singlesiteL[1])
        Uth.append( [1, Ubp, Ubm, k] )

    return Uth

#This implementation is only for NO PERIODIC BOUNDARY CONDITIONS!
def build_sys(nsteps=50,chi_max=4,deltalist=[0.01],Nsites=20, D=8, V=0,
        gammad=0, gammal=0, seed=3):
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
    parm.seed = seed

    #set hamiltonian parameters
    parm.t = 1.
    parm.V = V
    parm.Delta=D
    
    #number of sites and bonds for PBC and OBC on a chain
    parm.Nsites=Nsites
    # I add 'dummy' Lambdas at the begining and the end of the chain.
    parm.Nbonds = Nsites+1

    parm.basis = build_simple_basis(parm.d)
    #set two site Hamiltonians for every bond
    #function returns a list of hamiltonians for every bond
    parm.H=model.spinless_hopping_ham(parm)

    parm.chi_max=chi_max # maximal bond dimension
    parm.deltalist=deltalist #list of time steps
    parm.nsteps=nsteps #number of time steps

    #coupling for Lindbladian
    parm.gammal = gammal
    parm.gammad = gammad
    
    # build_loss returns a dictionary of site operators
    Lin = []
    if not parm.gammal == 0:
        Lin.append(model.build_loss(parm))

    # build_deph returns a dictionary of bond operators
    if not parm.gammad ==0:
        Lin.append(model.build_deph(parm))

    # this merges whatever dictionaries where loaded above
    parm.Lindblad = {}
    for l in Lin:
        for k in l.keys():
            parm.Lindblad[k] = parm.Lindblad.get(k, []) + l[k]

    parm.Lsite, parm.Lbond=build_lindbladian(parm)

    mps = MPS(parm.Nsites,parm.d,parm.chi_max)
    # initial state CDW
    state = np.zeros((parm.Nsites, parm.d**2))
    for i in range(parm.Nsites):
        state[i][0] = (i+1)%2
        state[i][3] = i%2
    mps.set_initial_state( state )
    
    return parm,mps


'''
MAIN functions to do stuff
'''
def energy(p,mps):
  E = 0
  for bond_nr in range(1,p.Nbonds-1):
    sH = build_superop(p.H[bond_nr-1], p)
    E+=mps.bond_expectation_value(p, sH, bond_nr)
  return E/p.Nsites

def num(p, mps):
  num = 0
  nop = build_superop(model.N, p)
  for s in range(p.Nsites):
      ns = mps.site_expectation_value(nop, s)
      #print s, tebdf.site_expectation_value(mps, p, nop, s)
      num+=ns
  #print 
  return num/p.Nsites

def CDW(p, mps):
  cdw = 0
  N = num(p, mps)*p.Nsites
  nop = build_superop(model.N, p)
  for s in range(p.Nsites):
    cdw -=(-1)**s*mps.site_expectation_value(nop, s)
  return cdw/N
 

def time_evolution(p,mps):
    #print_state(p, mps)
    #time evolve initial system
    print "initial conditions:"
    print "Tr:", mps.trace()
    print "CDW: ", CDW(p, mps)    
    print "N: ", num(p, mps)
    print "Ent:", mps.entanglement_entropy(int(p.Nsites/2))
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
        #Ut=build_evolution_op(p,delta)
        Ut=build_lndbld_evolution_op(p, delta)
        for timestep in range(p.nsteps):
            Err = 0
            for U in Ut:
                if U[0] == 2:
                    Err+=tebdf.bond_update_conserve(U[1], U[2]+1, mps, p)
                else:
                    Err+=tebdf.site_update_conserve(U[1], U[2], U[3], mps, p)
                
            times.append( (timestep+1) * delta )
            ntot.append( num(p, mps) )
            cdw.append( CDW(p, mps) )
            ent.append(mps.entanglement_entropy(int(p.Nsites/2)))
            err.append(Err)
            blocksizes.append( mps.Q[int(p.Nsites/2)] )

            #------------------------------------------------------------------
            # Checkpointing every 100 steps (including the very first)
            #------------------------------------------------------------------
            if timestep % 100 == 0:
                print "Saving timestep %d..."%timestep
                np.save("{0}".format(savedir) + "/results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-seed-{5}.npy".format(N,D,V,gammad,gammal,seed), np.array([np.real(times),np.real(ntot),np.real(cdw),np.real(ent),np.real(err),np.real(blocksizes)], dtype=object))
                mps.saveToFile("{0}".format(savedir) + "/results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-seed-{5}.mps".format(N,D,V,gammad,gammal,seed))

            if p.verb:
                print "Error: ", Err
                print p.chi
            
        print mps.Q

    #print_state(p, mps)
    return times, ntot, cdw, ent, err, blocksizes

def main(Tsteps=3, chi_max=100, deltalist=[0.1], D=10., V=0.7, N=4, gammad=0.2,
        gammal=0.0, seed=3, run=0):
    p,mps=build_sys(Tsteps, chi_max,deltalist, N, D,V, gammad, gammal,seed)
    t, ntot,cdw,ent,err, blocks = time_evolution(p,mps)
    return np.real(t), np.real(ntot), np.real(cdw), np.real(ent), np.real(err), np.real(blocks), mps

Tsteps      = 2000
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
np.save("{0}".format(savedir) + "/results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-seed-{5}.npy".format(N,D,V,gammad,gammal,seed), np.array([np.real(times),np.real(ntot),np.real(cdw),np.real(ent),np.real(err),np.real(blocksizes)], dtype=object))
mps.saveToFile("{0}".format(savedir) + "/results-N-{0}-h-{1}-V-{2}-gammas-{3}-{4}-seed-{5}.mps".format(N,D,V,gammad,gammal,seed))
#np.savetxt("res.txt",
#        np.column_stack([np.real(times),np.real(ntot),np.real(cdw),np.real(ent),np.real(err)]))
