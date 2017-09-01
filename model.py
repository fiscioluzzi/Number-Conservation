import numpy as np
import random

# Alias for 2x2 identity operator
Id2 = np.eye(2)

# Spin-1/2 operators
sx = 0.5*np.array([[0, 1], [1, 0]], dtype=np.complex128)
sy = 0.5*np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sz = 0.5*np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Operators for spinless Fermions
N = np.array([[0, 0], [0, 1]], dtype=np.complex128)
C = np.array([[0, 1], [0, 0]], dtype=np.complex128)
Cdag = np.array([[0, 0], [1, 0]], dtype=np.complex128)

def disorder(i, p):
  	if p.disorder == "Anderson":
		return p.Delta*(2*random.random()-1)
	if p.disorder == "AA":
	  	return p.Delta*np.cos(i*p.k + p.phi)
	return 0

def spinless_hopping_ham(p):
    '''
    Function to construct local Hamiltonian matrices for the pairs of sites of every bond 
    (in case of inequivalent sites/bonds this will be useful)
    Return:
    List of Hamiltonians for every pair of sites
    '''
    random.seed(int(p.seed))
    
    dis = np.zeros(p.Nsites)
    for i in range(p.Nsites):
        dis[i] = disorder(i, p)
   
    print dis
    Ham=[]
    H = -p.t*(np.kron(Cdag,C)+np.kron(C,Cdag))
    H += dis[0]*np.kron(N, Id2)
    H += dis[1]/2.*np.kron(Id2, N)
    H += p.V*np.kron(N, N)
    Ham.append(H)
    for i in range(1,p.Nbonds-3):
        H = -p.t*(np.kron(Cdag,C)+np.kron(C,Cdag))
        H += dis[i]/2.*np.kron(N, Id2)
        H += dis[i+1]/2.*np.kron(Id2, N)
        H += p.V*np.kron(N, N)
        Ham.append(H)
	
    if p.Nsites>2:
        H = -p.t*(np.kron(Cdag,C)+np.kron(C,Cdag))
        H += dis[p.Nsites-2]/2.*np.kron(N, Id2)
        H += dis[p.Nsites-1]*np.kron(Id2, N)
        H += p.V*np.kron(N, N)
        Ham.append(H)
	
    return Ham

def build_Ising_hxhz(p):
	'''
	Function to construct local Hamiltonian matrices for the pairs of sites of every bond 
	(in case of inequivalent sites/bonds this will be useful)
	Return:
	List of Hamiltonians for every pair of sites
	'''
	Ham=[]
        H =  p.J*np.kron(sz,sz)
        H += p.hx*np.kron(sx, Id) + p.hz*np.kron(sz, Id2)
        H += p.hx/2.*np.kron(Id, sx) + p.hz/2.*np.kron(Id2, sz)
        Ham.append(H)
	for i in range(1,p.Nbonds-3):
        	H = p.J*np.kron(sz,sz)
        	H += p.hx/2.*np.kron(sx, Id2) + p.hz/2.*np.kron(sz, Id2)
        	H += p.hx/2.*np.kron(Id2, sx) + p.hz/2.*np.kron(Id2, sz)
		Ham.append(H)
	if p.Nsites>2:
        	H = p.J*np.kron(sz,sz)
        	H += p.hx/2.*np.kron(sx, Id2) + p.hz/2.*np.kron(sz, Id2)
        	H += p.hx*np.kron(Id2, sx) + p.hz*np.kron(Id2, sz)
        	Ham.append(H)
	return Ham
    
def build_loss(p):
    LossLin = {}

    for i in range(p.Nsites):
        LossLin[i] = [np.sqrt(p.gammal)*C] 

    return LossLin

def build_deph(p):
    DephLin={}
    first_bond=1/np.sqrt(2)
    if p.Nsites==2: first_bond=1
    DephLin[1] = [np.sqrt(p.gammad)*np.kron(N, Id2), np.sqrt(p.gammad)*first_bond*np.kron(Id2,
        N)]
    for i in range(2,p.Nbonds-2):
        DephLin[i] = [np.sqrt(p.gammad)*1/np.sqrt(2) * np.kron(N, Id2),
                np.sqrt(p.gammad)*1/np.sqrt(2) *
                np.kron(Id2, N)]
    if p.Nsites>2:
        DephLin[p.Nbonds-2] = [np.sqrt(p.gammad)*1/np.sqrt(2)*np.kron(N, Id2),
                np.sqrt(p.gammad)*np.kron(Id2, N)]

    return DephLin
    


