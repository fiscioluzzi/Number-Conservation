#-------------------------------------------------------------------------------
# Filename: MPS.py
# Description: Contains all the classes related to Matrix Product States
# Authors: Mark Fischer & Evert van Nieuwenburg
# Copyright (c) 2016 ETH Zurich
#-------------------------------------------------------------------------------
from __future__ import division
import numpy as np
import cPickle
from scipy import linalg as lg

#-------------------------------------------------------------------------------
# A 'regular' MPS for a pure state, with quantum numbers
#-------------------------------------------------------------------------------
class MPS:
    """ Base class for an MPS representing a pure state with quantum numbers.
    """

    def __init__(self,N,d,chimax,pqn=0):
        '''
        Initialize an MPS

        Parameters
        ----------
            N : int
                The system size, or number of MPS matrices
            d : int
                Dimension of the local Hilbert space. Currently, every site is 
                restricted to the same dimension.
            chimax : int
                Maximal bond dimension of the MPS matrices. Currently, every 
                site is restricted to the same maximal bond dimension.
            pqn : array
                An array of objects (integers, tuples) describing the quantum
                numbers associated to each of the physical basis states.
        '''

        # Number of MPS Matrices
        self.N = N
        # Local Hilbert space dimension
        self.d = d
        # Maximal bond dimension
        self.chimax = chimax

        # We use the Vidal representation of the MPS, so that we have easy
        # acces to both the left- and right-canonical forms of the MPS.
        self.G   = []  # Gamma matrices
        self.L   = []  # Lambda's
        self.chi = []  # Current bond dimensions
        self.Q   = []  # Quantum numbers to the left

        # List of quantum numbers corresponding to the physical indices,
        # i.e. the first entry corresponds to the q-number for d = 0, etc.
        self.pqn = range(self.d) if pqn == 0 else pqn

    def set_initial_state(self, state):
        '''
        Set the initial state
            - state: an array of size L specifing the basis state 
                     to which the MPS is initialized.
        '''
        # Erase current MPS
        self.G   = []
        self.L   = []
        self.chi = []
        self.Q   = []

        Ls0 = np.zeros(self.chimax,complex)
        Ls0[0] = 1.
        self.L.append(Ls0)
        self.chi.append(1)

        # No quantum numbers to the left
        q_left = 0
        self.Q.append( {q_left:[0]} )
        
        for i in range(0,self.N):
            # Set G
            Gs = np.zeros( (self.d, self.chimax, self.chimax), dtype=np.complex128 )
            Gs[state[i], 0, 0] = 1
            self.G.append(Gs)
            # Set L and current bond dimension (chi)
            Ls = np.zeros(self.chimax,complex)
            Ls[0]=1.
            self.L.append(Ls)
            self.chi.append(1)
            # Set Q
            q_left = self.add_qn( q_left, self.pqn[state[i]], 1 )
            self.Q.append( { q_left : [0] } )

    def add_qn(self, q1, q2, reldir):
        '''
        Add quantum numbers q1 and q2, assuming a relative sign `reldir` = +/-1
        between the legs they are associated with.

        Parameters
        ----------
        q1,q2 : int
            Quantum numbers to be added
        reldir: int
            Equals +1 if both legs are incoming legs, -1 if either is outgoing.

        Returns
        -------
        Summed quantum numbers `q1` and `q2`
        '''
        return q1 + reldir*q2

    def entanglement_entropy(self, bond_nr):
        '''
        Compute the entanglement entropy when bipartitioned at given bond

        Parameters
        ----------
        bond_nr : int
            Bond at which partitioning happens

        Returns
        -------
        Entanglement entropy at given bond
        '''
        Ent = 0
        for i in range(int(self.chi[bond_nr])):
            Ent += - self.L[bond_nr][i]**2 *2* np.log(self.L[bond_nr][i])
        return Ent
    
    def site_expectation_value(self, site_nr, operator):
        '''
        Compute the expecation value of a single site operator, acting on site site_nr
        '''
        B=np.tensordot(np.diag(self.L[site_nr]),self.G[site_nr],axes=(1,1))
        B=np.tensordot(B,np.diag(self.L[site_nr+1]),axes=(2,0))
        C=np.tensordot(B,operator, axes=(1, 0))
        XOp = np.tensordot(C, np.conjugate(B), axes=([0,2,1], [0,1,2]))
        return XOp

    def bond_expectation_value(self, bond_nr, operator):
        '''
        Computation of bond observable
        '''
        B=np.tensordot(np.diag(self.L[bond_nr-1]),self.G[bond_nr-1],axes=(1,1))
        B=np.tensordot(B,np.diag(self.L[bond_nr]),axes=(2,0))
        B=np.tensordot(B,self.G[bond_nr],axes=(2,1))
        B=np.tensordot(B,np.diag(self.L[bond_nr+1]),axes=(3,0))
        C = np.tensordot(B,np.reshape(operator[bond_nr-1], (self.d, self.d, self.d, self.d)),axes=([1,2],[0,1]));
        XOp = np.tensordot(C, np.conj(B), axes=([0,2,3,1], [0,1,2,3]))
        return XOp

    def saveToFile(self, filename):
        """ Save MPS to a file """
        with open(filename, "wb") as f:
            cPickle.dump(self, f)
    
    @classmethod
    def loadFromFile(cls, filename):
        with open(filename, "rb") as f:
            return cPickle.load(f)


#-------------------------------------------------------------------------------
# MPS for super operators, i.e. an MPS representing a density matrix
# Also with quantum numbers, for dephasing Lindbladians
#-------------------------------------------------------------------------------
class sMPS:
    """ Class for an MPS representing a density matrix, with quantum
        numbers ready for Lindblad dephasing.    
    """

    def __init__(self,N,d,chimax,pqn=0):
        '''
        Initialize an MPS

        Parameters
        ----------
            N : int
                The system size, or number of MPS matrices
            d : int
                Dimension of the local Hilbert space. Currently, every site is 
                restricted to the same dimension.
            chimax : int
                Maximal bond dimension of the MPS matrices. Currently, every 
                site is restricted to the same maximal bond dimension.
            pqn : array
                An array of objects (integers, tuples) describing the quantum
                numbers associated to each of the physical basis states.
        '''
        # Number of MPS Matrices
        self.N = N
        # Local Hilbert space dimension
        self.d = d**2
        # Maximal bond dimension
        self.chimax = chimax

        # We use the Vidal representation of the MPS, so that we have easy
        # acces to both the left- and right-canonical forms of the MPS.
        self.G   = []  # Gamma matrices
        self.L   = []  # Lambda's
        self.chi = []  # Current bond dimensions
        self.Q   = []  # Quantum numbers to the left

        # List of quantum numbers corresponding to the physical indices,
        # i.e. the first entry corresponds to the q-number for d = 0, etc.
        self.pqn = [(0,0), (1, 0), (0,1), (1,1)]

    def set_initial_state(self, state):
        '''
        Set the initial state.

        Parameters
        ----------
        state : array
            An array of size Lxd^2 specifing the basis state 
            to which the MPS is initialized.
        '''
        # Erase current MPS
        self.G   = []
        self.L   = []
        self.chi = []
        self.Q   = []

        Ls0 = np.zeros(self.chimax,complex)
        Ls0[0] = 1.
        self.L.append(Ls0)
        self.chi.append(1)

        # No quantum numbers to the left
        q_left = (0,0)
        Q_left = {q_left:[0]}
        self.Q.append( Q_left )
        
        for i in range(0,self.N):
            # Set G
            L_right = np.zeros(self.chimax, complex)
            Q_right = {}
            chi_right = 0
            Gs = np.zeros( (self.d, self.chimax, self.chimax), dtype=np.complex128 )
            for n in range(self.d):
                if state[i,n] == 0: continue
                for key in Q_left.keys(): 
                    new_key = self.add_qn(key, self.pqn[n], 1)
                    k = Q_left[key]
                    l = Q_right.get(new_key, chi_right)
                    if l == chi_right:
                        Q_right[new_key] = [chi_right]
                        L_right[chi_right] = 1
                        chi_right += 1
                    Gs[n, k, l] = state[i,n]
            self.G.append(Gs)
            # Set L and current bond dimension (chi)
            self.L.append(L_right)
            self.chi.append(chi_right)

            # Set Q
            self.Q.append( Q_right )
            Q_left = Q_right

    def add_qn(self, q1, q2, reldir):
        '''
        Add quantum numbers q1 and q2, assuming a relative sign `reldir` = +/-1
        between the legs they are associated with.

        Parameters
        ----------
        q1,q2 : int, tuple
            Quantum numbers to be added
        reldir: int
            Equals +1 if both legs are incoming legs, -1 if either is outgoing.

        Returns
        -------
        Summed quantum numbers `q1` and `q2`
        '''
        # Add quantum numbers on legs with legs having relative direction reldir
        return tuple(np.array(q1) + reldir*np.array(q2))

    def trace(self):
        '''
        Compute the norm of the MPS
        Note that the local basis is chosen in a way such that the trace is given
        by  sum i in d: G[l][d*i + i]
        '''
        theta = np.dot(np.diag(self.L[0]), self.G[0][0]+self.G[0][3])
        for s in range(1, self.N):
            theta = np.dot(theta, np.diag(self.L[s]))
            theta = np.dot(theta, self.G[s][0]+self.G[s][3])
        lastL = np.zeros((self.chimax, self.chimax), complex)
        lastL[:,0] = self.L[self.N-1]
        theta = np.dot(theta, lastL)
        return np.trace(theta)
    
    def entanglement_entropy(self, bond_nr):
        '''
        Compute the entanglement entropy when bipartitioned at given bond

        Parameters
        ----------
        bond_nr : int
            Bond at which partitioning happens

        Returns
        -------
        Entanglement entropy at given bond
        '''
        Ent = 0
        for i in range(int(self.chi[bond_nr])):
            Ent += - self.L[bond_nr][i]**2 *2* np.log(self.L[bond_nr][i])
        return Ent

    def site_expectation_value(self, operator, site_nr):
        '''
        Compute the expectation value of a single site operator

        Parameters
        ----------
        operator : array
            Single site operator to be contracted with the MPS
        site_nr  : int
            Site at which to apply the operator

        Returns
        -------
        The single site expecation value Tr(operator*rho)/Tr(rho)
        '''
        norm = self.trace()
        theta = np.diag(self.L[0])
        for s in range(site_nr):
            theta = np.dot(theta, self.G[s][0]+self.G[s][3])
            theta = np.dot(theta, np.diag(self.L[s+1]))
        C = np.tensordot(self.G[site_nr],operator, axes=(0, 0))
        C = np.transpose(C, (2,0,1))
        theta = np.dot(theta, C[0]+C[3])
        for s in range(site_nr+1, self.N):
            theta = np.dot(theta, np.diag(self.L[s]))
            theta = np.dot(theta, self.G[s][0]+self.G[s][3])
        lastL = np.zeros((self.chimax, self.chimax), complex)
        lastL[:,0] = self.L[self.N-1]
        theta = np.dot(theta, lastL)
        return np.trace(theta)/(norm)
    
    def bond_expectation_value(self, operator, bond_nr):
        '''
        Compute the expectation value of a bond operator
        Note that the first bond is numbered 1.

        Parameters
        ----------
        operator : array
            Single site operator to be contracted with the MPS
        site_nr  : int
            Site at which to apply the operator

        Returns
        -------
        The bond expecation value Tr(operator*rho)/Tr(rho)
        '''

        theta = np.diag(self.L[0])
        for s in range(bond_nr-1):
            theta = np.dot(theta, self.G[s][0]+self.G[s][3])
            theta = np.dot(theta, np.diag(self.L[s+1]))

        #update just the Gammas and the Lambda of the bond bond_nr 
        chi_left   = self.chi[bond_nr-1]
        chi_middle = self.chi[bond_nr]
        chi_right  = self.chi[bond_nr+1]
        L_left   = self.L[bond_nr-1][:chi_left]
        G_left   = self.G[bond_nr-1][:, :chi_left, :chi_middle]
        L_middle = self.L[bond_nr][:chi_middle]
        G_right  = self.G[bond_nr][:, :chi_middle, :chi_right]
        L_right  = self.L[bond_nr+1][:chi_right]
   
        # Store dimensions for SVD
        m = min(self.chimax, self.d*chi_left)
        n = min(self.chimax, self.d*chi_right)

        # Contract matrices to form Theta
        #(1)->(chi,d,chi)
        thetat=np.tensordot(np.diag(L_left),G_left,axes=(1,1))
        #(2)->(chi,d,chi)
        thetat=np.tensordot(thetat,np.diag(L_middle),axes=(2,0))
        #(3)->(chi,d,d,chi)
        thetat=np.tensordot(thetat,G_right,axes=(2,1))
        #(4)->(chi,d,d,chi)
        thetat=np.tensordot(thetat,np.diag(L_right),axes=(3,0))
        #(chi, chi, d, d)
        # Then contract it with the operator and reshape
        thetat = np.tensordot(thetat,operator,axes=([1,2],[0,1]));
        thetat =np.reshape(np.transpose(thetat,(2,0,3,1)),(self.d*chi_left,self.d*chi_right))

        # SVD theta into X, Y and Z
        if thetat.dtype != np.complex128 :
            X,Y,Z = svd_dgesvd.svd_dgesvd(thetat, full_matrices = 1,compute_uv = 1)
        else:
            X,Y,Z = svd_zgesvd.svd_zgesvd(thetat, full_matrices = 1,compute_uv = 1)

        Z=Z.T

        # Reset MPS matrices
        G_left_new=np.zeros((self.d, self.chimax, self.chimax), dtype=np.complex128)
        G_right_new=np.zeros((self.d, self.chimax, self.chimax), dtype=np.complex128)

        # Truncate
        L_middle_new = Y[0:min(self.chimax, np.shape(Y)[0])]

        #Truncate X tensor and reshape it, then assign to new MPS matrix
        X=np.reshape(X[:self.d*chi_left, :m],(self.d, chi_left,m))
        G_left_new[:,:chi_left,:m]=np.transpose(np.tensordot(np.diag(L_left**(-1)),X,axes=(1,1)),(1,0,2))
        #Truncate Z tensor and reshape it, then assign to new MPS matrix

        Z=np.transpose(np.reshape(Z[:self.d*chi_right,:n] ,(self.d,chi_right,n)),(0,2,1))
        G_right_new[:,:n,:chi_right]=np.tensordot(Z,np.diag(L_right**(-1)),axes=(2,0));

        # Truncate tensors
        L_middle_0 = np.zeros(self.chimax)
        L_middle_0[:self.chi[bond_nr]] = L_middle_new[0:self.chi[bond_nr]]/np.sqrt(sum(L_middle_new[0:self.chi[bond_nr]]**2))

        G_left_0 = G_left_new[:,:,:]
        G_right_0 = G_right_new[:,:,:]

        theta = np.dot(theta, G_left_0[0]+G_left_0[3])
        theta = np.dot(theta, np.diag(L_middle_0))
        theta = np.dot(theta, G_right_0[0]+G_right_0[3])

        for s in range(bond_nr+1, self.N):
            theta = np.dot(theta, np.diag(self.L[s]))
            theta = np.dot(theta, self.G[s][0]+self.G[s][3])
        lastL = np.zeros((self.chimax, self.chimax), complex)
        lastL[:,0] = self.L[self.N-1]
        theta = np.dot(theta, lastL)
        norm = self.trace()
        return np.trace(theta)/norm

    def saveToFile(self, filename):
        """ Save MPS to a file """
        with open(filename, "wb") as f:
            cPickle.dump(self, f)
    
    @classmethod
    def loadFromFile(cls, filename):
        with open(filename, "rb") as f:
            return cPickle.load(f)

#-------------------------------------------------------------------------------
# MPS for super operators, i.e. an MPS representing a density matrix
# Also with quantum numbers, for fermionic partile loss Lindbladians
#-------------------------------------------------------------------------------
class spMPS(sMPS):
    """ Class for an MPS representing a density matrix, with quantum
        numbers ready for linbdlad loss operators.
    """

    def __init__(self,N,d,chimax,pqn=0):
        '''
        Initialize an MPS

        Parameters
        ----------
            N : int
                The system size, or number of MPS matrices
            d : int
                Dimension of the local Hilbert space. Currently, every site is 
                restricted to the same dimension.
            chimax : int
                Maximal bond dimension of the MPS matrices. Currently, every 
                site is restricted to the same maximal bond dimension.
            pqn : array
                An array of objects (integers, tuples) describing the quantum
                numbers associated to each of the physical basis states.
        '''
        # Number of MPS Matrices
        self.N = N
        # Local Hilbert space dimension
        self.d = d**2
        # Maximal bond dimension
        self.chimax = chimax

        # We use the Vidal representation of the MPS, so that we have easy
        # acces to both the left- and right-canonical forms of the MPS.
        self.G   = []  # Gamma matrices
        self.L   = []  # Lambda's
        self.chi = []  # Current bond dimensions
        self.Q   = []  # Quantum numbers to the left

        # List of quantum numbers corresponding to the physical indices,
        # i.e. the first entry corresponds to the q-number for d = 0, etc.
        self.pqn = [1, -1, -1, 1]

    def set_initial_state(self, state):
        '''
        Set the initial state.

        Parameters
        ----------
        state : array
            An array of size Lxd^2 specifing the basis state 
            to which the MPS is initialized.
        '''
        # Erase current MPS
        self.G   = []
        self.L   = []
        self.chi = []
        self.Q   = []

        Ls0 = np.zeros(self.chimax,complex)
        Ls0[0] = 1.
        self.L.append(Ls0)
        self.chi.append(1)

        # No quantum numbers to the left
        q_left = 1
        Q_left = {q_left:[0]}
        self.Q.append( Q_left )
        
        for i in range(0,self.N):
            # Set G
            L_right = np.zeros(self.chimax, complex)
            Q_right = {}
            chi_right = 0
            Gs = np.zeros( (self.d, self.chimax, self.chimax), dtype=np.complex128 )
            for n in range(self.d):
                if state[i,n] == 0: continue
                for key in Q_left.keys(): 
                    new_key = self.add_qn(key, self.pqn[n], 1)
                    k = Q_left[key]
                    l = Q_right.get(new_key, chi_right)
                    if l == chi_right:
                        Q_right[new_key] = [chi_right]
                        L_right[chi_right] = 1
                        chi_right += 1
                    Gs[n, k, l] = state[i,n]
            self.G.append(Gs)
            # Set L and current bond dimension (chi)
            self.L.append(L_right)
            self.chi.append(chi_right)

            # Set Q
            self.Q.append( Q_right )
            Q_left = Q_right

    def add_qn(self, q1, q2, reldir):
        '''
        Add quantum numbers q1 and q2, assuming a relative sign `reldir` = +/-1
        between the legs they are associated with.

        Parameters
        ----------
        q1,q2 : int, tuple
            Quantum numbers to be added
        reldir: int
            Equals +1 if both legs are incoming legs, -1 if either is outgoing.

        Returns
        -------
        Summed quantum numbers `q1` and `q2`
        '''
        # Add quantum numbers on legs with legs having relative direction reldir
        return q1*q2

