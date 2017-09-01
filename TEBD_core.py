import numpy as np
import scipy as sp
import scipy.linalg
import itertools
import svd_dgesvd, svd_zgesvd

# Time evolve MPS for a given bond
def bond_update(op, bond_nr, mps,p):
    '''
    Update a bond by applying a two-site operator.

    Parameters
    ----------
    op : array
        An array of shape (d,d,d,d) for the 2-site operator, with d 
        the local Hilbert space dimensions.
    bond_nr : int
        The bond to which the operator is applied.
    mps : object
        The mps object to be updated. 
    p : dictionary
        Dictionary containing simulation parameters

    Returns
    -------
        None
    ''' 

    #    Function to time evolve Tensors for one bond
    #    1. Merge tensors 
    #        (1)->L[i-1],G[i] 
    #        (2)-> L[i] 
    #        (3)-> G[i+1]
    #        (4)-> L[i+1] 
    #        into Theta (chi,d,d,chi)
    #    2. Contract it with tensor U
    #    3. SVD
    #    4. Create new site and bond tensors A, B 

    # Convenience
    dim = p.d 

    # We should truncate all the matrices to reduce overhead
    chi_left   = mps.chi[bond_nr-1]
    chi_middle = mps.chi[bond_nr]
    chi_right  = mps.chi[bond_nr+1]
    L_left   = mps.L[bond_nr-1][:chi_left]
    G_left   = mps.G[bond_nr-1][:, :chi_left, :chi_middle]
    L_middle = mps.L[bond_nr][:chi_middle]
    G_right  = mps.G[bond_nr][:, :chi_middle, :chi_right]
    L_right  = mps.L[bond_nr+1][:chi_right]
   
    # Store dimensions for SVD
    m = min(p.chi_max, dim*chi_left)
    n = min(p.chi_max, dim*chi_right)

    # Contract matrices to form Theta
    #(1)->(chi,d,chi)
    theta=np.tensordot(np.diag(L_left),G_left,axes=(1,1))
    #(2)->(chi,d,chi)
    theta=np.tensordot(theta,np.diag(L_middle),axes=(2,0))
    #(3)->(chi,d,d,chi)
    theta=np.tensordot(theta,G_right,axes=(2,1))
    #(4)->(chi,d,d,chi)
    theta=np.tensordot(theta,np.diag(L_right),axes=(3,0))
    #(chi, chi, d, d)
    # Then contract it with the operator and reshape
    theta = np.tensordot(theta,op,axes=([1,2],[0,1]));
    theta = np.reshape(np.transpose(theta,(2,0,3,1)),(dim*chi_left,dim*chi_right))

    # SVD theta into X, Y and Z
    if theta.dtype != np.complex128 :
        X,Y,Z = svd_dgesvd.svd_dgesvd(theta, full_matrices = 1,compute_uv = 1)
    else:
        X,Y,Z = svd_zgesvd.svd_zgesvd(theta, full_matrices = 1,compute_uv = 1)

    Z=Z.T

    # Reset MPS matrices
    G_left_new=np.zeros((dim, p.chi_max, p.chi_max), dtype=np.complex128)
    G_right_new=np.zeros((dim, p.chi_max, p.chi_max), dtype=np.complex128)

    # Truncate
    L_middle_new = Y[0:min(p.chi_max, np.shape(Y)[0])]
    Error        = np.sum(Y[min(p.chi_max, np.shape(Y)[0]):])

    #Truncate X tensor and reshape it, then assign to new MPS matrix
    X=np.reshape(X[:dim*chi_left, :m],(dim, chi_left,m))
    G_left_new[:,:chi_left,:m]=np.transpose(np.tensordot(np.diag(L_left**(-1)),X,axes=(1,1)),(1,0,2))
    #Truncate Z tensor and reshape it, then assign to new MPS matrix
    Z=np.transpose(np.reshape(Z[:dim*chi_right,:n] ,(dim,chi_right,n)),(0,2,1))
    G_right_new[:,:n,:chi_right]=np.tensordot(Z,np.diag(L_right**(-1)),axes=(2,0));

    # Store new bond dimension
    chi_new           = np.min([p.chi_max,L_middle_new[L_middle_new>p.max_error].shape[0]])
    mps.chi[bond_nr]  = chi_new
    Error            += np.sum(L_middle_new[chi_new:])

    # Output if requested
    if(p.verb):
      print 'Error: ', Error
      print 'new chi at ', bond_nr, ': ', chi_new

    # Truncate tensors
    mps.L[bond_nr][0:mps.chi[bond_nr]]=L_middle_new[0:mps.chi[bond_nr]]/np.sqrt(sum(L_middle_new[0:mps.chi[bond_nr]]**2))
    mps.G[bond_nr-1][:,:mps.chi[bond_nr-1],:] = G_left_new[:,:mps.chi[bond_nr-1],:]
    mps.G[bond_nr][:,:,:mps.chi[bond_nr+1]] = G_right_new[:,:,:mps.chi[bond_nr+1]]
    return Error

#time evolve MPS for a given bond using number conservation
def bond_update_conserve(op,bond_nr,mps,p):
    '''
    Update a bond by applying a two-site operator, using quantum numbers to
    perform SVD only within blocks of same quantum number. This provides a
    considerable speed-up over the non-conserving update.

    Parameters
    ----------
    op : array
        An array of shape (d,d,d,d) for the 2-site operator, with d 
        the local Hilbert space dimensions.
    bond_nr : int
        The bond to which the operator is applied.
    mps : object
        The mps object to be updated. 
    p : dictionary
        Dictionary containing simulation parameters

    Returns
    -------
        None
    '''    

    #    Function to time evolve Tensors for one bond
    #    1. Merge tensors
    #        (1)-> L[i-1], G[i]
    #        (2)-> L[i]
    #        (3)-> G[i+1]
    #        (4)-> L[i+1]
    #        into Theta (chi,d,d,chi)
    #    2. Contract it with tensor U
    #    3. Compute quantum numbers, so that we may decompose theta into blocks
    #    3. SVD
    #    4. Create new site and bond tensors A, B

    # Convenience
    dim = mps.d
    
    # We should truncate all the matrices to reduce overhead
    chi_left   = mps.chi[bond_nr-1]
    chi_middle = mps.chi[bond_nr]
    chi_right  = mps.chi[bond_nr+1]
    L_left     = mps.L[bond_nr-1][:chi_left]
    G_left     = mps.G[bond_nr-1][:, :chi_left, :chi_middle]
    L_middle   = mps.L[bond_nr][:chi_middle]
    G_right    = mps.G[bond_nr][:, :chi_middle, :chi_right]
    L_right    = mps.L[bond_nr+1][:chi_right]

    # Get the quantum numbers to the left of each bond
    # The q are dictionaries, that have as keys the q-numbers,
    # and a list of indices as corresponding values. 
    # The number of indices corresponds to the bond dimension.
    q_left   = mps.Q[bond_nr-1]
    q_middle = mps.Q[bond_nr]
    q_right  = mps.Q[bond_nr+1]
    
    # Construct theta
    #(1)->(chi,d,chi)
    theta=np.tensordot(np.diag(L_left),G_left,axes=(1,1))
    #(2)->(chi,d,chi)
    theta=np.tensordot(theta,np.diag(L_middle),axes=(2,0))
    #(3)->(chi,d,d,chi)
    theta=np.tensordot(theta,G_right,axes=(2,1))
    #(4)->(chi,d,d,chi)
    theta=np.tensordot(theta,np.diag(L_right),axes=(3,0))
    #(chi, chi, d, d)
    # Apply the operator and reshape
    theta = np.tensordot(theta,op,axes=([1,2],[0,1]));
    theta = np.reshape(np.transpose(theta,(2,0,3,1)),(dim*chi_left,dim*chi_right))
 
    # Possible new charge values:
    Q_left = {}
    for k in q_left.keys():
        for d in range(dim):
            newq = mps.add_qn(mps.pqn[d],k,1)
            Q_left[newq] = Q_left.get( newq, [] ) + list(np.array(q_left.get(k))+ d*int(chi_left))

    # Now let's do the same for the right. The slight difference here, is that 
    # the allowed q_values need to take into account that the rightmost leg is an outgoing one.
    # This is taken into account by the add_qn function of the MPS class, which provides the
    # relative sign between the contracted legs. 
    Q_right = {}
    for k in q_right.keys():
        for d in range(dim):
            newq = mps.add_qn(k,mps.pqn[d],-1)
            Q_right[newq] = Q_right.get( newq, [] ) + list(np.array(q_right.get(k))+ d*int(chi_right))

    # Create empty matrices for us to put the blocks in
    X = np.zeros((chi_left*dim, min(chi_left, chi_right) * dim), dtype=np.complex128)
    Y = np.zeros( min(chi_left, chi_right) * dim, dtype=np.complex128 )
    Z = np.zeros((min(chi_left, chi_right) * dim, chi_right*dim), dtype=np.complex128)

    # Create an empty list to store, for each index, the quantum number it corresponds to.
    # We will need this list later to reshape the SVD'ed blocks into the MPS matrices
    Qlist = []
    current_size = 0
    for ql in Q_left.keys():
        # Get the indices for the entries corresponding to these elements
        left_indices = Q_left.get(ql, [])
        right_indices = Q_right.get(ql, [])

        # If either of the two lists is empty, the corresponding legs
        # do not allow for the q_number, and so we can skip it. 
        if left_indices == [] or right_indices == []:
            continue

        # Extract the block
        theta_Q = theta[np.ix_(left_indices,right_indices)]

        # We can now SVD the block
        if theta_Q.dtype != np.complex128 :
            XQ,YQ,ZQ = svd_dgesvd.svd_dgesvd(theta_Q, full_matrices = 0,compute_uv = 1)
        else:
            XQ,YQ,ZQ = svd_zgesvd.svd_zgesvd(theta_Q, full_matrices = 0,compute_uv = 1)

        # And assign them back to the larger matrices, as if we had SVD'ed it in one go
        X[np.array(left_indices), current_size:current_size+len(YQ)] = XQ[:, :len(YQ)]
        Y[current_size:current_size+len(YQ)] = YQ
        Z[current_size:current_size+len(YQ), np.array(right_indices)] = ZQ[:len(YQ), :]
        
        # Keep track of the quantum number at a given index
        Qlist.append( [ql]*len(YQ) )
        
        # Update size
        current_size += len(YQ)

    # Truncate trailing zeros on Y
    Y = Y[:current_size]
    # Flatten Qlist (can find more elegant way?)
    #Qlist = list( itertools.chain.from_iterable(Qlist) )
    Qlist = [a for a in itertools.chain.from_iterable(Qlist)]
    # Get indices for sorting singular values
    sorted_idx = np.argsort( Y )[::-1]
    # Actually sort them
    Y = Y[sorted_idx]
    # But we needed the indices so that we can also sort Qlist
    #  since the corresponding indices for the q-sectors have moved
    Qlist = [Qlist[i] for i in sorted_idx]
    # Sort the other SVD matrices
    X = X[:, sorted_idx]
    Z = Z[sorted_idx, :].T
    
    # Make sure we never go over max bond-dim bound
    newchi = np.max( [np.min([ np.sum( Y > p.max_error ), p.chi_max]), 1])
    # Truncate
    L_middle_new = Y[:newchi]
    Qlist = Qlist[:newchi]
    # Track error in truncation
    Error = np.sum(Y[newchi:])
    
    # Compute the new quantum numbers for the middle bond that we have
    #  just updated. This is why we used Qlist in the first place.
    q_middle_new = {}
    for i,q in enumerate(Qlist):
        q_middle_new[q] = q_middle_new.get(q,[]) + [i]
                
    # Reset the MPS matrices
    G_left_new=np.zeros((dim, p.chi_max, p.chi_max), dtype=np.complex128)
    G_right_new=np.zeros((dim, p.chi_max, p.chi_max), dtype=np.complex128)
    
    #Truncate X tensor and reshape it
    X=np.reshape(X[:dim*chi_left, :newchi],(dim, chi_left,newchi))
    G_left_new[:,:chi_left,:newchi]=np.transpose(np.tensordot(np.diag(L_left**(-1)),X,axes=(1,1)),(1,0,2))
    #Truncate Z tensor and reshape it
    Z=np.transpose(np.reshape(Z[:dim*chi_right,:newchi], (dim,chi_right,newchi)),(0,2,1))
    #get G_right from Z
    G_right_new[:,:newchi,:chi_right]=np.tensordot(Z,np.diag(L_right**(-1)),axes=(2,0));

    # Set Lambda
    mps.L[bond_nr] = np.zeros(p.chi_max, dtype=np.complex128)
    mps.L[bond_nr][:newchi] = L_middle_new/np.sqrt(np.sum(L_middle_new**2))
    # Set new Q numbers    
    mps.Q[bond_nr] = q_middle_new
    # Update bond dimension
    mps.chi[bond_nr] = newchi    
    # Set G's
    mps.G[bond_nr-1][:,:chi_left,:] = G_left_new[:,:chi_left,:]
    mps.G[bond_nr][:,:,:chi_right] = G_right_new[:,:,:chi_right]
    
    return Error

def site_update_conserve( Ue, Uo, site_nr, mps, p ):

#    print "Site update on site %d"%site_nr
    # Convenience
    dim = mps.d
    
    # We should truncate all the matrices to reduce overhead
    chi_left   = mps.chi[site_nr]
    chi_right  = mps.chi[site_nr+1]
    G          = mps.G[site_nr][:, :chi_left, :chi_right]

    # Get the quantum numbers to the left of each bond
    # The q are dictionaries, that have as keys the q-numbers,
    # and a list of indices as corresponding values. 
    # The number of indices corresponds to the bond dimension.
    q_left   = mps.Q[site_nr]
#    print q_left

    indexplus = q_left.get(1, []) 
    if indexplus != []:
        Gp = np.tensordot( G[:,indexplus,:], Ue, axes=(0,0) ) # chi1 chi2 d
#        print indexplus
#        print Gp.shape
#        print mps.G[site_nr].shape
        mps.G[site_nr][:, indexplus, :chi_right] = np.transpose( Gp, (2,0,1) )
    indexmin = q_left.get(-1, [])
    if indexmin != []:
        Gm = np.tensordot( G[:,indexmin,:], Uo, axes=(0,0) ) # chi1 chi2 d
        mps.G[site_nr][:, indexmin, :chi_right]  = np.transpose( Gm, (2,0,1) )


    return 0
