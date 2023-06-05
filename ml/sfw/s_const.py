import numpy as np
import itertools
import torch
from itertools import product
import opt_einsum as oe

epsilon = torch.zeros(3*3*3).reshape(3,3,3)

index = list(itertools.permutations([0,1,2]))

for i in index:
 epsilon[i[0]][i[1]][i[2]] = (-i[0] + i[1])*(-i[0] + i[2])*(-i[1] + i[2])/2

def LeviCivita():
 return epsilon

"""
.. module:: gellmann.py
   :synopsis: Generate generalized Gell-Mann matrices
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

def gellmann(j, k, d):
    r"""Returns a generalized Gell-Mann matrix of dimension d. According to the
    convention in *Bloch Vectors for Qubits* by Bertlmann and Krammer (2008),
    returns :math:`\Lambda^j` for :math:`1\leq j=k\leq d-1`,
    :math:`\Lambda^{kj}_s` for :math:`1\leq k<j\leq d`,
    :math:`\Lambda^{jk}_a` for :math:`1\leq j<k\leq d`, and
    :math:`I` for :math:`j=k=d`.

    :param j: First index for generalized Gell-Mann matrix
    :type j:  positive integer
    :param k: Second index for generalized Gell-Mann matrix
    :type k:  positive integer
    :param d: Dimension of the generalized Gell-Mann matrix
    :type d:  positive integer
    :returns: A genereralized Gell-Mann matrix.
    :rtype:   numpy.array

    """

    if j > k:
        gjkd = np.zeros((d, d), dtype=np.complex128)
        gjkd[j - 1][k - 1] = 1
        gjkd[k - 1][j - 1] = 1
    elif k > j:
        gjkd = np.zeros((d, d), dtype=np.complex128)
        gjkd[j - 1][k - 1] = -1.j
        gjkd[k - 1][j - 1] = 1.j
    elif j == k and j < d:
        gjkd = np.sqrt(2/(j*(j + 1)))*np.diag([1 + 0.j if n <= j
                                               else (-j + 0.j if n == (j + 1)
                                                     else 0 + 0.j)
                                               for n in range(1, d + 1)])
    else:
        gjkd = np.diag([1 + 0.j for n in range(1, d + 1)])

    return gjkd


def get_basis(d):
    r'''Return a basis of orthogonal Hermitian operators on a Hilbert space of
    dimension d, with the identity element in the last place.

    '''
    return np.array([gellmann(j, k, d) for j, k in product(range(1, d + 1), repeat=2)])

def structure_const(d):
        #   Completely skew-symm s const:      
        #       ijk      1        i  j   k
        #      f      =  _   Tr [F, F ] F     (1)
        #                4i
        #       
        #       
        #   Completely symm s const:      
        #       ijk      1        i  j   k
        #      d      =  _   Tr {F, F } F     (2)
        #                4
        #       
        #       


        no_id = d**2-1
        Lambda = get_basis(d)[:no_id]
        abc = oe.contract('aij,bjk,cki->abc',Lambda,Lambda,Lambda ) 
        #  aij cjk bki ->  abc          
        acb = oe.contract('aij,bki,cjk->abc',Lambda,Lambda,Lambda )            
        f = np.real( -1j*0.25*(abc-acb))
        d = np.real(0.25* (abc+acb))
        return torch.from_numpy(f).float(),torch.from_numpy(d).float()



def pauli_s_const():
    sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
    sigma_z = np.array( [ [ 1. , 0. ] , [ 0. , -1. ] ] )
    sigma_y = np.array([[0, -1j], [1j, 0]])
    Id = np.array( [ [ 1. , 0. ] , [ 0. , 1. ] ] )   
    pauli_dict = {
       'X': sigma_x,
       'Y': sigma_y,
       'Z': sigma_z,
       'I': Id
    }
    names=['I','X','Y','Z']
   
   
    paulis2 = []
    for i in names: 
        for j in names:
          sigma_i = pauli_dict[i]
          sigma_j = pauli_dict[j]
          paulis2.append( np.kron(sigma_i,sigma_j))
    Lambda = paulis2[1:]
    abc = oe.contract('aij,bjk,cki->abc',Lambda,Lambda,Lambda ) 
    acb = oe.contract('aij,bki,cjk->abc',Lambda,Lambda,Lambda )            
    f = np.real( -1j*0.25*(abc-acb))
    d = np.real(0.25* (abc+acb))
    return torch.from_numpy(f).float(),torch.from_numpy(d).float()



