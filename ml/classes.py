'''Where all the classes used for the ML part of the project
are stored.
'''
import numpy as np
import torch
from torch import nn
import h5py
from torch.utils.data.dataset import Dataset

from ml.utils import pauli_s_const, get_arch_from_layer_list

class CustomDatasetFromHDF5(Dataset):
    '''Class implementing the Dataset object from HDF5 file.

    This class loads the data from HDF5 file.

    This class is implemented such that it can be passed to
    torch.utils.data.DataLoader.

    Parameters
    ----------
    path : str
        Path to where hdf5 file is
    group : str
        Group name or names of the desired data
    '''

    def __init__(self, path, group):
        with h5py.File(path, 'r') as f:
            self.X = []
            self.y = []
            self.t = []
            self.V = [] # dummy vector (legacy previous implementations)

            self.X.extend(f[group + '/X'][()])
            self.y.extend(f[group + '/y'][()])
            self.t.extend(f[group + '/t'][()])
            self.V.extend([0]*len(f[group + '/X'][()]))

    def __getitem__(self, index):
        # return the potential and the vector at t and t+dt
        # as tensors
        return torch.tensor(self.V[index]), torch.tensor(self.t[index]), \
            torch.tensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.X)


class MLLP(nn.Module):
    '''Machine learning model to parametrize the Lindbladian operator.

    Parametes
    ---------
    mlp_params : dict
        Dictionary containing all parameters needed to exp_LL
    potential : float
        Potential appearing in the H
    time_dependent = bool, default False
        Wheter or not to use the time dependent exp_LL
    '''

    def __init__(self, mlp_params):
        super().__init__()
        self.MLP = exp_LL(**mlp_params)  # multi(=1) layer perceptron

        self.dt = mlp_params['dt']

    def forward(self, **kwargs):
        '''Forward step of the model'''
        return self.MLP.forward_t(**kwargs)

    def generate_trajectory(self, v_0, T):
        '''Function that generates the trajectory v(t).

        Given an initial condition v_0, this function generates the trajectory,
        namely the time evolution v(t), using the learned model.

        Parameters
        ----------
        v_0 : array
            Initial conditions
        T : int
            Total time of the evolution
            
        Return
        ------
        vector of vectors representing the v(t) at each instant of time.
        '''
        X = torch.Tensor(v_0)

        length = int(T/self.dt)

        results = [v_0]
        with torch.no_grad():
            Lindblad = self.MLP.get_L()
            for i in range(length-1):
                exp_dt_L = torch.matrix_exp(i*self.dt*Lindblad )
                y = torch.add(0.5*exp_dt_L[1:,0], X @ torch.transpose(exp_dt_L[1:,1:],0,1))
                results.extend([y.numpy()])

        return results


class exp_LL(nn.Module):
    '''Custom Liouvillian layer to ensure positivity of the rho.

    Parameters
    ----------
    data_dim : int
        Dimension of the input data
    layers : arr
        Array containing for each layer the number of neurons
        (can be empty)
    nonlin : str
        Activation function of the layer(s)
    output_nonlin : str
        Activation function of the output layer
    dt : float
        Time step for the input data
    '''

    def __init__(self, data_dim, layers, nonlin, output_nonlin, dt):
        super().__init__()
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.data_dim = data_dim
        self.dt = dt

        # I want to build a single layer NN
        self.layers = get_arch_from_layer_list(1, data_dim**2, layers)
        # structure constants
        self.f, self.d = pauli_s_const()

        # Dissipative parameters v = Re(v) + i Im(v) = x + i y
        # NOTE: v is called Z on the paper, it represents the complx matrix
        #       used to build the kossakowski matrix c = Z^T Z
        v_re = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        v_im = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        self.v_x = nn.Parameter(v_re)
        self.v_y = nn.Parameter(v_im)

        # Hamiltonian parameters omega
        omega = torch.zeros([data_dim])
        self.omega = nn.Parameter(omega).float()

        # initialize omega and v
        nn.init.kaiming_uniform_(self.v_x, a=1)
        nn.init.kaiming_uniform_(self.v_y, a=1)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.v_x)
        bound = 1. / np.sqrt(fan_in)
        nn.init.uniform_(self.omega, -bound, bound)

    def get_L(self):
        ''' Function that returns the learned Lindbladian.
        '''
        v_x = self.v_x
        v_y = self.v_y
        # Structure constant for SU(n) are defined
        #
        # We define the real and imaginary part of the Kossakowsky's matrix c.
        #       +
        # c = v   v =  ∑  x     x    + y   y    + i ( x   y  - y   x   )
        #              k    ki   kj     ki  kj         ki  kj   ki  kj
        c_re = torch.add(torch.einsum('ki,kj->ij', v_x, v_x),\
                         torch.einsum('ki,kj->ij', v_y, v_y)  )
        c_im = torch.add(torch.einsum('ki,kj->ij', v_x, v_y),\
                         -torch.einsum('ki,kj->ij', v_y, v_x) )

        # Here I impose the fact c_re is symmetric and c_im antisymmetric
        re_1 = -4.*torch.einsum('mjk,nik,ij->mn', self.f, self.f, c_re )
        re_2 = -4.*torch.einsum('mik,njk,ij->mn', self.f, self.f, c_re )
        im_1 =  4.*torch.einsum('mjk,nik,ij->mn', self.f, self.d, c_im )
        im_2 = -4.*torch.einsum('mik,njk,ij->mn', self.f, self.d, c_im )
        d_super_x_re = torch.add(re_1, re_2 )
        d_super_x_im = torch.add(im_1, im_2 )
        d_super_x = torch.add(d_super_x_re, d_super_x_im)

        tr_id = 2.*torch.einsum('imj,ij ->m', self.f, c_im )

        h_commutator_x = -4.* torch.einsum('ijk,k->ij', self.f, self.omega)

        # building the Lindbladian operator
        L = torch.zeros(self.data_dim+1, self.data_dim+1)
        L[1:,1:] = torch.add(h_commutator_x, d_super_x)
        L[1:,0] = tr_id

        return L

    def forward(self, t, x):
        """Forward step of the Layer.
        The step: t -> t+dt

        Time is not used but present from compatibility reasons.
        """
        L = self.get_L()

        exp_dt_L = torch.matrix_exp(self.dt*L ).float()
        return torch.add(0.5*exp_dt_L[1:,0], x @ torch.transpose(exp_dt_L[1:,1:],0,1))

    def forward_t(self, t, x):
        """Forward step of the Layer.
        The step: 0 -> t
        """
        L = self.get_L()

        exp_dt_L = torch.matrix_exp( torch.einsum('b,ij->bij', t, L) ).float()
        return torch.add(0.5*exp_dt_L[:,1:,0], torch.einsum('bi,bji->bj', x, exp_dt_L[:,1:,1:]))

    def gap(self):
        '''Function to calculate the Lindblad gap, meaning
        the smallest real part of the modulus of the spectrum.
        '''
        L = self.get_L()
        # take the real part of the spectrum
        e_val = np.linalg.eigvals(L.detach().numpy()).real

        e_val.sort()
        return np.abs(e_val[-2])

