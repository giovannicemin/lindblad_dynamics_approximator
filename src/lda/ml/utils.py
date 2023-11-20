import numpy as np
import torch
from torch import nn
import os
from itertools import product
import opt_einsum as oe
import getopt
import sys
from torch.utils.data.sampler import SubsetRandomSampler

def calculate_error(results_ml, results_tebd, T, dt):
    '''Function to calculate the error defined as
    the normalized norm squared of the difference
    of the two coherence vectors averaged over time

    Perameters
    ----------
    results_ml : array
        Vector of vectors containing the dynamics
        predicted by the model
    results_tebd : array
        Vector of vectors containing the dynamics
        calculated using TEBD
    T : int
        Total time of the dynamics
    dt : float
        Time increase

    Return
    ------
        Return the error
    '''
    integral = 0

    # to do thigs right first and last element shoul be *1/2
    for v_ml, v_tebd in zip(results_ml, results_tebd):
        integral += (np.linalg.norm(v_ml - v_tebd) / np.linalg.norm(v_tebd) )**2

    return integral * (dt/T)

def get_arch_from_layer_list(input_dim, output_dim, layers):
    ''' Function returning the NN architecture from layer list
    '''
    layers_module_list = nn.ModuleList([])
    # layers represent the structure of the NN
    layers = [input_dim] + layers + [output_dim]
    for i in range(len(layers)-1):
        layers_module_list.append(nn.Linear(layers[i], layers[i+1]))
    return layers_module_list

def pauli_s_const():
    '''Function returning the structure constants
    for 2 spin algebra
    '''
    s_x = np.array([[ 0,  1 ], [ 1,  0 ]], dtype=np.complex64)
    s_z = np.array([[ 1,  0 ], [ 0, -1 ]], dtype=np.complex64)
    s_y = np.array([[ 0, -1j], [ 1j, 0 ]], dtype=np.complex64)
    Id  = np.eye(2)
    pauli_dict = {
       'X' : s_x,
       'Y' : s_y,
       'Z' : s_z,
       'I' : Id
    }

    # creating the elements of the base
    base_F = []
    for i, j in product(['I', 'X', 'Y', 'Z'], repeat=2):
        base_F.append( 0.5*np.kron(pauli_dict[i], pauli_dict[j]))

    base_F.pop(0) # don't want the identity
    abc = oe.contract('aij,bjk,cki->abc', base_F, base_F, base_F )
    acb = oe.contract('aij,bki,cjk->abc', base_F, base_F, base_F )

    # added -, and put 0.25 instead of 0.5
    f = np.real( -1j*0.25*(abc - acb) )
    d = np.real( 0.25*(abc + acb) )

    # return as a torch tensor
    return torch.from_numpy(f).float(), torch.from_numpy(d).float()


def ensure_empty_dir(directory):
    if len(os.listdir(directory)) != 0:
        raise Exception('Model dir not empty!')

def load_data(path, L, potential, N, M,
              num_traj, batch_size, validation_split):
    '''Function to load the data from hdf5 file.
    Reshuffling of data is performed. Then separates train
    from validation and return the iterables.

    Parameters
    ----------
    path : str
        Path to the hdf5 file
    potential : float
        Potential for the group name
    num_traj : int
    validation_split : float
        Number 0 < .. < 1 which indicates the relative
        sizes of validation and train

    Return
    ------
    train and validation loaders
    '''
    # put import here to avoid circular imports
    from ml.classes import CustomDatasetFromHDF5

    # list of group names
    gname = 'cohVec_L_' + str(L) + \
            '_V_' + str(int(potential*1e3)).zfill(4) + \
            '_N_' + str(int(N)) + '_M_' + str(int(M))

    dataset = CustomDatasetFromHDF5(path, gname)

    # creating the indeces for training and validation split
    dataset_size = len(dataset)
    indeces = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    print(f"Data points used in the training {dataset_size}")

    # shuffling the datesets
    np.random.seed(42)
    np.random.shuffle(indeces)
    train_indices, val_indices = indeces[split:], indeces[:split]

    # Creating PT data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=valid_sampler)
    return train_loader, val_loader


def get_params_from_cmdline(argv, default_params=None):
    '''Function that parse command line argments to dicitonary
    Parameters
    ----------
    argv : argv from sys.argv
    default_params : dict
        Dicionary to update

    Return
    ------
    Updated dictionary as in input
    '''
    arg_help = '{0} -L <length of spin chain> -b <beta> -V <potential> -w <working dir>'

    if default_params == None:
        raise Exception('Missing default parameters')

    try:
        opts, args = getopt.getopt(argv[1:], 'hL:b:V:w:', ['help', 'length', 'beta=', 'potential=', 'working_dir='])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(arg_help)
        elif opt in ('-L', '--length'):
            default_params['L'] = arg
        elif opt in ('-b', '--beta'):
            default_params['beta'] = arg
        elif opt in ('-p', '--potential'):
            default_params['potential'] = arg
        elif opt in ('-w', '--working_dir'):
            default_params['working_dir'] = arg

    return default_params
