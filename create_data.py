#!/usr/bin/env python
'''File for the generation of data by means of TEBD algorithm.
'''
import numpy as np
import sys
import multiprocessing
import h5py
import quimb as qu

from models import SpinChain
from utils import get_params_from_cmdline


prms = {'L' : 50,                # length of spin chain
        'sites' : [0, 1],        # sites of the subsystem S spins
        'omega' : 1,             # Rabi frequency
        'potential' : [0.5],     # interaction of subsystem's S spins
        'potential_' : None,     # interaction of bath spins, if None same as potential
        'T' : 10,                # total time for the evolution
        'dt' : 0.01,             # interval every which save the data
        'cutoff' : 1e-8,         # cutoff for TEBD algorithm
        'im_cutoff' : 1e-10,     # cutoff for TEBD algorithm, img t-e
        'tolerance' : 1e-5,      # Trotter tolerance for TEBD algorithm
        'verbose' : True,        # verbosity of the script
        'num_traj' : 30,         # how many trajectories to do
        # file to save the data
        'fname' : './data/data_tebd.hdf5'
        }

def generate_data(default_params, argv=[1]):
    '''Here I generate the data for the spin chain
    as in models.SpinChain

    Parameter
    --------
    argv : array
        Output from sys.argv, custom parameters
        default [1]
    default_params : dict
        Contains all the parameters necessary for the generation, e.g.
        {'L' : 10,                # length of spin chain
         'sites' : [0, 1],        # sites of the subsystem S spins
         'omega' : 1,             # Rabi frequency
          # interaction of subsystem's S spins
         'potential' : [0.1, 0.2, 0.3, 0.4, 0.5],
         'potential_' : None,     # interaction of bath spins, if None same as potential
         'T' : 10,                # total time for the evolution
         'dt' : 0.02,             # interval every which save the data
         'cutoff' : 1e-5,         # cutoff for TEBD algorithm
         'im_cutoff' : 1e-7,      # cutoff for TEBD algorithm, im t-e
         'tolerance' : 1e-5,      # Trotter tolerance for TEBD algorithm
         'verbose' : True,        # verbosity of the script
         'num_traj' : 20,         # how many trajectories to do
         # file to save the data
         'fname' : './data/data_tebd.hdf5'
         }
    '''
    # check if custom parameters are given
    if len(argv) == 1:
        print('Working with default params \n')
        prms = default_params.copy()
    else:
        prms = get_params_from_cmdline(argv, default_params)
        print('Working with custom parameters')

    print('   -----------------------')
    for key in prms.keys():
        print(f'   |{key} : {prms[key]}')
    print('   -----------------------\n')

    # system parameters
    sys_prms = prms.copy()
    sys_prms.pop('sites')
    sys_prms.pop('potential_')
    sys_prms.pop('num_traj')
    sys_prms.pop('fname')
    sys_prms['verbose'] = False

    ### ACTUAL GENERATION
    n_simulations = len(prms['potential'])
    count = 1
    for vv in prms['potential']:

        sys_prms['potential'] = vv

        # create the output array
        X = []

        print(f'==== {count}/{n_simulations}')
        print(f'== potential = {vv}')

        # random seed
        seed = np.random.randint(420)

        # parallelized code
        # with multiprocessing.Pool() as pool:
        #     # creating the list of inputs for the function
        #     items = [(sys_prms, seed*(i+1)) for i in range(1, prms['num_traj']+1)]
        #     # calling the function for each trajectory
        #     for results in pool.starmap(execute_trajectories, items):
        #         X.extend(results)

        # or normal loop
        for i in range(prms['num_traj']):
            X.extend(execute_trajectories(sys_prms, seed*(i+1)))

        # save to file
        file = h5py.File(prms['fname'], 'a')

        # group name in hdf5 file
        gname = 'cohVec_L_' + str(prms['L']) + \
            '_V_' + str(int(vv*1e3)).zfill(4) + \
            '_dt_' + str(int(prms['dt']*1e3)).zfill(4)
        # create the subgroup
        subg = file.create_group(gname)

        subg.create_dataset('X', data=X)

        file.close()

        count += 1

def execute_trajectories(sys_prms, seed):
    '''Little function needed for the
    parallelization
    '''
    # evolution of the spin chain
    system = SpinChain(**sys_prms)
    system.evolve(seed)

    return system.return_results()

if __name__ == '__main__':
    argv = sys.argv
    generate_data(prms, argv)
