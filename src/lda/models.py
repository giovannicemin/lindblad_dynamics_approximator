#!/usr/bin/env python
'''File containing the implementation of the models.
    - the spin chain, to perform TEBD simulations
    - the Lindbladian to bechmark the ML algorthm
'''
import numpy as np
import pandas as pd
import torch
import time

import quimb as qu
import quimb.tensor as qtn
import quimb.linalg.base_linalg as la
from itertools import product
from abc import ABC, abstractmethod

from lda.ml.utils import pauli_s_const


class SpinChain:
    '''Class implementing the spin chain with PBC.
    The evolution is done using the TEBD algorithm.

    Parameters
    ----------
    L : int
        Length of the spin chain
    omega : float
        Rabi frequency
    potential : float
        Interaction strength between the spins
    T : float
        Total time of the evolution
    dt : float
        Time step for observable measurement
    cutoff : float
        Cutoff for the TEBD algorithm
    im_cutoff : float
        Cutoff from TEBD imaginary time
    tolerance : float
        Trotter tolerance for TEBD algorithm
    '''

    def __init__(
        self,
        L,
        omega=1,
        potential=0.1,
        T=10,
        dt=0.01,
        cutoff=1e-10,
        im_cutoff=1e-10,
        tolerance=1e-3,
        verbose=True,
    ):
        # setting th parameters
        self.L = L
        self.vv = potential
        self.T = T
        self.t = [i for i in np.arange(0, T, dt)]
        self.cutoff = cutoff
        self.im_cutoff = im_cutoff
        self.tolerance = tolerance
        self._verbose = verbose

        # creating verboseprint
        self.verboseprint = print if verbose else lambda *a, **k: None

        # create the MPS of the spin chain
        self.verboseprint(f'System with potential = {self.vv}, L = {self.L}')
        self.verboseprint('Initial condition: \n')
        binary = '0' * L
        self.psi = qtn.MPS_computational_state(binary, cyclic=True)
        if self._verbose:
            self.psi.show()
            print('\n')

        # build the Hamiltonian of the system
        self.verboseprint('Building the Hamiltonian of the system \n')

        # dims = [2]*L # overall space of L qbits

        I = qu.pauli('I')
        X = qu.pauli('X')
        Z = qu.pauli('Z')

        O_Rabi = (omega / 2) * X
        N = (I + Z) / 2

        # the hamiltonian for the time evolution
        H1 = {None: O_Rabi}
        H2 = {None: self.vv * N & N}
        self.H = qtn.LocalHam1D(L=L, H2=H2, H1=H1, cyclic=True)

        # results
        self.results = None

    def evolve(self, seed):
        '''Perform time evolution of the System
        Parameter
        ---------
        seed : int
            Seed needed for random perturbation of thermal state
        '''

        self.verboseprint('Performing the time evolution \n')

        # random initial condition
        rand_uni = qu.gen.rand.random_seed_fn(qu.gen.rand.rand_uni)
        rand1 = rand_uni(2, seed=seed)
        rand2 = rand_uni(2, seed=3 * seed)
        self.psi_init = self.psi.gate(
            rand1 & rand2, (0, 1), contract='swap+split'
        )

        start = time.time()

        # first I build the observables and results dictionaries
        observables = {}
        self.results = {}
        for ob1, ob2 in product(['I', 'X', 'Y', 'Z'], repeat=2):
            key = ob1 + '1' + ob2 + '2'
            observables[key] = []
            self.results[key] = []

        # dropping the identity
        observables.pop('I1I2')
        self.results.pop('I1I2')

        # create the object
        tebd = qtn.TEBD(self.psi_init, self.H)

        # cutoff for truncating after each infinitesimal-time operator application
        tebd.split_opts['cutoff'] = self.cutoff

        self.keys = self.results.keys()

        for psit in tebd.at_times(self.t, tol=self.tolerance):
            for key in self.keys:
                ob1 = qu.pauli(key[0])
                ob2 = qu.pauli(key[2])
                self.results[key].append(
                    (psit.H @ psit.gate(ob1 & ob2, (0, 1))).real * 0.5
                )

        end = time.time()
        self.verboseprint(f'It took:{int(end - start)}s')

    def return_results(self):
        '''Return the results, which are the evolution of
        the coherence vector, as a vector of vectors
        '''
        if self.results == None:
            raise Exception('The object have not been evolved jet')
        else:
            length = len(self.results['I1X2'])
            return [
                [self.results[key][i] for key in self.keys]
                for i in range(length)
            ]

            # tried to return dataframe, but array is better
            # return pd.DataFrame(data=self.results, dtype=np.float32)

    def calculate_correlations(self, site, step=0.1, seed=0):
        """Function that calculates the spread of correlations
        over the spin chain
        """

        # stuff I need for the data generation
        I = qu.pauli('I')
        X = qu.pauli('X')
        Y = qu.pauli('Y')
        Z = qu.pauli('Z')

        # observables
        mag_x = X & I
        mag_y = Y & I
        mag_z = Z & I
        cx_t_j = []  # x-magnetization
        cy_t_j = []  # y-magnetization
        cz_t_j = []  # z-magnetization

        # initial codition obtained by means of a projection
        # and random unitary
        sigma_m = 0.5 * (qu.pauli('X') - 1j * qu.pauli('Y'))
        projection = sigma_m & qu.pauli('I')
        self.psi_th.gate_(
            projection & projection, (0, 1), contract='swap+split'
        )
        self.psi_th /= self.psi_th.norm()  # normalization

        rand_uni = qu.gen.rand.random_seed_fn(qu.gen.rand.rand_uni)
        rand1 = rand_uni(2, seed=seed) & qu.pauli('I')
        rand2 = rand_uni(2, seed=3 * seed) & qu.pauli('I')

        self.psi_init = self.psi_th.gate(
            rand1 & rand2, (0, 1), contract='swap+split'
        )

        # create the object
        tebd = qu.tensor.TEBD(self.psi_init, self.H)

        # cutoff for truncating after each infinitesimal-time operator application
        tebd.split_opts['cutoff'] = self.cutoff

        for psit in tebd.at_times(
            np.arange(0, self.T, step), tol=self.tolerance
        ):
            cx_j = []
            cy_j = []
            cz_j = []

            for j in range(0, self.L):
                # along each direction I calculate the correlations as:
                # <sig_{site} sig_{j}> - <sig_{site}> <sig_{j}>
                psi_H = psit.H
                corr = (
                    psi_H
                    @ psit.gate(
                        mag_x & mag_x, (site, j), contract='swap+split'
                    )
                ).real
                ex_site = (
                    psi_H @ psit.gate(mag_x, site, contract='swap+split')
                ).real
                ex_j = (
                    psi_H @ psit.gate(mag_x, j, contract='swap+split')
                ).real
                cx_j.append(corr - ex_site * ex_j)

                corr = (
                    psi_H
                    @ psit.gate(
                        mag_y & mag_y, (site, j), contract='swap+split'
                    )
                ).real
                ex_site = (
                    psi_H @ psit.gate(mag_y, site, contract='swap+split')
                ).real
                ex_j = (
                    psi_H @ psit.gate(mag_y, j, contract='swap+split')
                ).real
                cy_j.append(corr - ex_site * ex_j)

                corr = (
                    psi_H
                    @ psit.gate(
                        mag_z & mag_z, (site, j), contract='swap+split'
                    )
                ).real
                ex_site = (
                    psi_H @ psit.gate(mag_z, site, contract='swap+split')
                ).real
                ex_j = (
                    psi_H @ psit.gate(mag_z, j, contract='swap+split')
                ).real
                cz_j.append(corr - ex_site * ex_j)

            cx_t_j += [cx_j]
            cy_t_j += [cy_j]
            cz_t_j += [cz_j]

        return cx_t_j, cy_t_j, cz_t_j


class Lindbladian(ABC):
    """Parent class for time-dependent Lindbladian."""

    def __init__(self, dt=0.01):
        self.dt = dt
        self.f, self.d = pauli_s_const()

    @abstractmethod
    def kossakowski(self, t):
        pass

    @abstractmethod
    def omega(self, t):
        pass

    def get_rates(self, t):
        kossakowski = self.kossakowski(t)
        eigenval, _ = np.linalg.eig(kossakowski)

        return eigenval

    def get_L(self, t):
        re_c = self.kossakowski(t).real.float()
        im_c = self.kossakowski(t).imag.float()

        # Here I impose the fact c_re is symmetric and c_im antisymmetric
        re_1 = -4.0 * torch.einsum('mjk,nik,ij->mn', self.f, self.f, re_c)
        re_2 = -4.0 * torch.einsum('mik,njk,ij->mn', self.f, self.f, re_c)
        im_1 = 4.0 * torch.einsum('mjk,nik,ij->mn', self.f, self.d, im_c)
        im_2 = -4.0 * torch.einsum('mik,njk,ij->mn', self.f, self.d, im_c)
        d_super_x_re = torch.add(re_1, re_2)
        d_super_x_im = torch.add(im_1, im_2)
        d_super_x = torch.add(d_super_x_re, d_super_x_im)

        tr_id = 2.0 * torch.einsum('imj,ij->m', self.f, im_c)

        h_commutator_x = -4.0 * torch.einsum(
            'ijk,k->ij', self.f, self.omega(t)
        )

        # building the Lindbladian operator
        L = torch.zeros(16, 16)
        L[1:, 1:] = torch.add(h_commutator_x, d_super_x)
        L[1:, 0] = tr_id

        return L

    def forward_t(self, t, x):
        """Time dependent Lindbladian"""
        L = self.get_L(t)
        exp_dt_L = torch.matrix_exp(self.dt * L)
        return torch.add(
            0.5 * exp_dt_L[1:, 0], x @ torch.transpose(exp_dt_L[1:, 1:], 0, 1)
        )

    def forward(self, x):
        """Time independent Lindbladian"""
        exp_dt_L = torch.matrix_exp(self.dt * self.L)
        return torch.add(
            0.5 * exp_dt_L[1:, 0], x @ torch.transpose(exp_dt_L[1:, 1:], 0, 1)
        )

    def generate_trajectory(self, v_0, T):
        '''Function that generates the time evolution of
        the system, namely the trajectory of v(t) coherence
        vector

        Parameters
        ----------
        v_0 : array
            Initial conditions
        T : int
            Total time of the simulation
        Return
        ------
        vector of vectors representing the v(t) at each
        instant of time, and the time
        '''
        results = [v_0]
        t = [0]

        X = torch.tensor(v_0, dtype=torch.double)

        length = int(T / self.dt)

        self.L = self.get_L(1)

        for i in range(length - 1):
            y = self.forward(X.float())
            X = y.clone()
            results.extend([y.numpy()])
            t.append((i + 1) * self.dt)

        return results, t
