# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 00:56:31 2018

@author: sonam
"""

import numpy as np
import qutip as qt
import itertools as it
from cython.parallel import prange
import scipy.linalg as la
from math import pi
import pandas as pd
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import sympy as sp


class unitcell_hamiltonian(object):
    """
    Generalized connected triport system
    
    In this simulation we consider 3 unit cell for Benzene
    
    phi_a = phi_b = phi_c = phi
    
    @todo - Tweak phi's later for even more generalization
    
    @todo - add ability to put range of phis
    
    @todo - check for unitarity condition UU^\dag=  I

    @todo - check for hermitian condition H = H\dag
    
    """
    
    def __init__(self, phi=0, n=0, sites=3):
        # Error Checkers (incomplete)
        if (n < 0):
            raise ValueError('n must be greater than or equal to 0')
        else:
            n = int(n)  # Enfroce discretization of n

        # Constants
        self.exp_fact = np.exp(1j*phi)
        self.denom_fact = 2 + 1j*self.exp_fact
        self.exp_amp = 1j*np.exp(-1j*phi) - 1  # rho
        self.beta = self.exp_fact / self.denom_fact
        self.bohr_radius = 5.2917721067e-11  # meters , idk if will be used
        # Momentum Discretization (maybe later)
        #######################################
        # Transformation Matrix
        self.unit_mat = self.gen_unit_mat(phi, self.exp_amp)
        self.tot_mat = self.beta * self.unit_mat
        # Visual Purposes
        self.disp_unit = self.display_mat(self.unit_mat)
        self.disp_tot = self.display_mat(self.tot_mat)
        # Hamiltonians for position (n) [Add momentum later]
        self.pos_hamiltonian = self.gen_pos_hamiltonian(self.tot_mat)
        ############################################################
        # Solve Hamiltonian



        def gen_unit_mat(self, phi, amplitudes):
            return 'hi'