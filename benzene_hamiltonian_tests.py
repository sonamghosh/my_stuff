# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:10:37 2018

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


class benzene_hamiltonian(object):
    # Do stuff
    def __init__(self, phi_a=0, phi_t=0, k=0):
        # Constants
        self.exp_fact = np.exp(1j*phi_a)
        self.sqrt_fact = np.sqrt(1 - 8*np.power(np.cos(phi_t), 2))
        self.alpha = self.exp_fact * self.sqrt_fact
        #self.off_diag = -2*np.exp(1j*phi_t)*np.cos(phi_t)
        # Transition Matrices
        self.unit_mat = self.unitary_mat(phi_a, phi_t)
        self.tot_mat = self.alpha * self.unit_mat
        # Hamiltonians of Pos (n) and Momentum (K) space
        self.pos_hamiltonian = self.gen_pos_hamiltonian()
        self.quasimom_hamiltonian = self.gen_quasimom_hamiltonian()



    def unitary_mat(self, phi_a, phi_t):
        off_diag = -2*np.exp(1j*phi_t)*np.cos(phi_t)
        unit_mat = np.array([[1, off_diag, off_diag],
                         [off_diag, 1, off_diag],
                         [off_diag, off_diag, 1]
                         ])
        return unit_mat

    def total_mat(self, const, mat):
        total_mat = const*mat
        return total_mat

   # For 3 by 3 matrices U and alpha * U
    def display_mat(self, mat):
        df = pd.DataFrame(mat, columns=list('ABC'))
        return df
