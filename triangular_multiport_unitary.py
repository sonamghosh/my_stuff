# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:14:45 2017

@author: sonam
"""

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from string import ascii_lowercase, ascii_uppercase  #vertex
from collections import deque
from collections import OrderedDict
from qutip import Qobj
from scipy.linalg import block_diag
from numpy.random import random_sample as rand
from copy import deepcopy
import itertools as it
from cycler import cycler  #plotting


def triport_unit(iter_num = 1, phase_a = 45, phase_b = 45, phase_c = 45):
    """
    @brief This function runs N number of iterations of a triangular
    directionally-unbiased optical multiport and returns a matrix of
    probability amplitudes.

    @oaran iter_num The number of iterations this unitary transformation will
    occur within the system.

    @param phase_a The phase angle of port A. (units deg)

    @oaran phase_b The phase angle of port B. (units deg)

    @param phase_c The phase angle of port C. (units deg)
    """

    # Unitary Matrix U
    U = np.matrix('0 0 0 0 0; 0 0 0 0 0; 1 0 0 0 0;\
                  1j 0 0 0 0; 0 0 0 0 0')
    # Unitary Matrix Initialization
    U_zeros = np.matrix(np.zeros([5, 5]))

    # Unitary Matrix of all Transitions
    U_1 = 1/np.sqrt(2)*np.bmat([[U, U_zeros, U_zeros], 
                               [U_zeros, U, U_zeros],
                               [U_zeros, U_zeros, U]])
    # Transition Matrix from Port A to Port B
    U_AB = np.matrix([[0, 0, 0, 0, 0], [0, 0, 0, 1j*np.exp(1j*phase_b), 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]])

    # Transition Matrix from Port A to Port C
    U_AC = np.matrix([[0, 0, 0, 0, 0], [0, 0, 1j*np.exp(1j*phase_c), 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1j, 0, 0]])

    # Transition Matrix from Port B to Port A
    U_BA = np.matrix([[0, 0, 0, 0, 0], [0, 0, 1j*np.exp(1j*phase_a), 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1j, 0, 0]])

    # Transition Matrix from Port B to Port C
    U_BC = np.matrix([[0, 0, 0, 0, 0], [0, 0, 0, 1j*np.exp(1j*phase_c), 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]])

    # Transition Matrix from C to A
    U_CA = np.matrix([[0, 0, 0, 0, 0], [0, 0, 0, 1j*np.exp(1j*phase_a)],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]])

    # Transition Matrix from C to B
    U_CB = np.matrix([[0, 0, 0, 0, 0], [0, 0, 1j*np.exp(1j*phase_b), 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1j, 0, 0]])

    # Transition Matrix with A, B, C teansitions
    U_2 = 1/np.sqrt(2)*np.bmat([[U_zeros, U_BA, U_CA], [U_AB, U_zeros, U_CB],
                               [U_AC, U_BC, U_zeros]])

    # Transition matrix from A2 to A3 , A4
    U_Mout = np.matrix([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1j, 0, 0, 0],
                        [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]])

    # Transition Matrix for A transitions
    U_3 = 1/np.sqrt(2)*np.bmat([[U_Mout, U_zeros, U_zeros],
                               [U_zeros, U_Mout, U_zeros],
                               [U_zeros, U_zeros, U_Mout]])

    return U_2


# Testing
a = triport_unit()