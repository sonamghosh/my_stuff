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
    U = np.array(np.matrix('0 0 0 0 0; 0 0 0 0 0; 1 0 0 0 0;\
                  1j 0 0 0 0; 0 0 0 0 0'))
    # Unitary Matrix Initialization
    U_zeros = np.zeros([5, 5])

    # Unitary Matrix of all Transitions
    U_1 = np.array
    return U