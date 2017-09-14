# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy as sp
import qutip as qt
import matplotlib.pyplot as plt

"""
@brief This is a test run of the QuTip Simulation Toolbox
"""

# Define State psi = (1,0)^T
psi = qt.Qobj([[1], [0]])
print(psi)

# Define Up and Down States
psi_up = qt.Qobj([[1], [0]])
psi_down = qt.Qobj([[0], [1]])
psi_norm = (psi_up + psi_down)/np.sqrt(2)
print(psi_norm)