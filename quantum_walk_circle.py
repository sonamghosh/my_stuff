# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 23:30:04 2018

@author: sonam
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from math import *
import pandas as pd


ket0 = basis(2, 0).unit()  # |0>, |V>
ket1 = basis(2, 1).unit()  # |1>, |H>
psi_1 = (basis(2, 0)+basis(2, 1)*1j).unit()  #(|0> + i|1>)/sqrt(2)
psi_2 = (basis(2, 0)-basis(2, 1)*1j).unit()  #(|0> - i|1>)/sqrt(2)

def coin(angle):
    C_hat = qutip.Qobj([[cos(radians(angle)), sin(radians(angle))],
                        [sin(radians(angle)), -cos(radians(angle))]
                        ])
    return C_hat

def shift(nodes):
    # |(x+1) mod N>_w <x| tensprod |0>_c <0|
    shift_ccw = qutip.Qobj(np.roll(np.eye(nodes), 1, axis=0))
    ccw_prod = tensor(ket0*ket0.dag(), shift_ccw)
    # |(x+(N-2)+1)mod N>_w <x| tens prod |1>_c <1|
    shift_cw = qutip.Qobj(np.roll(np.eye(nodes), -1, axis=0))
    cw_prod = tensor(ket1*ket1.dag(), shift_cw)
    S_hat = ccw_prod + cw_prod
    return S_hat