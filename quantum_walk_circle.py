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
from cython.parallel import prange


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


def unitary_transform(nodes, angle):
    C_hat = coin(angle)
    S_hat = shift(nodes)
    U_hat = S_hat*(tensor(C_hat, qeye(nodes)))
    return U_hat

def quantum_walk(state, t_step, nodes, angle):
    position_state = basis(nodes, 0)
    psi = ket2dm(tensor(state, position_state))
    U_hat = unitary_transform(nodes, angle)
    for step in prange(t_step):
        if step > nodes:
            step = np.mod(step, nodes)
            psi = U_hat*psi*U_hat.dag()
        else:
            psi = U_hat*psi*U_hat.dag()
    return psi


def measurement(nodes, psi):
    prob = []
    for i in prange(nodes):
        m_p = basis(nodes, i)*basis(nodes, i).dag()  # |x><x| Outer Prod
        measure = tensor(qeye(2), m_p)
        prob_amp = abs((psi*measure).tr())  # Probability Amplitude
        prob.append(prob_amp)
    return prob


# Testing
#psi = quantum_walk(psi_1, 4, 3, 35.26)
psi = quantum_walk(psi_1, 2, 4, 45)
mat = measurement(4, psi)