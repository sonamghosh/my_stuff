# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:41:01 2017

@author: sonam
"""

# Coined Discrete Quantum Random Walk

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from math import *


# Basis states
"""
@def basis() - basis function with inputs d, n creates coloumn
vector with dimension d with n=0,1,..,d-1 being the
physical state (position for example)

@def unit() - Normalizes state

@ex basis(2,0): vector with '1' in 0th position of (1,0)^T
Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
Qobj data =
[[ 1.]
 [ 0.]]

"""

ket0 = basis(2,0).unit()  # |0>
ket1 = basis(2,1).unit()  # |1>
psip = (basis(2,0)+basis(2,1)*1j).unit()  # (|0> + i|1>)/sqrt(2)
psim = (basis(2,0)-basis(2,1)*1j).unit()  # (|0> - i|1>)/sqrt(2)

# Coin Operation
"""
@def qutip.Qobj - Creates a matrix or an array type object in
qutip. Note the ([[]]) structure for input.
"""


def coin(angle):
    """
    @brief Applies unitary rotation on coin basis (2D basis)

    @return QuTip matrix/array type object
    
    @ex coin(60): Quantum object: dims = [[2], [2]], shape = (2, 2),
    type = oper, isherm = True
    Qobj data =
    [[ 0.5        0.8660254]
     [ 0.8660254 -0.5      ]]
    """

    C_hat = qutip.Qobj([[cos(radians(angle)), sin(radians(angle))],
                         [sin(radians(angle)), -cos(radians(angle))]
                         ])
    return C_hat


# Translational Operation
def shift(t_step):
    """
    @brief Applies shift to the coin state one step forwards or
    backwards depending on the coin state.
    
    np.roll(array, shift) performs a shift operation on a given
    array input.
    a = [1,2,3,4] --> np.roll(a,2) = array([3,4,1,2])

    @return Rotated and Translated state.

    @ex shift(1)
    Quantum object: dims = [[2, 3], [2, 3]], shape = (6, 6), type = oper,
    isherm = False
    Qobj data =
    [[ 0.  0.  1.  0.  0.  0.]
     [ 1.  0.  0.  0.  0.  0.]
     [ 0.  1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  1.  0.]
     [ 0.  0.  0.  0.  0.  1.]
     [ 0.  0.  0.  1.  0.  0.]]
    """

    sites = 2*t_step+1
    shift_left = qutip.Qobj(np.roll(np.eye(sites), 1, axis=0))  # left coin
    shift_right = qutip.Qobj(np.roll(np.eye(sites), -1, axis=0))  # Right Coin
    left_prod = tensor(ket0*ket0.dag(), shift_left)
    right_prod = tensor(ket1*ket1.dag(), shift_right)
    S_hat = left_prod + right_prod
    return S_hat


# Walk Evolution Operator
def walk(t_step, coin_angle):
    """
    @brief Applies the total unitary transformation of the coin and translation
    operation on a state where
    U = S dot (C tensor_prod with Identity)
    |psi>_t_step = U^t_step |psi>_initial

    @note qeye is the Identity Operator

    @ex walk(1, 60)
    Quantum object: dims = [[2, 3], [2, 3]], shape = (6, 6), type = oper,
    isherm = False
    Qobj data =
    [[ 0.         0.         0.8660254  0.         0.         0.5      ]
     [ 0.8660254  0.         0.         0.5        0.         0.       ]
     [ 0.         0.8660254  0.         0.         0.5        0.       ]
     [ 0.         0.5        0.         0.        -0.8660254  0.       ]
     [ 0.         0.         0.5        0.         0.        -0.8660254]
     [ 0.5        0.         0.        -0.8660254  0.         0.       ]]
    """

    sites = 2*t_step + 1
    C_hat = coin(coin_angle)
    S_hat = shift(t_step)
    U_hat = S_hat*(tensor(C_hat, qeye(sites)))
    return U_hat

# QW generator which provides evolved wavefunction after steps @c t_step
def quantum_walk(t_step, qubit_state, coin_angle):
    """
    @Brief takes in a t_step, coin angle and applies the Unitary
    Transformation U = S dot (C tensor_prod with Identity) on a
    initial state to get a ouput |psi>_t_step = U^t_step |psi>_initial.

    @note ket2dm converts kets to density matrices.
    """

    sites = 2*t_step + 1
    position_state = basis(sites, t_step)  # Initialize the lattice
    psi = ket2dm(tensor(qubit_state, position_state))
    U_hat = walk(t_step, coin_angle)
    for i in range(t_step):
        psi = U_hat*psi*U_hat.dag()
    return psi

# Measurement
def measurement(t_step, psi, space):
    """
    @brief Projective measurement on the position basis.
    Walker will have zero probability at odd positions of the lattice.
    Odd positions will be avoided by measuring only even position if space is
    set to 2.

    @return probability
    """

    sites = 2*t_step + 1
    prob = []
    # Measurement on all 2*t+1 states
    for i in range(0, sites, space):
        m_p = basis(sites, i)*basis(sites, i).dag()  # Outer product
        measure = tensor(qeye(2), m_p)
        prob_amp = abs((psi*measure).tr())  # Probability
        prob.append(prob_amp)
    return prob

# Plot results
def plot_pdf(prob_mat):
    """
    @brief Plots the probability density function
    """

    lattice_position = range(-len(prob_mat)/2+1,len(prob_mat)/2+1)
    plt.plot(lattice_position, prob_mat)
    plt.xlim([-len(prob_mat)/2+2, len(prob_mat)/2+2])
    plt.ylim([min(prob_mat), max(prob_mat)+0.01])
    plt.ylabel('Probability')
    plt.xlabel('Position of particle')
    plt.show()

# Create insance
    if __name__ == "__main__":
        psi_t = quantum_walk(100, psip, 45)  # Performs walk
        prob_mat = measurement(100, psi_t, 2)  # Measures output
        plot_pdf(prob_mat)
