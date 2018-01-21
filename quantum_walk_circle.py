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
from cython.parallel import prange  # Optimized range function
import plotly.plotly as py  # Not Being Used Yet
import plotly.graph_objs as go  # Not Being Used Yet


# Input States
ket0 = basis(2, 0).unit()  # |0>, |V>, (1, 0)
ket1 = basis(2, 1).unit()  # |1>, |H>, (0, 1)
psi_1 = (basis(2, 0)+basis(2, 1)*1j).unit()  #(|0> + i|1>)/sqrt(2)
psi_2 = (basis(2, 0)-basis(2, 1)*1j).unit()  #(|0> - i|1>)/sqrt(2)


# Coin Operation
def coin(angle):
    """
    @brief Applies rotation on 2D coin state basis

    @param angle The angle to rotate the vector by

    @return 2D QuTip array
    """

    C_hat = qutip.Qobj([[cos(radians(angle)), sin(radians(angle))],
                        [sin(radians(angle)), -cos(radians(angle))]
                        ])
    return C_hat


def shift(nodes):
    """
    @brief Applies positional shift to coin state one step forward
    or backwards dpending on the coin state.

    @param nodes, the number of nodes on the circular surface.

    @return Translated state with dimensions 2*2n by 2*2n where n = nodes
    """

    # |(x+1) mod N>_w <x| tensprod |0>_c <0|
    shift_ccw = qutip.Qobj(np.roll(np.eye(nodes), 1, axis=0))
    ccw_prod = tensor(ket0*ket0.dag(), shift_ccw)
    # |(x+(N-2)+1)mod N>_w <x| tens prod |1>_c <1|
    shift_cw = qutip.Qobj(np.roll(np.eye(nodes), -1, axis=0))
    cw_prod = tensor(ket1*ket1.dag(), shift_cw)
    S_hat = ccw_prod + cw_prod
    return S_hat


def unitary_transform(nodes, angle):
    """
    @brief Applies the total unitary transformation of the coin and
    translation operator on a input state where
    U = S dot (C tensor_prod with Identity)
    |psi>_nodes = U^nodes |psi>_initial

    @param nodes, the number od nodes on the circular surface
    @param angle, the angle to rotate the coin state by

    @return Output transformed state
    """

    C_hat = coin(angle)
    S_hat = shift(nodes)
    U_hat = S_hat*(tensor(C_hat, qeye(nodes)))
    return U_hat

def quantum_walk(state, t_step, nodes, angle):
    """
    @brief Applies the unitary transformation t_step times on a line of
    length modulo the number of nodes to mimic a circle.

    @param state, the initial state of the particle |psi>_initial
    @param t_step, the number of steps to apply U to |psi>_initial
    @param nodes, the number of nodes of the circle
    @param angle, the angle to rotate the coin states

    @return Unmeasured Output state of the Quantum Walk
    """

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
    """
    @brief Makes a measurement of the state of the particle in the
    quantum walk with a matrix of probability values for each node
    in the circle.

    @param nodes, the number of nodes in the circle
    @param psi, the output state of the quantum walk

    @return array of probability amplitudes for each node on the circle
    """

    prob = []
    for i in prange(nodes):
        m_p = basis(nodes, i)*basis(nodes, i).dag()  # |x><x| Outer Prod
        measure = tensor(qeye(2), m_p)
        prob_amp = abs((psi*measure).tr())  # Probability Amplitude
        prob.append(prob_amp)
    return prob


def plot_pdf(prob_mat):
    """
    @brief Plots probability density function

    @param prob_mat the array of probability amplitudes after measurement

    @return Plot of Position vs Probability
    """

    #lattice_position = np.arange(-len(prob_mat)/2+1,len(prob_mat)/2+1)
    lattice_position = prange(len(prob_mat))
    #plt.plot(lattice_position, prob_mat)
    #plt.xlim([-len(prob_mat)/2+2, len(prob_mat)/2+2])
    #plt.ylim([min(prob_mat), max(prob_mat)+0.01])
    #plt.ylabel('Probability')
    #plt.xlabel('Position of particle')
    #plt.show()
    #(username = 'sonamghosh', key='ZWhMxlc9xwnfQp45Jh8y')
    """
    df = pd.DataFrame({'x': lattice_position, 'y': prob_mat})
    df.head()
    data = [go.Bar(x=df['x'], y=df['y'])]
    response = py.plot(data, filename='testing_123')
    url=response['url']
    filename=response['filename']
    print(url)
    print(filename)
    """
    df = pd.DataFrame(data={'Probability': prob_mat})
    print(df)
    ax = df.plot(kind='bar', colormap='jet', grid=True,
                 title='Quantum Walk around a N-Node Circle')
    ax.set_xlabel('Position of Photon')
    ax.set_ylabel('Probability')


if __name__ == "__main__":
    psi_t = quantum_walk(psi_1, 10, 4, 45)
    prob_mat = measurement(4, psi_t)
    plot_pdf(prob_mat)
# Testing
#psi = quantum_walk(psi_1, 4, 3, 35.26)
#psi = quantum_walk(psi_1, 2, 4, 45)
#mat = measurement(4, psi)