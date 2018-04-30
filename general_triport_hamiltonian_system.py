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
from scipy.linalg import logm, expm


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
        self.exp_amp = 1j*np.exp(-1j*phi) - 1  # rho = ie^(-iphi) - 1
        self.beta = self.exp_fact / self.denom_fact
        
        #self.reamp = (self.exp_amp * self.beta).real
        
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
        self.disp_ham = self.display_mat(self.pos_hamiltonian)
        ############################################################
        # Solve Hamiltonian
        self.pos_eig = self.solve_hamiltonian(self.pos_hamiltonian)
        # Projected Version without Left/Right
        #self.proj_energies = self.proj_mat(self.beta, self.exp_amp)
        #self.proj_hamiltonian = self.proj_ham(self.beta, self.exp_amp)
        self.proj_hamiltonian = self.proj_mat(self.pos_hamiltonian)
        self.proj_energies = self.solve_hamiltonian(self.proj_hamiltonian)


    def gen_unit_mat(self, phi, amp):
       # \Lambda_{1} matrix
        mat_1 = np.array([[0, 1, 0, 0],
                          [1, 0, 0, np.sqrt(2)*amp],
                          [np.sqrt(2)*amp, 0, 0, (1+amp)],
                          [0, 0, (1+amp), 0]
                          ])
        # \Lambda_{2} matrix
        mat_2 = np.array([[0, 0, np.sqrt(2)*amp, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]
                          ])
        # \Lambda_{3} matrix
        mat_3 = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, np.sqrt(2)*amp, 0, 0]
                          ])
        # Put it together
        unit_mat = np.block([[mat_1, mat_2, mat_3],
                             [mat_3, mat_1, mat_2],
                             [mat_2, mat_3, mat_1]
                             ])
        # Turn into Qobj
        unit_mat = qt.Qobj(unit_mat)
        
        return unit_mat


    def display_mat(self, mat):
        mat = mat.full()
        df = pd.DataFrame(mat, columns=list('SSDDSSDDSSDD'))
        return df


    def gen_pos_hamiltonian(self, mat):
        # First Order Taylor Approximation
        # Take T = 1 and hbar = 1
        # H = exp(-i*pi/2) ( I - U)
        """
        eye = qt.qeye(12)
        alpha = np.exp(-1j*np.pi/2)
        ham_mat = eye - mat
        ham_mat = alpha*ham_mat
        """
        
        ham_mat = 1j*logm(mat.full())
        ham_mat = qt.Qobj(ham_mat)
        ham_mat = ham_mat.tidyup()
        
        return ham_mat


    def solve_hamiltonian(self, mat):
        # Solve Eigenvalues
        eigvals = qt.Qobj.eigenenergies(mat)
        # Solve Eigenstates
        eigvecs = qt.Qobj.eigenstates(mat)
        # Store into dict
        eig_dict = {'Eigenenergies': eigvals,
                    'Eigenstates': eigvecs}
        return eig_dict
    
    
    def proj_mat(self, H_mat):
        # Convert QObj to numpy array
        H_mat = H_mat.full()
        # Projection Vector index pairs
        idx = np.arange(12)
        p_i = [(idx[2*i], idx[2*i+1]) for i in range(len(idx) // 2)]
        # Empty matrix
        p_mat = np.zeros([6,6], dtype=complex)
        # Perform projection
        for i in range(6):
            p_1 = np.zeros(12, dtype=complex)
            p_1[p_i[i][0]] = 1
            p_1[p_i[i][1]] = 1
            #print('p_1 ', p_1)  # Debugging
            for j in range(6):
                p_2 = np.zeros(12)
                p_2[p_i[j][0]] = 1
                p_2[p_i[j][1]] = 1
                #print('p_2 ' , p_2)  # Debugging
                p_mat[i][j] = 0.5*np.dot(np.dot(p_1, H_mat), p_2.T)
        
        # Convert back to Qobj
        p_mat = qt.Qobj(p_mat)
        return p_mat
    
        
                
    
    """
    
    def proj_mat(self, const, amp):
        e1 = 0.5 * const * (2 + amp * (1 - np.sqrt(3)))
        e2 = 0.5 * const * (2 + amp * (1 + np.sqrt(3)))
        e3 = const * (1 - amp)
        e4 = const * (1 + 2 * amp)
        
        e1 = 1j*np.log(e1)
        e2 = 1j*np.log(e2)
        e3 = 1j*np.log(e3)
        e4 = 1j*np.log(e4)
        
        e_arr = np.array([e1, e2, e3, e4])
        return e_arr
    
    def proj_ham(self, const, amp):
        m1 = np.array([[const, const*amp/np.sqrt(2)], [const*amp/np.sqrt(2), const+(amp*const).real]])
        m2 = const*np.array([[0, amp/np.sqrt(2)], [0, 0]])
        m3 = const*np.array([[0, 0], [amp/np.sqrt(2), 0]])
        unit_proj = np.block([[m1, m2, m3],
                                      [m3, m1, m2],
                                      [m2, m3, m1]
                                      ])
        #ham_proj = qt.Qobj(1j*logm(unit_proj)).tidyup()
        ham_proj = 1j*logm(unit_proj)
        ham_proj = qt.Qobj(ham_proj)
        ham_proj = ham_proj.tidyup()
        return unit_proj, ham_proj
    """
        
    
    
    


