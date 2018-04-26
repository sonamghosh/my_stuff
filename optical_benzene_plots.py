# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 02:33:04 2018

@author: sonam
"""

import general_triport_hamiltonian_system as gt  # has needed modules
import benzene_huckel_approx as bz  # huckel model
import seaborn as sb

"""
This Script Plots 
The Position Eigenvalues of the Optical Benzene Model
as a function of the phi angle where 
the three port angles of A B C are taken to be the same and equal to phi
"""

phi_angles = np.arange(0, 2*np.pi, np.pi/64)
colors = sb.color_palette("hls", 12)
eig_dict = []
ham_dict = []
penerg_dict = []


for i in phi_angles:
    photon = gt.unitcell_hamiltonian(phi = i)
    eig_dict.append(photon.pos_eig['Eigenenergies'])
    ham_dict.append(photon.pos_hamiltonian)
    penerg_dict.append(photon.proj_energies)
    
# Consider making interactive plot with Blokeh or Plotly

for i in range(12):
    e_list = [item[i] for item in eig_dict]  # E_1
    line = plt.plot(phi_angles, e_list, 'o', label='$E_{%i}$' % (i+1),
                    color=colors[i])
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$E_{n}(\phi)$")
    plt.title('Eigenenergies $E_{i}(\phi)$ of Position Hamiltonian')
    legend = plt.legend(loc = "center right", bbox_to_anchor=(1,0.5))
    #legend = plt.legend(loc="upper right", bbox_to_anchor=(0.5,1.05),
                #        ncol=3, fancybox=True, shadow=True)
    #legend = plt.legend(handles=[line])
    #ax = plt.gca().add_artist(legend)
    

# QuTip Visualizers
# Energy Level Diagram
#qt.plot_energy_levels(ham_dict)

# Hamiltonian Diagram

# Hamiltonian Amplitudes Diagram
    
# Projected Out Energy Spectrum
plt.figure(2)
for i in range(4):
    pe_list = [item[i] for item in penerg_dict]
    l2 = plt.plot(phi_angles, pe_list, '-o', label= '$E_{%i}$' % (i+1),
                    color=colors[i])
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$E_{n}(\phi)$")
    plt.title('Eigenenergies $E_{i}(\phi)$ of Projected Position Hamiltonian')
    legend_2 = plt.legend(loc = "center right", bbox_to_anchor=(1,0.5))


# EIgenstate visualization
my_xticks = ['SL1', 'SR1', 'DL1', 'DR1', 'SL2', 'SR2', 'DL2', 'DR2', 'SL3', 'SR3' , 'DL3', 'DR3']
colors_2 = sb.color_palette("hls", 8)
colors_3 = sb.color_palette("hls", 4)

p_test = gt.unitcell_hamiltonian(phi = 3*np.pi/2)
non_periodic = [p_test.pos_eig['Eigenstates'][1][0].full(),
                p_test.pos_eig['Eigenstates'][1][1].full(),
                p_test.pos_eig['Eigenstates'][1][4].full(),
                p_test.pos_eig['Eigenstates'][1][5].full(),
                p_test.pos_eig['Eigenstates'][1][6].full(),
                p_test.pos_eig['Eigenstates'][1][7].full(),
                p_test.pos_eig['Eigenstates'][1][10].full(),
                p_test.pos_eig['Eigenstates'][1][11].full()]

periodic = [p_test.pos_eig['Eigenstates'][1][2].full(),
            p_test.pos_eig['Eigenstates'][1][3].full(),
            p_test.pos_eig['Eigenstates'][1][8].full(),
            p_test.pos_eig['Eigenstates'][1][9].full(),
            ]

plt.figure(3)
for i in range(len(non_periodic)):
    pnp = plt.plot(np.arange(0,12), non_periodic[i], '-o', label= '$\psi_{%i}$' % (i+1),
                   color=colors_2[i])
    plt.xlabel('State of the photon |B, m, n>')
    plt.xticks(np.arange(0, 12), my_xticks)
    plt.ylabel('Probability Amplitude')
    legend_3 = plt.legend(loc = 'center right', bbox_to_anchor=(1,0.5))
    plt.title('Motion of Photon for non-periodic eigenstates for $\phi$ = $3\pi/2$')

plt.figure(4)
for i in range(len(periodic)):
    pnp = plt.plot(np.arange(0,12), periodic[i], '-o', label='$\psi_{%i}$' % (i+1),
                   color=colors_3[i])
    plt.xlabel('State of the photon |B, m, n>')
    plt.xticks(np.arange(0, 12), my_xticks)
    plt.ylabel('Probability Amplitude')
    legend_4 = plt.legend(loc = 'center right', bbox_to_anchor=(1,0.5))
    plt.title('Motion of Photon for periodic eigenstates for $\phi$ = $3\pi/2$')

