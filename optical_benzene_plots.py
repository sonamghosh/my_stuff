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

phi_angles = np.arange(0, 2*np.pi, np.pi/16)
colors = sb.color_palette("hls", 12)
eig_dict = []


for i in phi_angles:
    photon = gt.unitcell_hamiltonian(phi = i)
    eig_dict.append(photon.pos_eig['Eigenenergies'])
    
# Consider making interactive plot with Blokeh or Plotly

for i in prange(12):
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
    
