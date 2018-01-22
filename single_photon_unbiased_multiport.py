# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:45:16 2018

@author: sonam
"""

# Python 3.6 Implementation of Casey's Quantum Code

import numpy as np
import seaborn as sb
from cython.parallel import prange
from string import ascii_lowercase, ascii_uppercase  # Vertex Labeling
from collections import deque
from qutip import Qobj
from scipy.linalg import block_diag
from numpy.random import random_sample as rand
from collections import OrderedDict  # phase_sweep method on network
from copy import deepcopy  # For resetting network phases
from itertools import cycle  # for plotting and labeling
from cycler import cycler  # for plotting


# Set default phase angle for any unspecified vertex.
# Typically value from PRA paper, -i = exp(-i*pi/2)
global_default_phase_angle = -1*(np.pi/2)


class Vertex(object):
    def __init__(self, phase_angle=global_default_phase_angle):
        self.matrix = self._make_matrix(phase_angle)
        self.labels = self._make_label()

    def _make_matrix(self, phase_angle):
        phi = phase_angle

        def ei(x):
            return np.complex128(np.exp(1j*x))
        non_zero_entries = {
                            (0, 1): 1j, (0, 2): 1,
                            (1, 0): 1j, (1, 3): 1,
                            (2, 0): 1, (2, 3): 1j,
                            (3, 1): ei(phi), (3, 2): 1j*ei(phi)
                            }
        base_matrix = np.zeros((4, 4), dtype=np.complex128)
        for idx, non_zero_entry in non_zero_entries.items():
            base_matrix[idx] = non_zero_entry
        return np.multiply(1.0/np.sqrt(2), base_matrix)

    def _make_label(self):
        labels = {}
        # get columns by getting rows of transpose
        for i, row in enumerate(self.matrix):
            for j, val in enumerate(row):
                if val != 0:
                    labels[(str(i+1), str(j+1))] = val
        return labels


class SinglePhotonUnbiasedMultiPort(object):
    def __init__(self, num_vert=3, vertex_phase_angles=None,
                 transient_params=None):
        # Create matrices and quantum objects based on input parameters
        self.number_of_vertices = num_vert
        self.vertex_labels = ascii_uppercase[0:num_vert]
        self.state_labels = self._create_state_labels()
        self.state_label_dict = self._create_state_label_dict()
        self.label_neighbors_dict = self._create_neighbors_dict()
        self.vertex_phase_angles = self._create_vertex_phase_angles(
                vertex_phase_angles)
        self.base_matrix = self._create_base_matrix()
        self.iterated_matrix = self._create_iterated_matrix(transient_params)
        self.reduced_matrix = self._create_reduced_matrix()
        self.quantum = self._create_quantum()
        self.reduced_state_labels = self._create_reduced_state_labels()
        self.reduced_state_dict = self._create_reduced_state_dict()

    def _create_state_labels(self):
        num_vert = self.number_of_vertices
        # Implement a way in the future for >26 vertices multiport
        if num_vert > len(ascii_uppercase):
            raise KeyError(' This code does not generate multiports with more\
                           than one 26 vertices.')
        if num_vert < 3:
            raise KeyError('Multiport must have atleast 3 verices.')

        nums = cycle(['1', '3', '2', '4', 'X'])  # Iterable 1,3,2,4,X,1,3,2..
        label_array = np.chararray((num_vert, 5))  # num_vert by 5 arr of ''
        for row in prange(num_vert):
            vert = deque(self.vertex_labels)
            vert.rotate(-row)
            labs = list(vert)
            labs = [labs[x] for x in [0,1,-1]]  # connected vertices NN
            labs.extend(labs[0]*2)
            # Creates 3 by 5 arr of ABCAA, BCABB, CABCC
            label_array[row,:] = labs

        label_array = label_array.flatten().tolist()  # ABCAABCABBCABCC
        label_array = [label_array[i].decode() for i in
                       prange(len(label_array))]  # Convert Bytes to Str
        return [i+next(nums) for i in label_array]  # A3, B2, C4, AX etc

    def _create_state_label_dict(self):
        return {lab: val for (val, lab) in enumerate(self.state_labels)}

    def _create_neighbors_dict(self):
        vertex_labels = self.vertex_labels
        label_neighbors = {}
        for (i, vertex) in enumerate(vertex_labels):
            vert = deque(vertex_labels)
            vert.rotate(-i)
            labs = list(vert)
            # Create { A: [B , C], B: [C, A], C: [A, B]}
            label_neighbors[vertex] = [labs[x] for x in [1, -1]]
        return label_neighbors

    # @todo: Review all code below this line
    def _create_vertex_phase_angles(self, vertex_phase_angles=None):
        num_vert = self.number_of_vertices
        vertex_labels = self.vertex_labels

        # Fill with default (PRA paper) values if no phase angles given
        if vertex_phase_angles is None:
            return {label: global_default_phase_angle for label
                    in vertex_labels}
        # Check Type
        if not isinstance(vertex_phase_angles, dict):
            raise TypeError('vertex_phase_angles must be a dict of the form\
                            vertex_label:phase_angle')

        # Convert keys to uppercase if needed
        for key, val in vertex_phase_angles.items():
            vertex_phase_angles[key.upper()] = vertex_phase_angles.pop(key)

        # Check if labels correspond to multiport vertices
        if not all(key in vertex_labels for key in vertex_phase_angles.keys()):
            status = [key in vertex_labels for key in
                      vertex_phase_angles.keys()]
            bad_keys = [key for i, key in
                        enumerate(vertex_phase_angles.keys())
                        if status[i] is False]
            raise ValueError(' Some keys not found in vertex labels -->'
                             + str(bad_keys))

        # Fill in missing phase angles with default (PRA paper) values.
        if set(vertex_phase_angles.keys()).intersection(set(vertex_labels))\
                != set(vertex_labels):
                for label in vertex_labels:
                    if label not in vertex_phase_angles.keys():
                        vertex_phase_angles[label] = global_default_phase_angle
        return vertex_phase_angles

    def _rekey_vertex_labels(self, vertex_label_dict, vertex_input_label_key):
        state_labels = self.state_labels
        neighbors = self.label_neighbors_dict[vertex_input_label_key]
        new_dict = {}
        for key, val in vertex_label_dict.items():
            new_key = ['', '']
            new_key[1] = vertex_input_label_key+key[1]
            if key[0] == '1':
                new_key[0] = vertex_input_label_key + 'X'
            if key[0] == '4':
                new_key[0] = vertex_input_label_key + key[0]
            if key[0] == '2':
                new_key[0] = neighbors[1] + key[0]
            if key[0] == '3':
                new_key[0] = neighbors[0] + key[0]

            # The following lines compensate for changes to the vertex matrix
            # needed for connecting vertices
            # EG A4 to B3 should be 1, not i, but the code doesn't distinguish
            # this without the following if statement:
            if new_key[0][0] != new_key[1][0]:  # going from 1 vertex to other
                if np.imag(val) != 0:
                    val = 1/np.sqrt(2)
                else:
                    val = 1j*val
            new_dict[tuple(new_key)] = val
        return new_dict

    def _label_tup_to_dim(self, label_tup):
        state_labels = self.state_label_dict
        return (state_labels[label_tup[0]], state_labels[label_tup[1]])

    def _create_base_matrix(self):
        num_vert = self.number_of_vertices
        base_matrix = np.zeros((5*num_vert, 5*num_vert), dtype=np.complex128)
        # Build Non-Zero Entries
        non_zero_entries = {}
        for vertex in self.vertex_labels:
            phase_angle = self.vertex_phase_angles[vertex]
            vertex_single = Vertex(phase_angle=phase_angle)
            vertex_entry_dict = vertex_single.labels
            vertex_entry_dict = self._rekey_vertex_labels(vertex_entry_dict,
                                                          vertex)
            for label_tup, val in vertex_entry_dict.items():
                dims = self._label_tup_to_dim(label_tup)
                non_zero_entries[dims] = val
            # Add exit entry
            exit_dim = self._label_tup_to_dim((vertex + 'X', vertex + 'X'))
            non_zero_entries[exit_dim] = 1
        for idx, non_zero_entry in non_zero_entries.items():
            base_matrix[idx] = non_zero_entry
        return base_matrix

    def _create_iterated_matrix(self, transient_params=None):
        # iterate out transient states according to transient_params
        if transient_params:
            num_iter = transient_params[0]
            thresh = transient_params[1]
        else:
            # default values
            num_iter = 1000
            thresh = 10e-3
        # iterate
        iterated_matrix = np.linalg.matrix_power(self.base_matrix, num_iter)
        # threshold
        iterated_matrix[np.absolute(iterated_matrix) < thresh] = 0
        return iterated_matrix

    def _create_reduced_matrix(self):
        row_slice = [self.state_label_dict[state] for state in
                     self.state_labels if state[-1] == 'X']  # output rows
        col_slice = [self.state_label_dict[state] for state in
                     self.state_labels if state[-1] == '1']  # input columns

        # copy iterated
        reduced_matrix = self.iterated_matrix.copy()
        # reduce to unitary
        reduced_matrix = reduced_matrix[np.ix_(row_slice, col_slice)]
        return reduced_matrix

    def _create_reduced_state_labels(self):
        return [state[:-1] for state in self.state_labels if state[-1] == 'X']

    def _create_reduced_state_dict(self):
        return {state: dim for dim, state in
                enumerate(self.reduced_state_labels)}

    def _create_quantum(self):
        return {'base_matrix': Qobj(self.base_matrix),
                'iterated_matrix': Qobj(self.iterated_matrix),
                'reduced_matrix': Qobj(self.reduced_matrix)}

    def check_unitary_node(self, thresh=10e-3, decimal=3, display=False):
        op = self.quantum['reduced_matrix']
        should_be_id = op.dag()*op
        should_be_id = should_be_id.full()  # Params for full()?
        should_be_id[np.absolute(should_be_id) < thresh] = 0
        # Get shape for comparison
        shape = should_be_id.shape
        try:
            np.testing.assert_array_almost_equal(should_be_id,
                                                 np.eye(shape[0]), decimal)
            result = True
        except AssertionError:  # Look up later
            result = False
        if display:
            print('Reduced node operator unitary:   ', result)
            print('thresh                       :    ', thresh)
            print('decimal                      :    '. decimal)
            print('op.dag()*op                  :\n')
            print(np.absolute(should_be_id))
        return result

    def update_vertex_phase_angles(self, new_vertex_phase_angles=None,
                                   transient_params=None):
        vertex_labels = self.vertex_labels
        if new_vertex_phase_angles is None:
            print('Nothing to update')
            return None
        if not instance(new_vertex_phase_angles, dict):
            raise TypeError('Vertex phase updates must be entered as a dict')
        # Convert Keys to Uppercase if needed
        for key, val in new_vertex_phase_angles.items():
            new_vertex_phase_angles[key.upper()] =\
                new_vertex_phase_angles.pop(key)
        # Validate keys
        if not all(key in vertex_labels for key in
                   new_vertex_phase_angles.keys()):
            status = [key in vertex_labels for key in
                      new_vertex_phase_angles.keys()]
            bad_keys = [key for i, key in
                        enumerate(new_vertex_phase_angles.keys())
                        if status[i] is False]
            raise ValueError('Some keys not found in vertex_labels --->'
                             + str(bad_keys))
        # Set New values
        for old_key, new_val in new_vertex_phase_angles.items():
            self.vertex_phase_angles[old_key] = new_val
        # Propogate updates to the matrices
        self.base_matrix = self._create_base_matrix()
        self.iterated_matrix = self._create_iterated_matrix(transient_params)
        self.reduced_matrix = self._create_reduced_matrix()
        self.quantum = self._create_quantum()

    def randomly_update_phase_angles(self):
        rand_phase = 2*np.pi*rand(self.number_of_vertices,)
        new_vertex_phase_angles = {vertex: rand_phase[i] for i, vertex in
                                   enumerate(self.vertex_phase_angles.keys())}
        self.update_vertex_phase_angles(new_vertex_phase_angles)

    def cumulative_exit_probability(self, num_):  # Why pass?
        pass

    def get_number_vertices(self):  # Also why?
        return self.number_of_vertices

    def get_base_matrix(self, qobj=True):
        if qobj:
            return self.quantum['base_matrix']
        if not qobj:
            return self.base_matrix

    def get_iterated_matrix(self, qobj=True):
        if qobj:
            return self.quantum['iterated_matrix']
        if not qobj:
            return self.iterated_matrix

    def get_reduced_matrix(self, qobj=True):
        if qobj:
            return self.quantum['reduced_matrix']
        if not qobj:
            return self.reduced_matrix

    def get_state_labels(self):
        return self.state_labels

    def get_reduced_state_labels(self):
        return self.reduced_state_labels

    def get_reduced_state_dict(self):
        return self.reduced_state_dict

    def view_base_matrix_heatmap(self):
        return sb.heatmap(np.absolute(self.base_matrix))

    def view_iterated_matrix_heatmap(self):
        return sb.heatmap(np.absolute(self.iterated_matrix))

    def view_reduced_matrix_heatmap(self):
        return sb.heatmap(np.absolute(self.reduced_matrix))
