# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 00:59:22 2018

@author: sonam
"""

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse

# Activation Function 
# ON if s(x) > 0.5 , OFF if s(x) < 0.5 for inputs x
def sigmoid_activation(x):
    return 1.0/ (1 + np.exp(-x))

