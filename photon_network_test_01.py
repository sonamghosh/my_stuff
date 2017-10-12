# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 00:23:32 2017

@author: sonam
"""

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from string import ascii_lowercase, ascii_uppercase  #vertex
from collections import deque
from collections import OrderedDict
from qutip import Qobj
from scipy.linalg import block_diag
from numpy.random import random_sample as rand
from copy import deepcopy
import itertools as it
from cycler import cycler  #plotting