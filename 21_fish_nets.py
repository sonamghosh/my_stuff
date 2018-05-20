# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:19:31 2018

@author: sonam
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from sklearn import model_selection
from random import shuffle
import sys
import os
from scipy.special import expit

# Activation Functions
def relu(x):
    x[x < 0] = 0
    return x

def drelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.power(np.tanh(x), 2)

def softmax(x):
    e = np.exp(x - np.max(x))
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T

def dsoftmax(x):
    p = softmax(x)
    ds = np.empty((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            if i == j:
                ds[i][j] = p[i] * (1 - p[i])
            else:
                ds[i][j] = -1 * p[i] * p[j]
    return np.array(ds)

def identity(x):
    return x

def didentity(x):
    return 1/np.array([1] * len(x))

# Loss Functions
def cross_entropy(g_x, y):
    return -1 * np.dot(y, map(lambda x: np.log(x), g_x))

def dcross_entropy(g_x, y):
    return -1 * np.multiply(y, map(lambda x: 1/x, g_x))

def squared_loss(g_x, y):
    return 0.5 * np.power(y - g_x, 2)

def dsquared_loss(g_x, y):
    return g_x - y

def softmax_loss(y, y_hat):
    min_val = 0.000000000000000001
    # Number of samples
    m = y.shape[0]
    # Loss formula
    loss = -1/m * np.sum(y * np.log(y_hat.clip(min=min_val)))
    return loss


class Function:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative

    def __call__(self, x, y=None):
        if y != None:
            return self.func(x,y)
        else:
            return self.func(x)

relu_f = Function(relu, drelu)
sig_f = Function(sigmoid, dsigmoid)
soft_f = Function(softmax, dsoftmax)
tanh_f = Function(tanh, dtanh)
cross_ent_loss_f = Function(cross_entropy, dcross_entropy)
squared_loss_f = Function(squared_loss, dsqared_loss)
id_f = Function(identity, didentity)

class Layer:
    def __init__(self, weight, bias, act_func=None):
        self.w = weight
        self.b = bias
        self.A = act_func
        self.num_neurons = self.w.shape[0]
    def __call__(self, x):
        lin_step = np.dot(self.w, x) + self.b
        return lin_step, self.A(lin_step)


class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output,
                 n_hidden_layer=1, act_func=sig_f,
                 output_func=None, loss_func=None, random_seed=None):
        if random_seed:
            np.random_seed(random_seed)
        # Features
        self.n_input = n_input
        # Hidden Neurons
        self.n_hidden = n_hidden
        # Output Neurons
        self.n_output = n_output
        # Activation function for output layer
        self.output_func = output_func
        # Holds all the information of each layer
        self.layers = []
        # Total layers
        self.n_hidden_layer = n_hidden_layer
        self.tot_layers = n_hidden_layer + 2
