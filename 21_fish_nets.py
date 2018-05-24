# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:19:31 2018

@author: sonam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
#from itertools import izip

from sklearn import model_selection
from random import shuffle


# Turns arrays into diagonal matrix
def diag(x):
    mat = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        mat[i][i] = x[i]
    return mat


# Activation Functions and their respective derivatives
def relu(x):
    x[x < 0] = 0
    return x


def drelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def sigmoid(x):
    return 1.0/(1.0+np.exp(-(x.astype(float))))


def dsigmoid(x):
    s = sigmoid(x)
    return diag(s * (1 - s))


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
        for j in range(len(x)):
            if i == j:
                ds[i][j] = p[i] * (1 - p[i])
            else:
                ds[i][j] = -1 * p[i] * p[j]
    return np.array(ds)


def identity(x):
    return x


def didentity(x):
    return diag(1/np.array([1] * len(x)))


# Loss Functions
def cross_entropy(g_x, y):
    return -1 * np.dot(y, list(map(lambda x: np.log(x), g_x)))


def dcross_entropy(g_x, y):
    return -1 * np.multiply(y, list(map(lambda x: 1/x, g_x)))


def squared_loss(g_x, y):
    return 0.5 * np.power(g_x - y, 2)


def dsquared_loss(g_x, y):
    return g_x - y


def softmax_loss(y, y_hat):
    min_val = 0.000000000000000001
    # Number of samples
    m = y.shape[0]
    # Loss formula
    loss = -1/m * np.sum(y * np.log(y_hat.clip(min=min_val)))
    return loss


# Class that handles the value inputs of activation & loss functions
# and thei respective derivatives
class Function:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative

    def __call__(self, x, y=None):
        if y != None:
            return self.func(x,y)
        else:
            return self.func(x)

# Instances of all activation functions and loss functions
relu_f = Function(relu, drelu)
sig_f = Function(sigmoid, dsigmoid)
soft_f = Function(softmax, dsoftmax)
tanh_f = Function(tanh, dtanh)
cross_ent_loss_f = Function(cross_entropy, dcross_entropy)
squared_loss_f = Function(squared_loss, dsquared_loss)
id_f = Function(identity, didentity)


# Learning Schedule Hyperparameter for Stochastic Gradient Descent
def eta(t, tau_0=100, kappa=0.75):
    return (tau_0 + t)**(-1 * kappa)



# Implementation of a layer of neurons in a neural network
class Layer:
    def __init__(self, weight, bias, act_func):
        self.w = weight
        self.b = bias
        self.A = act_func
        self.num_neurons = self.w.shape[0]
    def __call__(self, x):
        # Forward Propagation
        lin_step = np.dot(self.w, x) + self.b
        return lin_step, self.A(lin_step)


# Handles all the training
class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output,
                 act_func=[sig_f],
                 output_func=None, loss_func=None, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        # Features
        self.n_input = n_input
        # Hidden Neurons
        self.n_hidden = n_hidden
        # Output Neurons
        self.n_output = n_output
        # Activation function for output layer
        self.output_func = output_func
        # Loss Function
        self.loss = loss_func
        # Holds all the information of each layer
        self.layers = []
        # Total layers
        #self.n_hidden_layer = n_hidden_layer
        #self.tot_layers = n_hidden_layer + 2
        self.L = len(n_hidden) + 2
        
        # Instantitate Input layer
        input_layer = Layer(np.identity(self.n_input),
                            np.zeros(self.n_input), id_f)

        self.layers.append(input_layer)
        
        # instantitate all hidden layers 
        for neurons, func in zip(n_hidden, act_func):
            prev_neuron = self.layers[-1].num_neurons
            weight_mat = self.init_weights((neurons, prev_neuron))
            biases = self.init_weights((neurons, ))
            current_layer = Layer(weight_mat, biases, func)
            self.layers.append(current_layer)

        # Instantiate Output layer
        out_prev_neuron = self.layers[-1].num_neurons
        out_weight_mat = self.init_weights((n_output, out_prev_neuron))
        out_bias = self.init_weights((n_output, ))
        output_layer = Layer(out_weight_mat, out_bias, output_func)
        self.layers.append(output_layer)


    def __call__(self, x):
        return self.predict(x)
    
    
    # Create the weights for the neurons
    def init_weights(self, shape):
        return np.random.normal(0, 1./np.sqrt(shape[0]), shape)

    # Calculate the prediction per layer 
    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)[1]
        return output

    
    # Training Method that calls the backpropagation through
    # Stochastic gradient descent
    def train(self, train_data, val_data, test_data,
              step=2.0, epochs=50, learning_time=eta):
        self.sgd(train_data, step, epochs, learning_time, val_data, test_data)

    # Calculates the average loss per prediction
    def avg_loss(self, data):
        avg_loss = 0
        for (x, y) in data:
            prediction = self.predict(x)
            avg_loss += self.loss(prediction, y)
        avg_loss /= len(data)
        return avg_loss

    # Calculate the accuracy for each set of data
    def check_acc(self, training_data, valid_data, test_data):
        correct_training = 0.
        correct_val = 0.
        correct_test = 0.
        for (x, y) in valid_data:
            prediction = np.argmax(self.predict(x))
            print("valid prediction: ", prediction)
            print("y = ", y)
            #if y[prediction] == 1:
            if y == prediction:
                correct_val += 1
        for (x, y) in training_data:
            prediction = np.argmax(self.predict(x))
            print(" training prediction: ", prediction)
            print("y = ", y)
            #if y[prediction] == 1:
            if y == prediction:
                correct_training += 1
        for (x, y) in test_data:
            prediction = np.argmax(self.predict(x))
            #if y[prediction] == 1:
            if y == prediction:
                correct_test += 1
        return correct_training / len(training_data), correct_val / len(valid_data), correct_test / len(test_data)

    # Stochastic Gradient Descent method for Backpropagation
    def sgd(self, train_data, step, epochs, learning_time, val_data, test_data):
        t = 0
        init_acc = self.check_acc(train_data, val_data, test_data)
        val_accs = []
        print("Initial Validation Accuracy: ", init_acc[0], init_acc[1], init_acc[2])
        val_accs.append(init_acc[1])
        # Iterate over time while updating the learning schedule
        for epoch in range(1, epochs+1):
            shuffle(train_data)
            for x, y in train_data:
                self.update_weights(x, y, step * learning_time(t))
                t += 1
            accuracies = self.check_acc(train_data, val_data, test_data)
            print(epoch, " Training Validation Test Accuracy: ",
                  accuracies[0], accuracies[1], accuracies[2])
            #self.which_n = accuracies[1]
            val_accs.append(accuracies[1])
            print("Loss: ", self.avg_loss(train_data))
            #plt.plot(epoch, self.avg_loss(train_data)[1])
            
            # Break validation if converged to a value
            if len(val_accs) > 10 and val_accs[epoch - 10] >= val_accs[epoch]:
                print(val_accs)
                break

    # Updates the weights by taking gradients of the weight and biases
    def update_weights(self, x, y, step):
        dW, db = self.grad(x, y)
        for i in range(1, self.L):
            layer = self.layers[i]
            layer.w -= step * dW[i]
            layer.b -= step * db[i]

    # Compute the gradient 
    def grad(self, x, y):
        L = self.L
        Z = []
        A = []
        err = [np.empty((1)) for _ in range(L)]
        output = x
        for layer in self.layers:
            z, a = layer(output)
            Z.append(z)
            A.append(a)
            output = a
        #print("L = ", L)

        # Debugging
        #print("Size of err array: ", len(err), "\n")
        #print("Size of Z: ", len(Z), "\n")
        #print("Size of A: ", len(A), "\n")
        try:
            err[L-1] = np.dot(self.output_func.derivative(Z[L-1]),
                              self.loss.derivative(A[L-1], y))
        except IndexError:
            print("Index: ", L, "caused an error \n")

        for i in range(L - 2, 0, -1):
            layer = self.layers[i]
            err[i] = np.dot(self.layers[i+1].w.T, err[i+1])
            err[i] = np.dot(layer.A.derivative(Z[i]), err[i])
        dW = [np.empty((1)) for _ in range(L)]
        db = [np.empty((1)) for _ in range(L)]
        for i in range(1, L):
            db[i] = err[i]
            #dW[i] = np.dot(A[i-1], err[i])
            dW[i] = np.dot(np.reshape(A[i-1], (A[i-1].shape[0],1)),
                           np.reshape(err[i], (1, err[i].shape[0]))).T
        return dW, db


#################################
# DEMO with Iris Training Set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

array = dataset.values
X = array[:, 0:4].astype(float)
Y = array[:, 4]

encoder = LabelEncoder()
encoded_y = encoder.fit_transform(Y)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_Y = encoded_y.reshape(len(encoded_y), 1)
onehot_Y = onehot_encoder.fit_transform(onehot_Y)


validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = \
model_selection.train_test_split(X, encoded_y, test_size=validation_size,
                                 random_state=seed)

test = NeuralNetwork(4, [5, 5], 3, [sig_f, sig_f], soft_f, cross_ent_loss_f, random_seed=4)
train_data = list(zip(X_train, Y_train))
val_data = list(zip(X_validation, Y_validation))
test_data = list(zip(X, encoded_y))
test.train(train_data, val_data, test_data)

"""
for i in range(200):
    test = NeuralNetwork(4, [i, i], 3, [sig_f, sig_f], soft_f, cross_ent_loss_f, random_seed=4)
    train_data = list(zip(X_train, Y_train))
    val_data = list(zip(X_validation, Y_validation))
    test_data = list(zip(X, encoded_y))
    test.train(train_data, val_data, test_data)
    if test.which_n > 0.9:
        print("neuron: ", test.which_n)
        break
        """