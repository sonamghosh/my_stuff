# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 00:59:22 2018

@author: sonam
"""

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse
from cython.parallel import prange

# Activation Function 
# ON if s(x) > 0.5 , OFF if s(x) < 0.5 for inputs x
def sigmoid_activation(x):
    return 1.0/ (1 + np.exp(-x))


# Command Line Parser
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type=float, default=100, help='# of epochs')
ap.add_argument('-a', '--alpha', type=float, default=0.01,
                help='learning rate')
args = vars(ap.parse_args())


# Create 2-Class classification with 250 data points
# Each data point corresponds to a 2D feature vector
# make_blobs generates N data points that are 2D
"""
    125 of the data points belong to CLass 0 , the other 125 belong to
    class 1. Goal is to train the classified such that it knows whether the
    point is CLass 0 or Class 1
"""
(X, y) = make_blobs(n_samples=250, n_features=2, centers=2,
                    cluster_std=1.05, random_state=20)

# Insert Column vector of 1s for first entry in feature vector
# Allows to treat bias as trainable paramter inside weight maatrix than
# a different variable
X = np.c_[np.ones((X.shape[0])), X]

# Initialize Weight Matrix (Should have same no. of columns as input fetures)
print('[INFO] starting training...')
W = np.random.uniform(size=(X.shape[1],))

# Initiliaze list to store loss value per epoch
lossInfo = []

# Loop over given amount of epochs
for epoch in prange(0, args['epochs']):
    # Dot Product between the full training data X and the weight matrix W
    # Take the dot product output and feed the values through the activation
    # function to give prediction values.
    preds = sigmoid_activation(X.dot(W))
    # Find the Error Difference between Prediction and True Values
    error = preds - y
    # Find squared loss sum of error
    loss = np.sum(error**2)
    lossInfo.append(loss)
    print('[INFO] epoch #{}, loss={:.7f}'.format(epoch+1, loss))
    # Compute Gradient by X.T dotp Error scaled by no. of data points in X
    gradient = X.T.dot(error)/X.shape[0]
    # Move the weight matrix in the neg. direction of the gradient
    W += -args['alpha'] * gradient

# Use Weight Matrix as a Classifier through the training sample
for i in np.random.choice(250, 10):
    """
    Compute Prediction via dot product of feature vector with
    weight matrix W, followed by passing it through the activation function
    """
    activation = sigmoid_activation(X[i].dot(W))
    """
    Specify classification, sigmoid function is in the range y = [0,1]
    Classify y>=0.5 as Class 1 , y < 0.5 as Class 0
    """
    label = 0 if activation < 0.5 else 1
    # Display Output Classification
    print('activation={:.4f}; predicted_label={}, true_label={}'.format(
            activation, label, y[i]))

# Compute Line of Best Fit by setting Sigmoidal Function to 0
# and solving for X2 in terms of X1
Y = (-W[0] - (W[1] * X))/W[2]

# Plot Original Data with Line of Best Fit
plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker='o', c=y)
plt.plot(X, Y, 'r-')

# Plot Loss Over Time
fig = plt.figure()
plt.plot(np.arange(0, args['epochs']), lossInfo)
fig.suptitle('Training Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()
