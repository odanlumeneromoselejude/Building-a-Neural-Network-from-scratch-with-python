import numpy as np


def sigmoid(x, back_propagation=False):
    if back_propagation:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1/(1+np.exp(-x))


def tanh(x, back_propagation=False):
    if back_propagation:
        return 1 - (tanh(x)) ** 2
    else:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    if relu(x) < 0:
        return 0
    else:
        return relu(x)
