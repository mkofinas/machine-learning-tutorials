#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def activation(x, activation_function):
    if activation_function == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation_function == 'tanh':
        return np.tanh(x)
    elif activation_function == 'relu':
        return np.maximum(0, x)

def d_activation(x, activation_function):
    if activation_function == 'sigmoid':
        s_x = self.sigmoid(x)
        return s_x * (1 - s_x)
    elif activation_function == 'tanh':
        return 1 - np.tanh(x) ** 2
    elif activation_function == 'relu':
        return (x > 0).astype(int)

