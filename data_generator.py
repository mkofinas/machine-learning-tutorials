#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def generate_spiral_data(N, D, K, seed=0):
    np.random.seed(seed)
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype='uint8')
    for j in xrange(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y

def generate_parabolic_data(N, D, K, seed=0):
    np.random.seed(seed)
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype='uint8')
    for j in xrange(K):
        ix = range(N * j, N * (j + 1))
        x_c = np.linspace(-1, 1, N)
        y_c = x_c ** 2 - 0.5 * j
        X[ix] = np.c_[x_c, y_c]
        y[ix] = j
    return X, y

def generate_circular_data(N, D, K, seed=0):
    np.random.seed(seed)
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype='uint8')
    for j in xrange(K):
        ix = range(N * j, N * (j + 1))
        r = (j + 1) * np.ones((N,)) + np.random.randn(N) * 0.1
        t = np.linspace(0, 2 * np.pi, N)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y
