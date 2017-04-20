#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

import sage.all

import data_generator
import symbolic_plot

def plot_loss(errors, num_epochs):
    x_vec = np.arange(0, num_epochs)[: , np.newaxis]
    loss_plot = sage.all.line(np.hstack((x_vec, errors)))
    sage.all.show(loss_plot, axes_labels=['Number of Epochs','Loss'],
                  title='Loss Function as Training Progresses')

def plot_classification_surfaces(X, y, W, num_classes):
    Ws = sage.all.Matrix(list(W))
    xs, ys = sage.all.var('xs ys')
    Xs = sage.all.Matrix([[xs], [ys], [1]])

    Zs = Ws * Xs
    x_lim = 1
    y_lim = 1
    plot_colors = ['blue','red']
    bg_colors = ['#89CFF0', '#FFC0CB']
    surfaces = [0, Zs[0, 0]]
    reg_plots = symbolic_plot.get_region_plots(surfaces, num_classes,
                                               x_lim, y_lim, bg_colors)
    scatter_plots = symbolic_plot.get_scatter_plots(X, y, num_classes,
                                                    plot_colors)

    sage.all.show(sum(reg_plots) + sum(scatter_plots))

N = 50
num_dims = 2
num_classes = 2
num_examples = num_classes * N

X, y = data_generator.generate_spiral_data(N, num_dims, num_classes)
X_p = np.hstack((X, np.ones((X.shape[0], 1))))

W = np.random.rand(num_dims+1)
delta = 0.2
num_epochs = 100
errors = np.zeros((num_epochs, 1))

for i in xrange(num_epochs):
    pred = np.array(np.dot(X_p, W) >= 0.0, dtype=np.int)
    error = y - pred
    errors[i] = np.sum(np.abs(error))
    W += delta * np.dot(X_p.T, error)

pred = np.array(np.dot(X_p, W) >= 0.0, dtype=np.int)
print('Training accuracy: {0}'.format(np.mean(pred == y)))

plot_loss(errors, num_epochs)
plot_classification_surfaces(X, y, W, num_classes)

