#!/usr/bin/env python
# -*- coding: utf-8 -*-

###########################
## EXPERIMENTAL/UNFINISHED
###########################

from __future__ import print_function

import numpy as np

from sage.all import *

import neural_network
import data_generator
import symbolic_plot
from symbolic_parametric_transformation import *

N = 100 # number of points per class
num_dims = 2 # dimensionality
num_classes = 3 # number of classes

X, y = data_generator.generate_circular_data(N, num_dims, num_classes, seed=0)

# Hyperparameters
hidden_layers_sizes = [2]
learning_rate = 1e-0
regularization_strength = 1e-3
num_epochs = 10000
activation_function = 'tanh'

neural_net = neural_network.NeuralNetwork(
        num_dims, num_classes, hidden_layers_sizes, activation_function,
        num_epochs, learning_rate, regularization_strength, seed=1)
neural_net.train(X, y, print_loss=1000)
# evaluate training set accuracy
predicted_class = neural_net.predict(X)
print('Training accuracy: {0}'.format(np.mean(predicted_class == y)))

# plot the resulting classifier
neural_net.plot_classification_surfaces(X, y)

t = var('t')
t_lim = 3.15
t_limits = (t, -t_lim, t_lim)
num_lines = 21

f = (cos(t), sin(t))
g = (2 * cos(t), 2 * sin(t))
h = (3 * cos(t), 3 * sin(t))

Ws1 = Matrix(np.concatenate((neural_net.W[0], neural_net.b[0])).T)
Ws2 = Matrix(np.concatenate((neural_net.W[1], neural_net.b[1])).T)
affine_transformation = Matrix(Ws1)
non_linear_transformations = [tanh, tanh, tanh]

xs, ys = sage.all.var('xs ys')
Xs = sage.all.Matrix([[xs], [ys], [1]])

bg_colors = ('#FFC0CB', '#89CFF0', '#90EE90')

Hs = Ws1 * Xs
Hs = Hs.apply_map(tanh)
Hs = Hs.stack(sage.all.ones_matrix(1, 1))
separating_planes = Ws2 * Hs
separating_plane_plot = symbolic_plot.get_region_plots(
        separating_planes, num_classes, t_lim, t_lim, bg_colors)

planes, boundary_planes = cartesian_grid(t, t_lim, num_lines)

grid_colors = ('gray', 'black')
grid_thickness = (1.0, 2.0)
plane_plots, boundary_plane_plots = grid_plots(
        (planes, boundary_planes), t_limits, grid_colors, grid_thickness)
function_colors = ('red', 'blue', 'green')
function_thickness = (1.5, 1.5, 1.5)
f_plot, g_plot, h_plot = function_plots((f, g, h), t_limits, function_colors,
                                        function_thickness)

## Transformations
parametric_transformation = ParametricTransformation(
        affine_transformation, non_linear_transformations)

transformed_planes = [parametric_transformation.transform(*plane) for plane in planes]
transformed_boundary_planes = [parametric_transformation.transform(*plane) for plane in boundary_planes]
transformed_f = parametric_transformation.transform(*f)
transformed_g = parametric_transformation.transform(*g)
transformed_h = parametric_transformation.transform(*h)
transformed_separating_planes = Ws2 * Xs

transformed_plane_plots, transformed_boundary_plane_plots = grid_plots(
        (transformed_planes, transformed_boundary_planes),
        t_limits, grid_colors, grid_thickness)
transformed_f_plot, transformed_g_plot, transformed_h_plot = function_plots(
        (transformed_f, transformed_g, transformed_h),
        t_limits, function_colors, function_thickness)

analytic_planes = [0.0 for _ in range(len(transformed_boundary_planes))]
xy_limits = [0.0 for _ in range(len(transformed_boundary_planes))]
for i in range(len(transformed_boundary_planes)):
    sol = solve(xs == transformed_boundary_planes[i][0], t)[0]
    yV = transformed_boundary_planes[i][1](sol.rhs())
    analytic_planes[i] = yV

    x_lims = transformed_boundary_planes[i][0](-t_lim), transformed_boundary_planes[i][0](t_lim)
    if x_lims[0] > x_lims[1]:
        x_lims = [x_lims[1], x_lims[0]]
    y_lims = transformed_boundary_planes[i][1](-t_lim), transformed_boundary_planes[i][1](t_lim)
    if y_lims[0] > y_lims[1]:
        y_lims = [y_lims[1], y_lims[0]]
    xy_limits[i] = [(xs, x_lims[0], x_lims[1]), (ys, y_lims[0], y_lims[1])]

transformed_separating_plane_plots = []
for i in range(len(transformed_boundary_planes)):
    # if i < len(transformed_boundary_planes) / 2:
        for j in range(3):
            transformed_separating_plane_plots.append(region_plot(
                transformed_separating_planes[j, 0] > 0,
                xy_limits[i][0], xy_limits[i][1], incol=bg_colors[0]))
    # else:
        # for j in range(3):
            # transformed_separating_plane_plots.append(region_plot(
                # [analytic_planes[i] > ys, transformed_separating_planes[j, 0] > 0],
                # xy_limits[i][0], xy_limits[i][1], incol=bg_colors[0]))

show(sum(plane_plots) +
     f_plot +
     g_plot +
     h_plot +
     sum(boundary_plane_plots) +
     sum(separating_plane_plot))

show(sum(transformed_plane_plots) +
     transformed_f_plot +
     transformed_g_plot +
     transformed_h_plot +
     sum(transformed_boundary_plane_plots) +
     sum(transformed_separating_plane_plots))

