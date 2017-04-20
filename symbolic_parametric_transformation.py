#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sage.all import *

class ParametricTransformation(object):
    def __init__(self, affine_transformation, non_linear_transformations):
        self.affine_transformation = copy(affine_transformation)
        self.non_linear_transformations = copy(non_linear_transformations)

    def transform(self, t, f):
        X_mat = Matrix([[t], [f], [1]])
        affine_transform = self.affine_transformation * X_mat
        transform = tuple(affine_transform[i].apply_map(self.non_linear_transformations[i])[0]
                          for i in range(len(affine_transform.rows())))
        return transform

def get_parametric_plots(planes, limits, **kwds):
    return [parametric_plot(plane, limits, **kwds) for plane in planes]

def cartesian_grid(t, t_lim, num_lines):
    xx = np.linspace(-t_lim, t_lim, num_lines)

    planes = []
    boundary_planes = []
    for i, x in enumerate(xx):
        v = vector((x, t))
        h = vector((t, x))
        if i == 0 or i == (num_lines-1):
            boundary_planes.append(v)
            boundary_planes.append(h)
        else:
            planes.append(v)
            planes.append(h)
    return planes, boundary_planes

def grid_plots(planes, t_limits, colors, thickness):
    plots = []
    for i in range(len(planes)):
        plots.append(get_parametric_plots(planes[i], t_limits, color=colors[i],
                                          thickness=thickness[i]))
    return plots

def function_plots(functions, t_limits, colors, thickness):
    plots = []
    for i in range(len(functions)):
        plots.append(parametric_plot(functions[i], t_limits,
                                     color=colors[i],
                                     thickness=thickness[i]))
    return plots

