#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sage.plot.scatter_plot import ScatterPlot
from sage.all import *

def get_surface_plots(surfaces, K, x_lim, y_lim, plot_colors):
    surface_plots = []
    xs, ys = var('xs ys')
    for k in range(K):
        surface_plots.append(plot3d(surfaces[k],
                                    (xs, -x_lim, x_lim),
                                    (ys, -y_lim, y_lim),
                                    color=plot_colors[k]))
    return surface_plots


def get_region_plots(surfaces, K, x_low_lim, y_low_lim, plot_colors,
                     x_high_lim=None, y_high_lim=None):
    if x_high_lim is None:
        x_high_lim = x_low_lim
    if y_high_lim is None:
        y_high_lim = y_low_lim
    region_plots = []
    xs, ys = var('xs ys')
    for k in range(K):
        surface_intersection = [surfaces[k] - surfaces[j] >= 0
                                for j in range(K) if j != k]
        region_plots.append(region_plot(surface_intersection,
                                        (xs, -x_low_lim, x_high_lim),
                                        (ys, -y_low_lim, y_high_lim),
                                        incol=plot_colors[k]))
    return region_plots


def get_scatter_plots(X, y, K, plot_colors):
    scatter_plots = []
    for k in range(K):
        scatter_plots.append(scatter_plot(X[y == k], facecolor=plot_colors[k]))
    return scatter_plots
