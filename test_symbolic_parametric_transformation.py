#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sage.all import *

from symbolic_parametric_transformation import *

def main():
    t = var('t')
    t_lim = 1
    t_limits = (t, -t_lim, t_lim)
    num_lines = 21

    f = t ** 2
    g = t ** 2 - 0.5

    affine_transformation = Matrix([[2.6, -0.4, 2.1], [1.0, 2.0, -0.5]])
    non_linear_transformations = [tanh, tanh]

    planes, boundary_planes = cartesian_grid(t, t_lim, num_lines)

    grid_colors = ('gray', 'black')
    grid_thickness = (1.0, 2.0)
    plane_plots, boundary_plane_plots = grid_plots(
            (planes, boundary_planes), t_limits, grid_colors, grid_thickness)
    function_colors = ('red', 'blue')
    function_thickness = (1.5, 1.5)
    f_plot, g_plot = function_plots(((t, f), (t, g)), t_limits, function_colors,
                                    function_thickness)

    ## Transformations
    parametric_transformation = ParametricTransformation(
            affine_transformation, non_linear_transformations)

    transformed_planes = [parametric_transformation.transform(*plane)
                          for plane in planes]
    transformed_boundary_planes = [parametric_transformation.transform(*plane)
                                   for plane in boundary_planes]
    transformed_f = parametric_transformation.transform(t, f)
    transformed_g = parametric_transformation.transform(t, g)

    transformed_plane_plots, transformed_boundary_plane_plots = grid_plots(
            (transformed_planes, transformed_boundary_planes),
            t_limits, grid_colors, grid_thickness)
    transformed_f_plot, transformed_g_plot = function_plots(
            (transformed_f, transformed_g),
            t_limits, function_colors, function_thickness)

    show(sum(plane_plots) + f_plot + g_plot + sum(boundary_plane_plots))

    show(sum(transformed_plane_plots) +
         transformed_f_plot +
         transformed_g_plot +
         sum(transformed_boundary_plane_plots))

if __name__ == "__main__":
    main()

