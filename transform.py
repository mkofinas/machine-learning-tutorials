#!/usr/bin/env python
# -*- coding: utf-8 -*-

###########################
## EXPERIMENTAL/UNFINISHED
###########################

import numpy as np
import matplotlib.pyplot as plt

x_lim = 1
y_lim = 1
eps = 1e-8
num_lines = 21

xx = np.linspace(-x_lim, x_lim, num_lines)
yy = np.linspace(-y_lim, y_lim, num_lines)

xv, yv = np.meshgrid(xx, yy)

for j in range(num_lines):
    for i in range(num_lines):
        if i+1 < num_lines:
            plt.plot([xv[j, i], xv[j, i+1]], [yv[j, i], yv[j, i+1]], 'k')
        if j+1 < num_lines:
            plt.plot([xv[j, i], xv[j+1, i]], [yv[j, i], yv[j+1, i]], 'k')
plt.show()

xtv, ytv = [np.tanh(0.6*xv-0.2*yv), np.sinh(xv+2*yv)]

for j in range(num_lines):
    for i in range(num_lines):
        if i+1 < num_lines:
            plt.plot([xtv[j, i], xtv[j, i+1]], [ytv[j, i], ytv[j, i+1]], 'k')
        if j+1 < num_lines:
            plt.plot([xtv[j, i], xtv[j+1, i]], [ytv[j, i], ytv[j+1, i]], 'k')

plt.scatter(xtv, ytv)
plt.show()
