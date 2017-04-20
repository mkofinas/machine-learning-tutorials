#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sage.all

def activation(H, activation_function):
    if activation_function == 'sigmoid':
        return H.apply_map(lambda x: 1 / (1 + sage.all.exp(-x)))
    elif activation_function == 'tanh':
        return H.apply_map(sage.all.tanh)
    elif activation_function == 'relu':
        return H.apply_map(lambda x: sage.all.max_symbolic(x, 0))

