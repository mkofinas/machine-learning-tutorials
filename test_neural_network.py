#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

import neural_network
import data_generator

def main():
    N = 100 # number of points per class
    num_dims = 2 # dimensionality
    num_classes = 3 # number of classes

    X, y = data_generator.generate_spiral_data(N, num_dims, num_classes, seed=0)

    # Hyperparameters
    hidden_layers_sizes = [100]
    learning_rate = 1e-0
    regularization_strength = 1e-3
    num_epochs = 10000
    activation_function = 'relu'

    neural_net = neural_network.NeuralNetwork(
            num_dims, num_classes, hidden_layers_sizes, activation_function,
            num_epochs, learning_rate, regularization_strength, seed=1)
    neural_net.train(X, y, print_loss=1000)
    # evaluate training set accuracy
    predicted_class = neural_net.predict(X)
    print('Training accuracy: {0}'.format(np.mean(predicted_class == y)))

    # plot the resulting classifier
    neural_net.plot_loss()
    neural_net.plot_classification_surfaces(X, y)

if __name__ == "__main__":
    main()

