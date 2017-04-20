#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

import softmax
import data_generator

def main():
    N = 100 # number of points per class
    num_dims = 2 # dimensionality
    num_classes = 3 # number of classes

    X, y = data_generator.generate_spiral_data(N, num_dims, num_classes, seed=0)

    # Hyperparameters
    learning_rate = 1e-0
    regularization_strength = 1e-3
    num_epochs = 200

    # Train a Linear Classifier
    softmax_cl = softmax.SoftmaxClassifier(num_dims, num_classes, num_epochs,
                                           learning_rate,
                                           regularization_strength)
    softmax_cl.train(X, y, print_loss=10)
    # Evaluate training set accuracy
    predicted_class = softmax_cl.predict(X)
    print('Training accuracy: {0}'.format(np.mean(predicted_class == y)))

    # Plot the resulting classifier
    softmax_cl.plot_classification_surfaces(X, y)
    softmax_cl.plot_loss()

if __name__ == "__main__":
    main()
