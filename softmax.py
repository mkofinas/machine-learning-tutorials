#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

import sage.all

import data_generator
import symbolic_plot

class SoftmaxClassifier(object):
    def __init__(self, num_dims, num_classes, num_epochs, learning_rate,
                 regularization_strength, seed=None):
        """
        """
        # Network Architecture
        self.num_dims = num_dims
        self.num_classes = num_classes

        # Training Parameters
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.losses = np.zeros((num_epochs, 1))

        # Weights & Biases Initialization
        np.random.seed(seed)
        self.W = 0.01 * np.random.randn(num_dims, num_classes)
        self.b = np.zeros((1, num_classes))

    def softmax_function(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def cross_entropy(self, probs, y):
        correct_logprobs = -np.log(probs[range(probs.shape[0]), y])
        data_loss = np.mean(correct_logprobs)
        return data_loss

    def weight_decay(self):
        regularization_loss = 0.5 * self.regularization_strength * np.sum(self.W * self.W)
        return regularization_loss

    def forward(self, X):
        scores = np.dot(X, self.W) + self.b
        return scores

    def backward(self, probs, y, X):
        num_examples = probs.shape[0]
        dscores = probs.copy()
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        # Backpropagate the gradient to the parameters (W,b)
        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)

        return dW, db

    def train(self, X, y, print_loss=10):
        # Gradient Descent loop
        for i in xrange(self.num_epochs):
            # Evaluate class scores, [N x K]
            scores = self.forward(X)

            # Compute the class probabilities
            probs = self.softmax_function(scores) # [N x K]

            # Compute the loss: average cross-entropy loss and regularization
            loss = self.cross_entropy(probs, y) + self.weight_decay()
            self.losses[i] = loss
            if print_loss > 0  and i % print_loss == 0:
                print('Iteration {0}: Loss {1}'.format(i, loss))

            # compute the gradient on scores
            dW, db = self.backward(probs, y, X)

            # Add regularization gradient
            dW += self.regularization_strength * self.W

            # perform a parameter update
            self.W += -self.learning_rate * dW
            self.b += -self.learning_rate * db

    def predict(self, X):
        scores = self.forward(X)
        predicted_class = np.argmax(scores, axis=1)
        return predicted_class

    def plot_classification_surfaces(self, X, y):
        Ws = sage.all.Matrix(np.concatenate((self.W, self.b)).T)
        xs, ys = sage.all.var('xs ys')
        Xs = sage.all.Matrix([[xs], [ys], [1]])

        Zs = Ws * Xs

        surfaces = []
        for k in range(self.num_classes):
            surfaces.append(Zs[k, 0])

        x_lim = 1
        y_lim = 1

        plot_colors = ['blue', 'green', 'red']
        bg_colors = ['#89CFF0', '#90EE90', '#FFC0CB']
        region_plots = symbolic_plot.get_region_plots(surfaces, self.num_classes, x_lim, y_lim, bg_colors)
        scatter_plots = symbolic_plot.get_scatter_plots(X, y, self.num_classes, plot_colors)
        sage.all.show(sum(scatter_plots)+sum(region_plots))

        surface_plots = symbolic_plot.get_surface_plots(surfaces, self.num_classes, x_lim, y_lim, plot_colors)
        sage.all.show(sum(surface_plots), viewer='tachyon')

    def plot_loss(self):
        x_vec = np.arange(0, self.num_epochs)[: , np.newaxis]
        loss_plot = sage.all.line(np.hstack((x_vec, self.losses)))
        sage.all.show(loss_plot, axes_labels=['Number of Epochs','Loss'],
                      title='Loss Function as Training Progresses')

