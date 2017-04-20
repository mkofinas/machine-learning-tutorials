#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from functools import partial
import numpy as np

import sage.all

import symbolic_plot
import activations
import symbolic_activations

class NeuralNetwork(object):
    def __init__(self, num_dims, num_classes, hidden_layers_sizes,
                 activation_function, num_epochs, learning_rate,
                 regularization_strength, seed=None):
        """
        """
        # Network Architecture
        self.num_dims = num_dims
        self.num_classes = num_classes
        self.num_layers = len(hidden_layers_sizes) + 1
        self.num_hidden_layers = len(hidden_layers_sizes)
        self.activation = partial(activations.activation,
                                  activation_function=activation_function)
        self.d_activation = partial(activations.d_activation,
                                    activation_function=activation_function)
        self.symbolic_activation = partial(
                symbolic_activations.activation,
                activation_function=activation_function)

        # Training Parameters
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.losses = np.zeros((num_epochs, 1))

        # Weights & Biases Initialization
        np.random.seed(seed)
        # layers = [num_dims] + hidden_layers_sizes + [num_classes]
        layers = [num_dims] + hidden_layers_sizes + [num_classes if num_classes > 2 else 1]
        self.W = [0.0 for _ in range(self.num_layers)]
        self.b = [0.0 for _ in range(self.num_layers)]
        for l in range(self.num_layers):
            self.W[l] = 0.01 * np.random.randn(layers[l], layers[l+1])
            self.b[l] = np.zeros((1, layers[l+1]))

        # Reserve memory for backpropagation
        self.dW = [0.0 for _ in range(self.num_layers)]
        self.db = [0.0 for _ in range(self.num_layers)]

    def forward(self, X):
        hidden_layers = [0.0 for _ in range(self.num_hidden_layers)]
        hidden_layers[0] = self.activation(np.dot(X, self.W[0]) + self.b[0])
        for l in range(1, self.num_hidden_layers):
            hidden_layers[l] = self.activation(
                    np.dot(hidden_layers[l-1], self.W[l]) + self.b[l])

        scores = np.dot(hidden_layers[-1], self.W[-1]) + self.b[-1]
        return scores, hidden_layers

    def backward(self, probs, y, X, hidden_layers):
        num_examples = probs.shape[0]
        # Compute the gradient on scores
        d_scores = probs.copy()
        if self.num_classes == 2:
            d_scores -= y[:, np.newaxis]
        else:
            d_scores[range(num_examples), y] -= 1
        d_scores /= num_examples

        # Backpropate the gradient to the parameters
        # First backprop into last layer weights and biases
        self.dW[-1] = np.dot(hidden_layers[-1].T, d_scores)
        self.db[-1] = np.sum(d_scores, axis=0, keepdims=True)

        # Backprop into Hidden Layers
        d_hidden_layers = [0.0 for _ in range(self.num_hidden_layers)]
        d_hidden_layers[-1] = (np.dot(d_scores, self.W[-1].T) *
                               self.d_activation(hidden_layers[-1]))
        for l in range(-2, -self.num_hidden_layers-1, -1):
            d_hidden_layers[l] = (np.dot(d_hidden_layers[l+1], self.W[l].T) *
                                  self.d_activation(hidden_layers[l]))
        # Finally backprop from second to last up to the first layer weights
        # and biases
        for l in range(-2, -self.num_layers, -1):
            self.dW[l] = np.dot(hidden_layers[l].T, d_hidden_layers[l+1])
            self.db[l] = np.sum(d_hidden_layers[l+1], axis=0, keepdims=True)
        self.dW[0] = np.dot(X.T, d_hidden_layers[0])
        self.db[0] = np.sum(d_hidden_layers[0], axis=0, keepdims=True)

    def softmax_function(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def cross_entropy(self, probs, y):
        correct_logprobs = -np.log(probs[range(probs.shape[0]), y])
        data_loss = np.mean(correct_logprobs)
        return data_loss

    def logloss(self, probs, y):
        correct_logprobs = -np.log(np.concatenate((probs[y==1], 1-probs[y==0])))
        data_loss = np.mean(correct_logprobs)
        return data_loss

    def weight_decay(self):
        regularization_loss = 0.0
        sum_sqr = lambda W: np.sum(W * W)
        for l in range(self.num_layers):
            regularization_loss += sum_sqr(self.W[l])
        regularization_loss *= 0.5 * self.regularization_strength
        return regularization_loss

    def train(self, X, y, print_loss=1000):
        # Gradient Descent loop
        for i in xrange(self.num_epochs):
            # evaluate class scores, [N x K]
            scores, hidden_layers = self.forward(X)

            # compute the class probabilities
            if self.num_classes > 2:
                probs = self.softmax_function(scores) # [N x K]
            else:
                probs = 1 / (1 + np.exp(-scores))

            # compute the loss: average cross-entropy loss and regularization
            if self.num_classes > 2:
                loss = self.cross_entropy(probs, y) + self.weight_decay()
            else:
                loss = self.logloss(probs, y) + self.weight_decay()
            self.losses[i] = loss
            if print_loss > 0  and i % print_loss == 0:
                print('Iteration {0}: Loss {1}'.format(i, loss))

            self.backward(probs, y, X, hidden_layers)
            # add regularization gradient contribution
            for l in range(self.num_layers):
                self.dW[l] += self.regularization_strength * self.W[l]

            # perform a parameter update
            for l in range(self.num_layers):
                self.W[l] -= self.learning_rate * self.dW[l]
                self.b[l] -= self.learning_rate * self.db[l]

    def predict(self, X):
        scores, _ = self.forward(X)
        if self.num_classes == 2:
            sigmoid_scores = 1 / (1 + np.exp(-scores))
            predicted_class = (sigmoid_scores >= 0.5).squeeze()
        else:
            predicted_class = np.argmax(scores, axis=1)
        return predicted_class

    def plot_classification_surfaces(self, X, y):
        Ws = [0.0 for _ in range(self.num_layers)]
        for l in range(self.num_layers):
            Ws[l] = sage.all.Matrix(np.concatenate((self.W[l], self.b[l])).T)

        xs, ys = sage.all.var('xs ys')
        Xs = sage.all.Matrix([[xs], [ys], [1]])

        Hs = Ws[0] * Xs
        Hs = self.symbolic_activation(Hs)
        Hs = Hs.stack(sage.all.ones_matrix(1, 1))
        for l in range(1, self.num_hidden_layers):
            Hs = Ws[l] * Hs
            Hs = self.symbolic_activation(Hs)
            Hs = Hs.stack(sage.all.ones_matrix(1, 1))

        Zs = Ws[-1] * Hs

        if self.num_classes == 2:
            surfaces = [0, Zs[0, 0]]
        else:
            surfaces = []
            for k in range(self.num_classes):
                surfaces.append(Zs[k, 0])

        x_lim = np.max(np.abs(X[:, 0]))
        y_lim = np.max(np.abs(X[:, 1]))

        plot_colors = ['blue', 'green', 'red']
        bg_colors = ['#89CFF0', '#90EE90', '#FFC0CB']
        region_plots = symbolic_plot.get_region_plots(
                surfaces, self.num_classes, x_lim, y_lim, bg_colors)
        scatter_plots = symbolic_plot.get_scatter_plots(
                X, y, self.num_classes, plot_colors)
        sage.all.show(sum(scatter_plots) + sum(region_plots))

        surface_plots = symbolic_plot.get_surface_plots(
                surfaces, self.num_classes, x_lim, y_lim, plot_colors)
        sage.all.show(sum(surface_plots), viewer='tachyon')

    def plot_loss(self):
        x_vec = np.arange(0, self.num_epochs)[: , np.newaxis]
        loss_plot = sage.all.line(np.hstack((x_vec, self.losses)))
        sage.all.show(loss_plot, axes_labels=['Number of Epochs','Loss'],
                      title='Loss Function as Training Progresses')

