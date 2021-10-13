import sys
import time
from abc import ABC, abstractmethod

import numpy as np


class Neuron(ABC):
    def __init__(self, w_range, alpha):
        self.w_range = w_range
        self.alpha = alpha
        self.w = []
        self.y = []
        self.epochs = 0
        self.time = 0

        self.net_error = 0

    @abstractmethod
    def activation_function(self, weighted_sum):
        pass

    def weighted_sums(self, x):
        return self.w[0] + np.dot(x, self.w[1:])

    def predictions(self, x):
        return self.activation_function(self.weighted_sums(x))

    def update_weights(self, x, error):
        self.w[0] += self.alpha * np.sum(error)
        self.w[1:] += self.alpha * np.dot(error, x)

    def error(self, d, y):
        return d - y

    def calculate_net_error(self, errors):
        self.net_error = np.sum(abs(errors))

    def stop_condition(self):
        return self.net_error == 0

    def train(self, x, d):
        self.w = np.random.uniform(low=-self.w_range, high=self.w_range, size=(len(x[0]) + 1,))
        self.y = np.empty(len(x))
        self.net_error = sys.maxsize

        self.epochs = 0
        self.time = time.time()

        while not self.stop_condition():
            self.net_error = 0

            for i, xi in enumerate(x):
                self.y[i] = self.predictions(xi)
                error = self.error(d[i], self.y[i])
                self.net_error += abs(error)
                if error != 0:
                    self.update_weights(xi, error)

            self.epochs += 1
        self.time = time.time() - self.time

    def evaluate(self, x):
        self.y = np.empty(len(x))

        for i, xi in enumerate(x):
            self.y[i] = self.predictions(xi)


class PerceptronUnipolar(Neuron):

    def activation_function(self, weighted_sum):
        if weighted_sum > 0:
            return 1
        else:
            return 0


class PerceptronBipolar(Neuron):

    def activation_function(self, weighted_sum):
        if weighted_sum > 0:
            return 1
        else:
            return -1


class Adaline(Neuron):
    def __init__(self, w_range, alpha, allowed_error):
        super().__init__(w_range, alpha)
        self.allowed_error = allowed_error

    def activation_function(self, weighted_sum):
        return weighted_sum

    def threshold_function(self, y):
        if y > 0:
            return 1
        else:
            return -1

    def stop_condition(self):
        return self.net_error < self.allowed_error

    def calculate_net_error(self, errors):
        self.net_error = np.sum(errors ** 2) / (len(self.w) - 1)

    def train(self, x, d):
        self.net_error = sys.maxsize
        self.w = np.random.uniform(low=-self.w_range, high=self.w_range, size=(len(x[0]) + 1,))
        self.epochs = 0
        self.time = time.time()

        while not self.stop_condition():
            self.y = self.predictions(x)
            errors = self.error(d, self.y)
            self.update_weights(x, errors)
            self.calculate_net_error(errors)

            self.epochs += 1

        self.time = time.time() - self.time
        self.y = np.array([self.threshold_function(y) for y in self.y])

    def evaluate(self, x):
        self.y = self.predictions(x)
        self.y = np.array([self.threshold_function(y) for y in self.y])
