import sys
import time
from abc import ABC, abstractmethod
import numpy as np

from data import x_train, y_train, x_test, y_test


class Neuron(ABC):
    def __init__(self, w_range, alpha):
        self.w_range = w_range
        self.alpha = alpha
        self.w = []
        self.y = []
        self.epochs = 0
        self.time = 0

        self.net_error = sys.maxsize

    @abstractmethod
    def activation_function(self, weighted_sum):
        pass

    def weighted_sums(self, x):
        return self.w[0] + np.dot(x, self.w[1:])

    def predictions(self, x):
        return np.array([self.activation_function(z) for z in self.weighted_sums(x)])

    def update_weights(self, x, error):
        self.w[0] = self.w[0] + self.alpha * np.sum(error)
        self.w[1:] += np.dot(error, x) * self.alpha

    def error(self, d, y):
        return d - y

    def calculate_net_error(self, errors):
        self.net_error = np.sum(abs(errors))

    def stop_condition(self):
        return self.net_error == 0

    def train(self, x, d):
        self.w = np.random.uniform(low=-self.w_range, high=self.w_range, size=(len(x[0]) + 1,))
        self.epochs = 0
        self.time = time.time()

        while not self.stop_condition():
            self.y = self.predictions(x)
            errors = self.error(d, self.y)
            self.calculate_net_error(errors)
            if self.net_error != 0:
                self.update_weights(x, errors)

            self.epochs += 1

        self.time = time.time() - self.time
        self.net_error = sys.maxsize

    def evaluate(self, x):
        self.y = self.predictions(x)


class PerceptronUnipolar(Neuron):

    def activation_function(self, weighted_sum):
        return int(weighted_sum > 0)


class PerceptronBipolar(Neuron):

    def activation_function(self, weighted_sum):
        if weighted_sum > 0:
            return 1
        else:
            return -1

    def train(self, x, d):
        x[x < 0.5] -= 1
        d[d == 0] = -1

        super().train(x, d)

        d[d == -1] = -0
        self.y[self.y == -1] = 0

    def evaluate(self, x):
        x[x < 0.5] -= 1

        super().evaluate(x)

        self.y[self.y == -1] = 0


class Adaline(PerceptronBipolar):
    def __init__(self, w_range, alpha, allowed_error):
        super().__init__(w_range, alpha)
        self.allowed_error = allowed_error

    def predictions(self, x):
        return self.weighted_sums(x)

    def stop_condition(self):
        return self.net_error < self.allowed_error

    def calculate_net_error(self, errors):
        self.net_error = np.sum(errors ** 2) / (len(self.w) - 1)

    def train(self, x, d):
        x[x < 0.5] -= 1
        d[d == 0] = -1

        Neuron.train(self, x, d)

        self.y = np.array([self.activation_function(y) for y in self.y])
        d[d == -1] = -0
        self.y[self.y == -1] = 0

    def evaluate(self, x):
        x[x < 0.5] -= 1

        Neuron.evaluate(self, x)

        self.y = np.array([self.activation_function(y) for y in self.y])
        self.y[self.y == -1] = 0


if __name__ == '__main__':
    learning_rate = 0.01
    weight_range = 0.01
    iterations = 100
    epochs = 0
    times = 0
    allowed_error = 0.1

    perceptron = PerceptronUnipolar(weight_range, learning_rate)

    for i in range(iterations):
        perceptron.train(x_train, y_train)
        epochs += perceptron.epochs
        times += perceptron.time

    print(epochs / iterations)
    print(times / iterations)
