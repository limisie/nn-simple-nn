import numpy as np

input_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
classes = [0, 1, 1, 1]
weights = np.random.rand(len(input_data[0]) + 1)
learning_rate = 0.01


def weighted_sum(x, w):
    sum = w[0]
    for i in range(len(x)):
        sum += w[i + 1] * x[i]
    return sum


def binary_activation_function(weighted_sum):
    return int(weighted_sum > 0)


def update_weights(x, w, error, alpha):
    w[0] = w[0] + alpha * error

    for i in range(len(w) - 1):
        w[i + 1] = w[i + 1] + error * alpha * x[i]

    return w


def stop_condition(delta_sum):
    return delta_sum == 0


def perceptron(x, w, alpha, d):
    delta_sum = -1
    predictions = np.empty(len(x))

    while not stop_condition(delta_sum):
        delta_sum = 0
        predictions = np.empty(len(x))

        for i, xi in enumerate(x):
            predictions[i] = binary_activation_function(weighted_sum(xi, w))
            error = d[i] - predictions[i]
            delta_sum += abs(error)
            update_weights(xi, w, error, alpha)

    print(w)
    return predictions


if __name__ == '__main__':
    print(perceptron(input_data, weights, learning_rate, classes))
