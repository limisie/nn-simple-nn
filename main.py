from data import x_train, y_train, x_test, y_test
from models import Adaline, PerceptronUnipolar, PerceptronBipolar


def convert_for_bipolar(arr):
    arr[arr < 0.5] -= 1


if __name__ == '__main__':
    learning_rate = 0.01
    weight_range = 0.01
    iterations = 100
    epochs = 0
    times = 0
    allowed_error = 0.5

    bipolar = False
    if bipolar:
        convert_for_bipolar(x_train)
        convert_for_bipolar(y_train)
        convert_for_bipolar(x_test)
        convert_for_bipolar(y_test)

    model = PerceptronUnipolar(weight_range, learning_rate)

    for i in range(iterations):
        model.train(x_train, y_train)
        epochs += model.epochs
        times += model.time

    print(epochs / iterations)
    print(times / iterations)

    # for lr in learning_rate:
    #     model = Adaline(weight_range, lr, allowed_error)
    #     epochs = 0
    #     times = 0
    #     for i in range(iterations):
    #         model.train(x_train, y_train)
    #         epochs += model.epochs
    #         times += model.time
    #
    #     print(f'lr: {lr}')
    #     print(f'epochs: {epochs / iterations}')
    #     print(f'time: {times / iterations}')
    #     print('---------------------------')
