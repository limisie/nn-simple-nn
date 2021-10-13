from data import x_train, y_train, x_test, y_test
from models import Adaline, PerceptronUnipolar, PerceptronBipolar


def convert_for_bipolar(arr):
    arr[arr < 0.5] -= 1


def n_iteration_stats(n=1000, learning_rate=0.005, weight_range=0.01, allowed_error=0.5):
    epochs = 0
    times = 0

    bipolar = True
    if bipolar:
        convert_for_bipolar(y_train)
        convert_for_bipolar(y_test)

    model = PerceptronBipolar(weight_range, learning_rate)

    for i in range(n):
        model.train(x_train, y_train)
        epochs += model.epochs
        times += model.time

    print(f'epochs: {epochs / n}')
    print(f'time: {times / n}')


def parameter_stats(iterations=10, learning_rate=0.5, weight_range=0.5, allowed_error=0.5,
                    researched_array=[1, 0.8, 0.5, 0.2, 0.1, 0.01, 0.001]):
    bipolar = False
    if bipolar:
        convert_for_bipolar(y_train)
        convert_for_bipolar(y_test)

    for variable in researched_array:
        model = PerceptronUnipolar(variable, learning_rate)
        epochs = 0
        times = 0
        for i in range(iterations):
            model.train(x_train, y_train)
            epochs += model.epochs
            times += model.time
            model.evaluate(x_test)
            print(model.y)
            print(y_test)

        print(f'variable: {variable}')
        print(f'epochs: {epochs / iterations}')
        print(f'time: {times / iterations}')
        print('---------------------------')


def adaline_test(learning_rate=0.01, weight_rate=0.5, allowed_error=0.5):
    convert_for_bipolar(y_train)
    convert_for_bipolar(y_test)

    model = Adaline(weight_rate, learning_rate, allowed_error)
    model.train(x_train, y_train)
    print(model.y)
    print(y_train)


if __name__ == '__main__':
    adaline_test()
