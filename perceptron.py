import numpy as np


class Perceptron:
    def __init__(self, input_size, lr):
        self.weight = np.random.uniform(low=-1, high=1, size=input_size)
        self.bias = np.random.uniform(low=-1, high=1, size=1)
        self.lr = lr

    def forward(self, x):
        return np.matmul(self.weight, x) + self.bias

    def tune_parameters(self, x, error):
        self.weight = self.weight - x * error * self.lr
        self.bias = self.bias - self.lr * error
