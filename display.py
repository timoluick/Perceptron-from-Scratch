import numpy as np
import matplotlib.pyplot as plt


class Display:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def display(self, perceptron):
        values = self.get_values(perceptron=perceptron)
        plt.pcolormesh(values)
        plt.show()

    def get_values(self, perceptron):
        values = []
        for i in range(self.x):
            column_values = []
            for j in range(self.y):
                column_values.append(perceptron.forward(np.array((i/self.x, j/self.y)))[0])
            values.append(column_values)
        return values
