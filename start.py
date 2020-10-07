import numpy as np
from perceptron import Perceptron
from display import Display

PERCEPTRON = Perceptron(input_size=2, lr=0.01)


x_y = np.array((((0, 0), 1), ((1, 1), 0)))


for i in range(1000):
    for (x, y) in x_y:
        output = PERCEPTRON.forward(x)
        error = output - y
        PERCEPTRON.tune_parameters(x, error)

DISPLAY = Display()
DISPLAY.display(perceptron=PERCEPTRON)
