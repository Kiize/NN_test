import numpy as np
import matplotlib.pyplot as plt 
import time

start = time.time()
np.random.seed(0)

X = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities        
        

def create_data(points, classes):
    x = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')

    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number + 1))
        r = np.linspace(.0, 1, points)
        t = np.linspace(class_number*4, (class_number + 1)*4, points) + np.random.randn(points) * .2
        x[ix] = np.c_[r*np.sin(2.5 * t), r*np.cos(2.5*t)]
        y[ix] = class_number

    return x, y

x, y = create_data(100, 3)

layer1 = Layer_Dense(4, 3)
activation1 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

print(activation1.output)



print(f"elapsed time: {(time.time() - start):.1f} seconds\n")