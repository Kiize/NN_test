import numpy as np
import matplotlib.pyplot as plt 
import time

start = time.time()
np.random.seed(0)

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

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

X = np.array([
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
])

x, y = create_data(100, 3)

layer1 = Layer_Dense(2, 3)
activation1 = Activation_Softmax()
loss_function = Loss_CategoricalCrossEntropy()

layer1.forward(x)
activation1.forward(layer1.output)

loss = loss_function.calculate(activation1.output, y)

plt.scatter(x[:, 0], x[:, 1], c = y, cmap='brg')

print(f"elapsed time: {(time.time() - start):.1f} seconds\n")

plt.show()