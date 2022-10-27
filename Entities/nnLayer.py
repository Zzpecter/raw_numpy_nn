import numpy as np


class NN_Layer:
    def __init__(self, weights, biases, activation_func):
        self.weights = weights
        self.biases = biases
        self.activation_func = activation_func

        self.size = weights.shape

    def propagate_forward(self, data):
        # Multiply the weights
        layer_data = np.dot(data, self.weights)
        # Add a bias
        layer_data += self.biases
        # Activate
        assert callable(self.activation_func), "Error, activation function is not callable!"

        return self.activation_func(layer_data)
