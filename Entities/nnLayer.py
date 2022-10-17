import numpy as np


class NN_Layer:
    def __init__(self, weights, biases, activation_func):
        self.weights = weights
        self.biases = biases
        self.activation_func = activation_func

        self.size = weights.shape

    def forward(self, input):
        # Propagate inputs through the network,
        # first multiply input by weights
        dot_prod = np.dot(input, self.weights)
        # add biases
        dot_prod += self.biases
        # pass the result to the activation function
        assert callable(self.activation_func), "Error, activation function is not callable!"

        return self.activation_func(dot_prod)
