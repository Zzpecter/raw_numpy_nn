import numpy as np


class ActivationFunctions:
    
    def sigmoid(self, value):
        return (1 / (1 + np.exp(-value)))

    def sigmoid_prime(self, value):
        # Derivative of Sigmoid gives the gradient
        return (np.exp(-value) / ((1 + np.exp(-value)) ** 2))
    