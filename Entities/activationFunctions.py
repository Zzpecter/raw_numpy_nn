import numpy as np


class ActivationFunctions:
    
    @staticmethod
    def sigmoid(value):
        return (1 / (1 + np.exp(-value)))

    @staticmethod
    def sigmoid_prime(value):
        # Derivative of Sigmoid gives the gradient
        return (np.exp(-value) / ((1 + np.exp(-value)) ** 2))
    