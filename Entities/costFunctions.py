import numpy as np


class CostFunctions:

    @staticmethod
    def mean_squared_error(actual_output, expected_output):
        # Compute the cost using MSE.
        cost = 0.5 * sum((expected_output - actual_output) ** 2)
        return cost

    @staticmethod
    def cost_function_prime(actual_output, expected_output):
        # TODO: Compute derivative with respect to the weights for a given X and y
        pass
