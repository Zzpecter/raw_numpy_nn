import numpy as np


class CostFunctions:

    def mean_squared_error(self, actual_output, expected_output):
        # Compute the cost using MSE.
        cost = 0.5 * sum((expected_output - actual_output) ** 2)
        return cost

    def CostFunctionPrime(self, actual_output, expected_output):
        # Compute derivative with respect to the weights for a given X and y:
        # TODOOOO

        delta3 = np.multiply(-(expected_output - actual_output), self.SigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.SigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2
