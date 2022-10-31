import numpy as np
from Entities.activationFunctions import ActivationFunctions as af


class MultilayerPerceptron(object):
    def __init__(self, cost_func, activation_func):
        self.input_layer_size = 784
        self.output_layer_size = 10

        self.cost_func = cost_func
        self.activation_func = activation_func
        self.layers = None
        self.actual_output = None

    def propagate_forward(self, data):
        print(data.shape)
        assert data.shape == (1, 784), "Data is not of the needed dimension [1, 784]"

        nn_data = data

        # Propagate the input data through each of the layers
        for idx, layer in enumerate(self.layers):
            print(f"Forward pass - Layer {idx + 1}.\ndata shape: {nn_data.shape} weight shape: {layer.size}")
            nn_data = layer.propagate_forward(nn_data)
            print(f"Forward pass - Layer {idx + 1}.\noutput: {nn_data}")

        print(f"result: {nn_data}")
        return nn_data

    def get_params(self):
        # Get the weights unrolled into vector:
        params = np.concatenate([layer.weights.ravel() for layer in self.layers])
        return params

    def set_params(self, params):
        # Set weights using a single parameter vector.

        current_index = 0
        current_layer = 0
        for layer in self.layers:
            w_start = current_index
            w_end = current_index + layer.length

            actual_layer_weights = params[w_start:w_end]
            layer_y = layer.size[0]
            layer_x = layer.size[1]

            reshaped_weights = np.reshape(actual_layer_weights, (layer_y, layer_x))
            self.layers[current_layer].weights = reshaped_weights

            current_index = w_end
            current_layer += 1

    def mean_squared_error(self, data, expected_output):
        # Compute the cost using MSE.
        self.actual_output = self.propagate_forward(data)
        cost = 0.5 * sum((expected_output - self.actual_output) ** 2)
        return cost

    def mse_prime(self, data, expected_output):
        # Compute derivative with respect to the weights for a given X and y
        self.actual_output = self.propagate_forward(data)
        weight_costs_per_layer = []

        for layer in reversed(self.layers):
            delta = np.multiply(-(expected_output - self.actual_output), af.sigmoid_prime(layer.propagation_data))
            weight_costs_per_layer.append(delta)

        return weight_costs_per_layer

    def compute_gradients(self, data, expected_output):
        costs = self.mse_prime(data, expected_output)
        return np.concatenate(layer.ravel() for layer in costs)
