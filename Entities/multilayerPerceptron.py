import numpy as np


class MultilayerPerceptron(object):
    def __init__(self, cost_func, activation_func):
        self.input_layer_size = 784
        self.output_layer_size = 10

        self.cost_func = cost_func
        self.activation_func = activation_func
        self.layers = None

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
