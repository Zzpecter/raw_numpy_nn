from Entities.dataLoader import DataLoader
from Entities.nnLayer import NN_Layer
from Entities.multilayerPerceptron import MultilayerPerceptron as mlp
from Entities.activationFunctions import ActivationFunctions
from Entities.costFunctions import CostFunctions
from trainer import Trainer

import numpy as np


if __name__ == "__main__":
    # First load data
    data, labels = DataLoader.load_data()
    print(data.shape)

    # Construct the NN
    neural_network = mlp(CostFunctions.mean_squared_error, ActivationFunctions.sigmoid)

    # Declare weights and biases for each layer
    w_1 = np.random.randn(neural_network.input_layer_size, 1000)
    b_1 = np.random.randn(1, 1000)
    w_2 = np.random.randn(1000, 100)
    b_2 = np.random.randn(1, 100)
    w_3 = np.random.randn(100, 10)
    b_3 = np.random.randn(1, 10)

    # Declare Layers and add them to the neural network
    first_layer = NN_Layer(w_1, b_1, neural_network.activation_func)
    second_layer = NN_Layer(w_2, b_2, neural_network.activation_func)
    third_layer = NN_Layer(w_3, b_3, neural_network.activation_func)
    layers = [first_layer, second_layer, third_layer]

    neural_network.layers = layers

    # Declare the trainer
    trainer = Trainer(neural_network)
    trainer.train(data, labels)
