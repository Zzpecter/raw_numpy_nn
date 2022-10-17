import numpy as np


class NeuralNetwork(object):
    def __init__(self, layers, cost_func, activation_func):
        self.inputLayerSize = 784
        self.outputLayerSize = 10

        self.hiddenLayers = layers
        self.cost_func = cost_func
        self.activation_func = activation_func



    def GetParams(self):
        # Get all weights unrolled into vector:
        params = np.concatenate([w.weights.ravel() for w in self.hiddenLayers])
        return params

    def SetParams(self, params):
        # Set W1 and W2 using single paramater vector.
        # TODOOO
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def ComputeGradients(self, X, y):
        dJdW1, dJdW2 = self.CostFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))