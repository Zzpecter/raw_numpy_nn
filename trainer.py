from scipy import optimize


class Trainer(object):
    def __init__(self, mlp):
        # Make reference to network:
        self.mlp = mlp

    def callback_function(self, params):
        self.mlp.set_params(params)
        self.costs.append(self.mlp.cost_func(self.data, self.labels))

    def cost_function_wrapper(self, params, data, y):
        self.mlp.set_params(params)
        cost = self.mlp.mean_squared_error(data, y)
        grad = self.mlp.compute_gradients(data, y)

        return cost, grad

    def train(self, data, labels):
        # Make an internal variable for the callback function:
        self.data = data
        self.labels = labels

        # Make empty list to store costs:
        self.costs = []
        params0 = self.mlp.get_params()
        self.mlp.set_params(params0)

        # optimize.minimize calls a BFGS minimization algorithm to calculate the descent
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.cost_function_wrapper, params0, jac=True, method='BFGS', args=(data, labels),
                                 options=options, callback=self.callback_function)
        self.mlp.set_params(_res.x)
        self.optimization_results = _res
