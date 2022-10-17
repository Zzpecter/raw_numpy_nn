from scipy import optimize


class Trainer(object):
    def __init__(self, N):
        # Make reference to network:
        self.N = N

    def CallbackF(self, params):
        self.N.SetParams(params)
        self.J.append(self.N.CostFunction(self.X, self.y))

    def CostFunctionWrapper(self, params, X, y):
        self.N.SetParams(params)
        cost = self.N.CostFunction(X, y)
        grad = self.N.ComputeGradients(X, y)

        return cost, grad

    def Train(self, X, y):
        # Make an internal variable for the callback function:
        self.X = X
        self.y = y

        # Make empty list to store costs:
        self.J = []

        params0 = self.N.GetParams()

        # optimize.minimize calls a BFGS minimization algorithm to calculate the descent
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.CostFunctionWrapper, params0, jac=True, method='BFGS', args=(X, y),
                                 options=options, callback=self.CallbackF)
        self.N.SetParams(_res.x)
        self.optimizationResults = _res