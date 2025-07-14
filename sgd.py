import numpy as np

class SGD:
    """Stochastic Gradient Descent optimizer with parameter shift method."""
    
    def __init__(self, ansatz, hamiltonian, estimator, learning_rate=0.01):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.estimator = estimator
        self.learning_rate = learning_rate

    def minimize(self, initial_params, num_iterations=100, verbose=True):
        params = initial_params
        for i in range(num_iterations):
            params = self.step(params)
            cost = self.cost_function(params)
            if verbose:
                print(f"Iteration {i+1}: Cost = {cost:.5f} Ha")
        return params, cost

    def step(self, params):
        grad = self.parameter_shift_grad(params)
        return params - self.learning_rate * grad

    def parameter_shift_grad(self, params):
        grad = np.zeros(len(params))
        for i in range(len(params)):
            shifted = params.copy()
            shifted[i] += np.pi/2
            plus = self.cost_function(shifted)
            
            shifted = params.copy()
            shifted[i] -= np.pi/2
            minus = self.cost_function(shifted)
            
            grad[i] = 0.5 * (plus - minus)
        return grad
    
    def cost_function(self, params):
        pub = (self.ansatz, [self.hamiltonian], [params])
        result = self.estimator.run(pubs=[pub]).result()[0]
        return result.data.evs[0]