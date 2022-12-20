class OptimizerLog:
    """Log to store optimizer's intermediate results"""

    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []
        self.count = 1

    def update(self, evaluation, parameter, cost, _stepsize):
        """Save intermediate results. Optimizer passes five values but we ignore the last two."""
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)
        self.count += 1
        print(self.count)
