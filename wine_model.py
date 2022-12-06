import numpy as np
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.datasets import wine
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit import BasicAer, execute
from qiskit.algorithms.optimizers import SPSA
from sklearn.utils import shuffle

class Model:
    def __init__(self, training_size, maxiter, feature_size, test_size, seed):
        self.training_size = training_size
        self.feature_size = feature_size
        self.test_size = test_size
        self.maxiter = maxiter
        self.seed = seed

    def prepare_data(self):
        # prepare data
        self.TRAIN_DATA, self.TRAIN_LABELS, self.TEST_DATA, self.TEST_LABELS = wine(
            training_size=self.training_size,
            test_size=self.test_size,
            n=self.feature_size,
            one_hot=False, 
        )
        self.TRAIN_DATA, self.TRAIN_LABELS = shuffle(self.TRAIN_DATA, self.TRAIN_LABELS)
        

        return self.TRAIN_DATA, self.TRAIN_LABELS, self.TEST_DATA, self.TEST_LABELS

    def prepare_circuit_structure(self):
        # prepare circuit components
        # data encoding
        self.FEATURE_MAP = ZZFeatureMap(feature_dimension=self.feature_size, reps=5)
        # variational circuit
        self.VAR_FORM = TwoLocal(self.feature_size, ["ry", "rz"], "cz", reps=5)

        # fügt die feature_map mit dem variational circuit zusammen
        self.WINE_CIRCUIT = self.FEATURE_MAP.compose(self.VAR_FORM)
        self.WINE_CIRCUIT.measure_all()

        # visualization of the circuit
        #FEATURE_MAP.decompose().draw("mpl").savefig("feature_map_3.png")
        #VAR_FORM.decompose().draw("mpl").savefig("var_form_3.png")

    def circuit_instance(self, data, variational):
        """Assigns parameter values to 'AD_HOC_CIRCUIT'
        Args:
            data(list):Data values for the feature map
            variational (list): Parameter values for 'VAR_FORM'
        Returns:
            Quantum Circuit: 'AD_HOC_CIRCUIT' with parameters assigned"""
        parameters = {}
        for i, p in enumerate(self.FEATURE_MAP.ordered_parameters):
            parameters[p] = data[i]
        for i, p in enumerate(self.VAR_FORM.ordered_parameters):
            parameters[p] = variational[i]
        return self.WINE_CIRCUIT.assign_parameters(parameters)

    def label_probability(self, results):
        """Converts a dict of bitstrings and their counts to parities and their counts"""
        shots = sum(results.values())
        probabilities = {0: 0.0, 1: 0.0, 2: 0.0}
        number_ones = 0
        for bitstring, counts in results.items():
            if bitstring[0] == "1":
                number_ones += counts
                probabilities[2] += counts 
            if bitstring[1] == "1":
                number_ones += counts
                probabilities[1] += counts 
            if bitstring[2] == "1":
                number_ones += counts
                probabilities[0] += counts 
        probabilities[0] = probabilities[0]/ number_ones
        probabilities[1] = probabilities[1]/ number_ones
        probabilities[2] = probabilities[2]/ number_ones
        return probabilities  
        # using 6 qubits, currently not helping
        """ for bitstring, counts in results.items():
        if bitstring[0] == "1":
            probabilities[2] += counts / shots
        if bitstring[1] == "1":
            probabilities[2] += counts / shots
        if bitstring[2] == "1":
            probabilities[1] += counts / shots
        if bitstring[3] == "1":
            probabilities[1] += counts / shots
        if bitstring[4] == "1":
            probabilities[0] += counts / shots
        if bitstring[5] == "1":
            probabilities[0] += counts / shots
        return probabilities """

        # use each probability only once, not helpful
        """ for bitstring, counts in results.items():
        if bitstring == "100":
            probabilities[2] += counts / shots
        if bitstring == "010":
            probabilities[1] += counts / shots
        if bitstring == "001":
            probabilities[0] += counts / shots
        return probabilities  """

    # quasi Ausführung des circuits
    def classification_probability(self, variational, data):
        """Classify data points using given parameters.
        Args:
            data(list): Set of data points to classify
            variational (list): Parameters for 'VAR_FORM
        Returns:
            list[dixt]: Probability of circuit classifyin each data point as 0 or 1
        """

        # selbstgeschriebene Funktion zur Parameterübergabe
        circuits = [self.circuit_instance(d, variational) for d in data]

        # Simulator
        backend = BasicAer.get_backend("qasm_simulator")
        results = execute(circuits, backend).result()
        classification = [
            self.label_probability(results.get_counts(c)) for c in circuits
        ]

        return classification

    # loss function
    def cross_entropy_loss(self, classification, expected):
        """Calculate accuracy of predictions using cross entropy loss."""
        p = classification.get(expected)
        print(p)
        return -np.log(p + 1e-10)

    # cost function
    def cost_function(self, variational):
        """Evaluates performance of our circuit with variational parameters on data"""
        classifications = self.classification_probability(variational=variational, data=self.TRAIN_DATA)
        cost = 0
        for i, classification in enumerate(classifications):
            cost += self.cross_entropy_loss(classification, self.TRAIN_LABELS[i])
        cost /= len(self.TRAIN_DATA)
        return cost

    def objective_function(self, variational):
        """Cost function of circuit parameters on training data.
        The optimizer will attempt to minimize this."""
        return self.cost_function(variational)

    def training(self):
        algorithm_globals.random_seed = self.seed
        np.random.seed(algorithm_globals.random_seed)

        initial_point = np.random.random(self.VAR_FORM.num_parameters)

        # Set up the optimization
        log = OptimizerLog()
        optimizer = SPSA(maxiter=self.maxiter, callback=log.update)

        # aktuelles Problem
        result = optimizer.minimize(self.objective_function, initial_point)
        return result.x, result.fun, log.evaluations, log.costs

    def test_classifier(self, variational):
        probability = self.classification_probability(variational, self.TEST_DATA)
        predictions = []
        for probabilities in probability:
            if probabilities[0] >= probabilities[1]:
                if probabilities[0] >= probabilities[2]:
                    predictions.append(0)
                    continue
            if probabilities[2] >= probabilities[1]:
                if probabilities[2] >= probabilities[0]:
                    predictions.append(2)
                    continue
            if probabilities[1] >= probabilities[0]:
                if probabilities[1] >= probabilities[2]:
                    predictions.append(1)
                    continue
        accuracy = 0
        for i, prediction in enumerate(predictions):
            if prediction == self.TEST_LABELS[i]:
                accuracy += 1
        accuracy /= len(self.TEST_LABELS)
        return accuracy, predictions


class OptimizerLog:
    """Log to store optimizer's intermediate results"""

    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []
        self.count = 1

    def update(self, evaluation, parameter, cost, _stepsize, _accept):
        """Save intermediate results. Optimizer passes five values but we ignore the last two."""
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)
        self.count += 1
        print(self.count)
