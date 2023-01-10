import numpy as np
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.datasets import wine
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, NLocal, PauliTwoDesign
from qiskit import BasicAer, execute
from qiskit.algorithms.optimizers import SPSA, GradientDescent
from sklearn.utils import shuffle
import optimizer_log


class Model:
    def __init__(
        self, training_size, maxiter, feature_size, test_size, seed, reps, dataset
    ):
        self.training_size = training_size
        self.feature_size = feature_size
        self.test_size = test_size
        self.maxiter = maxiter
        self.seed = seed
        self.reps = reps
        self.train_data = dataset[0]
        self.train_labels = dataset[1]
        self.test_data = dataset[2]
        self.test_labels = dataset[3]
        self._prepare_circuit_structure()

    def _prepare_circuit_structure(self):
        # prepare circuit components
        # data encoding
        self.FEATURE_MAP = ZZFeatureMap(
            feature_dimension=self.feature_size, reps=self.reps
        )
        # variational circuit TwoLocal
        # self.VAR_FORM = TwoLocal(self.feature_size, ["ry", "rz"], "cz", reps=self.reps)

        # variational circuit NLocal
        self.VAR_FORM = TwoLocal(self.feature_size, ["ry", "rz"], "cz", reps=self.reps)

        # variational circuit Pauli-Two-Design
        #self.VAR_FORM = PauliTwoDesign(self.feature_size, ["ry", "rz"], "cz", reps=self.reps)

        # fügt die feature_map mit dem variational circuit zusammen
        self.WINE_CIRCUIT = self.FEATURE_MAP.compose(self.VAR_FORM)
        self.WINE_CIRCUIT.measure_all()

        # visualization of the circuit
        # FEATURE_MAP.decompose().draw("mpl").savefig("feature_map_3.png")
        # VAR_FORM.decompose().draw("mpl").savefig("var_form_3.png")

    def _circuit_instance(self, data, variational):
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

    def _label_probability(self, results):
        """Converts a dict of bitstrings and their counts to parities and their counts"""
        shots = sum(results.values())
        probabilities = {0: 0.0, 1: 0.0, 2: 0.0}
        number_ones = 0
        """ for bitstring, counts in results.items():
            if bitstring[0] == "1":
                number_ones += counts
                probabilities[2] += counts / shots
            if bitstring[1] == "1":
                number_ones += counts
                probabilities[1] += counts / shots
            if bitstring[2] == "1":
                number_ones += counts
                probabilities[0] += counts / shots
        return probabilities """
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
        probabilities[0] = probabilities[0] / number_ones
        probabilities[1] = probabilities[1] / number_ones
        probabilities[2] = probabilities[2] / number_ones
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
    def _classification_probability(self, variational, data):
        """Classify data points using given parameters.
        Args:
            data(list): Set of data points to classify
            variational (list): Parameters for 'VAR_FORM
        Returns:
            list[dixt]: Probability of circuit classifyin each data point as 0 or 1
        """

        # selbstgeschriebene Funktion zur Parameterübergabe
        circuits = [self._circuit_instance(d, variational) for d in data]

        # Simulator
        backend = BasicAer.get_backend("qasm_simulator")
        results = execute(circuits, backend, seed_simulator=3142).result()
        classification = [
            self._label_probability(results.get_counts(c)) for c in circuits
        ]

        return classification

    # loss function
    def _cross_entropy_loss(self, classification, expected):
        """Calculate accuracy of predictions using cross entropy loss."""
        p = classification.get(expected)
        return -np.log(p + 1e-10)

    # cost function
    def _cost_function(self, variational):
        """Evaluates performance of our circuit with variational parameters on data"""
        classifications = self._classification_probability(
            variational=variational, data=self.train_data
        )
        cost = 0
        for i, classification in enumerate(classifications):
            cost += self._cross_entropy_loss(classification, self.train_labels[i])
        cost /= len(self.train_data)
        print(cost)
        return cost

    def _objective_function(self, variational):
        """Cost function of circuit parameters on training data.
        The optimizer will attempt to minimize this."""
        return self._cost_function(variational)

    def training(self):
        algorithm_globals.random_seed = self.seed
        np.random.seed(algorithm_globals.random_seed)

        initial_point = np.random.random(self.VAR_FORM.num_parameters)
        # initial_point = np.loadtxt(f"trained_models/opt_var_0.58_3_100.txt")
        # Set up the optimization
        log = optimizer_log.OptimizerLog()
        # optimizer = SPSA(maxiter=self.maxiter, callback=log.update)
        optimizer = GradientDescent(
            maxiter=self.maxiter, learning_rate=0.2, callback=log.update
        )
        # optimizer = SPSA(maxiter=self.maxiter, callback = log.update)
        # aktuelles Problem
        result = optimizer.minimize(self._objective_function, initial_point)
        return result.x, result.fun, log.evaluations, log.costs

    def test_classifier(self, variational):
        probability = self._classification_probability(variational, self.test_data)
        predictions = []
        # np.argmax
        for i in probability:
            predictions.append(np.argmax([i.get(0), i.get(1), i.get(2)]))
        accuracy = 0
        for i, prediction in enumerate(predictions):
            if prediction == self.test_labels[i]:
                accuracy += 1
        accuracy /= len(self.test_labels)
        return accuracy, predictions
