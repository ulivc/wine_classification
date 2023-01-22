import numpy as np
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
import circuit_library
from qiskit import BasicAer, execute
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import GradientDescent
import optimizer_log
import wine_model


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
        self.teacher_parameters = np.loadtxt(
            f"trained_models/teacher/opt_var_0.82_3_128_2022-12-21_10-53-22.txt"
        )

    def _prepare_circuit_structure(self):
        # teacher circuit components
        self.FEATURE_MAP = ZZFeatureMap(
            feature_dimension=self.feature_size, reps=self.reps
        )
        self.VAR_FORM = TwoLocal(self.feature_size, ["ry", "rz"], "cz", reps=self.reps)

        # student circuit components
        INVERSE_FEATURE_MAP = self.FEATURE_MAP.inverse()

        self.VAR_FORM_STUDENT = circuit_library.student_circuit_6()
        INVERSE_VAR_FORM_STUDENT = self.VAR_FORM_STUDENT.inverse()

        # fügt die feature_map mit dem variational circuit zusammen
        self.teacher_circuit = self.FEATURE_MAP.compose(self.VAR_FORM)
        self.student_circuit = INVERSE_VAR_FORM_STUDENT.compose(INVERSE_FEATURE_MAP)

    def _teacher_circuit_instance(self, d, variational):
        parameters = {}
        # d_student = d[::-1]
        for i, p in enumerate(self.FEATURE_MAP.ordered_parameters):
            parameters[p] = d[i]
        for i, p in enumerate(self.VAR_FORM.ordered_parameters):
            parameters[p] = variational[i]
        return self.teacher_circuit.assign_parameters(parameters)

    def _student_circuit_instance(self, circuit, d, variational):
        parameters = {}
        # d_student = d[::-1]
        for i, p in enumerate(self.FEATURE_MAP.ordered_parameters):
            parameters[p] = d[i]
        for i, p in enumerate(self.VAR_FORM_STUDENT.parameters):
            parameters[p] = variational[i]
        # return circuit.assign_parameters(parameters)
        return circuit.bind_parameters(parameters)

    def _objective_function(self, student_parameters):
        teacher_circuits = [
            self._teacher_circuit_instance(d, self.teacher_parameters)
            for d in self.train_data
        ]
        student_circuits = [
            self._student_circuit_instance(self.student_circuit, d, student_parameters)
            for d in self.train_data
        ]

        circuits = [teacher_circuits[i].compose(student_circuits[i]) for i in range(48)]
        [i.measure_all() for i in circuits]

        backend = BasicAer.get_backend("qasm_simulator")
        results = execute(circuits, backend, seed_simulator=3142).result()
        result_dictionary_array = [results.get_counts(i) for i in circuits]

        cost = 0
        for result_dictionary in result_dictionary_array:
            probability = result_dictionary.get("000", 0)
            probability /= 1024
            loss = -np.log(probability + 1e-10)
            cost += loss

        cost /= len(self.train_data)
        print(cost)
        return cost

    def training(self):
        algorithm_globals.random_seed = self.seed
        np.random.seed(algorithm_globals.random_seed)

        initial_point = np.random.random(self.VAR_FORM_STUDENT.num_parameters)
        log = optimizer_log.OptimizerLog()
        optimizer = GradientDescent(
            maxiter=self.maxiter, learning_rate=0.4, callback=log.update
        )
        result = optimizer.minimize(self._objective_function, initial_point)
        return result.x, result.fun, log.evaluations, log.costs, log.parameters

    def _label_probability(self, results):
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
        probabilities[0] = probabilities[0] / number_ones
        probabilities[1] = probabilities[1] / number_ones
        probabilities[2] = probabilities[2] / number_ones
        return probabilities

    def test_classifier(self, opt_parameters, test=True):
        data = self.train_data
        labels = self.train_labels
        if test:
            data = self.test_data
            labels = self.test_labels
        circuit = self.student_circuit.inverse()
        circuit.measure_all()
        circuits = [
            self._student_circuit_instance(circuit, d, opt_parameters) for d in data
        ]
        backend = BasicAer.get_backend("qasm_simulator")
        results = execute(circuits, backend, seed_simulator=3142).result()
        probability = [self._label_probability(results.get_counts(c)) for c in circuits]

        predictions = []
        # np.argmax
        for i in probability:
            predictions.append(np.argmax([i.get(0), i.get(1), i.get(2)]))
        accuracy = 0
        for i, prediction in enumerate(predictions):
            if prediction == labels[i]:
                accuracy += 1
        accuracy /= len(labels)
        return accuracy, predictions
