from qiskit.utils import algorithm_globals
import wine_model
import plotting
import dataset
import optimizer_log
import numpy as np
from qiskit import BasicAer, execute
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.algorithms.optimizers import SPSA, GradientDescent

from time import localtime, strftime

starting_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

## this is an unfinished draft ##


# configuration
feature_size = 3  # min 3
training_size = 128
test_size = 50
maxiter = 200
seed = 3142
reps = 1

data_array = dataset.prepare_data(training_size, test_size, feature_size)
""" model = wine_model.Model(
    training_size, maxiter, feature_size, test_size, seed, reps, data_array
) """

# teacher circuit components
FEATURE_MAP = ZZFeatureMap(feature_dimension=feature_size, reps=reps)
VAR_FORM = TwoLocal(feature_size, ["ry", "rz"], "cz", reps=reps)

# student circuit components
INVERSE_FEATURE_MAP = FEATURE_MAP.inverse()

VAR_FORM_STUDENT = TwoLocal(feature_size, "ry", "cz", reps=reps)
INVERSE_VAR_FORM = VAR_FORM_STUDENT.inverse()

# fügt die feature_map mit dem variational circuit zusammen
teacher_circuit = FEATURE_MAP.compose(VAR_FORM)
student_circuit = INVERSE_VAR_FORM.compose(INVERSE_FEATURE_MAP)


teacher_parameters = np.loadtxt(
    f"trained_models/opt_var_0.8_3_128_2022-12-21_23-15-19.txt"
)
# algorithm_globals.random_seed = 3142
# np.random.seed(algorithm_globals.random_seed)

# variational = np.random.random(VAR_FORM.num_parameters)
# student_parameters = teacher_parameters[::-1]
# composed_circuit.decompose().draw("mpl").savefig("circuit.png")


def teacher_circuit_instance(d, variational):
    parameters = {}
    # d_student = d[::-1]
    for i, p in enumerate(FEATURE_MAP.ordered_parameters):
        parameters[p] = d[i]
    for i, p in enumerate(VAR_FORM.ordered_parameters):
        parameters[p] = variational[i]
    return teacher_circuit.assign_parameters(parameters)


def student_circuit_instance(circuit, d, variational):
    parameters = {}
    # d_student = d[::-1]
    for i, p in enumerate(FEATURE_MAP.ordered_parameters):
        parameters[p] = d[i]
    for i, p in enumerate(VAR_FORM_STUDENT.ordered_parameters):
        parameters[p] = variational[i]
    return circuit.assign_parameters(parameters)


# composed_circuit.decompose().draw("mpl").savefig("circuit1.png")


def objective_function(student_parameters):
    teacher_circuits = [
        teacher_circuit_instance(d, teacher_parameters) for d in data_array[2]
    ]
    student_circuits = [
        student_circuit_instance(student_circuit, d, student_parameters)
        for d in data_array[2]
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

    cost /= len(data_array[0])
    print(cost)
    return cost


def training():
    algorithm_globals.random_seed = seed
    np.random.seed(algorithm_globals.random_seed)

    initial_point = np.random.random(VAR_FORM_STUDENT.num_parameters)
    log = optimizer_log.OptimizerLog()
    optimizer = GradientDescent(maxiter=maxiter, learning_rate=0.4, callback=log.update)
    result = optimizer.minimize(objective_function, initial_point)
    return result.x, result.fun, log.evaluations, log.costs


def label_probability(results):
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


def test(opt_parameters):
    circuit = student_circuit.inverse()
    circuit.measure_all()
    circuits = [
        student_circuit_instance(circuit, d, opt_parameters) for d in data_array[2]
    ]
    backend = BasicAer.get_backend("qasm_simulator")
    results = execute(circuits, backend, seed_simulator=3142).result()
    probability = [label_probability(results.get_counts(c)) for c in circuits]

    predictions = []
    # np.argmax
    for i in probability:
        predictions.append(np.argmax([i.get(0), i.get(1), i.get(2)]))
    accuracy = 0
    for i, prediction in enumerate(predictions):
        if prediction == data_array[3][i]:
            accuracy += 1
    accuracy /= len(data_array[3])
    print(accuracy)
    return accuracy, predictions


opt_parameters, opt_value, evaluation, costs = training()

accuracy, predictions = test(opt_parameters)

plotting.plot_loss_student(
    evaluation,
    costs,
    accuracy,
    feature_size,
    training_size,
    maxiter,
    reps,
    test_size,
    starting_time,
)

plotting.plot_results_student(
    data_array[3],
    predictions,
    accuracy,
    feature_size,
    training_size,
    maxiter,
    reps,
    test_size,
    starting_time,
)

np.savetxt(
    f"trained_models/student/opt_var_{accuracy}_{feature_size}_{training_size}_{starting_time}.txt",
    opt_parameters,
)
