from qiskit.utils import algorithm_globals
import numpy as np
from qiskit_machine_learning.datasets import wine
from sklearn.preprocessing import OneHotEncoder
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit import BasicAer, execute
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.algorithms.classifiers import VQC
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# configuration
feature_size = 2
training_size = 10

algorithm_globals.random_seed = 3142
np.random.seed(algorithm_globals.random_seed)

# prepare data
TRAIN_DATA, train_labels_oh, TEST_DATA, test_labels_oh = wine(
    training_size=training_size, test_size=20, n=feature_size, one_hot=False
)

# prepare circuit components
# data encoding
FEATURE_MAP = ZZFeatureMap(feature_dimension=feature_size, reps=2)
# variational circuit
VAR_FORM = TwoLocal(feature_size, ["ry", "rz"], "cz", reps=2)

# fügt die feature_map mit dem variational circuit zusammen
WINE_CIRCUIT = FEATURE_MAP.compose(VAR_FORM)
WINE_CIRCUIT.measure_all()

# wir stellen sicher dass die daten und parameter an die richtigen Parameter übergeben werden?
#
def circuit_instance(data, variational):
    """Assigns parameter values to 'AD_HOC_CIRCUIT'
    Args:
        data(list):Data values for the feature map
        variational (list): Parameter values for 'VAR_FORM'
    Returns:
        Quantum Circuit: 'AD_HOC_CIRCUIT' with parameters assigned"""
    # ?? überschreibt variational nicht data
    parameters = {}
    for i, p in enumerate(FEATURE_MAP.ordered_parameters):
        parameters[p] = data[i]
    for i, p in enumerate(VAR_FORM.ordered_parameters):
        parameters[p] = variational[i]
    return WINE_CIRCUIT.assign_parameters(parameters)


def label_probability(results):
    """Converts a dict of bitstrings and their counts to parities and their counts"""
    shots = sum(results.values())
    probabilities = {0: 0.0, 1: 0.0, 2: 0.0}
    for bitstring, counts in results.items():
        if bitstring[0] == "1":
            probabilities[2] += counts / shots
        if bitstring[1] == "1":
            probabilities[1] += counts / shots
        if bitstring[1] == "1":
            probabilities[0] += counts / shots
    return probabilities


# quasi Ausführung des circuits
def classification_probability(data, variational):
    """Classify data points using given parameters.
    Args:
        data(list): Set of data points to classify
        variational (list): Parameters for 'VAR_FORM
    Returns:
        list[dixt]: Probability of circuit classifyin each data point as 0 or 1
    """

    # selbstgeschriebene Funktion zur Parameterübergabe
    circuits = [circuit_instance(d, variational) for d in data]

    # Simulator
    backend = BasicAer.get_backend("qasm_simulator")
    results = execute(circuits, backend).result()
    classification = [label_probability(results.get_counts(c)) for c in circuits]

    return classification


# loss function
def cross_entropy_loss(classification, expected):
    """Calculate accuracy of predictions using cross entropy loss."""
    p = classification.get(expected)
    return -np.log(p + 1e-10)


# cost function
def cost_function(data, labels, variational):
    """Evaluates performance of our circuit with variational parameters on data"""
    classifications = classification_probability(data, variational)
    cost = 0
    print(classifications)
    for i, classification in enumerate(classifications):
        cost += cross_entropy_loss(classification, labels[i])
    cost /= len(data)
    print(cost)
    return cost


# store results for plotting
class OptimizerLog:
    """Log to store optimizer's intermediate results"""

    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []
        self.count = 1
        print("hallo")

    def update(self, evaluation, parameter, cost, _stepsize, _accept):
        """Save intermediate results. Optimizer passes five values but we ignore the last two."""
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)
        self.count += 1
        print(self.count)


# Set up the optimization
log = OptimizerLog()
optimizer = SPSA(maxiter=100, callback=log.update)

initial_point = np.random.random(VAR_FORM.num_parameters)


def objective_function(variational):
    """Cost function of circuit parameters on training data.
    The optimizer will attempt to minimize this."""
    print("objective_function")
    return cost_function(TRAIN_DATA, train_labels_oh, variational)


result = optimizer.minimize(objective_function, initial_point)

opt_var = result.x


opt_value = result.fun


def test_classifier(data, labels, variational):

    probability = classification_probability(data, variational)

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
        if prediction == labels[i]:
            accuracy += 1
    accuracy /= len(labels)
    return accuracy, predictions


accuracy, predictions = test_classifier(TEST_DATA, test_labels_oh, opt_var)
print(accuracy)
np.savetxt(f"trained_models/opt_var_{accuracy}_{feature_size}_{training_size}.txt", opt_var)
print("ergebnis")
print(opt_var)
y = np.loadtxt(f"trained_models/opt_var_{accuracy}_{feature_size}_{training_size}.txt")
print("neues Ergebnis")
print(y)
# das gespeicherte modell kann anschließend über "accuracy, predictions = test_classifier(TEST_DATA, test_labels_oh, y)" aufgerufen werden

#
# PLOT
#

# plot loss
fig = plt.figure()
plt.plot(log.evaluations, log.costs)
plt.xlabel("Steps")
plt.ylabel("Cost")
plt.title(f"{accuracy}_{feature_size}_wine_classification")
print(plt.show())
fig.savefig(f"plots/qfullcosts_{accuracy}_{feature_size}_{training_size}.png")

# create array for plot
correct_result = [0, 0, 0]
wrong_result = [0, 0, 0]
for label, pred in zip(test_labels_oh, predictions):
    if np.array_equal(label, pred):
        print(pred)
        correct_result[pred] += 1

    else:
        wrong_result[pred] += 1

fig1 = plt.figure()
# Daten für drei Teams erstellen
classes = ["0", "1", "2"]

# Diagrammparameter definieren
N = 3
barWidth = 0.5
xloc = np.arange(N)

# Gestapeltes Balkendiagramm erstellen
p1 = plt.bar(xloc, correct_result, width=barWidth)
p2 = plt.bar(xloc, wrong_result, bottom=correct_result, width=barWidth)

# Beschriftungen, Titel, Striche und Legende hinzufügen
plt.ylabel("count")
plt.xlabel("classes")
plt.title("Results")
plt.xticks(xloc, ("0", "1", "2"))
plt.yticks(np.arange(0, 41, 5))
plt.legend((p1[0], p2[0]), ("True", "False"))
print(plt.show())
fig1.savefig(f"plots/qfullresults_{accuracy}_{feature_size}_{training_size}.png")
