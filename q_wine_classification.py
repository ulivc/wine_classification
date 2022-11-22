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
feature_size = 10
training_size = 500
algorithm_globals.random_seed = 3142
np.random.seed(algorithm_globals.random_seed)

# prepare data
TRAIN_DATA, train_labels_oh, TEST_DATA, test_labels_oh = wine(
    training_size=training_size, test_size=20, n=feature_size, one_hot=True
)

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
    

# prepare circuit components
# data encoding
FEATURE_MAP = ZZFeatureMap(feature_dimension=feature_size, reps=2)
# variational circuit
VAR_FORM = TwoLocal(feature_size, ["ry", "rz"], "cz", reps=2)
print(VAR_FORM.decompose().draw())
initial_point = np.random.random(VAR_FORM.num_parameters)
log = OptimizerLog()

# initialize circuit
vqc = VQC(feature_map=FEATURE_MAP,
          ansatz=VAR_FORM,
          loss='cross_entropy',
          optimizer=SPSA(callback=log.update),
          initial_point=initial_point,
          quantum_instance=BasicAer.get_backend('qasm_simulator'))

# training (100 evaluations, why?)
vqc.fit(TRAIN_DATA, train_labels_oh)

# calculate accuracy
accuracy = vqc.score(TEST_DATA, test_labels_oh)
print(accuracy)

# plot loss 
fig = plt.figure()
plt.plot(log.evaluations, log.costs)
plt.xlabel('Steps')
plt.ylabel('Cost')
plt.title(f"{accuracy}_{feature_size}_wine_classification")
print(plt.show())
fig.savefig(f"plots/qcosts_{accuracy}_{feature_size}_{training_size}.png")

# create array for plot
correct_result = [0,0,0]
wrong_result = [0,0,0]
for label, pred in zip(test_labels_oh, vqc.predict(TEST_DATA)):
    if np.array_equal(label, pred):
        print(pred)
        if pred[0] == 1:
            correct_result[0] += 1
        if pred[1] == 1:
            correct_result[1] += 1
        if pred[2] == 1:
            correct_result[2] += 1   
            
    else:
        if pred[0] == 1:
            wrong_result[0] += 1
        if pred[1] == 1:
            wrong_result[1] += 1
        if pred[2] == 1:
            wrong_result[2] += 1  

fig1 = plt.figure()
# Daten für drei Teams erstellen
classes = ['0', '1', '2']

# Diagrammparameter definieren
N = 3 
barWidth = .5
xloc = np.arange(N)

# Gestapeltes Balkendiagramm erstellen
p1 = plt.bar(xloc, correct_result, width=barWidth)
p2 = plt.bar(xloc, wrong_result, bottom=correct_result, width=barWidth)

# Beschriftungen, Titel, Striche und Legende hinzufügen
plt.ylabel('count')
plt.xlabel('classes')
plt.title('Results')
plt.xticks(xloc, ('0', '1', '2'))
plt.yticks(np.arange(0, 41, 5))
plt.legend((p1[0], p2[0]), ('True', 'False'))
print(plt.show())
fig1.savefig(f"plots/qresults_{accuracy}_{feature_size}_{training_size}.png")

