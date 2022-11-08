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
training_size = 5
algorithm_globals.random_seed = 3142
np.random.seed(algorithm_globals.random_seed)

# prepare data
TRAIN_DATA, train_labels_oh, TEST_DATA, test_labels_oh = wine(
    training_size=training_size, test_size=5, n=feature_size, one_hot=True
)
""" encoder = OneHotEncoder()
train_labels_oh = encoder.fit_transform(TRAIN_LABELS.reshape(-1, 1)).toarray()
test_labels_oh = encoder.fit_transform(TEST_LABELS.reshape(-1,1)).toarray()
 """
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


fig1 = plt.figure(figsize=(9, 6))

for feature, label in zip(TRAIN_DATA, train_labels_oh):
    COLOR = "C1" if label[0] == 0 else "C0"
    plt.scatter(feature[0], feature[1], marker="o", s=100, color=COLOR)

for feature, label, pred in zip(TEST_DATA, test_labels_oh, vqc.predict(TEST_DATA)):
    COLOR = "C1" if pred[0] == 0 else "C0"
    plt.scatter(feature[0], feature[1], marker="s", s=100, color=COLOR)
    if not np.array_equal(label, pred):  # mark wrongly classified
        plt.scatter(
            feature[0],
            feature[1],
            marker="o",
            s=500,
            linewidths=2.5,
            facecolor="none",
            edgecolor="C3",
        )

legend_elements = [
    Line2D([0], [0], marker="o", c="w", mfc="C1", label="A", ms=10),
    Line2D([0], [0], marker="o", c="w", mfc="C0", label="B", ms=10),
    Line2D([0], [0], marker="s", c="w", mfc="C1", label="predict A", ms=10),
    Line2D([0], [0], marker="s", c="w", mfc="C0", label="predict B", ms=10),
    Line2D(
        [0],
        [0],
        marker="o",
        c="w",
        mfc="none",
        mec="C3",
        label="wrongly classified",
        mew=2,
        ms=15,
    ),
]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc="upper left")

plt.title("Training & Test Data")
plt.xlabel("x")
plt.ylabel("y")
print(plt.show())
fig1.savefig(f"plots/qresults_{accuracy}_{feature_size}_{training_size}.png")

