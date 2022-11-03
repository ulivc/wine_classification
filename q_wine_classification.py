from qiskit.utils import algorithm_globals
import numpy as np
from qiskit_machine_learning.datasets import wine
from sklearn.preprocessing import OneHotEncoder
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit import BasicAer, execute
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.algorithms.classifiers import VQC
import matplotlib.pyplot as plt

# 1. prepare data
#TRAIN_DATA, TRAIN_LABEL, TEST_DATA, TEST_LABEL = wine(training_size=140, test_size=30, n=13)


algorithm_globals.random_seed = 3142
np.random.seed(algorithm_globals.random_seed)

TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS = wine(
    training_size=20, test_size=5, n=13, one_hot=False
)

# data encoding
FEATURE_MAP = ZZFeatureMap(feature_dimension=13, reps=2)
# variational circuit
VAR_FORM = TwoLocal(13, ["ry", "rz"], "cz", reps=2)

# fügt die feature_map mit dem variational circuit zusammen
AD_HOC_CIRCUIT = FEATURE_MAP.compose(VAR_FORM)
AD_HOC_CIRCUIT.measure_all()

encoder = OneHotEncoder()
train_labels_oh = encoder.fit_transform(TRAIN_LABELS.reshape(-1, 1)).toarray()
test_labels_oh = encoder.fit_transform(TEST_LABELS.reshape(-1,1)).toarray()

initial_point = np.random.random(VAR_FORM.num_parameters)
""" initial_point = np.array([0.3200227 , 0.6503638 , 0.55995053,
                          0.96566328, 0.38243769, 0.90403094,
                          0.82271449, 0.26810137, 0.61076489,
                          0.82301609, 0.11789148, 0.29667125]) """
class OptimizerLog:
    """Log to store optimizer's intermediate results"""
    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []
    def update(self, evaluation, parameter, cost, _stepsize, _accept):
        """Save intermediate results. Optimizer passes five values but we ignore the last two."""
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)
    
log = OptimizerLog()
vqc = VQC(feature_map=FEATURE_MAP,
          ansatz=VAR_FORM,
          loss='cross_entropy',
          optimizer=SPSA(callback=log.update),
          initial_point=initial_point,
          quantum_instance=BasicAer.get_backend('qasm_simulator'))

# training?
vqc.fit(TRAIN_DATA, train_labels_oh)

fig = plt.figure()
plt.plot(log.evaluations, log.costs)
plt.xlabel('Steps')
plt.ylabel('Cost')
print(plt.show())
# 1.1 one hot encode data
# create logger
# 2. create circuit
# 3. plot circuit
