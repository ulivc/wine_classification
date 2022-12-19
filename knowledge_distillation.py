from qiskit.utils import algorithm_globals
import wine_model
import plotting
import numpy as np
from qiskit import BasicAer, execute

## this is an unfinished draft ##


# configuration
feature_size = 3  # min 3
training_size = 100
test_size = 100
maxiter = 300
seed = 3142
reps = 1



model = wine_model.Model(training_size, maxiter, feature_size, test_size, seed, reps)
TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS = model.prepare_data()
model.prepare_circuit_structure()

opt_parameters = np.loadtxt(f"trained_models/opt_var_0.74_3_100.txt")

teacher_circuits = [model.circuit_instance(d, opt_parameters) for d in TEST_DATA]
student_circuits = [i.inverse() for i in teacher_circuits]

circuits = []

# Simulator
backend = BasicAer.get_backend("qasm_simulator")
results = execute(circuits, backend).result()

