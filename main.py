from qiskit.utils import algorithm_globals
import wine_model
import plotting
import numpy as np
import matplotlib.pyplot as plt

from time import localtime, strftime

starting_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

# configuration
feature_size = 3  # min 3
training_size = 100
test_size = 100
maxiter = 100
seed = 3142
reps = 1
train = False
test = True

# .npz


# training
model = wine_model.Model(training_size, maxiter, feature_size, test_size, seed, reps)
TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS = model.prepare_data()
model.prepare_circuit_structure()

if train:
    opt_parameters, opt_value, evaluation, costs = model.training()

if train == 0 and test == True:
    opt_parameters = np.loadtxt(f"trained_models/opt_var_0.74_3_100.txt")
# test
accuracy, predictions = model.test_classifier(opt_parameters)
print(accuracy)

# plot
if train:
    plotting.plot_loss(
        evaluation, costs, accuracy, feature_size, training_size, maxiter, reps, test_size
    )

plotting.plot_results(
    TEST_LABELS,
    predictions,
    accuracy,
    feature_size,
    training_size,
    maxiter,
    reps,
    test_size,
)

# save model
if train:
    np.savetxt(
        f"trained_models/opt_var_{accuracy}_{feature_size}_{training_size}_{starting_time}.txt",
        opt_parameters,
    )

 
# load model
# y muss in model.training als initial_point übergeben werden oder als opt_parameters bei model.test_classifier(opt_parameters)
# y = np.loadtxt(f"trained_models/opt_var_{accuracy}_{feature_size}_{training_size}.txt")
