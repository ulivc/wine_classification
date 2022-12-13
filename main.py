from qiskit.utils import algorithm_globals
import wine_model
import plotting
import numpy as np

# configuration
feature_size = 3  # min 3
training_size = 20
test_size = 100
maxiter = 1
seed = 3142
reps = 1


# training
model = wine_model.Model(training_size, maxiter, feature_size, test_size, seed, reps)
TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS = model.prepare_data()
model.prepare_circuit_structure()
opt_parameters, opt_value, evaluation, costs = model.training()

# opt_parameters = np.loadtxt(f"trained_models/opt_var_0.68_3_100.txt")
# test
accuracy, predictions = model.test_classifier(opt_parameters)
print(accuracy)

""" # plot
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
np.savetxt(
    f"trained_models/opt_var_{accuracy}_{feature_size}_{training_size}.txt",
    opt_parameters,
)
 """

# load model
# y muss in model.training als initial_point übergeben werden oder als opt_parameters bei model.test_classifier(opt_parameters)
# y = np.loadtxt(f"trained_models/opt_var_{accuracy}_{feature_size}_{training_size}.txt")
