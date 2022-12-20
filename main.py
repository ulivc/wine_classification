import wine_model
import plotting
import dataset
import numpy as np
import matplotlib.pyplot as plt

from time import localtime, strftime

starting_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

# configuration
training_size = 128  # fixed
test_size = 50

feature_size = 3  # min 3
maxiter = 100
seed = 3142
reps = 1
train = True
save = True

# .npz

# training


dataset = dataset.prepare_data(training_size, test_size, feature_size)

model = wine_model.Model(
    training_size, maxiter, feature_size, test_size, seed, reps, dataset
)


if train:
    opt_parameters, opt_value, evaluation, costs = model.training()
else:
    opt_parameters = np.loadtxt(f"trained_models/opt_var_0.74_3_100.txt")

# test
accuracy, predictions = model.test_classifier(opt_parameters)
print(accuracy)

# plot
if train:
    plotting.plot_loss(
        evaluation,
        costs,
        accuracy,
        feature_size,
        training_size,
        maxiter,
        reps,
        test_size,
    )

plotting.plot_results(
    dataset[3],
    predictions,
    accuracy,
    feature_size,
    training_size,
    maxiter,
    reps,
    test_size,
)

# save model
if save and train:
    np.savetxt(
        f"trained_models/opt_var_{accuracy}_{feature_size}_{training_size}_{starting_time}.txt",
        opt_parameters,
    )


# load model
# y muss in model.training als initial_point übergeben werden oder als opt_parameters bei model.test_classifier(opt_parameters)
# y = np.loadtxt(f"trained_models/opt_var_{accuracy}_{feature_size}_{training_size}.txt")
