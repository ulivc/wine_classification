from qiskit.utils import algorithm_globals
import wine_model
import plotting
import numpy as np
import matplotlib.pyplot as plt
from time import localtime, strftime

# why is the accuracy changing with the test size???????????????


starting_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

# configuration
feature_size = 3  # min 3
training_size = 100
test_size = 170
maxiter = 100
seed = 3142
reps = 1
train = False
test = True

# .npz

size = []
accu = []
for i in range(test_size):
    if i:
        test_size = i
        # training
        model = wine_model.Model(
            training_size, maxiter, feature_size, test_size, seed, reps
        )
        TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS = model.prepare_data()
        model.prepare_circuit_structure()

        number = 0
        for p in TEST_LABELS:
            if p == 2:
                number += 1

        opt_parameters = np.loadtxt(
            f"trained_models/opt_var_0.76_3_120_2022-12-13_14-56-19.txt"
        )
        # test
        # accuracy, predictions = model.test_classifier(opt_parameters)
        # print(accuracy)

        # accu.append(accuracy)
        accu.append(number)
        size.append(test_size)
        print(i)
fig = plt.figure()
plt.plot(size, accu)
plt.xlabel("Size")
plt.ylabel("Accuracy")
plt.title(f"test_test_size")
print(plt.show())
fig.savefig(f"testing_class2_test_size.png")
