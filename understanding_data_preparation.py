import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from qiskit.utils import optionals

from qiskit_machine_learning.datasets.dataset_helper import (
    features_and_labels_transform,
)


def wine(training_size, test_size, n, plot_data=False, one_hot=True):

    class_labels = [r"A", r"B", r"C"]

    data, target = datasets.load_wine(return_X_y=True)
    sample_train, sample_test, label_train, label_test = train_test_split(
        data, target, test_size=test_size, random_state=7
    )

    # Now we standardize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(data)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Now reduce number of features to number of qubits
    pca = PCA(n_components=n).fit(data)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    # Scale to the range (-1,+1)
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)
    # Pick training size number of samples from each distro
    training_input = {
        key: (sample_train[label_train == k, :])[:training_size]
        for k, key in enumerate(class_labels)
    }
    test_input = {
        key: (sample_test[label_test == k, :])[:test_size]
        for k, key in enumerate(class_labels)
    }

    training_feature_array, training_label_array = features_and_labels_transform(
        training_input, class_labels, one_hot
    )
    test_feature_array, test_label_array = features_and_labels_transform(
        test_input, class_labels, one_hot
    )

    if plot_data:
        optionals.HAS_MATPLOTLIB.require_now("wine")
        # pylint: disable=import-error
        import matplotlib.pyplot as plt

        for k in range(0, 3):
            plt.scatter(
                sample_train[label_train == k, 0][:training_size],
                sample_train[label_train == k, 1][:training_size],
            )

        plt.title("PCA dim. reduced Wine dataset")
        plt.show()

    return (
        training_feature_array,
        training_label_array,
        test_feature_array,
        test_label_array,
    )


""" training_size = 400
test_size = 1
n = 3
one_hot = False

"""

"""
class_labels = [r"A", r"B", r"C"]

data, target = datasets.load_wine(return_X_y=True)


# Now reduce number of features to number of qubits
pca = PCA(n_components=n).fit(data)
data = pca.transform(data)

std_scale = StandardScaler().fit(data)
data = std_scale.transform(data)

# Scale to the range (-1,+1)
samples = data
minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
samples = minmax_scale.transform(samples)

training_input = {
        key: (samples[target == k, :])[:training_size]
        for k, key in enumerate(class_labels)
    }

training_feature_array, training_label_array = features_and_labels_transform(
    training_input, class_labels, one_hot
)

print(training_feature_array)
print(training_label_array)

print(len(training_feature_array)) """
