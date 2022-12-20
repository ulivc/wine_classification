import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from qiskit.utils import optionals

from qiskit_machine_learning.datasets.dataset_helper import features_and_labels_transform

training_size = 400
test_size = 1
n = 3
one_hot = False

"""
Returns wine dataset.
This function is deprecated in version 0.4.0
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

print(len(training_feature_array))