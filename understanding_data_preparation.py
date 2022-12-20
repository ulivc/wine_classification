import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from qiskit.utils import optionals
#from .dataset_helper import features_and_labels_transform
#from ..deprecation import deprecate_function

test_size = 51

data, target = datasets.load_wine(return_X_y=True)

sample_train, sample_test, label_train, label_test = train_test_split(
        data, target, test_size=3, random_state=7
    )

print(sample_test)

# Now we standardize for gaussian around 0 with unit variance
std_scale = StandardScaler().fit(sample_train)
sample_train = std_scale.transform(sample_train)
sample_test = std_scale.transform(sample_test)

print(sample_test)