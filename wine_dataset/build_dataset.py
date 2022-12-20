from sklearn.utils import shuffle
from qiskit_machine_learning.datasets import wine
from sklearn.model_selection import train_test_split
import numpy as np

SEED = 3142

def _prepare_data(training_size, feature_size):
    data = np.loadtxt(f"wine_dataset/q_wine_data")
    target = np.loadtxt(f"wine_dataset/q_wine_target")

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, target, test_size=TEST_SIZE, random_state=7
    )
    dataset = [train_data, train_labels, test_data, test_labels]
    return dataset


data, target, unused_data, unused_labels = wine(
        training_size=177,
        test_size=1,
        n=3,
        one_hot=False,
    )
#data, target = shuffle(data, target, random_state=SEED)
set = [data, target]


np.savetxt(
        f"wine_dataset/q_wine_data",
        data,
    ) 
np.savetxt(
        f"wine_dataset/q_wine_target",
        target,
    ) 

