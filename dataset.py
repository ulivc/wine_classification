from sklearn.utils import shuffle
from qiskit_machine_learning.datasets import wine
import numpy as np
import understanding_data_preparation
from sklearn.model_selection import train_test_split

MAX_TEST_SIZE = 50
SEED = 3142


def prepare_data(training_size, test_size, feature_size):
    # prepare data
    train_data, train_labels, test_data, test_labels = wine(
        training_size=training_size,
        test_size=MAX_TEST_SIZE,
        n=feature_size,
        one_hot=False,
    )
    train_data, train_labels = shuffle(train_data, train_labels, random_state=SEED)
    test_data, test_labels = shuffle(test_data, test_labels, random_state=SEED)

    test_data = test_data[:test_size]
    test_labels = test_labels[:test_size]

    dataset = [train_data, train_labels, test_data, test_labels]
    return dataset


def _prepare_data(training_size, test_size, feature_size):
    # prepare data
    (
        train_data,
        train_labels,
        test_data,
        test_labels,
    ) = understanding_data_preparation.wine(
        training_size=training_size,
        test_size=test_size,
        n=feature_size,
        one_hot=False,
    )
    train_data, train_labels = shuffle(train_data, train_labels, random_state=SEED)
    # test_data, test_labels = shuffle(test_data, test_labels, random_state=SEED)

    # test_data = test_data[:test_size]
    # test_labels = test_labels[:test_size]

    dataset = [train_data, train_labels, test_data, test_labels]
    return dataset


if __name__ == "__main__":
    dataset = prepare_data(400, 12, 3)
    print(dataset[3])
    print(len(dataset[3]))
