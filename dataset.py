from sklearn.utils import shuffle
from qiskit_machine_learning.datasets import wine

TEST_SIZE = 50
SEED = 3142


def prepare_data(training_size, feature_size):
    # prepare data
    train_data, train_labels, test_data, test_labels = wine(
        training_size=training_size,
        test_size=TEST_SIZE,
        n=feature_size,
        one_hot=False,
    )
    train_data, train_labels = shuffle(train_data, train_labels, random_state=SEED)
    dataset = [train_data, train_labels, test_data, test_labels]
    return dataset
