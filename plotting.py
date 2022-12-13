import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def plot_loss(
    evaluations, costs, accuracy, feature_size, training_size, maxiter, reps, test_size
):
    # plot loss
    fig = plt.figure()
    plt.plot(evaluations, costs)
    plt.xlabel("Steps")
    plt.ylabel("Cost")
    plt.title(
        f"Accuracy: {accuracy}, Feature size: {feature_size}, Training_size: {training_size}, Maxiter {maxiter}, Reps: {reps}"
    )
    print(plt.show())
    fig.savefig(
        f"plots/qfullcosts_{accuracy}_{feature_size}_{training_size}_{test_size}.png"
    )


def plot_results(
    TEST_LABELS,
    predictions,
    accuracy,
    feature_size,
    training_size,
    maxiter,
    reps,
    test_size,
):
    # create array for plot
    correct_result = [0, 0, 0]
    wrong_result = [0, 0, 0]
    for label, pred in zip(TEST_LABELS, predictions):
        if np.array_equal(label, pred):
            print(pred)
            correct_result[pred] += 1

        else:
            wrong_result[pred] += 1

    fig1 = plt.figure()
    # Daten für drei Teams erstellen
    classes = ["0", "1", "2"]

    # Diagrammparameter definieren
    N = 3
    barWidth = 0.5
    xloc = np.arange(N)

    # Gestapeltes Balkendiagramm erstellen
    p1 = plt.bar(xloc, correct_result, width=barWidth)
    p2 = plt.bar(xloc, wrong_result, bottom=correct_result, width=barWidth)

    # Beschriftungen, Titel, Striche und Legende hinzufügen
    plt.ylabel("count")
    plt.xlabel("classes")
    plt.title(
        f"Accuracy: {accuracy}, Feature size: {feature_size}, Training_size: {training_size}, Maxiter {maxiter}, Reps: {reps}"
    )
    plt.xticks(xloc, ("0", "1", "2"))
    plt.yticks(np.arange(0, 41, 5))
    plt.legend((p1[0], p2[0]), ("True", "False"))
    print(plt.show())
    fig1.savefig(
        f"plots/qfullresults_{accuracy}_{feature_size}_{training_size}_{test_size}.png"
    )
