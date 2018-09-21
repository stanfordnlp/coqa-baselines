import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt


################################################################################
# Graphing Functions #
################################################################################

def plot_learn(values, yAxis, xAxis, title=None, saveTo=None):
    """
    Plots the learning curve with train/val for all values. Limited to
    7 learning curves on the same graph as we only have 7 colours.

    Args:
        1. values: Dictionary of tuples of lists, where the tuple is ([train values], [dev values])
            and the key is the name of the model for the graph label.
        2. yAxis: Either 'Loss', 'F1' or 'Exact Match'
        3. xAxis: 'Epochs' or 'Iterations'
        4. title: optional title to the graph
        5. saveTo: save location for graph
    """
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, (k, (train_values, dev_values)) in enumerate(values.items()):

        plt.plot(map(float, train_values), linewidth=2, color=colours[i],
                 linestyle='--', label="Train {} {}".format(yAxis, k))
        if dev_values:
            plt.plot(map(float, dev_values), linewidth=2, color=colours[i],
                     linestyle='-', label="Dev {} {}".format(yAxis, k))

    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    if title:
        plt.title(title)

    if yAxis == "Loss":
        plt.legend(loc='upper right', shadow=True, prop={'size': 6})
    else:
        plt.legend(loc='upper left', shadow=True, prop={'size': 6})

    assert saveTo
    plt.savefig("{}".format(saveTo))
    plt.cla()
    plt.clf()
    plt.close()


def plot_metrics(values, yAxis, xAxis, title=None, saveTo=None):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, (train_values, dev_values, metric) in enumerate(values):
        plt.plot(map(float, train_values), linewidth=2, color=colours[i],
                 linestyle='-', label="Train {}".format(metric))
        if dev_values:
            plt.plot(map(float, dev_values), linewidth=2, color=colours[i],
                     linestyle='--', label="Dev {}".format(metric))

    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    if title:
        plt.title(title)

    if yAxis == "Loss":
        plt.legend(loc='upper right', shadow=True, prop={'size': 6})
    else:
        plt.legend(loc='upper left', shadow=True, prop={'size': 6})

    assert saveTo
    plt.savefig("{}".format(saveTo))
    plt.cla()
    plt.clf()
    plt.close()
