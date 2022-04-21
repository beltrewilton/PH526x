import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def find_nearest_neighborns(p, points, k=5):
    """
    Find the k nearest of point p, return the their indices.
    :param p:
    :param points:
    :param k:
    :return:
    """
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = np.sqrt(np.power(p - points[i], 2))
    ind = np.argsort(distances)
    return ind[0:k]


def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighborns(p, points, k)
    mode, count = stats.mode(outcomes[ind])
    return mode, count


def generate_synth_data(n=50):
    """
    Create 2 sets of points from bivariate normal distribution.
    :param n:
    :return:
    """
    points = np.concatenate((stats.norm(0,1).rvs((n,2)), stats.norm(1,1).rvs((n,2))), axis=0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)), axis=0)
    return points, outcomes


def plot_synth_data(points, outcomes, n=50):
    """
    helper plot function
    :param poins:
    :param outcomes:
    :return:
    """
    plt.figure('synth data')
    plt.plot(points[:n, 0], points[:n, 1], "ro")
    plt.plot(points[n:, 0], points[n:, 1], "bo")
    plt.show()


print()