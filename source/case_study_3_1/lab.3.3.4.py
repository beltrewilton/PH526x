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
        distances[i] = np.sum(np.sqrt(np.power(p - points[i], 2)))
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
    points = np.concatenate((stats.norm(0, 1).rvs((n, 2)), stats.norm(1, 1).rvs((n, 2))), axis=0)
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


def plot_prediction_grid (xx, yy, prediction_grid):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(xx[:,0], yy [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.show()


def make_prediction_grid(predictors, outcomes, limits, h, k):
    """
    Classsify each point in the prediction grid.
    :param predictors:
    :param outcomes:
    :param limits:
    :param h:
    :param k:
    :return:
    """
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype=int)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)

    return xx, yy, prediction_grid

points, outcomes = generate_synth_data()

k=5; limits = (-3, 4, -3, 4); h = .1
(xx, yy, prediction_grid) = make_prediction_grid(points, outcomes, limits, h=h, k=k)
plot_prediction_grid(xx, yy, prediction_grid)
