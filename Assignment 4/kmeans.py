import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def initialize_centers(X, k):
    """
    Initialize random centers
    `centers` is a matrix with K rows, each row is one center and each column is a feature
    """
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)

    x_values = np.random.uniform(low=x_min, high=x_max, size=(k, 1))
    y_values = np.random.uniform(low=y_min, high=y_max, size=(k, 1))

    centers = np.hstack((x_values, y_values))
    return centers


def euclidean_distance(entry, center):
    return np.linalg.norm(entry - center)


def find_closest_centers(X, centers):
    idx = []
    for entry in X:
        distances = [euclidean_distance(entry, center) for center in centers]
        idx.append(np.argmin(distances))
    return np.array(idx)


def compute_means(X, idx, k):
    centers = []
    for i in range(k):
        cluster_entries = X[idx == i]
        if len(cluster_entries) == 0:
            new_center = initialize_centers(X, k=1).tolist()
        else:
            new_center = cluster_entries.mean(axis=0)
        centers.append(new_center)
    return np.array(centers)


def k_means(X, k, n):
    """
    :param X: The matrix of input data, each row is a data point and each column is a data feature
    :param k: the number of desired clusters
    :param n: the number of iteration that we want the k-means to run on the data
    :return: Vector idx such that idx[i] is the index of the center assigned to example data point i
    """
    centers_evolutions = []

    centers = initialize_centers(X, k)
    centers_evolutions.append(centers)
    for iteration in range(n):
        idx = find_closest_centers(X, centers)
        centers = compute_means(X, idx, k)

    return idx


if __name__ == '__main__':
    X = pd.read_csv('Dataset1.csv').to_numpy()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
    for i, k in enumerate(range(2, 5)):
        idx = k_means(X, k, n=15)
        axes[i].scatter(X[:, 0], X[:, 1], c=idx)
        axes[i].set(title=f"K-means with k = {k}", xlabel='x', ylabel='y')
    plt.tight_layout(h_pad=3)
    plt.show()
