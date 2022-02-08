import pandas as pd
import numpy as np


def train_test_split(data, train_ratio, seed=None):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    if train_ratio > 1 or train_ratio < 0:
        print("ERROR: TRAIN RATIO SHOULD BE BETWEEN 0 AND 1")
        exit()

    data_length = data.shape[0]
    train_size = int(train_ratio * data_length)
    # https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros
    if seed:
        np.random.seed(seed)
    indices = np.random.permutation(data_length)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    return data[train_indices, :], data[test_indices, :]
