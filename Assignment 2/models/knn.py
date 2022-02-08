import numpy as np
import pandas as pd


class KNearestNeighbor:
    def __init__(self, train_data, test_data, k):
        self.train_data, self.test_data = train_data, test_data
        self.k = k

    @staticmethod
    def euclidean_distance(row1, row2):
        return np.sqrt(np.sum(np.power(row1 - row2, 2)))

    def get_k_neighbors(self, target_row):
        distances = []
        for train_row in self.train_data:
            if train_row.size == target_row.size:
                distance = self.euclidean_distance(train_row[:-1], target_row[:-1])
            else:
                distance = self.euclidean_distance(train_row[:-1], target_row)
            distances.append(distance)

        sorted_indices = sorted(range(len(self.train_data)), key=lambda i: distances[i])
        neighbors = self.train_data[sorted_indices[:self.k]]
        return neighbors

    def predict_class(self, target_row):
        neighbors = self.get_k_neighbors(target_row)
        output_values = neighbors[:, -1]

        unique_output_values, counts = np.unique(output_values, return_counts=True)
        most_frequent_index = np.argmax(counts)
        most_frequent_output_value = unique_output_values[most_frequent_index]
        return most_frequent_output_value

    def predict(self):
        predictions = []
        for test_row in self.test_data:
            prediction = self.predict_class(test_row)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions

    def get_accuracy(self):
        predictions = self.predict()

        test_labels = self.test_data[:, -1]
        true_estimations = np.count_nonzero(predictions == test_labels)
        return (true_estimations / len(self.test_data)) * 100

    def change_test_set(self, new_test_set):
        if isinstance(new_test_set, pd.DataFrame):
            new_test_set = new_test_set.to_numpy()
        self.test_data = new_test_set
