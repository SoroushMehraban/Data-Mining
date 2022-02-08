import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self, train_data, test_data):
        self.train_data, self.test_data = train_data, test_data
        self.classes = np.unique(self.train_data[:, -1])  # unique is always sorted (Important for argmax)

    def predict(self):
        predictions = []
        for test_row in self.test_data:
            prediction = self.predict_class(test_row)
            predictions.append(prediction)
        return predictions

    def predict_class(self, test_row):
        """
        P(y|x1, x2, ..., xn) = P(x1|y)P(x2|y)...P(xn|y)P(y)
        We don't consider denominator because it's the same for different classes and we want to compare them.
        """
        P_y_x = []
        for y in self.classes:
            class_data = self.train_data[np.where(self.train_data[:, -1] == y)]

            P_y = class_data.size / self.test_data.size

            P_x_y = 1
            for col in range(self.test_data.shape[1] - 1):  # except the last column
                P_x_y *= np.where(class_data[:, col] == test_row[col])[0].size / class_data.size

            P_y_x.append(P_y * P_x_y)

        return np.argmax(P_y_x)

    def get_accuracy(self):
        predictions = self.predict()

        test_labels = self.test_data[:, -1]
        true_estimations = np.count_nonzero(predictions == test_labels)
        return (true_estimations / len(self.test_data)) * 100

    def change_test_set(self, new_test_set):
        if isinstance(new_test_set, pd.DataFrame):
            new_test_set = new_test_set.to_numpy()
        self.test_data = new_test_set
