import numpy as np


class LinearRegression:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        ones_vector = np.ones((train_data.shape[0], 1))
        self.X = np.hstack((ones_vector, train_data))

        ones_vector = np.ones((test_data.shape[0], 1))
        self.X_test = np.hstack((ones_vector, test_data))

        self.y = train_labels
        self.y_test = test_labels

        self.beta = np.zeros((self.X.shape[1], 1))

    def loss(self, mode='train'):
        if mode == 'train':
            output = self.X @ self.beta
            return np.sum((np.power(output - self.y, 2))) / 2 * output.shape[0]
        else:
            output = self.X_test @ self.beta
            return np.sum((np.power(output - self.y_test, 2))) / 2 * output.shape[0]

    def predict(self, mode='train'):
        if mode == 'train':
            return self.X @ self.beta
        else:
            return self.X_test @ self.beta

    def reset_beta(self):
        self.beta = np.zeros((self.X.shape[1], 1))

    def gradient_descent(self, iterations=1000, learning_rate=0.01):
        for iteration in range(iterations):
            output = self.X @ self.beta
            grad_beta = self.X.T @ (output - self.y) / self.X.shape[0]
            self.beta -= learning_rate * grad_beta

    def stochastic_gradient_descent(self, iterations=1000, learning_rate=0.01, batch_size=200):
        Xy = np.hstack((self.X, self.y))
        for iteration in range(iterations):
            for start_index in range(0, self.X.shape[0], batch_size):
                stop_index = start_index + batch_size
                X_batch, y_batch = Xy[start_index:stop_index, :-1], Xy[start_index:stop_index, -1:]

                output = X_batch @ self.beta
                grad_beta = X_batch.T @ (output - y_batch) / batch_size
                self.beta -= learning_rate * grad_beta
