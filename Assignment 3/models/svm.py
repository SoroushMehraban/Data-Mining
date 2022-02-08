import numpy as np
import matplotlib.pyplot as plt


class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iter=1000):
        """
        LinearSVM using hinge loss and gradient descent

        :param learning_rate: Learning rate of gradient descent optimization
        :param lambda_param: Regularization Parameter
        :param n_iter: Number of iterations
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iter = n_iter

        self.w = None
        self.b = None

    def train(self, X, y):
        """
        Trains the Linear SVM model using gradient descent

        :param X: Training features
        :param y: Training labels
        """
        n_features = X.shape[1]

        """
        Since naturally SVM classifies our data as 1 and -1 and given dataset's labels are 0 and 1, we use the following
        function to convert 0 labels to -1 and others to 1
        """
        y = np.where(y == 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for iteration in range(self.n_iter):
            for i, x_i in enumerate(X):
                classifies_correctly = y[i] * (np.dot(x_i, self.w) - self.b) >= 1
                if classifies_correctly:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[i]))
                    self.b -= self.learning_rate * y[i]

    def predict(self, X):
        """
        Classifies given data as 1 and -1

        :param X: given data
        :return: Labels of each data as 1 and -1
        """
        try:
            return np.sign(np.dot(X, self.w) - self.b)
        except TypeError:
            print("ERROR: You have to train the model first")
            exit()

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """
        Calculates accuracy of our model

        :param y_true: Our labels
        :param y_pred: Model prediction
        :return: Accuracy fo our model
        """
        y_true = np.where(y_true == 0, -1, 1)

        correct_classifications = np.count_nonzero(y_true == y_pred)
        n_samples = y_true.shape[0]

        return (correct_classifications / n_samples) * 100

    def visualize(self, X):
        def get_hyperplane_point(x, w, b, offset):
            """
            Finds a point on the following hyperplane:
            w0x + w1y - b = offset => y = (offset - w0x + b) / w1

            :param x: independent variable of the hyperplane
            :param w: weights of hyperplane
            :param b: y-intercept
            :param offset: Either 1, 0, or -1
            :return: A point on the hyperplane
            """
            y = (offset - w[0] * x + b) / w[1]
            return y

        y_pred = self.predict(X)

        fig, ax = plt.subplots()
        ax.scatter(x=X[:, 0], y=X[:, 1], c=y_pred)

        min_x0 = np.amin(X[:, 0])
        max_x0 = np.amax(X[:, 0])

        x1_1_1 = get_hyperplane_point(min_x0, self.w, self.b, offset=-1)
        x1_1_2 = get_hyperplane_point(max_x0, self.w, self.b, offset=-1)

        x1_mid_1 = get_hyperplane_point(min_x0, self.w, self.b, offset=0)
        x1_mid_2 = get_hyperplane_point(max_x0, self.w, self.b, offset=0)

        x1_2_1 = get_hyperplane_point(min_x0, self.w, self.b, offset=1)
        x1_2_2 = get_hyperplane_point(max_x0, self.w, self.b, offset=1)

        ax.plot([min_x0, max_x0], [x1_1_1, x1_1_2], 'k')
        ax.plot([min_x0, max_x0], [x1_mid_1, x1_mid_2], 'y--')
        ax.plot([min_x0, max_x0], [x1_2_1, x1_2_2], 'k')

        plt.show()
