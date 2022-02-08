import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self, train_set, test_set, max_depth=3, min_size=10, criterion='GINI'):
        """
        :param train_set: train set of our dataset which is a numpy array
        :param test_set: test set of our dataset which is a numpy array
        :param max_depth: maximum depth that we want to expand. Default is 3
        :param min_size: minimum size of each node. Default is 10
        :param criterion: Either GINI or ENTROPY. Default is GINI
        """
        self.max_depth = max_depth
        self.min_size = min_size
        self.criterion = criterion

        self.train_set = train_set
        self.test_set = test_set

        self.tree = self.build_tree(self.train_set)

    def get_accuracy(self):
        predictions = self.predict()
        labels = self.test_set[:, -1]

        accuracy = ((predictions == labels).sum() / labels.shape[0]) * 100
        return accuracy

    def change_test_set(self, new_test_set):
        if isinstance(new_test_set, pd.DataFrame):
            new_test_set = new_test_set.to_numpy()
        self.test_set = new_test_set

    def predict(self):
        predictions = []
        for row in self.test_set:
            predictions.append(self.predict_row(self.tree, row))
        return np.array(predictions)

    def cost(self, groups):
        if self.criterion == "GINI":
            return self.gini_index(groups)
        elif self.criterion == "ENTROPY":
            return self.entropy(groups)
        else:
            print("ERROR: INVALID CRITERION")
            exit()

    @staticmethod
    def gini_index(groups):
        number_of_instances = np.sum([group.shape[0] for group in groups])

        gini = 0
        for group in groups:
            group_size = group.shape[0]
            if group_size == 0:
                continue

            group_classes = group[:, -1]
            _, counts = np.unique(group_classes, return_counts=True)
            p = counts / group_size
            sigma = np.sum(p * p)
            group_weight = group_size / number_of_instances
            gini += (1 - sigma) * group_weight

        return gini

    @staticmethod
    def entropy(groups):
        number_of_instances = np.sum([group.shape[0] for group in groups])

        entropy = 0
        for group in groups:
            group_size = group.shape[0]
            if group_size == 0:
                continue

            group_classes = group[:, -1]
            _, counts = np.unique(group_classes, return_counts=True)
            p = counts / group_size
            sigma = np.sum(p * np.log2(p))
            group_weight = group_size / number_of_instances
            entropy += (- sigma) * group_weight

        return entropy


    @staticmethod
    def binary_split(feature_index, data):
        left = data[data[:, feature_index] == 0]
        right = data[data[:, feature_index] == 1]
        return np.array([left, right], dtype=object)

    def get_split(self, data):
        best_index, best_score, best_groups = float('inf'), float('inf'), None
        for feature_index in range(data.shape[1] - 1):  # except last column
            groups = self.binary_split(feature_index, data)
            cost_score = self.cost(groups)
            if cost_score < best_score:
                best_index, best_score, best_groups = feature_index, cost_score, groups
        return {
            'index': best_index,
            'groups': best_groups
        }

    @staticmethod
    def to_terminal(group):
        outcomes = group[:, -1]
        unique_outcomes, counts = np.unique(outcomes, return_counts=True)
        most_frequent_index = np.argmax(counts)
        return unique_outcomes[most_frequent_index]

    def split(self, node, depth):
        left, right = node['groups']
        del (node['groups'])

        if left.shape[0] == 0 or right.shape[0] == 0:
            node['left'] = node['right'] = self.to_terminal(np.vstack((left, right)))
            return
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return

        if left.shape[0] <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth + 1)

        if right.shape[0] <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth + 1)

    def build_tree(self, train_set):
        root = self.get_split(train_set)
        self.split(root, 1)
        return root

    def predict_row(self, node, row):
        if row[node['index']] == 0:
            if isinstance(node['left'], dict):
                return self.predict_row(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_row(node['right'], row)
            else:
                return node['right']

    def visualize_tree(self, current_node=None, depth=0):
        if current_node is None:
            if self.tree is not None:
                print(f"|--- feature_{self.tree['index']} == 0")
                self.visualize_tree(self.tree['left'], depth=1)
                print(f"|--- feature_{self.tree['index']} == 1")
                self.visualize_tree(self.tree['right'], depth=1)
            else:
                print("ERROR: TRAIN DATA FIRST")
        else:
            if isinstance(current_node, dict):
                print("|  " * depth + f"|--- feature_{current_node['index']} == 0")
                self.visualize_tree(current_node['left'], depth=depth + 1)
                print("|  " * depth + f"|--- feature_{current_node['index']} == 1")
                self.visualize_tree(current_node['right'], depth=depth + 1)
            else:
                print("|  " * depth + f"|--- class {current_node}")
