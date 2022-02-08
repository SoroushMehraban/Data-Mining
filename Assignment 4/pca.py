import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def pca(D, r):
    n = D.shape[0]

    miu = D.mean(axis=0)
    Z = D - miu
    cov_matrix = (1 / n) * (Z.T @ Z)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    Ur = eigen_vectors[:, :r]

    A = (Ur.T @ D.T).T
    return A


if __name__ == '__main__':
    df = pd.read_excel("dataset2.xlsx")
    data = df.drop('class', axis=1).to_numpy()
    classes = df['class'].to_numpy()

    scaler = StandardScaler()
    scaler.fit(data)
    D = scaler.transform(data)

    A = pca(D, r=2)
    X_train, X_test, y_train, y_test = train_test_split(A, classes, test_size=0.2)

    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)

    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)

    print(f"Train accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
