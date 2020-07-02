# Code courtesy: https://towardsdatascience.com/support-vector-machine-python-example-d67d9b63f1c8
# Theory: https://www.youtube.com/watch?v=_PwhiWxHK8o
import numpy as np
import cvxopt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


class SVM:
    """

    """
    def __init__(self):
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_y = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        # P = X_train^T X_train
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X_train[i], X_train[j])

        P = cvxopt.matrix(np.outer(y_train, y_train) * K)
        # q = -1 (1xN)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        # A = y_train^T
        A = cvxopt.matrix(y_train, (1, n_samples))
        # b = 0
        b = cvxopt.matrix(0.0)
        # -1 (NxN)
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        # 0 (1xN)
        h = cvxopt.matrix(np.zeros(n_samples))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        # Lagrange have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.alpha = a[sv]
        self.support_vectors = X_train[sv]
        self.support_vector_y = y_train[sv]
        # Intercept
        self.b = 0
        for n in range(len(self.alpha)):
            self.b += self.support_vector_y[n]
            self.b -= np.sum(self.alpha * self.support_vector_y * K[ind[n], sv])
        self.b /= len(self.alpha)
        # Weights
        self.w = np.zeros(n_features)
        for n in range(len(self.alpha)):
            self.w += self.alpha[n] * self.support_vector_y[n] * self.support_vectors[n]

    def predict(self, X_test):
        return self.sign(np.dot(X_test, self.w) + self.b)

    def f1_score(self, X_test, y_test):
        pass


if __name__ == '__main__':
    X, y = make_blobs(n_samples=250, centers=2, random_state=100, cluster_std=1)
    y[y == 0] = -1
    tmp = np.ones(len(X))
    y = tmp * y

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    svm = SVM()
    svm.fit(X_train, y_train)


    def f(x, w, b, c=0):
        return (-w[0] * x - b + c) / w[1]


    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter')
    # w.x + b = 0
    a0 = -4
    a1 = f(a0, svm.w, svm.b)
    b0 = 4;
    b1 = f(b0, svm.w, svm.b)
    plt.plot([a0, b0], [a1, b1], 'k')
    # w.x + b = 1
    a0 = -4;
    a1 = f(a0, svm.w, svm.b, 1)
    b0 = 4;
    b1 = f(b0, svm.w, svm.b, 1)
    plt.plot([a0, b0], [a1, b1], 'k--')
    # w.x + b = -1
    a0 = -4;
    a1 = f(a0, svm.w, svm.b, -1)
    b0 = 4;
    b1 = f(b0, svm.w, svm.b, -1)
    plt.plot([a0, b0], [a1, b1], 'k--')
    plt.show()