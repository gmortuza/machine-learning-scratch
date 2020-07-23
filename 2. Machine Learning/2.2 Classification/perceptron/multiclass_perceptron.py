import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as sklearn_perceptron


class multiClassPerceptron:

    def __init__(self, num_iter=1000):
        """

        :param num_iter:
        """
        self.w = None
        self.b = None
        self.num_class = None
        self.num_iter = num_iter

    def fit(self, X_train, y_train):
        m, num_feature = X_train.shape
        self.num_class = len(np.unique(y_train))
        # Every class will have its own weights and intercept/bias
        self.w = np.zeros((self.num_class, num_feature))
        self.b = np.zeros(self.num_class)
        for _ in range(self.num_iter):
            # We will check every training example
            for i in range(m):
                # Make prediction for this training example
                y_pred = self.predict(np.expand_dims(X_train[i], axis=0))
                y_true = y_train[i]
                if y_pred != y_true:
                    # if our prediction is wrong we will update the weights only for this true label
                    self.w[y_true, :] += X_train[i, :]
                    self.b[y_true] += 1.0
                    # We will also penalize the wrong predicted label
                    self.w[y_pred, :] -= X_train[i, :]
                    self.b[y_pred] -= 1.0

    def predict(self, X_test):
        # will contain the score for each class
        scores = np.zeros((X_test.shape[0], self.num_class))
        for i in range(self.num_class):
            scores[:, i] = np.dot(X_test, self.w[i]) + self.b[i]
        return np.argmax(scores) if X_test.shape[0] == 1 else np.argmax(scores, axis=1)

    def get_f1_score(self, X_test, y_test):
        return f1_score(y_test, self.predict(X_test), average='weighted')

    def show_decision_boundary(self, x, y):
        # Determine the x1- and x2- limits of the plot
        x1min = min(x[:, 0]) - 1
        x1max = max(x[:, 0]) + 1
        x2min = min(x[:, 1]) - 1
        x2max = max(x[:, 1]) + 1
        plt.xlim(x1min, x1max)
        plt.ylim(x2min, x2max)
        # Plot the data points
        k = int(max(y)) + 1
        cols = ['ro', 'k^', 'b*', 'gx']
        for label in range(k):
            plt.plot(x[(y == label), 0], x[(y == label), 1], cols[label % 4], markersize=8)
        # Construct a grid of points at which to evaluate the classifier
        grid_spacing = 0.05
        xx1, xx2 = np.meshgrid(np.arange(x1min, x1max, grid_spacing), np.arange(x2min, x2max, grid_spacing))
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        Z = self.predict(grid)
        # Show the classifier's boundary using a color plot
        Z = Z.reshape(xx1.shape)
        plt.pcolormesh(xx1, xx2, Z, cmap=plt.cm.Pastel1, vmin=0, vmax=k)
        plt.show()


if __name__ == '__main__':
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, n_classes=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    perceptron = multiClassPerceptron()
    perceptron.fit(X_train, y_train)

    pred = perceptron.predict(X_test)
    print(f"Training f1 score: {perceptron.get_f1_score(X_train, y_train)}")
    print(f"Testing f1 score: {perceptron.get_f1_score(X_test, y_test)}")
    print("=============================================")
    perceptron.show_decision_boundary(X, y)

    # measure performance with sklearn model
    clf = sklearn_perceptron()
    clf.fit(X_train, y_train)
    print(f"Train F1 score for sklearn model: {f1_score(y_train, clf.predict(X_train), average='weighted')}")
    print(f"Test F1 score for sklearn model: {f1_score(y_test, clf.predict(X_test), average='weighted')}")