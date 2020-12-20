import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as sklearn_perceptron


class Perceptron:
    """
    Implementation of perceptron algorithm.
    This algorithm works only for binary classification
    """
    def __init__(self, num_iter=100):
        """
        :parameter
        w: contains the weight of the model. shape -> (number_of_input_feature, )
        b: intercept of the model. shape -> (1, )
        num_iter: total number of iteration through the training set
        """
        self.w = None
        self.b = 0
        self.num_iter = num_iter

    def fit(self, x_train, y_train):
        """
        This method trains the model
        :param x_train: training data of shape (number_of_training_example, number_of_input_feature)
        :param y_train: label of the training example of shape (number of training_example, )
        :return:
        """
        # m represent the number of training example
        # n_feature represent of number of input feature
        m, n_feature = x_train.shape
        # Binary classification will have threshold either -1 or 1.
        # most of the training label have threshold 0 and 1.
        # So we are making sure labels 0 is actually label -1
        # convert 0 to -1
        y_train[y_train == 0] = -1
        # Initialize w with 0 vector
        self.w = np.zeros(n_feature)
        # Number of time we will iter through the training example
        for _ in range(self.num_iter):
            # We will update w and b for all training example
            for i in range(m):
                # We will update w and b only if our prediction is wrong for this particular training example
                if self.predict(x_train[i, :]) != y_train[i]:
                    # we will update w and b here
                    self.w = self.w + y_train[i] * x_train[i, :]
                    self.b = self.b + y_train[i]

    def predict(self, x_test):
        """
        Make prediction of our model
        :param x_test: Data for which we will make prediction
        :return:
        """
        # * 2) -1 ensures that we have either -1 or 1 in the prediction
        return (((np.dot(x_test, self.w) + self.b) > 0) * 2) - 1

    def get_f1_score(self, x_test, y_test):
        """
        measure f1 score for our model
        :param x_test: data for which we will calculate the f1_score
        :param y_test: true label of x_test
        :return:
        """
        return f1_score(y_test, self.predict(x_test))

    def show_decision_boundary(self, X, y):
        """
        Show the decision boundary that was predicted by our model
        :param X: Data to plot on the decision boundary
        :param y: label of x
        :return:
        """
        # Make some rooms around the plot
        x1min = min(X[:, 0]) - 1
        x1max = max(X[:, 0]) + 1
        x2min = min(X[:, 1]) - 1
        x2max = max(X[:, 1]) + 1
        plt.xlim(x1min, x1max)
        plt.ylim(x2min, x2max)
        # Plot the data
        grid_spacing = 0.05
        plt.plot(X[(y == 1), 0], X[(y == 1), 1], 'ro')
        plt.plot(X[(y == -1), 0], X[(y == -1), 1], 'k^')
        # plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
        xx1, xx2 = np.meshgrid(np.arange(x1min, x1max, grid_spacing), np.arange(x2min, x2max, grid_spacing))
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        Z = self.predict(grid)
        # Show the classifier's boundary using a color plot
        Z = Z.reshape(xx1.shape)
        plt.pcolormesh(xx1, xx2, Z, cmap=plt.cm.PRGn, vmin=-3, vmax=3)
        plt.show()


if __name__ == '__main__':
    # Generate random data
    X, Y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1, n_samples=100)
    # Convert 0 label to -1
    Y[Y == 0] = -1
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    # Train our model
    our_perceptron = Perceptron()
    our_perceptron.fit(X_train, y_train)
    print(f"Train F1 score is: {our_perceptron.get_f1_score(X_train, y_train)}")
    print(f"Test F1 score is {our_perceptron.get_f1_score(X_test, y_test)}")
    print(f"===========================================")
    our_perceptron.show_decision_boundary(X, Y)

    # measure performance with sklearn model
    clf = sklearn_perceptron()
    clf.fit(X_train, y_train)
    print(f"Train F1 score for sklearn model: {f1_score(y_train, clf.predict(X_train))}")
    print(f"Test F1 score for sklearn model: {f1_score(y_test, clf.predict(X_test))}")
