# Import necessary module
import itertools
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import random


class KernelPerceptron:
    """
    Kernel perceptron
    provide polynomial decision boundary
    """
    def __init__(self, epoch=100):
        """
        :param epoch: number of iteration
        self.alpha --> dual form of w
        self.b --> bias of our model
        self.x_train --> train dataset. will be updated on the fit
        self.y_train --> test dataset. Will be updated on the fit
        """
        self.alpha = None
        self.b = None
        self.epoch = epoch
        self.x_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Train the model
        :param X_train: train data
        :param y_train: label of train data
        :return:
        """
        # m --> number of training example
        # num_features --> number of input feature
        m, num_features = X_train.shape
        # Set the input feature
        self.x_train = X_train
        self.y_train = y_train
        # Set the alpha to all zeros
        self.alpha = np.zeros(m)
        self.b = 0
        for _ in range(self.epoch):
            for i in range(m):
                # Update on each training example
                y_true = y_train[i]
                # Get prediction for this training example
                y_pred = self.predict(np.expand_dims(X_train[i], axis=0))[0]
                # We will update only if our prediction doesn't match with the true label
                if y_true != y_pred:
                    # Update alpha and b
                    self.alpha[i] = self.alpha[i] + 1.0
                    self.b = self.b + y_train[i]

    def predict(self, X_test):
        """
        Make prediction
        :param X_test: Data to predict
        :return:
        """
        # Set all the prediction to zero first
        y_pred = np.zeros(X_test.shape[0])
        # We don't need all the training exmaple
        # We will only use the training example that have non zero alpha value
        # Those are our support vector we will only use those to make prediction
        x_train_used = self.x_train[self.alpha != 0]
        alpha_used = self.alpha[self.alpha != 0]
        y_train_used = self.y_train[self.alpha != 0]
        for i, x in enumerate(X_test):
            # if all the alpha value is zero(training initialization) then we will use set wx = 0. Cause we don't have
            # any training data at this point
            if x_train_used.shape[0] > 0:
                wx = (1 + np.dot(x_train_used, x)**2) * alpha_used * y_train_used
            else:
                wx = [0.]
            # Make prediction based on the sign
            y_pred[i] = np.sign(np.sum(wx) + self.b)
        # Making sure our return is an integer
        return y_pred.astype(int)

    def get_f1_score(self, X_test, y_test):
        """
        return the f1 accuracy of our model
        :param X_test: data for which we will measure accuracy
        :param y_test: true label of X_test
        :return:
        """
        return f1_score(y_test, self.predict(X_test))

    def show_decision_boundary(self, x, y):
        """
        Show the decision boundary that was predicted by our model
        :param X: Data to plot on the decision boundary
        :param y: label of x
        :return:
        """
        colors = itertools.cycle(['r', 'g', 'b', 'c', 'y', 'm', 'k'])
        markers = itertools.cycle(['o', 'v', '+', '*', 'x', '^', '<', '>'])
        # Determine the x1- and x2- limits of the plot
        x1min = min(x[:, 0]) - 1
        x1max = max(x[:, 0]) + 1
        x2min = min(x[:, 1]) - 1
        x2max = max(x[:, 1]) + 1
        plt.xlim(x1min, x1max)
        plt.ylim(x2min, x2max)
        # Plot the data points
        k = int(max(y)) + 1
        for label in list(np.unique(y)):
            plt.plot(x[(y == label), 0], x[(y == label), 1], marker=next(markers), color=next(colors), linestyle='',
                     markersize=8)
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
    # Seeding for getting consistant data
    random.seed(0)
    # Get data
    X, y = make_circles(noise=0.1, factor=0.5, random_state=1)
    # convert zero prediction to -1
    y[y == 0] = -1
    # Split training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # Initialize our model
    clf = KernelPerceptron()
    # Train our model
    clf.fit(X_train, y_train)
    # Show the decision boundary that we predicted
    clf.show_decision_boundary(X, y)
    print(f"F1 score of our model for train data: {clf.get_f1_score(X_train, y_train)}")
    print(f"F1 score of our model for test data: {clf.get_f1_score(X_test, y_test)}")
