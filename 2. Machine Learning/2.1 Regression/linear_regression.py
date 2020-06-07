import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score


class LinearRegression:

    def __init__(self, learning_rate=0.001, num_iteration=3000, verbose=False, method="gradient_descent"):
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.verbose = verbose
        self.method = method
        self.w = None
        self.b = None

    def propagate(self, w, b, x, y):
        m = x.shape[0]
        # Calculating cost
        h_x = np.dot(x, w) + b
        cost = np.sum((h_x - y) ** 2) / 2
        cost = np.squeeze(cost)

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = np.dot(x.T, (h_x - y)) / m
        db = np.sum(h_x - y) / m

        return dw, db, cost

    def optimize(self, w, b, x, y):
        costs = {}
        for i in range(self.num_iteration):
            dw, db, cost = self.propagate(w, b, x, y)
            # Updating parameter
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db

            if i % 100 == 0:
                costs[i] = cost
                if self.verbose:
                    print("Cost after iteration ", i, " is ", cost)
        return w, b, costs

    def optimize_by_normal_equation(self, x, y):
        return np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)

    def predict(self, x):
        return np.dot(x, self.w) + self.b

    def fit(self, x_train, y_train, x_test, y_test):
        # Resizing the y input
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        # initialize parameters with zeros
        w = np.zeros((x_train.shape[1], 1))
        b = 0

        w, b, costs = self.optimize(w, b, x_train, y_train)
        if self.method == 'normal_equation':
            w = self.optimize_by_normal_equation(x_train, y_train)
        self.w = w
        self.b = b
        train_r2 = r2_score(y_train, self.predict(x_train))
        test_r2 = r2_score(y_test, self.predict(x_test))
        print("Train r2 score: ", train_r2)
        print("Test r2 score: ", test_r2)

        return {"costs": costs,  "train_r2": train_r2, "test_r2": test_r2}


if __name__ == "__main__":
    data = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.33)
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    linearRegression = LinearRegression()
    history = linearRegression.fit(X_train, y_train, X_test, y_test)
    plt.plot(list(history["costs"].keys()), list(history["costs"].values()))
    plt.show()

    # Test it with

