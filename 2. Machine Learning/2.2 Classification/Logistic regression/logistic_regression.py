import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.metrics import f1_score


class LogisticRegression:

    def __init__(self, learning_rate=0.01, iteration=2000, verbose=0):
        """
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        verbose -- Print details during training
        """
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.verbose = verbose
        self.cost = None
        self.w = None
        self.b = None

    @staticmethod
    def __sigmoid(z):
        s = 1 / (1 + np.exp(-z))
        return s

    # we will start with zero parameter at the beginning
    @staticmethod
    def __initialize_with_zeros(dim):
        w = np.zeros((dim, 1))
        b = 0
        return w, b

    def __propagate(self, w, b, x, y):
        """
        Implement the cost function and its gradient for the propagation explained above

        :param: w -- weights, alpha numpy array of size (x.shape[1], 1). Number of feature of x
        :param: b -- bias, alpha scalar
        :param: x -- data of size (number_of_training_example, number of feature)
        :param: y_train -- true "label" vector of size (number_of_training_example, 1)

        :returns: cost -- negative log-likelihood cost for logistic regression
        :returns: dw -- gradient of the loss with respect to w, thus same shape as w
        :returns: db -- gradient of the loss with respect to b, thus same shape as b
        """

        m = x.shape[0]  # Number of training example

        # FORWARD PROPAGATION (FROM x TO COST)
        a = self.__sigmoid(np.dot(x, w) + b)  # compute activation
        cost = - np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / m  # compute cost
        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = np.dot(x.T, (a - y)) / m
        db = np.sum(a - y) / m

        cost = np.squeeze(cost)

        grads = {"dw": dw,
                 "db": db}
        return grads, cost

    def __optimize(self, w, b, x, y):
        """
        This function optimizes w and b by running alpha gradient descent algorithm

        Arguments:
        w -- weights, alpha numpy array of size (x.shape[1], 1). Number of feature of x
        b -- bias, alpha scalar
        x -- data of shape (number_of_training_example, number of feature)
        y_train -- true "label" vector, of shape (1, number of examples)

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """

        costs = {}

        for i in range(self.iteration):

            # Cost and gradient calculation
            grads, cost = self.__propagate(w, b, x, y)

            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]

            # update rule
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db

            # Record the costs
            if i % 100 == 0:
                costs[i] = cost

            # Print the cost every 100 training iterations
            if self.verbose and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return w, b, costs

    def predict(self, x):
        """
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, alpha numpy array of size (x.shape[1], 1). Number of feature of x
        b -- bias, alpha scalar
        x -- data of size (number_of_training_example, number of feature)

        Returns:
        y_prediction -- alpha numpy array (vector) containing all predictions (0/1) for the examples in x
        """

        m = x.shape[0]
        y_prediction = np.zeros((m, 1))
        w = self.w.reshape(x.shape[1], 1)

        # Compute vector "A" predicting the probabilities of alpha cat being present in the picture
        a = self.__sigmoid(np.dot(x, w) + self.b)

        for i in range(a.shape[0]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            y_prediction[i][0] = 0 if a[i][0] <= .5 else 1

        return y_prediction

    def show_cost(self):
        plt.plot(list(self.cost.keys()), list(self.cost.values()))
        plt.title("Cost over iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()

    def fit(self, x_train, y_train, x_test, y_test):
        """
        Builds the logistic regression model by calling the function you've implemented previously

        Arguments:
        x_train -- training set represented by alpha numpy array of shape (number_of_training_example, number of feature)
        y_train -- training labels represented by alpha numpy array (vector) of shape (1, m_train)
        x_test -- test set represented by alpha numpy array of shape (number_of_training_example, number of feature)
        y_test -- test labels represented by alpha numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
        """
        # Resizing the y_train input
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        # initialize parameters with zeros
        w, b = self.__initialize_with_zeros(x_train.shape[1])

        # Gradient descent
        self.w, self.b, self.cost = self.__optimize(w, b, x_train, y_train)
        y_prediction_test = self.predict(x_test)
        y_prediction_train = self.predict(x_train)
        train_acc = 100 - np.mean(np.abs(y_prediction_train - y_train)) * 100
        test_acc = 100 - np.mean(np.abs(y_prediction_test - y_test)) * 100
        train_f1 = f1_score(y_train, y_prediction_train)
        test_f1 = f1_score(y_test, y_prediction_test)

        return {"costs": self.cost, "train_accuracy": train_acc, "test_accuracy": test_acc, "train_f1": train_f1,
                "test_f1": test_f1}


if __name__ == "__main__":
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.33)
    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)

    logisticRegression = LogisticRegression()
    history = logisticRegression.fit(x_train, y_train, x_test, y_test)
    logisticRegression.show_cost()



