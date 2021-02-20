from collections import Counter
from models.tree.decision_tree.decision_tree import DecisionTree
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder


class AdaBoost:
    def __init__(self, base_estimator=None, n_estimator=20):
        """

        :param base_estimator: estimator from which the boosted ensemble is build. e.g. "decision, tree, linear regression"
        :param n_estimator: Number of base estimator
        :param learning_rate:
        """
        self.n_estimator = n_estimator
        # Number of training example
        self.m = None
        # Number of input feature
        self.n_features = None
        # Number of output class
        self.n_class = None
        # will contains the all the weak estimators
        self.estimators = []
        # If no estimator is given then we will use decision tree as base estimator
        if not base_estimator:
            # This base estimator will work as a stump.
            # This will be the weak classifier
            base_estimator = DecisionTree(maximum_depth=1)
            # base_estimator = DecisionTreeClassifier(max_depth=1)
        self.base_estimator = base_estimator

    def fit(self, X_train, y_train):
        """
        Train the model
        :param X_train:
        :param y_train:
        :return:
        """
        self.m, self.n_features = X_train.shape
        self.n_class = np.unique(y_train).size
        # set the training weight
        training_weight = np.full((self.m, ), 1/self.m)
        for _ in range(self.n_estimator):
            self.base_estimator.fit(X_train, y_train, training_weight)
            # Compute weighted error for this estimator
            # Predict using the trained estimator
            single_estimator_prediction = self.base_estimator.predict(X_train)
            misclassified_datas = single_estimator_prediction != y_train
            r = np.sum(training_weight[misclassified_datas]) / np.sum(training_weight)
            alpha = np.log((1-r)/r) / 2
            # re-weight the dataset
            # training_weight = training_weight * np.exp(alpha * misclassified_datas)
            training_weight = np.exp(alpha * misclassified_datas * ((training_weight > 0) | (alpha < 0)))
            self.estimators.append((self.base_estimator, alpha))
            if r == 0:
                return

    def predict(self, X_test):
        overall_prediction = np.zeros((X_test.shape[0], self.n_class))
        for estimator, alpha in self.estimators:
            single_prediciton = np.asarray(estimator.predict(X_test))
            # Convert into one hot encoding
            single_prediciton_one_hot = np.zeros((X_test.shape[0], self.n_class))
            single_prediciton_one_hot[np.arange(single_prediciton.shape[0]), single_prediciton] = alpha
            overall_prediction = np.add(overall_prediction, single_prediciton_one_hot)
        return np.argmax(overall_prediction, axis=1)


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    #X = X[y != 2]
    #y = y[y != 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # ===================================
    # ======= Test our model here =======
    # ===================================
    our_model = AdaBoost()
    our_model.fit(X_train, y_train)
    print("Training f1 score of sklearn adaboost model is: ",
          str(f1_score(our_model.predict(X_train), y_train, average='micro')))
    print("Testing f1 score of sklearn adaboost model is: ",
          str(f1_score(our_model.predict(X_test), y_test, average='micro')))
    # ====================================
    # ===== Test sklearn model here ======
    # ====================================
    # This will use decision tree classifier with maximum depth 1
    sklearn_adaboost = AdaBoostClassifier()
    sklearn_adaboost.fit(X_train, y_train)
    print("Training f1 score of sklearn adaboost model is: ",
          str(f1_score(sklearn_adaboost.predict(X_train), y_train, average='micro')))
    print("Testing f1 score of sklearn adaboost model is: ",
          str(f1_score(sklearn_adaboost.predict(X_test), y_test, average='micro')))



