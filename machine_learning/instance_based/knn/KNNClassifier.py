# Import necessary modules
import heapq

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    """
    K nearest neighbor is a supervised machine learning algorithm. The main intuition of this algorithm is check your
    neighbor points and make prediction based on your neighbour value. This algorithm can be used both for classification
    and regression task.
    There is no training phase of this algorithm. During training it only store the training data. But during testing
    it goes over all the training data and find it's neighbor. That's why when the training data is huge it's infeasible
    to use this algorithm. Cause for each prediction it have to go over all the training data
    """

    def __init__(self, n_neighbors=5, metric="euclidean"):
        """

        :param n_neighbors: number of neighbor to consider for labeling
        :param metric: distance metric to use euclidean/manhattan
        """
        self.n_neighbors = n_neighbors
        self.x_train = None
        self.y_train = None
        self.distance = self.__get_distance(metric)  # distance function

    def __get_distance(self, metric):
        """
        Return the distance metric
        :param metric: the distance metric to use. {euclidean/manhattan}
        :return: method for measuring distance of two points
        """
        if metric == "manhattan":
            return self.__manhattan_distance
        else:
            return self.__euclidean_distance

    @staticmethod
    def __manhattan_distance(point2):
        """
        Calculate manhattan distance
        :param point2:
        :return:
        """

        def measure_distance(point1):
            total_distance = 0
            for i, j in zip(point1, point2):
                total_distance += abs(i - j)
            return total_distance

        return measure_distance

    @staticmethod
    def __euclidean_distance(point2):
        """
        Measure the euclidean distance of two points
        :param point2:
        :return:
        """

        def measure_distance(point1):
            total_distance = 0
            for i, j in zip(point1, point2):
                total_distance += (i - j) ** 2
            return total_distance ** .5

        return measure_distance

    def fit(self, X_train, y_train):
        """
        Store the training data
        :param X_train:
        :param y_train:
        :return:
        """
        # Store the training data
        self.x_train = X_train
        self.y_train = np.expand_dims(y_train, axis=1)

    def get_label(self, neighbor_points):
        """
        Find out the label based on the nearest neighbor
        :param neighbor_points:
        :return:
        """
        labels = np.asarray(neighbor_points)[:, -1].astype(int)
        label = np.argmax(np.bincount(labels))
        return label

    def predict(self, X_test):
        """
        Make prediction of this dataset
        :param X_test:
        :return:
        """
        # Add the label with the training data itself so that we have that information on our nearest neighbor points
        data_with_label = np.concatenate((self.x_train, self.y_train), axis=1)
        # All the prediction
        prediction = []
        for single_test in X_test:
            # make prediction for this point only
            neighbor_points = heapq.nsmallest(self.n_neighbors, data_with_label, key=self.distance(single_test))
            # extract label
            label = self.get_label(neighbor_points)
            prediction.append(label)
        return np.asarray(prediction)


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # =========================================
    print("=" * 20 + "\tTest of our model\t" + "=" * 20)
    knn = KNNClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print(f"F1 score of our model is: {f1_score(y_test, knn.predict(X_test), average='macro')}")
    # =========================================
    print("=" * 20 + "\tTest of sklearn model model\t" + "=" * 20)
    knn_sklearn = KNeighborsClassifier(n_neighbors=5)
    knn_sklearn.fit(X_train, y_train)
    print(f"F1 score of sklearn model is: {f1_score(y_test, knn_sklearn.predict(X_test), average='macro')}")
