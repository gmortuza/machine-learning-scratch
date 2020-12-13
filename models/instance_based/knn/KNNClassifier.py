import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import heapq


class KNNClassifier:
    def __init__(self, n_neighbors=5, algorithm="bruteforce", metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.x_train = None
        self.y_train = None
        self.distance = self.__get_distance(metric)

    def __get_distance(self, metric):
        if metric == "manhattan":
            return self.__manhattan_distance
        else:
            return self.__euclidean_distance

    @staticmethod
    def __manhattan_distance(point1, point2):
        total_distance = 0
        for i, j in zip(point1, point2):
            total_distance += abs(i - j)
        return total_distance

    @staticmethod
    def __euclidean_distance(point2):
        def measure_distance(point1):
            total_distance = 0
            for i, j in zip(point1, point2):
                total_distance += (i - j) ** 2
            return total_distance ** .5
        return measure_distance

    def fit(self, X_train, y_train):
        self.x_train = X_train
        self.y_train = np.expand_dims(y_train, axis=1)

    @staticmethod
    def __get_label(neighbor_points):
        labels = np.asarray(neighbor_points)[:, -1].astype(int)
        label = np.argmax(np.bincount(labels))
        return label

    def predict(self, X_test):
        data_with_label = np.concatenate((self.x_train, self.y_train), axis=1)
        # Check distance in each position
        prediction = []
        for single_test in X_test:
            # make prediction for this point only
            neighbor_points = heapq.nsmallest(self.n_neighbors, data_with_label, key=self.distance(single_test))
            # extract label
            label = self.__get_label(neighbor_points)
            prediction.append(label)
        return np.asarray(prediction).astype(int)


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
    print(f"F1 score of our model is: {f1_score(y_test, knn_sklearn.predict(X_test), average='macro')}")