import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from KNNClassifier import KNNClassifier


class KNNRegressor(KNNClassifier):
    """
    K nearest neighbor as a regressor
    """

    def get_label(self, neighbor_points):
        """
        Nearest neighbor regression and classification works almost the same way.
        Only different will be the way to get our label.
        This method we will extract the labels first from our neighbors and take their mean as the label of this point
        :param neighbor_points:
        :return:
        """
        labels = np.asarray(neighbor_points)[:, -1]
        label = np.mean(labels)
        return label


if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # =========================================
    print("=" * 20 + "\tTest of our model\t" + "=" * 20)
    knn = KNNRegressor(n_neighbors=10)
    knn.fit(X_train, y_train)
    print(f"F1 score of our model is: {r2_score(y_test, knn.predict(X_test))}")
    # =========================================
    print("=" * 20 + "\tTest of sklearn model model\t" + "=" * 20)
    knn_sklearn = KNeighborsRegressor(n_neighbors=10)
    knn_sklearn.fit(X_train, y_train)
    print(f"F1 score of sklearn model is: {r2_score(y_test, knn_sklearn.predict(X_test))}")
