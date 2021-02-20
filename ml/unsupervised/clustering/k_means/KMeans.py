import random
import numpy as np
from sklearn.cluster import KMeans as sklearnKMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics


class KMeans:
    """
    KMeans Clustering is a unsupervised machine learning algorithm. This algorithm tries to find the k number of cluster
    in the dataset. Where value of k is user defined parameter.
    - The "cluster center" is the arithmetic mean of all the points belonging to the cluster.
    - Each point is closer to its own cluster center than to other cluster centers.
    The main intuition of this algorithm is:
        - Randomly choose k points in the dataset ( initial cluster center )
        - Repeat until convergence:
            - Assign all the training data point to it's nearest cluster
            - Update a single cluster centroid by taking the mean of all the points in that cluster
    K-means algorithm is one of the fastest clustering algorithms. But it falls into local minima depending on the
    initial cluster center point. For that reason it's better to rerun the whole algorithms several times to find it's
    global minima. This also doesn't guarantee the global minimum.
    """
    def __init__(self, n_clusters, init="random", n_init=10, num_iter=300, tol=1e-4, random_state=None):
        """

        :param n_clusters: Number of cluster
        :param init: Initial cluster initialization method
        :param n_init: number of times to rerun the algorithm to find the global minimum
        :param num_iter: Maximum number of iteration for convergence
        :param tol: Minimum distance between the centroid to consider that as converged
        :param random_state: random seed for random case for reproducibility
        """
        self.x_train = None
        self.n_cluster = n_clusters
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.num_iter = num_iter
        self.random_state = random_state

        # Attributes
        self.inertia_ = None
        self.cluster_centers_ = None
        self.n_iter_ = None
        self.labels_ = None

    def __get_initial_centroid(self, x_train, seed=None):
        """
        Returns initial centroid points of the clusters
        :param x_train:
        :return: random cluster points
        """
        if self.init == "random":
            # randomly select n_cluster point from the input dataset
            if seed:
                random.seed(seed)
            return np.asarray(random.choices(x_train, k=self.n_cluster))

    @staticmethod
    def __get_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """
        return the euclidean distance between two points
        :param point1: coordinate of point 1
        :param point2: co ordinate of point 2
        :return:
        """
        return np.sqrt(np.sum(np.square(point1 - point2)))

    def __get_cluster_centroid_distance(self, single_training: np.ndarray, cluster_center: np.ndarray) -> (int, float):
        """
        Return on which cluster an individual training points belongs to and distance of that point to it's nearest
        cluster center.
        :param single_training:
        :param cluster_center:
        :return:
        """
        training_label, training_distance = None, float('inf')
        # Check the distance of this point from all the cluster point.
        # This training point belongs to a cluster, which ever cluster centroid have the lowest distance from this point
        for cluster_label, single_cluster in enumerate(cluster_center):
            # Distance from the this training point to this cluster centroid
            this_distance = self.__get_distance(single_cluster, single_training)
            if this_distance < training_distance:
                training_label = cluster_label
                training_distance = this_distance
        return training_label, training_distance

    def __get_inertia_label(self, x_train: np.ndarray, cluster_center: np.ndarray) -> (np.ndarray, float):
        """
        Get label for each training point. This label doesn't mean the actual label. The labels are created based on the
        clusters.
        Inertia is the summation of the distance of a single training point to it's nearest centroid(it's label)
        :param x_train: Training data
        :param cluster_center: centroid of the cluster
        :return:
        """
        inertia = 0
        labels = []
        for single_training in x_train:
            # check distance with all the points and get the minimum distance
            label, distance = self.__get_cluster_centroid_distance(single_training, cluster_center)
            labels.append(label)
            inertia += distance
        return np.asarray(labels), inertia

    def __fit(self, x_train: np.ndarray, seed: int) -> (float, np.ndarray, int, np.ndarray):
        """
        Train the model
        :param x_train:
        :return:
        """
        # add the seed to the random for reproducibility
        # Adding the value i so that random choice doesn't return same random cluster point all the time
        if seed:
            random.seed(seed)
        # Random cluster point
        cluster_center = self.__get_initial_centroid(x_train, seed=seed)
        # Keep iterating until convergence
        for i in range(self.num_iter):

            # move centroid as the average of the labels
            labels, inertia = self.__get_inertia_label(x_train, cluster_center)
            new_cluster_center = []
            for centroid_label in range(self.n_cluster):
                # Update the centroid of the cluster.
                # new cluster = average of all the points in that clusters
                # if there is no training data in this cluster then choose another random cluster centroid
                if np.any(labels == centroid_label):
                    new_cluster_center.append(np.mean(x_train[labels == centroid_label], axis=0))
                else:
                    new_cluster_center.append(random.choice(x_train))
            new_cluster_center = np.asarray(new_cluster_center)
            # check if converged or not
            if np.sum(np.abs(new_cluster_center - cluster_center)) < self.tol:
                # Stop training if reach convergence
                break
            cluster_center = new_cluster_center
        return inertia, cluster_center, i, labels

    def fit(self, x_train):
        # restart the training n_init times. To reach the global minimum instead of local minimum. Though it's not
        # guaranteed to reach global minimum after n_init iteration.
        for i in range(self.n_init):
            seed = self.random_state + i if self.random_state else None
            inertia, cluster_center, n_iter, labels = self.__fit(x_train, seed)
            # If inertia is less then previous inertia then keep this model instead of the previous model
            if not self.inertia_ or inertia < self.inertia_:
                self.inertia_ = inertia
                self.cluster_centers_ = cluster_center
                self.n_iter_ = n_iter
                self.labels_ = labels


if __name__ == '__main__':
    X_train, y_train = load_digits(return_X_y=True)
    X_train = scale(X_train)

    # =========================================
    print("=" * 20 + "\tTest of our model\t" + "=" * 20)
    KMeans = KMeans(n_clusters=10, n_init=10, random_state=50)
    KMeans.fit(X_train)
    print(f"V measure of our model is: {metrics.v_measure_score(y_train, KMeans.labels_)}")

    # =========================================
    print("=" * 20 + "\tTest of Sklearn model\t" + "=" * 20)
    sklearnKMeans = sklearnKMeans(n_clusters=10, init="random", n_init=10, max_iter=300, verbose=0, random_state=50)
    sklearnKMeans.fit(X_train)
    print(f"V measure of sklearn model is: {metrics.v_measure_score(y_train, sklearnKMeans.labels_)}")
