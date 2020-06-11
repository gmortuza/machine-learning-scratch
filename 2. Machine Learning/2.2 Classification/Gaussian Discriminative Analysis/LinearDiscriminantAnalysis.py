import numpy as np
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.stats import multivariate_normal


class LinearDiscriminantAnalysis:
    """
    Linear Discriminant analysis is one of the gaussian generative analysis. Which fits a gaussian for each of the
    training feature.
    """

    def __init__(self):
        """

        """
        self.phi = None
        self.sigma = None
        self.mu = None

    def fit(self, x_train, y_train):
        m = y_train.shape[0]
        input_feature = x_train.shape[1]
        class_label = len(np.unique(y_train.reshape(-1)))

        self.mu = np.zeros((class_label, input_feature))
        self.sigma = np.zeros((class_label, input_feature, input_feature))
        self.phi = np.zeros(class_label)

        for label in range(class_label):
            indices = (y_train == label)
            self.phi[label] = float(np.sum(indices)) / m
            self.mu[label] = np.mean(x_train[indices, :], axis=0)
            self.sigma[label] = np.cov(x_train[indices, :], rowvar=0)


    def predict(self, x_test):
        class_label = self.mu.shape[0]
        scores = np.zeros((x_test.shape[0], class_label))
        for label in range(class_label):
            normal_distribution_prob = multivariate_normal(mean=self.mu[label], cov=self.sigma[label])
            for i, x in enumerate(x_test):
                scores[i, label] = np.log(self.phi[label]) + normal_distribution_prob.logpdf(x)
        predictions = np.argmax(scores, axis=1)
        return predictions


if __name__ == '__main__':
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target)
    # y_test = y_test.reshape(-1, 1)
    # y_train = y_train.reshape(-1, 1)
    LinearDiscriminantAnalysis = LinearDiscriminantAnalysis()
    LinearDiscriminantAnalysis.fit(x_train, y_train)
    y_predict = LinearDiscriminantAnalysis.predict(x_test)
    score = f1_score(y_test, y_predict, average="weighted")

