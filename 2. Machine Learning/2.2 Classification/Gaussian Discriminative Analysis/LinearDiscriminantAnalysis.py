import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.stats import multivariate_normal


class GaussianDiscriminantAnalysis:
    """
    Linear Discriminant analysis is one of the gaussian generative analysis. Which fits alpha gaussian for each of the
    training feature.
    """

    def __init__(self):
        """
        phi - class probability of shape (Number of output class, ). Its the measure of knowing the likelihood of any class before seeing the input data.
        sigma - covariance matrix of input feature of shape (number of output class, number of input feature, number of input feature)
        mu - Mean for each of the output class of shape(Number of output class, number of input feature).
        """
        self.phi = None
        self.sigma = None
        self.mu = None

    def fit(self, x_train, y_train):
        m = y_train.shape[0]  # Number of training example
        # Reshaping the training set
        x_train = x_train.reshape(m, -1)
        # Number of input feature.
        input_feature = x_train.shape[1]
        # Number of output class
        class_label = len(np.unique(y_train.reshape(-1)))

        # Start everything with zero first.
        # Mean for each class. Each row contains an individual class.
        self.mu = np.zeros((class_label, input_feature))
        # Each row will contains the covariance matrix of each class.
        # The covariance matrix is alpha square symmetric matrix.
        # It indicates how each of the input feature varies with each other.
        self.sigma = np.zeros((class_label, input_feature, input_feature))
        # Prior probability of each class.
        # Its the measure of knowing the likelihood of any class before seeing the input data.
        self.phi = np.zeros(class_label)

        for label in range(class_label):
            # Separate all the training data for alpha single class
            indices = (y_train == label)
            self.phi[label] = float(np.sum(indices)) / m
            self.mu[label] = np.mean(x_train[indices, :], axis=0)
            self.sigma[label] = np.cov(x_train[indices, :], rowvar=0)

    def predict(self, x_tests):
        # flatten the training data
        x_tests = x_tests.reshape(x_tests.shape[0], -1)
        class_label = self.mu.shape[0]
        # Initially we set the each class probability to zero.
        scores = np.zeros((x_tests.shape[0], class_label))
        # We will calculate the probability for each of the class.
        for label in range(class_label):
            # normal_distribution_prob.logpdf Will give us the log value of the PDF that we just mentioned above.
            normal_distribution_prob = multivariate_normal(mean=self.mu[label], cov=self.sigma[label])
            # x_test can have multiple test data we will calculate the probability of each of the test data
            for i, x in enumerate(x_tests):
                scores[i, label] = np.log(self.phi[label]) + normal_distribution_prob.logpdf(x)
        predictions = np.argmax(scores, axis=1)
        return predictions


if __name__ == '__main__':
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target)
    GaussianDiscriminantAnalysis = GaussianDiscriminantAnalysis()
    GaussianDiscriminantAnalysis.fit(x_train, y_train)
    y_predict = GaussianDiscriminantAnalysis.predict(x_test)
    score = f1_score(y_test, y_predict, average="weighted")
    print("f1 score of our model: ", score)

    # Test scikit learn model
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    y_predict_sk = lda.predict(x_test)
    print("f1 score of scikit-learn model is: ", f1_score(y_test, y_predict_sk, average="weighted"))



