import numpy as np
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


class NaiveBayes:
    """
    Naive Bayes implementation based on Multi-variate Bernoulli using python
    """
    def __init__(self):
        """
        self.class_probability --> Class probability of shape (output_label, ). It indicates the probability of a label appearing
                       without seeing the input data.
        self.phi --> Probability of a input feature given a output label. P(x|y). shape (output_label, input_feature)
        self.output_label --> Number of output class
        self.input_feature --> Number of input feature
        """
        self.class_probability = None
        self.phi = None
        self.output_label = None

    def fit(self, x_train, y_train):
        """
        Train the model
        :param x_train: Input training example of shape (number of training data, input feature)
        :param y_train: Output training example of shape (number of training data, )
        :return:
        """
        m = x_train.shape[0]  # Number of training example
        # Flatten the training set
        x_train = x_train.reshape(m, -1)
        input_features = x_train.shape[1]
        self.output_label = len(np.unique(y_train.reshape(-1)))
        # Initialize everything with zero
        self.class_probability = np.zeros(self.output_label)
        self.phi = np.zeros((self.output_label, input_features))
        # Calculate class probability and phi
        for label in range(self.output_label):
            # Extract the training data from an individual labels
            current_label_data = x_train[y_train == label]
            # Number of occurarances of this particular label in the training set
            current_label_occur = current_label_data.shape[0]
            # Class label of this training data
            self.class_probability[label] = (current_label_occur + 1) / (m + self.output_label)
            # Calculate phi for an individual label
            # How many times each of the input feature appeared for this label
            # One is added for laplace smoothing
            input_feature_occur = np.sum(current_label_data, axis=0) + 1
            # Fix the denominator according to the laplace smoothing
            curr_label_laplace_smoothing = current_label_occur + self.output_label
            # Calculate phi
            self.phi[label, :] = input_feature_occur / curr_label_laplace_smoothing

    def predict(self, x_test):
        """
        Make prediction
        :param x_test: data to predict of shape (number of prediction, input feature)
        :return:
        """
        # Number of prediction
        num_of_test = x_test.shape[0]
        # Probability of each of the class.
        # Initially each of the label will have zero probability
        probabilities = np.zeros((num_of_test, self.output_label))
        # Calculate for all test
        for test_index in range(num_of_test):
            # Count probabilities for each of the classes
            for label in range(self.output_label):
                # First get all the words present in this test example
                words_for_this_example = x_test[test_index] == 1
                # Get the calculated probabilities for this label and this ese words example
                words_probabilities = self.phi[label][words_for_this_example]
                # Multiply all these probability
                words_probability_multiply = np.prod(words_probabilities)
                # Multiply this with class_probability probabilities/class probabilities
                # to get the overall probability of this example
                probabilities[test_index, label] = words_probability_multiply * self.class_probability[label]
            # Normalize the probabilities
            probabilities[test_index] /= np.sum(probabilities[test_index])
        # return the maximum probability index
        return np.argmax(probabilities, axis=1)

    def get_f1_score(self, x_test, y_test):
        """
        Calculate the f1 score of our model
        :param x_test:
        :param y_test:
        :return:
        """
        return f1_score(y_test, self.predict(x_test))


if __name__ == '__main__':
    # Test our model with amazon cell review dataset.
    # Where model will predict if a review is positive or negetive
    # Read the dataset
    with open("amazon_cells_labelled.txt", "r") as file:
        # This will contain all the sentences
        sentences = []
        # This will contain all the output label(0, 1)
        labels = []
        for line in file.readlines():
            # The label and sentences are separated with a tab
            line_arr = line.strip().split("\t")
            # Remove stop words
            sentences.append(line_arr[0])
            labels.append(int(line_arr[1]))

    # Vectorize the training sentences
    vectorizer = CountVectorizer(analyzer="word", lowercase=True, stop_words="english", max_features=4500)
    data = vectorizer.fit_transform(sentences).toarray()

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(data, np.array(labels))

    # Fit to our model
    naive_bayes = NaiveBayes()
    naive_bayes.fit(x_train, y_train)
    model_f1_score = naive_bayes.get_f1_score(x_test, y_test)
    print("F1 score of test set of our model is : ", str(model_f1_score))

    # Compare with scikit-learn model
    sci_naive_bayes = BernoulliNB()
    sci_naive_bayes.fit(x_train, y_train)
    sk_prediction = sci_naive_bayes.predict(x_test)
    print("F1 score of the test set of scikit learn model is : ", str(f1_score(y_test, sk_prediction)))


