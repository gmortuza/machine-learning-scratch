from collections import Counter
from models.tree.decision_tree.decision_tree import DecisionTree
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class RandomForest(DecisionTree):
    def __init__(self, n_estimator=100, criterion="gini", max_depth=3, min_leaf_size=4, max_sample=.5):
        """
        Random forest is a supervised learning algorithm. Some week estimator works together to build a strong
        estimator.
        :param n_estimator: Total number of tree that will be created
        :param criterion: The loss function that will be used. gini/entropy
        :param max_depth: Maximum depth of each tree.
        :param min_leaf_size: Minimum number of data in any individual tree nodes
        :param max_sample: Number of training example that will be used to train an individual tree. If float
                            value is passed then it will use the percentages of the overall training example.
                            If integer value is passed that number of training example will be used
        """
        # Total number of trees for our random forest
        self.n_estimator = n_estimator
        # This will contain the list of the decision tree for the random forest
        self.trees = []
        # number/percentage of sample that will be used to build a single tree
        self.max_sample = max_sample
        # parameters for decision tree
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size

    def fit(self, X_train, y_train):
        """
        Train the model
        :param X_train: Training data
        :param y_train: Training label
        :return:
        """
        # m --> # number of training example
        # n_feature --> number of input feature
        m, n_feature = X_train.shape
        # if user pass max_sample as a float value then it means that it's an percentage value
        # otherwise it is an exact amount each tree will be trained on
        sample = self.max_sample if isinstance(self.max_sample, int) else int(self.max_sample * m)
        for _ in range(self.n_estimator):
            # Bootstrapping --> get # sample from training set with replacement
            indices = np.random.choice(m, sample, replace=True)
            individual_train_x = X_train[indices, :]
            individual_train_y = y_train[indices]
            # Do not use all th feature use only a few feature at any time.
            # This makes sure that even if the dataset matches in some weak estimator
            # they don't calculate the same thing.
            # Let's say we will use 40% of the total feature
            # we could tune this 40% by analyzing out of bag error
            # For now to keep this simple let's implement without tuning that
            n_feature_to_be_used = int(n_feature*.5)
            # Get the random feature with which this tree will be trained on
            feature_indices = np.random.choice(n_feature, n_feature_to_be_used, replace=False)
            individual_train_x = individual_train_x[:, feature_indices]
            # make the individual tree/estimator
            single_estimator = DecisionTree(self.criterion, self.max_depth, self.min_leaf_size)
            # Train that tree with bootstrapped dataset
            single_estimator.fit(individual_train_x, individual_train_y)
            # save the trained tree
            self.trees.append([single_estimator, feature_indices])

    def predict(self, X_test):
        """
        Make prediction
        :param X_test: Data to predict
        :return:
        """
        # Will contain prediction from all the trees
        all_prediction = []
        for tree in self.trees:
            single_estimator, feature_indices = tree[0], tree[1]
            # Extract the feature with which that tree was trained on
            X_data = X_test[:, feature_indices]
            # Make the prediction for a single tree
            single_prediction = single_estimator.predict(X_data)
            # Add the prediction
            all_prediction.append(single_prediction)
        # Use the majority voting to make final prediction
        return [Counter(col).most_common(1)[0][0] for col in zip(*all_prediction)]


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rand_forest = RandomForest()
    rand_forest.fit(X_train, y_train)
    print(f"Training F1 score of our model is : {f1_score(y_train, rand_forest.predict(X_train), average='micro')}")
    print(f"Testing F1 score of our model is : {f1_score(y_test, rand_forest.predict(X_test), average='micro')}")
    # Calculate the accuracy for sk-learn model
    sk_learn_rand_forest = RandomForestClassifier()
    sk_learn_rand_forest.fit(X_train, y_train)
    print(f"Training F1 score of sk-learn model is : {f1_score(y_train, sk_learn_rand_forest.predict(X_train), average='micro')}")
    print(f"Testing F1 score of sk-learn model is : {f1_score(y_test, sk_learn_rand_forest.predict(X_test), average='micro')}")
